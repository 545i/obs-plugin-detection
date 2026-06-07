// control.cpp -- independent high-rate auto-tracking controller.
// See control.h. Faithful port of the reference aim_control::snap_to, with the
// only change being the actuator: the reference does SendInput absolute to
// (cursor + dx*ease); here we send the SAME eased vector as a RELATIVE device
// move (sendDelta), which is what works for in-game mouse-look.
#include "control.h"

#include <obs-module.h>  // blog (temporary diagnostics)
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

// Win10 1803+ high-resolution waitable timer flag.
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif

static constexpr int kTickIntervalMs = 8;  // ~125 Hz control loop

Controller::Controller() = default;

Controller::~Controller()
{
	stop();
}

void Controller::set_send_fn(SendDeltaFn fn)
{
	send_fn_ = std::move(fn);
}

void Controller::set_config(const ControlConfig &cfg)
{
	std::lock_guard<std::mutex> lk(cfg_mtx_);
	cfg_ = cfg;
}

void Controller::publish(std::vector<ControlTarget> targets, int fw, int fh)
{
	std::lock_guard<std::mutex> lk(snap_mtx_);
	targets_ = std::move(targets);
	frame_w_ = fw;
	frame_h_ = fh;
}

void Controller::start()
{
	if (running_.exchange(true))
		return;  // already running
	stop_event_ = ::CreateEventW(NULL, TRUE, FALSE, NULL);
	thread_ = std::thread(&Controller::thread_main, this);
}

void Controller::stop()
{
	if (!running_.exchange(false)) {
		// not running, but a handle may linger if start() half-ran
		if (stop_event_) { ::CloseHandle((HANDLE)stop_event_); stop_event_ = nullptr; }
		return;
	}
	if (stop_event_)
		::SetEvent((HANDLE)stop_event_);
	if (thread_.joinable())
		thread_.join();
	if (stop_event_) {
		::CloseHandle((HANDLE)stop_event_);
		stop_event_ = nullptr;
	}
}

void Controller::thread_main()
{
	HANDLE stop  = (HANDLE)stop_event_;
	HANDLE timer = ::CreateWaitableTimerExW(
		NULL, NULL, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
	if (timer) {
		LARGE_INTEGER due;
		due.QuadPart = -1LL;  // first tick almost immediately
		::SetWaitableTimer(timer, &due, kTickIntervalMs, NULL, NULL, FALSE);
	}
	HANDLE objs[2] = { stop, timer };
	while (running_.load(std::memory_order_acquire)) {
		DWORD r = timer ? ::WaitForMultipleObjects(2, objs, FALSE, 50)
		                : ::WaitForSingleObject(stop, kTickIntervalMs);
		if (r == WAIT_OBJECT_0)  // stop event
			break;
		tick();
	}
	if (timer) { ::CancelWaitableTimer(timer); ::CloseHandle(timer); }
}

// One control iteration (control thread only).
void Controller::tick()
{
	static unsigned s_dbg = 0;
	const bool dbg = (++s_dbg % 120u) == 0;  // ~1.5s at 8ms ticks (temporary)

	ControlConfig cfg;
	{
		std::lock_guard<std::mutex> lk(cfg_mtx_);
		cfg = cfg_;
	}
	if (!cfg.enable) {
		if (dbg) blog(LOG_INFO, "[control] idle: enable=0");
		locked_id_ = -1; return;
	}

	// Fire/trigger key: -1 = always on, 0 = off, >0 = VK held.
	const bool key_held =
		(cfg.trigger_vk == -1) ? true
		: (cfg.trigger_vk > 0) ? ((::GetAsyncKeyState(cfg.trigger_vk) & 0x8000) != 0)
		                       : false;
	if (!key_held) {
		if (dbg) blog(LOG_INFO, "[control] idle: key not held (vk=%d)", cfg.trigger_vk);
		locked_id_ = -1; return;
	}

	std::vector<ControlTarget> cs;
	int fw, fh;
	{
		std::lock_guard<std::mutex> lk(snap_mtx_);
		cs = targets_;
		fw = frame_w_;
		fh = frame_h_;
	}
	if (cs.empty() || fw <= 0 || fh <= 0) {
		if (dbg) blog(LOG_INFO, "[control] idle: no targets (n=%zu fw=%d fh=%d)",
		              cs.size(), fw, fh);
		locked_id_ = -1; return;
	}

	const int sw = ::GetSystemMetrics(SM_CXSCREEN);
	const int sh = ::GetSystemMetrics(SM_CYSCREEN);
	if (sw <= 0 || sh <= 0) return;

	// POV gate (centered rect, source px) — outside-POV targets are not tracked.
	const float icx = fw * 0.5f, icy = fh * 0.5f;
	const bool  pov_on = cfg.pov_enable;
	const float pw = (float)std::min(cfg.pov_w, fw);
	const float ph = (float)std::min(cfg.pov_h, fh);
	const float pl = icx - pw * 0.5f, pr = icx + pw * 0.5f;
	const float pt = icy - ph * 0.5f, pb = icy + ph * 0.5f;
	auto in_pov = [&](const ControlTarget &c) {
		return !pov_on || (c.cx >= pl && c.cx <= pr && c.cy >= pt && c.cy <= pb);
	};

	// Target select: nearest eligible + locked, with hysteresis (lock keeps the
	// target unless a rival is clearly nearer -> no stealing by far targets).
	const ControlTarget *nearest = nullptr, *locked = nullptr;
	float nd = std::numeric_limits<float>::max();
	for (const ControlTarget &c : cs) {
		if (!in_pov(c)) continue;
		if (cfg.target_mode == 1 && c.cls != cfg.target_cls) continue;
		if (c.dist < nd) { nd = c.dist; nearest = &c; }
		if (locked_id_ >= 0 && c.id == locked_id_) locked = &c;
	}
	if (!nearest) {
		for (const ControlTarget &c : cs)
			if (in_pov(c) && c.dist < nd) { nd = c.dist; nearest = &c; }
		locked = nullptr;
	}
	const ControlTarget *sel = locked
		? ((nearest && nearest != locked &&
		    nearest->dist < locked->dist * cfg.switch_ratio) ? nearest : locked)
		: nearest;
	locked_id_ = sel ? sel->id : -1;
	if (!sel) {
		if (dbg) blog(LOG_INFO, "[control] idle: no eligible target (POV/class)");
		return;
	}

	// Error = target's offset from the CROSSHAIR (= fixed screen center), in
	// SCREEN px. NO GetCursorPos: in a cursor-locked game it doesn't reflect the
	// crosshair. Linear extrapolation (pos + vel*lead) predicts the target's
	// next on-screen position; because the detection re-measures the screen-space
	// offset every frame, the camera's own motion is auto-accounted (this is why
	// "relative velocity" no longer reverses -- we never use absolute velocity).
	const double sxr = (double)sw / (double)fw;
	const double syr = (double)sh / (double)fh;
	const double dx = (sel->track_dx + sel->track_vx * cfg.lead) * sxr;
	const double dy = (sel->track_dy + sel->track_vy * cfg.lead) * syr;

	// Engage radius (normalized 0-1) measured from the crosshair. 0 = always.
	const double dist = std::sqrt(dx * dx + dy * dy);
	if (cfg.snap_radius > 0.0f) {
		const double dnx = dx / sw, dny = dy / sh;
		if (std::sqrt(dnx * dnx + dny * dny) > cfg.snap_radius) {
			if (dbg) blog(LOG_INFO, "[control] idle: target outside snap_radius "
			              "(distn=%.3f > %.3f)",
			              std::sqrt(dnx*dnx+dny*dny), cfg.snap_radius);
			return;
		}
	}
	if (dist < 1.0) {
		if (dbg) blog(LOG_INFO, "[control] on-target: dist<1px, hold");
		return;
	}

	// ease-out cubic toward the target, per-axis speed.
	double t = std::clamp((double)cfg.smooth, 0.05, 1.0);
	const double omt = 1.0 - t;
	const double ease = 1.0 - omt * omt * omt;
	double mvx = dx * ease * (double)cfg.speed_x;
	double mvy = dy * ease * (double)cfg.speed_y;

	// Small humanizing jitter scaled by distance (as in the reference).
	const double js = std::min(1.0, dist / 200.0);
	mvx += ((double)(::rand() % 7) - 3.0) * js;
	mvy += ((double)(::rand() % 7) - 3.0) * js;

	mvx = std::clamp(mvx, -30000.0, 30000.0);
	mvy = std::clamp(mvy, -30000.0, 30000.0);
	const short sdx = (short)mvx, sdy = (short)mvy;
	if (sdx == 0 && sdy == 0) return;

	const bool sent = send_fn_ ? send_fn_(sdx, sdy) : false;
	if (dbg)
		blog(LOG_INFO, "[control] MOVE err=(%.0f,%.0f) -> send=(%d,%d) sent=%d "
		     "(fw=%d sw=%d smooth=%.2f)",
		     dx, dy, (int)sdx, (int)sdy, (int)sent, fw, sw, cfg.smooth);
}
