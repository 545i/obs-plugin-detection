// control.h -- independent high-rate auto-tracking controller.
//
// Decoupled from OBS and from the detection pipeline: the host feeds it the
// latest detections (publish) and live config (set_config), and supplies a
// send-delta callback. The controller owns its own ~125 Hz timer thread and
// drives the device with RELATIVE eased moves (ported from the reference
// aim_control::snap_to). It knows nothing about OBS, ONNX or the device DLL.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// One target candidate, in SOURCE pixels. (track_dx/dy already include the
// box-display-mode aim point, e.g. head offset, relative to the image center.)
struct ControlTarget {
	int   id   = -1;
	int   cls  = 0;
	float cx   = 0.0f, cy = 0.0f;            // box center (for the POV gate)
	float track_dx = 0.0f, track_dy = 0.0f;  // aim-point offset from image center
	float track_vx = 0.0f, track_vy = 0.0f;  // aim-point velocity, source px/sec
	float dist = 0.0f;                        // distance from image center
};

// Live controller config (mirrors the UI knobs). Copied under a lock.
struct ControlConfig {
	bool  enable       = false;
	int   trigger_vk   = -1;     // -1 = always on, 0 = off, >0 = VK held
	int   target_mode  = 0;      // 0 = nearest, 1 = by-class
	int   target_cls   = 0;
	float smooth       = 0.30f;  // ease-out factor 0..1 (higher = snappier)
	float snap_radius  = 0.0f;   // engage radius 0..1 (0 = whole frame)
	float lead         = 0.0f;   // linear extrapolation lead time (seconds)
	float max_step     = 0.0f;   // per-move clamp (device units); 0 = unlimited
	float speed_x      = 1.0f, speed_y = 1.0f;
	bool  pov_enable   = false;
	int   pov_w        = 600, pov_h = 600;
	float switch_ratio = 0.80f;  // lock hysteresis: rival must be this much nearer
	// Humanize (grounded in fnhum-16-979293: human aim = reaction delay + sigmoid
	// move = slow start -> peak mid -> slow end). On lock-acquire: wait react_ms,
	// then ease-in over accel_ms (the sigmoid's rising edge; ease-out comes free
	// as the error shrinks near the target).
	bool  humanize     = false;
	float react_ms     = 150.0f;  // reaction delay before engaging a new target
	float accel_ms     = 120.0f;  // ease-in (acceleration) ramp duration
	// Valorant-style sensitivity calibration. The device sends raw COUNTS
	// (measured 1:1 on the desktop), and the game turns 1 count into
	// sens*0.07 degrees of view rotation (DPI-independent; verified against pro
	// cm/360 ~= 45cm). So to bring a target that is N screen-px from the
	// crosshair onto it we send N*(fov_deg/screen_width)/(sens*0.07) counts.
	// Off -> the move is taken as 1 count = 1 screen px.
	bool  calib_enable = false;
	float game_sens    = 0.40f;   // in-game sensitivity (e.g. Valorant)
	float fov_deg      = 103.0f;  // HORIZONTAL FOV (Valorant 16:9 = 103)
	// Vertical FOV. 0 = auto (square pixels: focal_y = focal_x), correct for a
	// normal un-stretched render. A STRETCHED resolution (e.g. a 16:9 image forced
	// into a 21:9 screen) scales one axis by a fixed factor, making pixels
	// non-square -> the single-focal assumption breaks. Set the real vertical FOV
	// here to keep both axes exact under stretch.
	float fov_v_deg    = 0.0f;
};

class Controller {
public:
	// Sends a RELATIVE device move. Returns true if it actually went out.
	using SendDeltaFn = std::function<bool(short dx, short dy)>;

	Controller();
	~Controller();

	Controller(const Controller &) = delete;
	Controller &operator=(const Controller &) = delete;

	void set_send_fn(SendDeltaFn fn);  // set before start()
	void start();
	void stop();

	void set_config(const ControlConfig &cfg);                       // any thread
	void publish(std::vector<ControlTarget> targets, int fw, int fh); // host -> ctl

private:
	void thread_main();
	void tick();

	SendDeltaFn   send_fn_;

	std::mutex    cfg_mtx_;
	ControlConfig cfg_;

	std::mutex                 snap_mtx_;
	std::vector<ControlTarget> targets_;
	int frame_w_ = 0, frame_h_ = 0;

	std::thread       thread_;
	void             *stop_event_ = nullptr;  // Win32 HANDLE
	std::atomic<bool> running_{false};

	int locked_id_ = -1;  // control thread only
	int lock_prev_id_ = -1;  // for detecting a fresh lock (reaction-delay reset)
	std::chrono::steady_clock::time_point lock_time_;  // when current lock began
};
