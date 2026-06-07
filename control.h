// control.h -- independent high-rate auto-tracking controller.
//
// Decoupled from OBS and from the detection pipeline: the host feeds it the
// latest detections (publish) and live config (set_config), and supplies a
// send-delta callback. The controller owns its own ~125 Hz timer thread and
// drives the device with RELATIVE eased moves (ported from the reference
// aim_control::snap_to). It knows nothing about OBS, ONNX or the device DLL.
#pragma once

#include <atomic>
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
	float speed_x      = 1.0f, speed_y = 1.0f;
	bool  pov_enable   = false;
	int   pov_w        = 600, pov_h = 600;
	float switch_ratio = 0.80f;  // lock hysteresis: rival must be this much nearer
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
};
