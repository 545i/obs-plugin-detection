// ============================================================================
//  obs-detection-overlay.cpp  -- real object-detection overlay (OUTLINE-ONLY)
//
//  Pipeline:
//    1. video_render() passes the parent through the filter chain unchanged.
//    2. Every N frames (user "infer_interval"), the filter target is rendered
//       STRETCHED into a 640x640 GS_BGRA texrender on the GRAPHICS THREAD, then
//       gs_stage_texture() kicks off an async GPU->CPU copy into a staging surf.
//    3. On the NEXT video_render() (one-frame deferral) the staging surface is
//       mapped, BGRA rows copied into a shared buffer, and the worker signaled.
//    4. A single WORKER THREAD wakes, builds the CHW float tensor, runs the
//       model (Detector, ONNX Runtime / DirectML), decodes + NMS, stores
//       results (640-net-space).
//    5. video_render() copies the latest results out and draws them as GREEN
//       OUTLINE boxes -- no filled background.
//
//  Settings (live, via update()): model_path, conf_threshold, iou_threshold,
//  infer_interval. The MODEL is (re)loaded on the WORKER thread so neither the
//  UI nor the graphics thread blocks on the ~0.5s DirectML session init.
//
//  Coordinate mapping: the 640x640 render is a STRETCH (aspect squashed), so
//  detections are decoded in 640-net-space and scaled back per-axis:
//      sx = src_w / 640,  sy = src_h / 640.
//
//  Threading invariants (HARD):
//    - ALL gs_* calls happen ONLY on the graphics thread.
//    - The worker thread NEVER touches a gs_* object; it owns `detector` and only
//      consumes the copied std::vector and writes latest_results under a mutex.
//
//  get_width/get_height use obs_source_get_base_width/height on the TARGET;
//  obs_source_get_width would re-enter this filter and recurse (stack overflow).
// ============================================================================

#include <obs-module.h>
#include <graphics/graphics.h>
#include <graphics/matrix4.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "detector.hpp"               // class Detector, Backend
#include "detection_postprocess.hpp"  // struct Detection, decode(), nms(), coco_names
#include "tracker.hpp"                // BoxTracker (Kalman smoothing)
#include "device.h"                   // DeviceController (motion/gimbal device DLL)
#include "control.h"                  // Controller (independent auto-tracking loop)
// NOTE: device.h #includes "device_symbols.h" (the SYM_* export-name strings for
// the real device.dll). That file is supplied separately by the device owner and
// is REQUIRED to compile this plugin. Until it is present in this folder the
// build will fail at the device.h include — this is expected.

// ---------------------------------------------------------------------------
// OBS module entry points need C linkage in a C++ translation unit.
// ---------------------------------------------------------------------------
extern "C" {
OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-detection-overlay", "en-US")

MODULE_EXPORT const char *obs_module_description(void)
{
	return "Object detection box overlay filter";
}
}  // extern "C"

// ---------------------------------------------------------------------------
// Tunables / setting keys + defaults.
// ---------------------------------------------------------------------------
static constexpr uint32_t kNet = 640;             // network input is 640x640
static constexpr int      kNumAnchors = 8400;
static constexpr int      kNumClasses = 80;
static constexpr int      kIntraOpThreads = 2;
static constexpr Backend  kBackend = Backend::DML;  // GPU via DirectML

#define S_MODEL      "model_path"
#define S_CONF       "conf_threshold"
#define S_IOU        "iou_threshold"
#define S_INTERVAL   "infer_interval"
#define S_SMOOTH     "smoothing"
#define S_SOFT_NMS   "soft_nms"   // Soft-NMS: keep close-but-distinct targets apart
#define S_INFER_REGION "infer_region"  // centered SQUARE crop fed to the model (px)
#define S_ROI_ENABLE "roi_enable"
#define S_ROI_W      "roi_width"
#define S_ROI_H      "roi_height"
#define S_ROI_SHOW   "roi_show_box"  // draw the red ROI rectangle (visual only)
// POV: a centered rect that gates AUTO-TRACKING only. Targets whose center is
// OUTSIDE the POV are still detected and drawn, but are never selected for
// tracking (fixes "stealing a far target"). Distinct from ROI, which removes
// outside detections entirely.
#define S_POV_ENABLE "pov_enable"
#define S_POV_W      "pov_width"
#define S_POV_H      "pov_height"
#define S_POV_SHOW   "pov_show_box"   // draw the POV rectangle (visual only)
#define S_DEV_PATH   "device_path"    // device.dll path (UI file picker)
#define S_DEV_LOAD   "device_load"    // "load device" button
#define S_DEV_STATUS "device_status"  // connection-status label (OBS_TEXT_INFO)

// Auto-tracking settings
#define S_TRACK_ENABLE   "track_enable"
#define S_TRACK_TRIGGER  "track_trigger_key"  // VK code; 0=always active
#define S_TRACK_TARGET   "track_target_mode"  // 0=nearest, 1=by-class
#define S_TRACK_CLS      "track_target_cls"
#define S_TRACK_GAIN     "track_gain"
#define S_TRACK_KI       "track_ki"
#define S_TRACK_KD       "track_kd"
#define S_TRACK_DEADZONE "track_deadzone"
#define S_TRACK_INTERVAL "track_send_interval"
#define S_TRACK_SMOOTH   "track_smooth_ema"
#define S_TRACK_SPEED_X  "track_speed_x"      // per-axis output multiplier (X)
#define S_TRACK_SPEED_Y  "track_speed_y"      // per-axis output multiplier (Y)
#define S_TRACK_LEAD     "track_lead"         // velocity lead/prediction (seconds)

// Box display mode (ported from ScreenCapture::setBoxDisplayMode):
//   0 = head+body  : process ALL detections (== "head OR body" for the 2-class
//                    head/body model; also keeps a COCO model fully usable).
//   1 = head only  : process only the head class.
//   2 = head trk   : process only the head class AND move the auto-tracking
//                    point up onto the head (offset 0.11*h from the box top).
#define S_BOX_MODE     "box_display_mode"
#define S_CLS_HEAD     "cls_head"     // class index treated as HEAD (model-specific)
#define S_TRACK_OFFSET "track_offset" // mode-2 tracking point: frac of box height from top
#define S_OFFSET0       "offset_mode0" // independent aim offset for box mode 0
#define S_OFFSET1       "offset_mode1" // independent aim offset for box mode 1
#define S_OFFSET_PREVIEW "offset_preview" // draw the aim point on the overlay

static constexpr double kDefConf = 0.25;
static constexpr double kDefIou = 0.45;
static constexpr int    kDefInterval = 2;  // infer every 2nd render frame
static constexpr bool   kDefSmooth = true; // Kalman smoothing on by default
static constexpr bool   kDefSoftNms = true; // Soft-NMS on by default (anti-merge)
static constexpr int    kDefInferRegion = 640; // centered square crop side (px); 0=auto
static constexpr bool   kDefRoiEnable = false; // ROI gating off by default
static constexpr int    kDefRoiW = 500;    // ROI is a rect CENTERED on the source
static constexpr int    kDefRoiH = 500;    // center; width/height in SOURCE pixels
static constexpr bool   kDefRoiShow = true; // draw the red ROI box when gating on
static constexpr bool   kDefPovEnable = false; // POV tracking-gate off by default
static constexpr int    kDefPovW = 600;     // POV rect (centered) in SOURCE pixels
static constexpr int    kDefPovH = 600;
static constexpr bool   kDefPovShow = true;  // draw the POV box when gating on

// Auto-tracking defaults
static constexpr bool   kDefTrackEnable   = false;
// Trigger key sentinel values stored in S_TRACK_TRIGGER (int):
//   -1 = always-on / test mode (no key check, tracking runs every frame)
//    0 = none / disabled
//   >0 = VK code; device->queryState(vk) must be non-zero while held
static constexpr int    kDefTrackTrigger  = -1;    // default: always-on for easy first setup
static constexpr int    kDefTrackTarget   = 0;     // 0=nearest-to-center, 1=by-class
static constexpr int    kDefTrackCls      = 0;
static constexpr double kDefTrackGain     = 0.3;   // sensitivity: device units per pixel
static constexpr double kDefTrackKi       = 0.0;   // engage radius (0 = whole frame)
static constexpr double kDefTrackKd       = 150.0; // per-move clamp (device units); 0=unlimited
static constexpr double kDefTrackDeadzone = 5.0;
static constexpr int    kDefTrackInterval = 2;     // send every N rendered frames
static constexpr bool   kDefTrackSmooth   = false; // EMA smoothing on output
static constexpr double kDefTrackSpeedX   = 1.0;   // per-axis output multiplier (X)
static constexpr double kDefTrackSpeedY   = 1.0;   // per-axis output multiplier (Y)
static constexpr double kDefTrackLead     = 0.0;   // velocity lead/prediction (seconds);
                                                   // 0 by default -> integral does the work

static constexpr int    kDefBoxMode = 0;           // 0=head+body,1=head,2=head-track
static constexpr int    kDefClsHead = 0;           // head class index in the model
static constexpr double kDefTrackOffset = 0.11;    // mode-2 track point: frac of h from top
static constexpr double kDefOffset0 = 0.25;        // mode-0 aim offset (frac of h)
static constexpr double kDefOffset1 = 0.20;        // mode-1 aim offset (frac of h)
static constexpr bool   kDefOffsetPreview = false; // draw aim point on overlay

// Target-lock hysteresis: once locked onto a track, only switch to another
// target that is at least (1 - ratio) NEARER than the locked one. A farther
// target can therefore never steal the lock; switching happens only toward a
// clearly nearer target. 0.80 => a rival must be >=20% closer to take over.
static constexpr float  kTrackSwitchRatio = 0.80f;

// ---------------------------------------------------------------------------
// Box display mode helpers (ported from ScreenCapture).
// ---------------------------------------------------------------------------
// Whether a detection of class `cls` is processed (drawn + tracked) under `mode`.
// mode 0 keeps everything (== "head OR body" on the 2-class model, and keeps a
// COCO model fully usable); mode 1/2 keep only the head class.
static inline bool box_mode_should_process(int cls, int mode, int cls_head)
{
	return mode == 0 ? true : (cls == cls_head);
}

// Tracking point for the auto-tracker, in the box's own pixel space:
//   (x + w/2, y + offset*h)   -- offset is the per-box-mode value (0..1 of the
//   box height from the top). Each box display mode (0/1/2) has its own offset.
static inline void box_mode_track_point(float bx, float by, float bw, float bh,
                                        float offset, float &tx, float &ty)
{
	tx = bx + bw * 0.5f;
	ty = by + bh * offset;
}

// Centered SQUARE inference crop (ported from ScreenCapture::calculateCaptureRegion).
// Feeding the WHOLE source stretched into the square network input distorts aspect
// (people squashed) and shrinks targets -> poor detection AND more box merging.
// Instead we feed only a centered square of side `side` at native resolution.
//   region<=0 -> auto: side = min(cx,cy) (full height, aspect fixed, no zoom)
//   region>0  -> side = clamp(region, 64, min(cx,cy))  (zoom into the center)
// (x0,y0) is the crop's top-left in SOURCE pixels; detections map back via this.
static inline void infer_crop(int cx, int cy, int region,
                              int &side, int &x0, int &y0)
{
	const int m = (cx < cy) ? cx : cy;
	side = (region <= 0) ? m : std::min(region, m);
	if (side < 64) side = std::min(64, m);
	x0 = (cx - side) / 2;
	y0 = (cy - side) / 2;
}

// One detection box, in SOURCE/CANVAS pixel coordinates. (Outline color is set
// via the solid effect's "color" param at draw time, not stored per-box.)
struct box_px {
	float x, y, w, h;
};

// Center point of one detected object, in SOURCE pixel coordinates, plus its
// position relative to the SOURCE-IMAGE CENTER expressed as polar coordinates.
// Saved every frame into detector_filter::centers (see detector_get_centers()).
//
// Origin is ALWAYS the source-image center (src_w/2, src_h/2), so it adapts to
// any source resolution automatically. Angle convention: degrees, 0 deg = +X
// (right), +90 deg = UP (screen-y is down, so dy is negated), CCW-positive,
// range (-180, 180]. dist is in SOURCE pixels.
struct ObjCenter {
	int   id;           // stable track id (>=0 when smoothing on; -1 for raw)
	int   cls;          // COCO class index; name = coco_names[cls]
	float cx, cy;       // center, in source pixels (absolute)
	float dx, dy;       // center MINUS image-center, in source pixels
	float track_dx, track_dy; // box-mode tracking point MINUS image-center (px)
	float track_vx, track_vy; // target velocity at the tracking point (source px/sec)
	float angle;        // polar angle in degrees (see convention above)
	float dist;         // polar distance from image center, in source pixels
	float score;        // confidence
};

// ---------------------------------------------------------------------------
// Filter instance.
// ---------------------------------------------------------------------------
struct detector_filter {
	obs_source_t *context = nullptr;

	// ---- GPU readback objects (graphics thread ONLY) ----
	gs_texrender_t *texrender = nullptr;
	gs_stagesurf_t *stage     = nullptr;
	bool            stage_pending = false;
	uint64_t        frame_counter = 0;

	// ---- live settings ----
	std::atomic<float> conf_thresh{(float)kDefConf};
	std::atomic<float> iou_thresh{(float)kDefIou};
	std::atomic<int>   infer_interval{kDefInterval};  // read on graphics thread
	std::atomic<bool>  soft_nms{kDefSoftNms};         // read on worker thread
	std::atomic<int>   infer_region{kDefInferRegion}; // centered crop side (graphics)

	// ---- ROI gating (graphics thread): a rect CENTERED on the source image,
	// roi_w x roi_h in SOURCE pixels. Detections whose center falls outside are
	// fully removed (not drawn, not in centers). ----
	std::atomic<bool>  roi_enable{kDefRoiEnable};
	std::atomic<int>   roi_w{kDefRoiW};
	std::atomic<int>   roi_h{kDefRoiH};
	std::atomic<bool>  roi_show_box{kDefRoiShow};  // draw the red ROI rectangle

	// ---- POV gating (graphics thread): a centered rect that limits which
	// targets the AUTO-TRACKER may select. Outside-POV targets are still drawn. ----
	std::atomic<bool>  pov_enable{kDefPovEnable};
	std::atomic<int>   pov_w{kDefPovW};
	std::atomic<int>   pov_h{kDefPovH};
	std::atomic<bool>  pov_show_box{kDefPovShow};

	// ---- model (re)load request (worker owns the actual Detector) ----
	std::mutex        cfg_mtx;
	std::string       requested_model;   // guarded by cfg_mtx
	std::string       loaded_model;      // guarded by cfg_mtx
	std::atomic<bool> model_dirty{false};

	// ---- producer/consumer handoff (double buffered) ----
	std::mutex              in_mtx;
	std::condition_variable in_cv;
	std::vector<uint8_t>    in_buf;
	bool                    frame_ready = false;

	// ---- results (graphics thread reads, worker writes) ----
	std::mutex              res_mtx;
	std::vector<Detection>  latest_results;  // in 640-network-space
	std::atomic<uint64_t>   result_seq{0};   // bumped each time worker publishes

	// ---- SYNCHRONOUS inference handshake: the graphics thread submits a frame
	// then blocks (briefly) on out_cv until the worker publishes THIS frame's
	// result, so tracking uses a current-frame detection (minimal dead-time). ----
	std::mutex              out_mtx;
	std::condition_variable out_cv;
	std::atomic<double>     dbg_pipe_ms{0.0};  // measured acquire->ready latency

	// ---- model output shape, derived per loaded model (worker writes) ----
	std::atomic<int>  model_num_classes{kNumClasses}; // C-4 from the model output
	std::atomic<bool> out_shape_logged{false};        // log the shape once/model

	// ---- object centers (SOURCE pixels), refreshed every rendered frame.
	// Kept for the periodic log and detector_get_centers(); the auto-tracking
	// controller gets its own snapshot via controller.publish() each frame. ----
	std::mutex             centers_mtx;
	std::vector<ObjCenter> centers;  // guarded by centers_mtx

	// ---- Kalman smoothing (graphics thread ONLY: tick + render) ----
	std::atomic<bool> smoothing{kDefSmooth};
	BoxTracker        tracker;       // operates in 640-network-space
	uint64_t          tick_seq = 0;  // last result_seq consumed by the tracker

	// ---- worker thread (inference) ----
	std::thread        worker;
	std::atomic<bool>  running{false};

	// ---- independent auto-tracking controller (owns its own thread) ----
	Controller         controller;

	Detector *detector = nullptr;  // worker thread ONLY

	// ---- motion/gimbal device (connected on demand via the "Load device"
	// button in the UI; guarded by dev_mtx). null when not connected. ----
	std::mutex        dev_mtx;
	DeviceController *device = nullptr;   // owned; dtor releases + frees the DLL
	std::string       dev_loaded_path;    // path of the currently-open device
	std::string       dev_ui_path;        // latest path from the UI (filter_update)

	// ---- auto-tracking (graphics thread ONLY: driven from video_render) ----
	std::atomic<bool>  track_enable{kDefTrackEnable};
	std::atomic<int>   track_trigger_vk{kDefTrackTrigger}; // 0=always, else VK code
	std::atomic<int>   track_target_mode{kDefTrackTarget};
	std::atomic<int>   track_target_cls{kDefTrackCls};
	std::atomic<float> track_gain{(float)kDefTrackGain};
	std::atomic<float> track_ki{(float)kDefTrackKi};
	std::atomic<float> track_kd{(float)kDefTrackKd};
	std::atomic<float> track_deadzone{(float)kDefTrackDeadzone};
	std::atomic<int>   track_send_interval{kDefTrackInterval};
	std::atomic<bool>  track_smooth_ema{kDefTrackSmooth};
	std::atomic<float> track_speed_x{(float)kDefTrackSpeedX};
	std::atomic<float> track_speed_y{(float)kDefTrackSpeedY};
	std::atomic<float> track_lead{(float)kDefTrackLead};

	// ---- box display mode (graphics thread): class filtering + track point ----
	std::atomic<int>   box_mode{kDefBoxMode};
	std::atomic<int>   cls_head{kDefClsHead};
	std::atomic<float> track_offset{(float)kDefTrackOffset};  // mode 2
	std::atomic<float> offset0{(float)kDefOffset0};           // mode 0
	std::atomic<float> offset1{(float)kDefOffset1};           // mode 1
	std::atomic<bool>  offset_preview{kDefOffsetPreview};
};

// One filled axis-aligned quad (two triangles) in pixel space.
static void draw_filled_rect(float x, float y, float w, float h)
{
	gs_render_start(true);
	gs_vertex2f(x,     y);
	gs_vertex2f(x + w, y);
	gs_vertex2f(x,     y + h);
	gs_vertex2f(x + w, y + h);
	gs_render_stop(GS_TRISTRIP);
}

// Thick rectangle OUTLINE as four edge quads (hollow frame, empty interior).
static void draw_box_outline(const box_px *b, float t)
{
	const float x = b->x, y = b->y, w = b->w, h = b->h;
	draw_filled_rect(x,        y,        w, t);  // top
	draw_filled_rect(x,        y + h - t, w, t);  // bottom
	draw_filled_rect(x,        y,        t, h);  // left
	draw_filled_rect(x + w - t, y,        t, h);  // right
}

// ===========================================================================
//  Worker thread: (re)load model on request, then wait -> infer -> publish.
//  NEVER touches a gs_* object.
// ===========================================================================
static void inference_worker(detector_filter *f)
{
	std::vector<uint8_t> local_bgra;
	std::vector<float>   chw(size_t(3) * kNet * kNet);

	while (f->running.load(std::memory_order_acquire)) {
		bool have_frame = false;
		{
			std::unique_lock<std::mutex> lk(f->in_mtx);
			f->in_cv.wait(lk, [f] {
				return f->frame_ready ||
				       f->model_dirty.load(std::memory_order_acquire) ||
				       !f->running.load(std::memory_order_acquire);
			});
			if (!f->running.load(std::memory_order_acquire))
				break;
			have_frame = f->frame_ready;
			if (have_frame) {
				std::swap(local_bgra, f->in_buf);
				f->frame_ready = false;
			}
		}

		// --- model (re)load (worker thread; no UI/graphics stall) ---
		if (f->model_dirty.exchange(false, std::memory_order_acq_rel)) {
			std::string path;
			{
				std::lock_guard<std::mutex> lk(f->cfg_mtx);
				path = f->requested_model;
			}
			if (!path.empty()) {
				try {
					Detector *nw = new Detector(path, kIntraOpThreads,
					                          kBackend);
					delete f->detector;
					f->detector = nw;
					f->out_shape_logged.store(false);  // re-log shape
					{
						std::lock_guard<std::mutex> lk(f->cfg_mtx);
						f->loaded_model = path;
					}
					blog(LOG_INFO,
					     "[obs-detection-overlay] model loaded: %s (EP=%s)",
					     path.c_str(), f->detector->provider().c_str());
					// stale detections from the previous model
					std::lock_guard<std::mutex> lk(f->res_mtx);
					f->latest_results.clear();
				} catch (const std::exception &e) {
					blog(LOG_ERROR,
					     "[obs-detection-overlay] failed to load model '%s': %s",
					     path.c_str(), e.what());
				}
			}
		}

		if (!have_frame)
			continue;
		if (!f->detector || local_bgra.size() != size_t(kNet) * kNet * 4)
			continue;

		// --- build CHW float tensor (BGRA640 -> RGB, /255, HWC->CHW) ---
		// 0..255 -> 0..1 via a LUT (exact: i/255), avoiding ~1.2M per-frame
		// float divisions. Built once on first use.
		static const std::array<float, 256> kU8ToUnit = [] {
			std::array<float, 256> t{};
			for (int i = 0; i < 256; ++i)
				t[(size_t)i] = (float)i / 255.0f;
			return t;
		}();
		const size_t plane = size_t(kNet) * kNet;
		for (size_t i = 0; i < plane; ++i) {
			const uint8_t *p = &local_bgra[i * 4];  // BGRA
			chw[0 * plane + i] = kU8ToUnit[p[2]];   // R
			chw[1 * plane + i] = kU8ToUnit[p[1]];   // G
			chw[2 * plane + i] = kU8ToUnit[p[0]];   // B
		}

		// --- inference + decode/nms (live conf/iou thresholds) ---
		// Output shape drives everything: [1, C, A] (channel-major, YOLOv8) or
		// [1, A, C] (anchor-major). num_classes = C - 4 — so any class count
		// (COCO-80, a 2-class body/head model, ...) works automatically.
		std::vector<Detection> dets;
		try {
			Detector::Output out = f->detector->run(chw.data());
			if (out.data && out.shape.size() == 3 && out.shape[0] == 1) {
				const int64_t d1 = out.shape[1], d2 = out.shape[2];
				const bool channel_major = d1 <= d2;  // C is the smaller dim
				const int  C  = (int)(channel_major ? d1 : d2);
				const int  A  = (int)(channel_major ? d2 : d1);
				const int  nc = C - 4;

				if (!f->out_shape_logged.exchange(true)) {
					blog(LOG_INFO,
					     "[obs-detection-overlay] model output [1,%lld,%lld] -> "
					     "classes=%d anchors=%d (%s)",
					     (long long)d1, (long long)d2, nc, A,
					     channel_major ? "channel-major" : "anchor-major");
				}

				if (nc >= 1 && A > 0 &&
				    out.element_count >= size_t(C) * A) {
					f->model_num_classes.store(nc);
					const float conf = f->conf_thresh.load();
					dets = decode(out.data, A, nc, conf,
					              1.0f, 0.0f, 0.0f, (int)kNet, (int)kNet,
					              channel_major);
					// Soft-NMS keeps close-but-distinct targets from merging;
					// classic hard-NMS as a fallback when the toggle is off.
					dets = f->soft_nms.load()
					           ? soft_nms(dets, f->iou_thresh.load(), conf)
					           : nms(dets, f->iou_thresh.load());
				} else if (!f->out_shape_logged.load()) {
					blog(LOG_WARNING,
					     "[obs-detection-overlay] unsupported output shape");
				}
			}
		} catch (const std::exception &e) {
			blog(LOG_ERROR, "[obs-detection-overlay] inference failed: %s",
			     e.what());
		}

		{
			std::lock_guard<std::mutex> lk(f->res_mtx);
			f->latest_results.swap(dets);
		}
		f->result_seq.fetch_add(1, std::memory_order_release);
		// Wake a graphics thread that is blocked waiting for THIS result
		// (synchronous inference path).
		{
			std::lock_guard<std::mutex> lk(f->out_mtx);
		}
		f->out_cv.notify_all();
	}
}

static const char *filter_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return obs_module_text("DetectionOverlayFilter");
}

static ControlConfig build_control_config(detector_filter *f);  // defined below

// Read live settings and request a model reload if the path changed.
static void filter_update(void *data, obs_data_t *s)
{
	auto *f = static_cast<detector_filter *>(data);

	f->conf_thresh.store((float)obs_data_get_double(s, S_CONF));
	f->iou_thresh.store((float)obs_data_get_double(s, S_IOU));
	int iv = (int)obs_data_get_int(s, S_INTERVAL);
	f->infer_interval.store(iv < 1 ? 1 : iv);
	f->smoothing.store(obs_data_get_bool(s, S_SMOOTH));
	f->soft_nms.store(obs_data_get_bool(s, S_SOFT_NMS));
	f->infer_region.store((int)obs_data_get_int(s, S_INFER_REGION));

	f->roi_enable.store(obs_data_get_bool(s, S_ROI_ENABLE));
	f->roi_show_box.store(obs_data_get_bool(s, S_ROI_SHOW));
	int rw = (int)obs_data_get_int(s, S_ROI_W);
	int rh = (int)obs_data_get_int(s, S_ROI_H);
	f->roi_w.store(rw < 2 ? 2 : rw);
	f->roi_h.store(rh < 2 ? 2 : rh);

	f->pov_enable.store(obs_data_get_bool(s, S_POV_ENABLE));
	f->pov_show_box.store(obs_data_get_bool(s, S_POV_SHOW));
	{
		int pw = (int)obs_data_get_int(s, S_POV_W);
		int ph = (int)obs_data_get_int(s, S_POV_H);
		f->pov_w.store(pw < 2 ? 2 : pw);
		f->pov_h.store(ph < 2 ? 2 : ph);
	}

	// Remember the UI's device path live, so the "Load device" button has a
	// reliable source even if obs_source_get_settings lags the dialog edits.
	{
		const char *dp = obs_data_get_string(s, S_DEV_PATH);
		std::lock_guard<std::mutex> lk(f->dev_mtx);
		f->dev_ui_path = dp ? dp : "";
	}

	// Auto-tracking parameters: write to atomics only. The independent control
	// thread reads these atomics every tick (no locking needed).
	f->track_enable.store(obs_data_get_bool(s, S_TRACK_ENABLE));
	f->track_trigger_vk.store((int)obs_data_get_int(s, S_TRACK_TRIGGER));
	f->track_target_mode.store((int)obs_data_get_int(s, S_TRACK_TARGET));
	f->track_target_cls.store((int)obs_data_get_int(s, S_TRACK_CLS));
	f->track_gain.store((float)obs_data_get_double(s, S_TRACK_GAIN));
	f->track_ki.store((float)obs_data_get_double(s, S_TRACK_KI));
	f->track_kd.store((float)obs_data_get_double(s, S_TRACK_KD));
	f->track_deadzone.store((float)obs_data_get_double(s, S_TRACK_DEADZONE));
	{
		int ti = (int)obs_data_get_int(s, S_TRACK_INTERVAL);
		f->track_send_interval.store(ti < 1 ? 1 : ti);
	}
	f->track_smooth_ema.store(obs_data_get_bool(s, S_TRACK_SMOOTH));
	f->track_speed_x.store((float)obs_data_get_double(s, S_TRACK_SPEED_X));
	f->track_speed_y.store((float)obs_data_get_double(s, S_TRACK_SPEED_Y));
	f->track_lead.store((float)obs_data_get_double(s, S_TRACK_LEAD));

	f->box_mode.store((int)obs_data_get_int(s, S_BOX_MODE));
	{
		int ch = (int)obs_data_get_int(s, S_CLS_HEAD);
		f->cls_head.store(ch < 0 ? 0 : ch);
	}
	f->track_offset.store((float)std::clamp(
		obs_data_get_double(s, S_TRACK_OFFSET), 0.0, 1.0));
	f->offset0.store((float)std::clamp(obs_data_get_double(s, S_OFFSET0), 0.0, 1.0));
	f->offset1.store((float)std::clamp(obs_data_get_double(s, S_OFFSET1), 0.0, 1.0));
	f->offset_preview.store(obs_data_get_bool(s, S_OFFSET_PREVIEW));

	const char *mp = obs_data_get_string(s, S_MODEL);
	if (mp && *mp) {
		bool changed = false;
		{
			std::lock_guard<std::mutex> lk(f->cfg_mtx);
			if (f->loaded_model != mp && f->requested_model != mp) {
				f->requested_model = mp;
				changed = true;
			}
		}
		if (changed) {
			f->model_dirty.store(true, std::memory_order_release);
			f->in_cv.notify_one();  // wake the worker to reload now
		}
	}

	// Push the latest knobs to the auto-tracking controller.
	f->controller.set_config(build_control_config(f));
}

// Build the controller config from the filter's live atomics (reused gain/ki
// fields: track_gain = smoothing, track_ki = engage radius).
static ControlConfig build_control_config(detector_filter *f)
{
	ControlConfig c;
	c.enable       = f->track_enable.load();
	c.trigger_vk   = f->track_trigger_vk.load();
	c.target_mode  = f->track_target_mode.load();
	c.target_cls   = f->track_target_cls.load();
	c.smooth       = f->track_gain.load();
	c.snap_radius  = f->track_ki.load();
	c.lead         = f->track_lead.load();
	c.max_step     = f->track_kd.load();   // repurposed: per-move clamp
	c.speed_x      = f->track_speed_x.load();
	c.speed_y      = f->track_speed_y.load();
	c.pov_enable   = f->pov_enable.load();
	c.pov_w        = f->pov_w.load();
	c.pov_h        = f->pov_h.load();
	c.switch_ratio = kTrackSwitchRatio;
	return c;
}

// ===========================================================================
//  create(): GPU objects on the graphics thread; model load deferred to worker.
// ===========================================================================
static void *filter_create(obs_data_t *settings, obs_source_t *context)
{
	auto *f = new detector_filter();
	f->context = context;

	obs_enter_graphics();
	f->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	f->stage     = gs_stagesurface_create(kNet, kNet, GS_BGRA);
	obs_leave_graphics();

	f->in_buf.assign(size_t(kNet) * kNet * 4, 0);

	// Pull initial settings; request the initial model load (worker performs it).
	f->conf_thresh.store((float)obs_data_get_double(settings, S_CONF));
	f->iou_thresh.store((float)obs_data_get_double(settings, S_IOU));
	int iv = (int)obs_data_get_int(settings, S_INTERVAL);
	f->infer_interval.store(iv < 1 ? 1 : iv);
	f->smoothing.store(obs_data_get_bool(settings, S_SMOOTH));
	f->soft_nms.store(obs_data_get_bool(settings, S_SOFT_NMS));
	f->infer_region.store((int)obs_data_get_int(settings, S_INFER_REGION));
	f->roi_enable.store(obs_data_get_bool(settings, S_ROI_ENABLE));
	f->roi_show_box.store(obs_data_get_bool(settings, S_ROI_SHOW));
	{
		int rw = (int)obs_data_get_int(settings, S_ROI_W);
		int rh = (int)obs_data_get_int(settings, S_ROI_H);
		f->roi_w.store(rw < 2 ? 2 : rw);
		f->roi_h.store(rh < 2 ? 2 : rh);
	}
	f->pov_enable.store(obs_data_get_bool(settings, S_POV_ENABLE));
	f->pov_show_box.store(obs_data_get_bool(settings, S_POV_SHOW));
	{
		int pw = (int)obs_data_get_int(settings, S_POV_W);
		int ph = (int)obs_data_get_int(settings, S_POV_H);
		f->pov_w.store(pw < 2 ? 2 : pw);
		f->pov_h.store(ph < 2 ? 2 : ph);
	}

	// Auto-tracking initial values. Write atomics only; the control thread reads
	// them every tick.
	f->track_enable.store(obs_data_get_bool(settings, S_TRACK_ENABLE));
	f->track_trigger_vk.store((int)obs_data_get_int(settings, S_TRACK_TRIGGER));
	f->track_target_mode.store((int)obs_data_get_int(settings, S_TRACK_TARGET));
	f->track_target_cls.store((int)obs_data_get_int(settings, S_TRACK_CLS));
	f->track_gain.store((float)obs_data_get_double(settings, S_TRACK_GAIN));
	f->track_ki.store((float)obs_data_get_double(settings, S_TRACK_KI));
	f->track_kd.store((float)obs_data_get_double(settings, S_TRACK_KD));
	f->track_deadzone.store((float)obs_data_get_double(settings, S_TRACK_DEADZONE));
	{
		int ti = (int)obs_data_get_int(settings, S_TRACK_INTERVAL);
		f->track_send_interval.store(ti < 1 ? 1 : ti);
	}
	f->track_smooth_ema.store(obs_data_get_bool(settings, S_TRACK_SMOOTH));
	f->track_speed_x.store((float)obs_data_get_double(settings, S_TRACK_SPEED_X));
	f->track_speed_y.store((float)obs_data_get_double(settings, S_TRACK_SPEED_Y));
	f->track_lead.store((float)obs_data_get_double(settings, S_TRACK_LEAD));

	f->box_mode.store((int)obs_data_get_int(settings, S_BOX_MODE));
	{
		int ch = (int)obs_data_get_int(settings, S_CLS_HEAD);
		f->cls_head.store(ch < 0 ? 0 : ch);
	}
	f->track_offset.store((float)std::clamp(
		obs_data_get_double(settings, S_TRACK_OFFSET), 0.0, 1.0));
	f->offset0.store((float)std::clamp(obs_data_get_double(settings, S_OFFSET0), 0.0, 1.0));
	f->offset1.store((float)std::clamp(obs_data_get_double(settings, S_OFFSET1), 0.0, 1.0));
	f->offset_preview.store(obs_data_get_bool(settings, S_OFFSET_PREVIEW));

	const char *mp = obs_data_get_string(settings, S_MODEL);
	{
		std::lock_guard<std::mutex> lk(f->cfg_mtx);
		f->requested_model = (mp && *mp) ? mp : "";
	}
	f->model_dirty.store(true, std::memory_order_release);

	f->running.store(true, std::memory_order_release);
	f->worker     = std::thread(inference_worker, f);

	// Independent auto-tracking controller: relative device moves via a callback
	// that locks dev_mtx (the device may be (re)loaded/freed on the UI thread).
	f->controller.set_send_fn([f](short dx, short dy) -> bool {
		std::lock_guard<std::mutex> lk(f->dev_mtx);
		if (f->device && f->device->isReady())
			return f->device->sendDelta(dx, dy);
		return false;
	});
	f->controller.set_config(build_control_config(f));
	f->controller.start();
	return f;
}

static void filter_destroy(void *data)
{
	auto *f = static_cast<detector_filter *>(data);

	f->running.store(false, std::memory_order_release);

	// Stop the control thread first (its send callback touches the device).
	f->controller.stop();

	{
		std::lock_guard<std::mutex> lk(f->in_mtx);
		f->frame_ready = true;  // force the wait predicate true
	}
	f->in_cv.notify_all();
	f->out_cv.notify_all();  // wake a graphics thread blocked on sync inference
	if (f->worker.joinable())
		f->worker.join();

	delete f->detector;  // worker is gone

	{
		std::lock_guard<std::mutex> lk(f->dev_mtx);
		delete f->device;  // dtor releases the device + frees the DLL
		f->device = nullptr;
	}

	obs_enter_graphics();
	if (f->stage)
		gs_stagesurface_destroy(f->stage);
	if (f->texrender)
		gs_texrender_destroy(f->texrender);
	obs_leave_graphics();

	delete f;
}

// Graphics-thread: render TARGET stretched into 640x640 and stage it.
static void capture_target_to_stage(detector_filter *f)
{
	obs_source_t *target = obs_filter_get_target(f->context);
	obs_source_t *parent = obs_filter_get_parent(f->context);
	if (!target)
		return;

	const uint32_t cx = obs_source_get_base_width(target);
	const uint32_t cy = obs_source_get_base_height(target);
	if (cx == 0 || cy == 0)
		return;

	// Use obs_source_default_render when we are the last filter (target==parent)
	// on a plain source: obs_source_video_render would re-enter THIS filter and
	// stage a BLACK frame (pattern from nv-filters).
	const uint32_t flags = obs_source_get_output_flags(target);
	const bool custom_draw = (flags & OBS_SOURCE_CUSTOM_DRAW) != 0;
	const bool async = (flags & OBS_SOURCE_ASYNC) != 0;

	gs_texrender_reset(f->texrender);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

	if (gs_texrender_begin(f->texrender, kNet, kNet)) {
		struct vec4 clear_color;
		vec4_zero(&clear_color);
		gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);
		// Crop a centered SQUARE region (native res, undistorted) into the
		// square network input -- NOT the whole stretched source. Must match the
		// mapping in video_render exactly (same infer_crop()).
		int side, x0, y0;
		infer_crop((int)cx, (int)cy, f->infer_region.load(), side, x0, y0);
		gs_ortho((float)x0, (float)(x0 + side),
		         (float)y0, (float)(y0 + side), -100.0f, 100.0f);

		if (target == parent && !custom_draw && !async)
			obs_source_default_render(target);
		else
			obs_source_video_render(target);

		gs_texrender_end(f->texrender);

		gs_texture_t *tex = gs_texrender_get_texture(f->texrender);
		if (tex) {
			gs_stage_texture(f->stage, tex);
			f->stage_pending = true;
		}
	}
	gs_blend_state_pop();
}

// Graphics-thread: map the just-staged surface (blocks for the GPU copy), copy
// the BGRA rows into in_buf, and signal the worker. Called synchronously right
// after capture_target_to_stage so the detection reflects the current frame.
static void map_stage_to_worker(detector_filter *f)
{
	uint8_t *src = nullptr;
	uint32_t linesize = 0;
	if (!gs_stagesurface_map(f->stage, &src, &linesize))
		return;

	const uint32_t row_bytes = kNet * 4;
	{
		std::lock_guard<std::mutex> lk(f->in_mtx);
		f->in_buf.resize(size_t(kNet) * kNet * 4);
		uint8_t *dst = f->in_buf.data();
		if (linesize == row_bytes) {
			memcpy(dst, src, size_t(row_bytes) * kNet);
		} else {
			for (uint32_t y = 0; y < kNet; ++y)
				memcpy(dst + size_t(y) * row_bytes,
				       src + size_t(y) * linesize, row_bytes);
		}
		f->frame_ready = true;
	}
	gs_stagesurface_unmap(f->stage);
	f->in_cv.notify_one();
}

// ===========================================================================
//  video_tick (once per frame, BEFORE render, same thread): advance the Kalman
//  tracks and, when a new detection set is ready, associate + correct them.
// ===========================================================================
static void filter_video_tick(void *data, float seconds)
{
	auto *f = static_cast<detector_filter *>(data);
	if (!f->smoothing.load())
		return;  // raw mode -> video_render draws latest_results directly

	float dt = seconds;
	if (dt <= 0.0f || dt > 0.25f)
		dt = 1.0f / 60.0f;  // clamp implausible dt
	f->tracker.predict(dt);
	// NOTE: the Kalman CORRECTION (tracker.update) now happens in video_render,
	// right after the SYNCHRONOUS inference, so control acts on a current-frame
	// detection instead of one consumed a frame later here.
}

static void filter_video_render(void *data, gs_effect_t *effect)
{
	auto *f = static_cast<detector_filter *>(data);
	UNUSED_PARAMETER(effect);

	// Step 1: pass-through render of the parent through the filter chain.
	if (!obs_source_process_filter_begin(f->context, GS_RGBA,
	                                     OBS_ALLOW_DIRECT_RENDERING))
		return;
	obs_source_process_filter_end(
		f->context, obs_get_base_effect(OBS_EFFECT_DEFAULT), 0, 0);

	// Step 2: SYNCHRONOUS inference (low dead-time). Render the crop, map it THIS
	// frame, submit to the worker, and block briefly until it publishes -> the
	// tracker corrects on a CURRENT-frame detection instead of one 2-3 frames old
	// (the old deferred-map + async-consume path). The gimbal is fast, so the
	// vision pipeline was the whole latency; this removes most of it.
	int interval = f->infer_interval.load();
	if (interval < 1)
		interval = 1;
	if ((f->frame_counter++ % (uint64_t)interval) == 0) {
		const auto t0 = std::chrono::steady_clock::now();
		capture_target_to_stage(f);            // render crop + stage
		if (f->stage_pending) {
			f->stage_pending = false;
			const uint64_t before =
				f->result_seq.load(std::memory_order_acquire);
			map_stage_to_worker(f);            // map (GPU sync) + signal worker
			std::unique_lock<std::mutex> lk(f->out_mtx);
			f->out_cv.wait_for(lk, std::chrono::milliseconds(20), [&] {
				return f->result_seq.load(std::memory_order_acquire) !=
				           before ||
				       !f->running.load(std::memory_order_acquire);
			});
		}
		// Measured capture -> detection-ready latency (GPU readback + preprocess
		// + inference + decode/NMS), i.e. the "acquire target" cost each frame.
		const double ms = std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - t0).count();
		f->dbg_pipe_ms = ms;
		if ((f->frame_counter % 60) == 0)
			blog(LOG_INFO, "[obs-detection-overlay] acquire->ready latency: "
			     "%.2f ms (capture+readback+infer+nms); send adds <1.2 ms",
			     ms);
	}

	// Apply the freshest detection set to the Kalman tracker NOW (this frame),
	// so the controller below acts on it immediately.
	if (f->smoothing.load()) {
		const uint64_t seq = f->result_seq.load(std::memory_order_acquire);
		if (seq != f->tick_seq) {
			f->tick_seq = seq;
			std::vector<Detection> dets;
			{
				std::lock_guard<std::mutex> lk(f->res_mtx);
				dets = f->latest_results;
			}
			f->tracker.update(dets);
		}
	}

	// Step 3: draw boxes from latest results (OUTLINE ONLY).
	obs_source_t *target = obs_filter_get_target(f->context);
	uint32_t cx = target ? obs_source_get_base_width(target) : 0;
	uint32_t cy = target ? obs_source_get_base_height(target) : 0;
	if (cx == 0 || cy == 0)
		return;

	// Map detections from network space back to SOURCE pixels. Detections come
	// from the centered SQUARE crop (see capture_target_to_stage) -> scale by
	// side/kNet and offset by the crop's top-left. MUST match infer_crop() there.
	int crop_side, crop_x0, crop_y0;
	infer_crop((int)cx, (int)cy, f->infer_region.load(),
	           crop_side, crop_x0, crop_y0);
	const float sx = (float)crop_side / (float)kNet;
	const float sy = (float)crop_side / (float)kNet;
	const float ox = (float)crop_x0;
	const float oy = (float)crop_y0;

	// Boxes to draw: smoothed tracks (+ stable ids) when smoothing is on, else
	// the raw detections (id = -1).
	std::vector<TrackedBox> boxes;
	if (f->smoothing.load()) {
		boxes = f->tracker.active();
	} else {
		std::lock_guard<std::mutex> lk(f->res_mtx);
		boxes.reserve(f->latest_results.size());
		for (const Detection &d : f->latest_results)
			boxes.push_back({d, -1, 0.0f, 0.0f});
	}

	// Source-image CENTER is the ALWAYS-on origin. Recomputed every frame from
	// cx,cy, so it adapts to any source resolution automatically.
	const float icx = (float)cx * 0.5f;
	const float icy = (float)cy * 0.5f;

	// ROI: a rect CENTERED on the image center, roi_w x roi_h in SOURCE pixels,
	// clamped to the source. When enabled, detections whose center falls outside
	// are FULLY filtered out (not drawn, not saved).
	const bool  roi_on = f->roi_enable.load();
	float roi_w = (float)std::min<int>(f->roi_w.load(), (int)cx);
	float roi_h = (float)std::min<int>(f->roi_h.load(), (int)cy);
	if (roi_w < 2.0f) roi_w = 2.0f;
	if (roi_h < 2.0f) roi_h = 2.0f;
	const float roi_l = icx - roi_w * 0.5f;
	const float roi_t = icy - roi_h * 0.5f;
	const float roi_r = roi_l + roi_w;
	const float roi_b = roi_t + roi_h;

	// POV: centered rect gating AUTO-TRACKING only (targets outside are still
	// drawn). Bounds computed the same way as ROI.
	const bool  pov_on = f->pov_enable.load();
	float pov_w = (float)std::min<int>(f->pov_w.load(), (int)cx);
	float pov_h = (float)std::min<int>(f->pov_h.load(), (int)cy);
	if (pov_w < 2.0f) pov_w = 2.0f;
	if (pov_h < 2.0f) pov_h = 2.0f;
	const float pov_l = icx - pov_w * 0.5f;
	const float pov_t = icy - pov_h * 0.5f;
	const float pov_r = pov_l + pov_w;
	const float pov_b = pov_t + pov_h;

	constexpr float kRad2Deg = 57.29577951308232f;

	// One pass: scale to source pixels, apply ROI gating, compute polar coords
	// relative to the image center, and collect the boxes that survive.
	const int   box_mode     = f->box_mode.load();
	const int   cls_head     = f->cls_head.load();
	// Independent aim offset per box display mode (0/1/2), fraction of box height.
	const float mode_offset  = (box_mode == 0) ? f->offset0.load()
	                         : (box_mode == 1) ? f->offset1.load()
	                                           : f->track_offset.load();
	const bool  offset_preview = f->offset_preview.load();

	std::vector<box_px>    draw_boxes;
	std::vector<ObjCenter> cs;
	std::vector<box_px>    aim_pts;  // aim points for the preview overlay
	draw_boxes.reserve(boxes.size());
	cs.reserve(boxes.size());
	for (const TrackedBox &tb : boxes) {
		// Box display mode: drop classes this mode doesn't process (mode 1/2
		// keep only the head class; mode 0 keeps everything).
		if (!box_mode_should_process(tb.box.cls, box_mode, cls_head))
			continue;

		const float bx = ox + tb.box.x * sx, by = oy + tb.box.y * sy;
		const float bw = tb.box.w * sx, bh = tb.box.h * sy;
		const float ocx = bx + bw * 0.5f;
		const float ocy = by + bh * 0.5f;
		if (roi_on && (ocx < roi_l || ocx > roi_r ||
		               ocy < roi_t || ocy > roi_b))
			continue;  // outside ROI -> fully filtered out

		// Tracking point: (cx, top + mode_offset*h) -> offset from image center.
		float tpx, tpy;
		box_mode_track_point(bx, by, bw, bh, mode_offset, tpx, tpy);

		// Target velocity at the tracking point, in SOURCE px/sec (Kalman
		// estimate scaled from net space). Used for lead/prediction below.
		const float vsx = tb.vx * sx;
		const float vsy = tb.vy * sy;

		ObjCenter c;
		c.id = tb.id;
		c.cls = tb.box.cls;
		c.score = tb.box.score;
		c.cx = ocx;
		c.cy = ocy;
		c.dx = ocx - icx;
		c.dy = ocy - icy;
		c.track_vx = vsx;
		c.track_vy = vsy;
		// Tracking point offset from image center (in source px). The CONTROL
		// thread does velocity prediction (pos + vel*age) at its own high rate,
		// so no lead is pre-applied here.
		c.track_dx = tpx - icx;
		c.track_dy = tpy - icy;
		c.dist = std::sqrt(c.dx * c.dx + c.dy * c.dy);
		// screen-y points DOWN: negate dy so +90 deg points UP.
		c.angle = std::atan2(-c.dy, c.dx) * kRad2Deg;
		cs.push_back(c);

		box_px b;
		b.x = bx; b.y = by; b.w = bw; b.h = bh;
		draw_boxes.push_back(b);

		if (offset_preview) {
			box_px a;  // store the aim point (x,y); w/h unused
			a.x = tpx; a.y = tpy; a.w = 0.0f; a.h = 0.0f;
			aim_pts.push_back(a);
		}
	}

	{
		std::lock_guard<std::mutex> lk(f->centers_mtx);
		f->centers = cs;  // keep for the periodic log + detector_get_centers()
	}

	// Hand this frame's targets to the independent auto-tracking controller.
	std::vector<ControlTarget> targets;
	targets.reserve(cs.size());
	for (const ObjCenter &c : cs)
		targets.push_back({c.id, c.cls, c.cx, c.cy,
		                   c.track_dx, c.track_dy, c.track_vx, c.track_vy, c.dist});
	f->controller.publish(std::move(targets), (int)cx, (int)cy);

	// Periodic log so the saved centers (polar, ROI-filtered) can be verified.
	if ((f->frame_counter % 120) == 0) {
		std::lock_guard<std::mutex> lk(f->centers_mtx);
		if (!f->centers.empty()) {
			const ObjCenter &c0 = f->centers.front();
			// COCO names only when the model actually has 80 classes; else show
			// the raw class index (e.g. a 2-class body/head model).
			char clsname[16];
			if (f->model_num_classes.load() == 80 && c0.cls >= 0 &&
			    c0.cls < kNumClasses)
				snprintf(clsname, sizeof clsname, "%s", coco_names[c0.cls]);
			else
				snprintf(clsname, sizeof clsname, "cls%d", c0.cls);
			blog(LOG_INFO,
			     "[obs-detection-overlay] %zu objects%s; e.g. id=%d %s "
			     "angle=%.1fdeg dist=%.0fpx (dx=%.0f,dy=%.0f) score=%.2f",
			     f->centers.size(), roi_on ? " in ROI" : "", c0.id, clsname,
			     c0.angle, c0.dist, c0.dx, c0.dy, c0.score);
		} else {
			blog(LOG_INFO, "[obs-detection-overlay] 0 objects%s",
			     roi_on ? " in ROI" : "");
		}
	}

	// ---- draw overlay: detection boxes, ROI rect, center marker ----
	gs_effect_t *solid = obs_get_base_effect(OBS_EFFECT_SOLID);
	gs_eparam_t *color_param = gs_effect_get_param_by_name(solid, "color");
	const float thick = std::max(2.0f, 0.004f * (float)std::max(cx, cy));

	// detection boxes (green outline)
	if (!draw_boxes.empty()) {
		struct vec4 green;
		vec4_set(&green, 0.0f, 1.0f, 0.0f, 1.0f);
		gs_effect_set_vec4(color_param, &green);
		while (gs_effect_loop(solid, "Solid")) {
			for (const box_px &b : draw_boxes)
				draw_box_outline(&b, thick);
		}
	}

	// ROI rectangle (red outline): drawn when gating is on AND the "show box"
	// toggle is on -- lets you gate without the rectangle cluttering the view.
	if (roi_on && f->roi_show_box.load()) {
		struct vec4 red;
		vec4_set(&red, 1.0f, 0.0f, 0.0f, 1.0f);
		gs_effect_set_vec4(color_param, &red);
		box_px r;
		r.x = roi_l; r.y = roi_t; r.w = roi_w; r.h = roi_h;
		while (gs_effect_loop(solid, "Solid"))
			draw_box_outline(&r, thick);
	}

	// POV rectangle (cyan outline): the auto-tracking gate. Targets outside are
	// drawn but never tracked. Shown when gating + its "show box" toggle are on.
	if (pov_on && f->pov_show_box.load()) {
		struct vec4 cyan;
		vec4_set(&cyan, 0.0f, 1.0f, 1.0f, 1.0f);
		gs_effect_set_vec4(color_param, &cyan);
		box_px r;
		r.x = pov_l; r.y = pov_t; r.w = pov_w; r.h = pov_h;
		while (gs_effect_loop(solid, "Solid"))
			draw_box_outline(&r, thick);
	}

	// aim-point preview (magenta crosses) -- where the controller aims on each
	// box, given the current mode's offset. Lets you tune the offset visually.
	if (offset_preview && !aim_pts.empty()) {
		struct vec4 magenta;
		vec4_set(&magenta, 1.0f, 0.0f, 1.0f, 1.0f);
		gs_effect_set_vec4(color_param, &magenta);
		const float L  = std::max(6.0f, thick * 3.0f);
		const float ht = thick;
		while (gs_effect_loop(solid, "Solid")) {
			for (const box_px &a : aim_pts) {
				draw_filled_rect(a.x - L, a.y - ht * 0.5f, 2.0f * L, ht);
				draw_filled_rect(a.x - ht * 0.5f, a.y - L, ht, 2.0f * L);
			}
		}
	}

	gs_load_vertexbuffer(NULL);
}

// ---------------------------------------------------------------------------
// Public accessor: copy out the latest object centers (SOURCE pixels). Safe to
// call from any thread. Intended entry point for later features that consume
// the coordinates.
// ---------------------------------------------------------------------------
std::vector<ObjCenter> detector_get_centers(detector_filter *f)
{
	std::lock_guard<std::mutex> lk(f->centers_mtx);
	return f->centers;
}

static uint32_t filter_get_width(void *data)
{
	auto *f = static_cast<detector_filter *>(data);
	obs_source_t *target = obs_filter_get_target(f->context);
	return target ? obs_source_get_base_width(target) : 0;
}

static uint32_t filter_get_height(void *data)
{
	auto *f = static_cast<detector_filter *>(data);
	obs_source_t *target = obs_filter_get_target(f->context);
	return target ? obs_source_get_base_height(target) : 0;
}

// ---------------------------------------------------------------------------
// Motion/gimbal device: connect / disconnect / status. Driven by the UI
// "Load device" button (filter_get_properties), never auto-loaded.
// ---------------------------------------------------------------------------
// Open the device at `path` (empty -> disconnect). Re-opens only when the path
// changes. Returns true if connected AND ready. Runs on the UI thread.
static bool device_apply(detector_filter *f, const char *path)
{
	std::lock_guard<std::mutex> lk(f->dev_mtx);
	const std::string want = path ? path : "";

	// Tear down if cleared or the path changed.
	if (f->device && (want.empty() || want != f->dev_loaded_path)) {
		delete f->device;            // dtor releases device + frees the DLL
		f->device = nullptr;
		f->dev_loaded_path.clear();
	}
	if (want.empty())
		return false;
	if (f->device)                   // already open to this exact path
		return f->device->isReady();

	try {
		auto *dev = new DeviceController(want);  // throws on load/resolve failure
		const bool ok = dev->initialize();        // fnInitById(0,0); active->true
		f->device = dev;
		f->dev_loaded_path = want;
		blog(LOG_INFO,
		     "[obs-detection-overlay] motion device connected: %s "
		     "(ready=%d id=%ld build=%d type=%d)",
		     want.c_str(), dev->isReady() ? 1 : 0, dev->queryUnitId(),
		     dev->queryBuildNum(), dev->queryUnitType());
		return ok && dev->isReady();
	} catch (const std::exception &e) {
		blog(LOG_ERROR,
		     "[obs-detection-overlay] failed to open motion device '%s': %s",
		     want.c_str(), e.what());
		f->device = nullptr;
		f->dev_loaded_path.clear();
		return false;
	}
}

// Human-readable connection status for the UI label (localized).
static std::string device_status_text(detector_filter *f)
{
	std::lock_guard<std::mutex> lk(f->dev_mtx);
	if (f->device && f->device->isReady())
		return std::string(obs_module_text("DeviceConnected")) + " (id=" +
		       std::to_string(f->device->queryUnitId()) + " build=" +
		       std::to_string(f->device->queryBuildNum()) + ")";
	if (f->device)
		return obs_module_text("DeviceNotReady");
	return obs_module_text("DeviceDisconnected");
}

// "Load device" button: (re)connect to the path in the UI, refresh the label.
static bool device_load_clicked(obs_properties_t *props, obs_property_t *property,
                                void *data)
{
	UNUSED_PARAMETER(property);
	auto *f = static_cast<detector_filter *>(data);
	if (!f)
		return false;

	obs_data_t *s = obs_source_get_settings(f->context);
	std::string path = obs_data_get_string(s, S_DEV_PATH);
	obs_data_release(s);
	if (path.empty()) {  // dialog edits may not be committed to source settings yet
		std::lock_guard<std::mutex> lk(f->dev_mtx);
		path = f->dev_ui_path;
	}
	blog(LOG_INFO, "[obs-detection-overlay] load-device button: path='%s'",
	     path.c_str());
	device_apply(f, path.c_str());

	obs_property_t *st = obs_properties_get(props, S_DEV_STATUS);
	if (st) {
		const std::string txt = device_status_text(f);
		obs_property_set_description(st, txt.c_str());
	}
	return true;  // refresh the properties view so the label updates
}

// ---------------------------------------------------------------------------
// Defaults + properties UI.
// ---------------------------------------------------------------------------
static void filter_get_defaults(obs_data_t *s)
{
	char *def = obs_module_file("models/model.onnx");
	if (def) {
		obs_data_set_default_string(s, S_MODEL, def);
		bfree(def);
	}
	obs_data_set_default_double(s, S_CONF, kDefConf);
	obs_data_set_default_double(s, S_IOU, kDefIou);
	obs_data_set_default_int(s, S_INTERVAL, kDefInterval);
	obs_data_set_default_bool(s, S_SMOOTH, kDefSmooth);
	obs_data_set_default_bool(s, S_SOFT_NMS, kDefSoftNms);
	obs_data_set_default_int(s, S_INFER_REGION, kDefInferRegion);
	obs_data_set_default_bool(s, S_ROI_ENABLE, kDefRoiEnable);
	obs_data_set_default_bool(s, S_ROI_SHOW, kDefRoiShow);
	obs_data_set_default_int(s, S_ROI_W, kDefRoiW);
	obs_data_set_default_int(s, S_ROI_H, kDefRoiH);
	obs_data_set_default_bool(s, S_POV_ENABLE, kDefPovEnable);
	obs_data_set_default_bool(s, S_POV_SHOW, kDefPovShow);
	obs_data_set_default_int(s, S_POV_W, kDefPovW);
	obs_data_set_default_int(s, S_POV_H, kDefPovH);
	obs_data_set_default_string(s, S_DEV_PATH, "");
	obs_data_set_default_bool(s, S_TRACK_ENABLE, kDefTrackEnable);
	obs_data_set_default_int(s, S_TRACK_TRIGGER, kDefTrackTrigger);
	obs_data_set_default_int(s, S_TRACK_TARGET, kDefTrackTarget);
	obs_data_set_default_int(s, S_TRACK_CLS, kDefTrackCls);
	obs_data_set_default_double(s, S_TRACK_GAIN, kDefTrackGain);
	obs_data_set_default_double(s, S_TRACK_KI, kDefTrackKi);
	obs_data_set_default_double(s, S_TRACK_KD, kDefTrackKd);
	obs_data_set_default_double(s, S_TRACK_DEADZONE, kDefTrackDeadzone);
	obs_data_set_default_int(s, S_TRACK_INTERVAL, kDefTrackInterval);
	obs_data_set_default_bool(s, S_TRACK_SMOOTH, kDefTrackSmooth);
	obs_data_set_default_double(s, S_TRACK_SPEED_X, kDefTrackSpeedX);
	obs_data_set_default_double(s, S_TRACK_SPEED_Y, kDefTrackSpeedY);
	obs_data_set_default_double(s, S_TRACK_LEAD, kDefTrackLead);
	obs_data_set_default_int(s, S_BOX_MODE, kDefBoxMode);
	obs_data_set_default_int(s, S_CLS_HEAD, kDefClsHead);
	obs_data_set_default_double(s, S_TRACK_OFFSET, kDefTrackOffset);
	obs_data_set_default_double(s, S_OFFSET0, kDefOffset0);
	obs_data_set_default_double(s, S_OFFSET1, kDefOffset1);
	obs_data_set_default_bool(s, S_OFFSET_PREVIEW, kDefOffsetPreview);
}

static obs_properties_t *filter_get_properties(void *data)
{
	auto *f = static_cast<detector_filter *>(data);
	obs_properties_t *p = obs_properties_create();

	// ============================ 1) Detection ============================
	obs_properties_t *gd = obs_properties_create();
	char *mdir = obs_module_file("models");
	obs_properties_add_path(gd, S_MODEL, obs_module_text("Model"), OBS_PATH_FILE,
	                        "ONNX model (*.onnx);;All Files (*.*)", mdir);
	if (mdir)
		bfree(mdir);
	obs_properties_add_float_slider(gd, S_CONF, obs_module_text("ConfThreshold"),
	                                0.05, 0.95, 0.01);
	obs_properties_add_float_slider(gd, S_IOU, obs_module_text("IoUThreshold"),
	                                0.10, 0.90, 0.01);
	obs_property_t *ir = obs_properties_add_int(
		gd, S_INFER_REGION, obs_module_text("InferRegion"), 0, 3840, 16);
	obs_property_int_set_suffix(ir, " px");
	obs_properties_add_int_slider(gd, S_INTERVAL,
	                              obs_module_text("InferInterval"), 1, 8, 1);
	obs_properties_add_bool(gd, S_SOFT_NMS, obs_module_text("SoftNms"));
	obs_properties_add_bool(gd, S_SMOOTH, obs_module_text("Smoothing"));
	obs_properties_add_group(p, "grp_detect", obs_module_text("GrpDetect"),
	                         OBS_GROUP_NORMAL, gd);

	// ===================== 2) Box display / classes =======================
	obs_properties_t *gb = obs_properties_create();
	obs_property_t *bm = obs_properties_add_list(
		gb, S_BOX_MODE, obs_module_text("BoxMode"),
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(bm, obs_module_text("BoxModeHeadBody"), 0);
	obs_property_list_add_int(bm, obs_module_text("BoxModeHead"),     1);
	obs_property_list_add_int(bm, obs_module_text("BoxModeHeadTrack"), 2);
	obs_properties_add_int(gb, S_CLS_HEAD, obs_module_text("ClsHead"), 0, 255, 1);
	// Independent aim offset per box display mode (0/1/2), + a preview marker.
	obs_properties_add_float_slider(gb, S_OFFSET0,
	                                obs_module_text("Offset0"), 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(gb, S_OFFSET1,
	                                obs_module_text("Offset1"), 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(gb, S_TRACK_OFFSET,
	                                obs_module_text("Offset2"), 0.0, 1.0, 0.01);
	obs_properties_add_bool(gb, S_OFFSET_PREVIEW, obs_module_text("OffsetPreview"));
	obs_properties_add_group(p, "grp_box", obs_module_text("GrpBox"),
	                         OBS_GROUP_NORMAL, gb);

	// ===================== 3) Regions (ROI / POV) =========================
	obs_properties_t *gr = obs_properties_create();
	obs_properties_add_bool(gr, S_ROI_ENABLE, obs_module_text("RoiEnable"));
	obs_property_t *rw = obs_properties_add_int(
		gr, S_ROI_W, obs_module_text("RoiWidth"), 2, 7680, 2);
	obs_property_int_set_suffix(rw, " px");
	obs_property_t *rh = obs_properties_add_int(
		gr, S_ROI_H, obs_module_text("RoiHeight"), 2, 7680, 2);
	obs_property_int_set_suffix(rh, " px");
	obs_properties_add_bool(gr, S_ROI_SHOW, obs_module_text("RoiShowBox"));
	obs_properties_add_bool(gr, S_POV_ENABLE, obs_module_text("PovEnable"));
	obs_property_t *pw = obs_properties_add_int(
		gr, S_POV_W, obs_module_text("PovWidth"), 2, 7680, 2);
	obs_property_int_set_suffix(pw, " px");
	obs_property_t *ph = obs_properties_add_int(
		gr, S_POV_H, obs_module_text("PovHeight"), 2, 7680, 2);
	obs_property_int_set_suffix(ph, " px");
	obs_properties_add_bool(gr, S_POV_SHOW, obs_module_text("PovShowBox"));
	obs_properties_add_group(p, "grp_region", obs_module_text("GrpRegion"),
	                         OBS_GROUP_NORMAL, gr);

	// ============================ 4) Device ===============================
	obs_properties_t *gv = obs_properties_create();
	obs_properties_add_path(gv, S_DEV_PATH, obs_module_text("DevicePath"),
	                        OBS_PATH_FILE, "DLL (*.dll);;All Files (*.*)", NULL);
	obs_properties_add_button2(gv, S_DEV_LOAD, obs_module_text("DeviceLoad"),
	                           device_load_clicked, f);
	obs_property_t *st = obs_properties_add_text(
		gv, S_DEV_STATUS, obs_module_text("DeviceDisconnected"), OBS_TEXT_INFO);
	if (f) {
		const std::string txt = device_status_text(f);
		obs_property_set_description(st, txt.c_str());
	}
	obs_properties_add_group(p, "grp_device", obs_module_text("GrpDevice"),
	                         OBS_GROUP_NORMAL, gv);

	// ======================== 5) Auto-tracking ============================
	obs_properties_t *gt = obs_properties_create();
	obs_properties_add_bool(gt, S_TRACK_ENABLE, obs_module_text("TrackEnable"));

	obs_property_t *tkey = obs_properties_add_list(
		gt, S_TRACK_TRIGGER, obs_module_text("TrackTriggerKey"),
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(tkey, obs_module_text("TrackKeyAlways"),   -1);
	obs_property_list_add_int(tkey, obs_module_text("TrackKeyNone"),      0);
	obs_property_list_add_int(tkey, obs_module_text("TrackKeyLButton"), VK_LBUTTON);
	obs_property_list_add_int(tkey, obs_module_text("TrackKeyRButton"), VK_RBUTTON);
	obs_property_list_add_int(tkey, obs_module_text("TrackKeyMButton"), VK_MBUTTON);
	obs_property_list_add_int(tkey, obs_module_text("TrackKeyXButton1"),VK_XBUTTON1);
	obs_property_list_add_int(tkey, obs_module_text("TrackKeyXButton2"),VK_XBUTTON2);

	obs_property_t *tmode = obs_properties_add_list(
		gt, S_TRACK_TARGET, obs_module_text("TrackTargetMode"),
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(tmode, obs_module_text("TrackTargetNearest"), 0);
	obs_property_list_add_int(tmode, obs_module_text("TrackTargetCls"),     1);
	obs_properties_add_int(gt, S_TRACK_CLS, obs_module_text("TrackTargetCls0"),
	                       0, 255, 1);

	// Controller (absolute-positioning + ease, 125Hz control thread):
	//   smoothing -> prediction -> per-axis speed -> deadzone.
	obs_properties_add_float_slider(gt, S_TRACK_GAIN,
	                                obs_module_text("TrackSmoothFactor"), 0.01, 1.0, 0.01);
	obs_properties_add_float_slider(gt, S_TRACK_KI,
	                                obs_module_text("TrackSnapRadius"), 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(gt, S_TRACK_LEAD,
	                                obs_module_text("TrackLead"), 0.0, 0.5, 0.005);
	obs_properties_add_float_slider(gt, S_TRACK_KD,
	                                obs_module_text("TrackMaxStep"), 0.0, 500.0, 5.0);
	obs_properties_add_float_slider(gt, S_TRACK_SPEED_X,
	                                obs_module_text("TrackSpeedX"), 0.05, 5.0, 0.05);
	obs_properties_add_float_slider(gt, S_TRACK_SPEED_Y,
	                                obs_module_text("TrackSpeedY"), 0.05, 5.0, 0.05);
	obs_property_t *tdz = obs_properties_add_float_slider(gt, S_TRACK_DEADZONE,
	                                obs_module_text("TrackDeadzone"), 0.0, 200.0, 1.0);
	obs_property_float_set_suffix(tdz, " px");
	obs_properties_add_group(p, "grp_track", obs_module_text("GrpTrack"),
	                         OBS_GROUP_NORMAL, gt);

	return p;
}

static obs_source_info make_filter_info()
{
	obs_source_info info = {};
	info.id             = "detection_overlay_filter";
	info.type           = OBS_SOURCE_TYPE_FILTER;
	info.output_flags   = OBS_SOURCE_VIDEO;
	info.get_name       = filter_get_name;
	info.create         = filter_create;
	info.destroy        = filter_destroy;
	info.video_tick     = filter_video_tick;
	info.video_render   = filter_video_render;
	info.get_width      = filter_get_width;
	info.get_height     = filter_get_height;
	info.get_defaults   = filter_get_defaults;
	info.get_properties = filter_get_properties;
	info.update         = filter_update;
	return info;
}

static obs_source_info filter_info = make_filter_info();

extern "C" MODULE_EXPORT bool obs_module_load(void)
{
	obs_register_source(&filter_info);
	blog(LOG_INFO, "[obs-detection-overlay] loaded (v3: DirectML + properties)");
	return true;
}
