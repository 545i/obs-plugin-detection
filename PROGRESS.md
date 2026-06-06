# obs-detection-overlay — Development Progress

Last updated: 2026-06-04

## Overview
In-tree OBS Studio video **filter** plugin that runs **object detection**
(ONNX Runtime, DirectML GPU) and draws outline-only detection boxes over a
source in the OBS preview. Built and verified on Windows x64 / RTX 5070.

Independent git repo, branch `main` (decoupled from the OBS source tree).

## Status: working & verified
- Builds in-tree (RelWithDebInfo, VS2022) against the OBS source.
- Loads in OBS, appears as filter **"Detection Overlay" / "偵測疊加"**.
- Real-time object detection on **DirectML GPU**, ~1.8 ms/frame.
- **60.0 fps, 0 skipped frames at 1080p60** (obs-websocket GetStats), even at
  inference interval = 1.
- Kalman smoothing for stable, fluid boxes on moving sources.
- Live filter settings (model / confidence / IoU / interval / smoothing).

## Commits (branch `main`)
```
4965fd3 Fix model combo path separators (use generic_string)
018760f Add Kalman (SORT-style) tracking for smooth detection boxes
b9a455e Add filter properties (model, conf, IoU, interval); 60fps verified
5a30fc8 Use DirectML GPU backend (Blackwell-compatible); add Backend enum
9ec1d12 obs-yolo-detect: YOLOv8 detection overlay filter for OBS (CPU)
```

## Architecture / pipeline
1. `video_render` (graphics thread): pass the parent through unchanged.
2. Every `infer_interval` frames: render the target STRETCHED into a 640x640
   GS_BGRA texrender, then `gs_stage_texture` (async GPU->CPU).
3. Next frame: map the staged surface, copy BGRA into a shared buffer, signal
   the worker (deferred-map avoids a same-frame GPU stall).
4. **Worker thread**: BGRA->RGB/normalize/CHW -> ONNX (DirectML) -> decode
   `[1,84,8400]` + per-class NMS -> publish results + bump `result_seq`.
5. `video_tick` (graphics thread, once/frame before render): Kalman `predict`
   every frame; on a new `result_seq`, `update` (greedy per-class IoU assoc).
6. `video_render`: draw the smoothed tracks (or raw detections if smoothing off)
   as green outline boxes, scaled 640-space -> source pixels (independent sx/sy).

## Files
| File | Purpose |
|------|---------|
| `obs-detection-overlay.cpp` | filter: registration, readback, worker, tick/render, properties |
| `detector.hpp` | ONNX Runtime C++ wrapper; Backend enum (CPU/DML/CUDA); preprocess |
| `detection_postprocess.hpp` | decode `[1,84,8400]` + per-class NMS + COCO-80 names |
| `tracker.hpp` | SORT-style tracker: 4 decoupled 1-D Kalman filters per box |
| `CMakeLists.txt` | links ONNX Runtime DirectML; stages DLLs into rundir |
| `data/models/model.onnx` | bundled model (images[1,3,640,640] -> output0[1,84,8400]) |
| `data/locale/*.ini` | en-US, zh-TW strings |

## Execution provider — important findings
- **DirectML (default)**: works on the RTX 5070 (Blackwell sm_120), ~1.8 ms.
- **CUDA: NOT usable here.** ONNX Runtime 1.26 (CUDA-12 build, CUDA-13 build,
  AND the official `onnxruntime-gpu` pip package) all HARD-CRASH (`0xC0000409`)
  inside CUDA session creation on Blackwell. Confirmed in a clean venv with
  official wheels — it is an ORT-vs-Blackwell issue, not a config/toolkit
  problem. `Backend::CUDA` code is kept but not selected; revisit when a
  Blackwell-compatible ORT ships.

## Dependencies
- ONNX Runtime **DirectML** C++ build at `C:/Users/mense/dev/deps/ort-dml`
  (CMake `ONNXRUNTIME_ROOT`): onnxruntime.dll + DirectML.dll + onnxruntime.lib
  + headers. From `Microsoft.ML.OnnxRuntime.DirectML` NuGet 1.24.4 + DirectML.dll
  from the `onnxruntime-directml` wheel.
- DirectML.dll is staged next to the plugin DLL by a POST_BUILD step.

## Build & run
See README.md. Short version:
```
cmake -S <obs> -B <obs>/build_x64
cmake --build <obs>/build_x64 --config RelWithDebInfo --target obs-detection-overlay
```
Run `<obs>/build_x64/rundir/RelWithDebInfo/bin/64bit/obs64.exe`; add the filter
to a source.

## Object coordinates (center points)

Every rendered frame, each detected object's center point is computed in
**source pixel coordinates** and saved to a thread-safe variable for later use.

```cpp
struct ObjCenter {
    int   id;       // stable Kalman track id (>=0 with smoothing; -1 raw)
    int   cls;      // COCO class index; name = coco_names[cls]
    float cx, cy;   // center, in source pixels
    float score;    // confidence
};
```

- Stored in `detector_filter::centers` (guarded by `centers_mtx`), refreshed each frame.
- Read via `std::vector<ObjCenter> detector_get_centers(detector_filter *f)` (returns a
  thread-safe copy).
- IDs are stable across frames (from the Kalman tracker) when smoothing is on.
- Coordinate space is SOURCE pixels (the object's real position on the captured
  source), not 640 network space.
- Verified: e.g. `5 objects; e.g. id=1 person center=(740,633) score=0.90`.

Status: the accessor is a foundation (no consumer yet). To make the coordinates
available to an external program, a delivery channel is still needed — options:
obs-websocket vendor request/event, a JSON file, or shared memory.

## Code-review findings
- FIXED: model-combo path separators (generic_string) so the active model shows
  as selected and doesn't trigger a redundant reload.
- Verified correct: worker-only `detector` ownership; tick/render same-thread
  ordering (no lock needed for the tracker); Kalman predict/update math;
  `result_seq` memory ordering; deferred-map readback; coordinate mapping.

## Known limitations / not done
- Async/webcam source readback uses the nv-filters pattern but was only tested
  with image/display-capture sources — verify once with a webcam.
- `obs_module_file` returns a path relative to the bin dir here ("../../data/..")
  so the model resolves relative to OBS's working dir — fine for a normal launch,
  fragile otherwise. Consider resolving to an absolute path.
- After a model switch, old tracks linger ~`max_misses` (30) frames.
- Only `model.onnx` is bundled, so the model picker has one entry.

## Possible next steps (legitimate)
- Draw class-name labels (`coco_names` is available; intentionally off).
- Verify on a live webcam / moving source.
- Filter property for smoothing strength; absolute model path.
- For protecting a high-value model commercially: server-side inference
  (thin client + your API) and software licensing/activation — the model never
  ships to the client. (Discussed; not started.)
