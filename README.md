# obs-detection-overlay

**English** · [繁體中文](README.zh-TW.md)

An OBS Studio video filter that runs object detection with ONNX Runtime
(DirectML), draws detection boxes on the preview, and can output a relative move
command to an external device for auto-tracking. Standalone repo, built
**in-tree** against the OBS Studio source.

## Deploy

### Build (in-tree)

1. Put this folder at `<obs-studio>/plugins/obs-detection-overlay`.
2. Add one line to `<obs-studio>/plugins/CMakeLists.txt` (alphabetical order):
   ```cmake
   add_obs_plugin(obs-detection-overlay PLATFORMS WINDOWS)
   ```
3. Point `ONNXRUNTIME_ROOT` at an ONNX Runtime **DirectML** distribution
   (with `onnxruntime.dll` + `DirectML.dll` + `dml_provider_factory.h`, e.g. the
   `Microsoft.ML.OnnxRuntime.DirectML` NuGet package).
4. Configure and build just this target:
   ```
   cmake -S <obs-studio> -B <obs-studio>/build_x64
   cmake --build <obs-studio>/build_x64 --config RelWithDebInfo --target obs-detection-overlay
   ```
   The plugin DLL and its dependencies land in
   `build_x64/rundir/<config>/obs-plugins/64bit/`.

### Use

Launch `build_x64/rundir/<config>/bin/64bit/obs64.exe`, right-click a source →
Filters → add **Detection Overlay**. Settings are grouped into 5 sections
(1. Detection, 2. Box display/classes, 3. Regions ROI/POV, 4. Device,
5. Auto-tracking) and all apply live.

### Swap the model

Drop any `*.onnx` into `data/models/` (it appears in the model picker). Input
must be `1×3×640×640`.

## Troubleshooting

- **Plugin not loading / OBS in safe mode**: a previous crash makes OBS enter
  safe mode (all third-party plugins disabled). Delete the `run_*` files under
  `%APPDATA%\obs-studio\.sentinel\`, then relaunch.
- **DLL becomes `libxxx.dll` and won't load**: CMake needs `PREFIX ""`.
- **Poor detection / small targets missed**: lower the **Inference region** so
  only a centered square is fed to the model at native resolution — targets get
  bigger and detection improves (stretching the whole frame distorts and shrinks
  them).
- **Close targets merge into one box**: enable **Soft-NMS**, or adjust the NMS
  IoU threshold.
- **CUDA crash on RTX 50-series (Blackwell, sm_120)**: ONNX Runtime 1.26 CUDA
  hard-crashes on this platform — use the default **DirectML**.
- **Auto-tracking can't keep up with moving targets**: ⚠️ **known limitation** —
  the controller (`PredCtl`) still leaves residual lag on fast targets and does
  not fully center them; static/slow targets lock stably. **Contributions to
  improve the controller are welcome** — the control law is isolated in
  `PredCtl` and the tracking block of `filter_video_render()`, decoupled from the
  detection/tracking pipeline and easy to swap or experiment with.

## Contributing

Do not edit/push `main` directly. Fork or branch, then open a Pull Request — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

GPLv2 — see [LICENSE](LICENSE).
