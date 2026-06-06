# obs-detection-overlay

[English](README.md) · **繁體中文**

OBS Studio 影像濾鏡：以 ONNX Runtime（DirectML）執行物件偵測，在預覽畫面畫出偵測框，
並可輸出相對移動指令給外部裝置做自動追蹤。獨立倉庫，以 **in-tree** 方式對 OBS Studio
原始碼編譯。

## 部署

### 編譯（in-tree）

1. 把此資料夾放到 `<obs-studio>/plugins/obs-detection-overlay`。
2. 在 `<obs-studio>/plugins/CMakeLists.txt` 加一行（依字母順序）：
   ```cmake
   add_obs_plugin(obs-detection-overlay PLATFORMS WINDOWS)
   ```
3. 把 `ONNXRUNTIME_ROOT` 指向一份 ONNX Runtime **DirectML** 發行版
   （含 `onnxruntime.dll` + `DirectML.dll` + `dml_provider_factory.h`，例如
   `Microsoft.ML.OnnxRuntime.DirectML` NuGet 套件）。
4. 設定並只編譯此目標：
   ```
   cmake -S <obs-studio> -B <obs-studio>/build_x64
   cmake --build <obs-studio>/build_x64 --config RelWithDebInfo --target obs-detection-overlay
   ```
   外掛 DLL 與相依檔會落在 `build_x64/rundir/<config>/obs-plugins/64bit/`。

### 使用

從 `build_x64/rundir/<config>/bin/64bit/obs64.exe` 啟動，對來源右鍵 → 濾鏡 →
新增 **偵測疊加**。設定分為 5 組（① 偵測 ② 框顯示/類別 ③ 範圍 ROI/POV ④ 設備
⑤ 自動追蹤），全部即時生效。

### 更換模型

把任意 `*.onnx` 放進 `data/models/`（會出現在模型選單）。輸入需為 `1×3×640×640`。

## 問題解決

- **外掛沒載入 / 進入安全模式**：上次崩潰會讓 OBS 進安全模式（停用所有第三方外掛）。
  刪除 `%APPDATA%\obs-studio\.sentinel\` 下的 `run_*` 檔後再啟動。
- **DLL 變成 `libxxx.dll` 而無法載入**：CMake 需設 `PREFIX ""`。
- **偵測效果差 / 目標太小漏偵測**：把「**推理範圍**」調小，只取畫面中央方形區域 —— 原
  比例、原解析度，目標變大、偵測更準（整張壓縮會失真且縮小目標）。
- **相近目標的框融合成一個**：開啟 **Soft-NMS**，或調整 NMS IoU 閾值。
- **RTX 50 系列（Blackwell, sm_120）用 CUDA 崩潰**：ONNX Runtime 1.26 的 CUDA 在此平台
  會硬崩潰，請用預設的 **DirectML**。
- **自動追蹤追不上移動目標**：⚠️ **已知限制** —— 控制器（`PredCtl`）對快速移動目標仍有
  殘餘落後，無法完全置中；靜止/慢速目標可穩定鎖定。**歡迎二次開發優化控制器**（控制律
  集中在 `obs-detection-overlay.cpp` 的 `PredCtl` 與 `filter_video_render()` 追蹤區塊，
  與偵測/追蹤管線解耦，易於替換或實驗）。

## 授權

GPLv2 —— 見 [LICENSE](LICENSE)。
