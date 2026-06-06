// detector.hpp -- object-detection inference via ONNX Runtime C++ API (v1.26),
// no OpenCV.
// Verified against deps/onnxruntime-win-x64-1.26.0/include/onnxruntime_cxx_api.h
//
// Header-only. Single class Detector. Owns Ort::Env / SessionOptions / Session.
// run() takes a caller-owned CHW float buffer (1x3x640x640) and returns a
// NON-OWNING view of the model output (e.g. [1,84,8400]). The view is valid only
// until the next run()/runAll() on the same instance -- copy anything you keep.
//
// Threading: ONE instance per worker thread. Do NOT call run() concurrently on
// the same instance. preprocessBGRA() is a static pure function, safe anywhere.
#pragma once

#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>  // DirectML EP (GPU on any DX12 device)

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>  // MultiByteToWideChar for std::string -> std::wstring path
#endif

// Inference backend / execution provider preference. DML = DirectML (GPU via
// DX12, works on Blackwell where the ORT CUDA build crashes). All fall back to
// CPU if the GPU EP cannot be created.
enum class Backend { CPU, DML, CUDA };

class Detector {
 public:
  struct Output {
    const float* data = nullptr;   // raw output (e.g. [1,84,8400] flattened)
    std::vector<int64_t> shape;    // typically {1,84,8400}
    size_t element_count = 0;
  };

  // Un-letterbox helper (unused in the stretched-render path, kept for parity):
  //   src_x = (mx - pad_x) / scale, src_y = (my - pad_y) / scale
  struct Letterbox {
    float scale = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
    int src_w = 0;
    int src_h = 0;
  };

  static constexpr int kNetW = 640;
  static constexpr int kNetH = 640;
  static constexpr int kNetC = 3;
  static constexpr size_t kInputElems =
      static_cast<size_t>(kNetC) * kNetH * kNetW;

  explicit Detector(const std::string& model_path, int intra_op_threads = 1,
                   Backend backend = Backend::DML)
      : env_(ORT_LOGGING_LEVEL_WARNING, "Detector") {
#ifdef _WIN32
    Init(ToWide(model_path).c_str(), intra_op_threads, backend);
#else
    Init(model_path.c_str(), intra_op_threads, backend);
#endif
  }

#ifdef _WIN32
  explicit Detector(const std::wstring& model_path, int intra_op_threads = 1,
                   Backend backend = Backend::DML)
      : env_(ORT_LOGGING_LEVEL_WARNING, "Detector") {
    Init(model_path.c_str(), intra_op_threads, backend);
  }
#endif

  Detector(const Detector&) = delete;
  Detector& operator=(const Detector&) = delete;
  Detector(Detector&&) = default;
  Detector& operator=(Detector&&) = default;

  // chw_640x640: kInputElems contiguous floats (NCHW, N=1). NOT copied -- must
  // stay valid for the call. Returned pointer invalidated by next run().
  Output run(const float* chw_640x640) {
    last_outputs_ = runRaw(chw_640x640);
    if (last_outputs_.empty())
      return Output{};
    return MakeView(last_outputs_.front());  // pass by const ref, no copy
  }

  std::vector<Output> runAll(const float* chw_640x640) {
    last_outputs_ = runRaw(chw_640x640);
    std::vector<Output> views;
    views.reserve(last_outputs_.size());
    for (auto& v : last_outputs_) views.push_back(MakeView(v));
    return views;
  }

  // BGRA8 src -> letterboxed 1x3x640x640 CHW float (dst holds kInputElems).
  // linesize = bytes per source row (frame buffers are often padded).
  // NOTE: the obs-detection-overlay filter uses a GPU STRETCHED 640x640 render and
  // does NOT call this; it builds CHW directly. This static helper is retained
  // for callers that want CPU-side letterboxing.
  static Letterbox preprocessBGRA(const uint8_t* src, int width, int height,
                                  int linesize, float* dst) {
    if (!src || !dst || width <= 0 || height <= 0)
      throw std::invalid_argument("preprocessBGRA: bad arguments");

    Letterbox lb;
    lb.src_w = width;
    lb.src_h = height;

    const float scale = std::min(static_cast<float>(kNetW) / width,
                                 static_cast<float>(kNetH) / height);
    lb.scale = scale;

    const int new_w = static_cast<int>(width * scale + 0.5f);
    const int new_h = static_cast<int>(height * scale + 0.5f);
    const int pad_x = (kNetW - new_w) / 2;
    const int pad_y = (kNetH - new_h) / 2;
    lb.pad_x = static_cast<float>(pad_x);
    lb.pad_y = static_cast<float>(pad_y);

    const size_t plane = static_cast<size_t>(kNetW) * kNetH;
    float* dst_r = dst + 0 * plane;
    float* dst_g = dst + 1 * plane;
    float* dst_b = dst + 2 * plane;

    const float kPad = 114.0f / 255.0f;
    for (size_t i = 0; i < plane; ++i) {
      dst_r[i] = kPad;
      dst_g[i] = kPad;
      dst_b[i] = kPad;
    }
    if (new_w <= 0 || new_h <= 0) return lb;

    const float inv_scale_x = static_cast<float>(width) / new_w;
    const float inv_scale_y = static_cast<float>(height) / new_h;

    for (int dy = 0; dy < new_h; ++dy) {
      int sy = static_cast<int>((dy + 0.5f) * inv_scale_y);
      if (sy >= height) sy = height - 1;
      const uint8_t* srow = src + static_cast<size_t>(sy) * linesize;

      const int canvas_y = pad_y + dy;
      float* row_r = dst_r + static_cast<size_t>(canvas_y) * kNetW;
      float* row_g = dst_g + static_cast<size_t>(canvas_y) * kNetW;
      float* row_b = dst_b + static_cast<size_t>(canvas_y) * kNetW;

      for (int dx = 0; dx < new_w; ++dx) {
        int sx = static_cast<int>((dx + 0.5f) * inv_scale_x);
        if (sx >= width) sx = width - 1;
        const uint8_t* px = srow + static_cast<size_t>(sx) * 4;  // BGRA
        const int canvas_x = pad_x + dx;
        row_r[canvas_x] = px[2] / 255.0f;  // R
        row_g[canvas_x] = px[1] / 255.0f;  // G
        row_b[canvas_x] = px[0] / 255.0f;  // B
      }
    }
    return lb;
  }

  const std::string& inputName() const { return input_name_; }
  const std::vector<std::string>& outputNames() const { return output_names_; }
  size_t numOutputs() const { return output_names_.size(); }
  const std::string& provider() const { return provider_; }  // "CUDA" / "CPU"

 private:
  // Configure `so` for the requested backend; sets provider_ and returns true if
  // a GPU EP was appended (so the caller can retry on CPU if session creation
  // fails). Falls back to CPU within this call if the GPU EP cannot be appended.
  bool ConfigureProvider(Ort::SessionOptions& so, int threads, Backend backend) {
    so.SetIntraOpNumThreads(threads);
    so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    if (backend == Backend::DML) {
      try {
        // DirectML EP requirements: no arena mem pattern, sequential exec.
        so.DisableMemPattern();
        so.SetExecutionMode(ORT_SEQUENTIAL);
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(so, 0));
        provider_ = "DML";
        return true;
      } catch (const std::exception&) {
        // rebuild clean options for the CPU fallback below
        so = Ort::SessionOptions();
        so.SetIntraOpNumThreads(threads);
        so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
      }
    } else if (backend == Backend::CUDA) {
      // NOTE: ORT 1.26's CUDA build hard-crashes at session creation on
      // Blackwell (sm_120) -- uncatchable. Only select CUDA on known-good GPUs.
      try {
        OrtCUDAProviderOptionsV2* opts = nullptr;
        Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&opts));
        const char* k[] = {"cudnn_conv_algo_search"};
        const char* v[] = {"DEFAULT"};
        Ort::ThrowOnError(Ort::GetApi().UpdateCUDAProviderOptions(opts, k, v, 1));
        so.AppendExecutionProvider_CUDA_V2(*opts);
        Ort::GetApi().ReleaseCUDAProviderOptions(opts);
        provider_ = "CUDA";
        return true;
      } catch (const std::exception&) {
      }
    }
    provider_ = "CPU";
    return false;
  }

  void Init(const ORTCHAR_T* model_path, int intra_op_threads, Backend backend) {
    if (intra_op_threads < 1) intra_op_threads = 1;
    if (intra_op_threads > 2) intra_op_threads = 2;

    const bool gpu =
        ConfigureProvider(session_options_, intra_op_threads, backend);

    try {
      session_ = Ort::Session(env_, model_path, session_options_);
    } catch (const std::exception&) {
      if (!gpu) throw;  // CPU-only failure is a real error
      // GPU EP failed at session creation -> rebuild CPU-only and retry.
      session_options_ = Ort::SessionOptions();
      ConfigureProvider(session_options_, intra_op_threads, Backend::CPU);
      provider_ += " (GPU unavailable)";
      session_ = Ort::Session(env_, model_path, session_options_);
    }

    Ort::AllocatorWithDefaultOptions allocator;

    const size_t in_count = session_.GetInputCount();
    if (in_count != 1)
      throw std::runtime_error("Detector: expected 1 input, got " +
                               std::to_string(in_count));
    {
      Ort::AllocatedStringPtr n = session_.GetInputNameAllocated(0, allocator);
      input_name_ = n.get();
    }

    const size_t out_count = session_.GetOutputCount();
    if (out_count == 0)
      throw std::runtime_error("Detector: model has no outputs");
    output_names_.reserve(out_count);
    for (size_t i = 0; i < out_count; ++i) {
      Ort::AllocatedStringPtr n = session_.GetOutputNameAllocated(i, allocator);
      output_names_.emplace_back(n.get());
    }

    input_name_cstr_.push_back(input_name_.c_str());
    output_names_cstr_.reserve(output_names_.size());
    for (const auto& s : output_names_) output_names_cstr_.push_back(s.c_str());

    memory_info_ =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  }

  std::vector<Ort::Value> runRaw(const float* chw_640x640) {
    if (!chw_640x640)
      throw std::invalid_argument("Detector::run: null input buffer");
    static const int64_t kInputShape[4] = {1, kNetC, kNetH, kNetW};
    Ort::Value input = Ort::Value::CreateTensor<float>(
        memory_info_, const_cast<float*>(chw_640x640), kInputElems,
        kInputShape, 4);
    return session_.Run(Ort::RunOptions{nullptr}, input_name_cstr_.data(),
                        &input, 1, output_names_cstr_.data(),
                        output_names_cstr_.size());
  }

  static Output MakeView(const Ort::Value& v) {
    Output out;
    if (!v) return out;
    auto info = v.GetTensorTypeAndShapeInfo();
    out.shape = info.GetShape();
    out.element_count = info.GetElementCount();
    out.data = v.GetTensorData<float>();
    return out;
  }

#ifdef _WIN32
  static std::wstring ToWide(const std::string& s) {
    if (s.empty()) return std::wstring();
    int len = ::MultiByteToWideChar(CP_UTF8, 0, s.data(),
                                    static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(len), L'\0');
    ::MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()),
                          &w[0], len);
    return w;
  }
#endif

  Ort::Env env_;                              // must outlive session_
  Ort::SessionOptions session_options_;
  Ort::Session session_{nullptr};
  Ort::MemoryInfo memory_info_{nullptr};
  std::string provider_ = "CPU";
  std::string input_name_;
  std::vector<std::string> output_names_;
  std::vector<const char*> input_name_cstr_;
  std::vector<const char*> output_names_cstr_;
  std::vector<Ort::Value> last_outputs_;      // keeps Output::data valid
};
