cmake_minimum_required(VERSION 3.28...3.30)

add_library(obs-detection-overlay MODULE)
add_library(OBS::detection-overlay ALIAS obs-detection-overlay)

target_sources(
  obs-detection-overlay
  PRIVATE
    obs-detection-overlay.cpp
    control.cpp
    control.h
    detector.hpp
    detection_postprocess.hpp
    tracker.hpp
    device.h
    device_symbols.h)
# NOTE: device_symbols.h (the real device.dll export-name strings) is supplied
# separately by the device owner and is REQUIRED to build. Until that file is
# placed in this folder, CMake configure/build will fail on this target by
# design. Do not auto-generate or read it — the owner adds it.

set_target_properties(obs-detection-overlay PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)

# --- ONNX Runtime (DirectML build: GPU via DX12, CPU fallback) -------------
# DirectML works on this Blackwell RTX 5070 (~1.8 ms/frame); the ORT CUDA build
# hard-crashes on sm_120. Root holds include/ + lib/ (onnxruntime.dll[.lib],
# DirectML.dll, onnxruntime_providers_shared.dll).
set(ONNXRUNTIME_ROOT
    "C:/Users/mense/dev/deps/ort-dml"
    CACHE PATH "Root directory of the ONNX Runtime DirectML distribution")

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/dml_provider_factory.h")
  message(FATAL_ERROR "ONNX Runtime DirectML not found: '${ONNXRUNTIME_ROOT}/include/dml_provider_factory.h' is missing. Set -DONNXRUNTIME_ROOT to a DirectML ORT distribution.")
endif()

# IMPORTED SHARED target wrapping onnxruntime.lib (import lib) + onnxruntime.dll.
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(
  onnxruntime
  PROPERTIES
    IMPORTED_IMPLIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib"
    IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")

target_include_directories(obs-detection-overlay PRIVATE "${ONNXRUNTIME_ROOT}/include")

target_link_libraries(
  obs-detection-overlay
  PRIVATE
    OBS::libobs
    onnxruntime
    $<$<PLATFORM_ID:Windows>:OBS::w32-pthreads>)

# Stage the ONNX Runtime + DirectML DLLs into the rundir plugin dir (where OBS
# loads the plugin from) so the loader resolves them at runtime.
# $<TARGET_FILE_DIR:...> is only the build dir; OBS installs the plugin to
# ${OBS_OUTPUT_DIR}/<config>/obs-plugins/64bit.
set(_ort_runtime_dlls
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
    "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
    "${ONNXRUNTIME_ROOT}/lib/DirectML.dll")

# Also stage device.dll (the motion-device DLL; loaded at runtime by full path,
# not linked) when it sits next to the sources, so it ships with the plugin and
# can be picked from the plugin folder.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/device.dll")
  list(APPEND _ort_runtime_dlls "${CMAKE_CURRENT_SOURCE_DIR}/device.dll")
endif()

add_custom_command(
  TARGET obs-detection-overlay
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
          "${OBS_OUTPUT_DIR}/$<CONFIG>/${OBS_PLUGIN_DESTINATION}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${_ort_runtime_dlls}
          "${OBS_OUTPUT_DIR}/$<CONFIG>/${OBS_PLUGIN_DESTINATION}/"
  COMMENT "Staging ONNX Runtime + DirectML (+ device.dll if present) into rundir obs-plugins/64bit"
  VERBATIM)

if(OS_WINDOWS)
  configure_file(cmake/windows/obs-module.rc.in obs-detection-overlay.rc)
  target_sources(obs-detection-overlay PRIVATE obs-detection-overlay.rc)
endif()

set_target_properties_obs(obs-detection-overlay PROPERTIES FOLDER plugins PREFIX "")
