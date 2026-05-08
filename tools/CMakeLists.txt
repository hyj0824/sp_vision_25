cmake_minimum_required(VERSION 3.16)

find_path(
    TENSORRT_INCLUDE_DIR
    NvInfer.h
    HINTS
        $ENV{TENSORRT_ROOT}/include
        $ENV{TENSORRT_ROOT}/usr/include
        $ENV{TENSORRT_ROOT}/usr/include/x86_64-linux-gnu
    PATHS
        /usr/include
        /usr/include/aarch64-linux-gnu
        /usr/include/x86_64-linux-gnu
)

find_path(
    CUDA_RUNTIME_INCLUDE_DIR
    cuda_runtime_api.h
    HINTS
        $ENV{CUDA_HOME}/include
        $ENV{CUDA_ROOT}/include
    PATHS
        /usr/local/cuda/include
        /usr/local/cuda-12.2/include
        /usr/local/cuda-11.8/include
        /usr/include
)

find_library(
    TENSORRT_LIB
    nvinfer
    HINTS
        $ENV{TENSORRT_ROOT}/lib
        $ENV{TENSORRT_ROOT}/lib/x86_64-linux-gnu
        $ENV{TENSORRT_ROOT}/usr/lib
        $ENV{TENSORRT_ROOT}/usr/lib/x86_64-linux-gnu
)
find_library(
    TENSORRT_ONNXPARSER_LIB
    nvonnxparser
    HINTS
        $ENV{TENSORRT_ROOT}/lib
        $ENV{TENSORRT_ROOT}/lib/x86_64-linux-gnu
        $ENV{TENSORRT_ROOT}/usr/lib
        $ENV{TENSORRT_ROOT}/usr/lib/x86_64-linux-gnu
)
find_library(
    TENSORRT_PLUGIN_LIB
    nvinfer_plugin
    HINTS
        $ENV{TENSORRT_ROOT}/lib
        $ENV{TENSORRT_ROOT}/lib/x86_64-linux-gnu
        $ENV{TENSORRT_ROOT}/usr/lib
        $ENV{TENSORRT_ROOT}/usr/lib/x86_64-linux-gnu
)
find_library(
    CUDA_RUNTIME_LIB
    cudart
    HINTS
        $ENV{CUDA_HOME}/lib64
        $ENV{CUDA_HOME}/targets/x86_64-linux/lib
        $ENV{CUDA_ROOT}/lib64
        $ENV{CUDA_ROOT}/targets/x86_64-linux/lib
    PATHS
        /usr/local/cuda/lib64
        /usr/local/cuda-12.2/targets/x86_64-linux/lib
        /usr/local/cuda-11.8/targets/x86_64-linux/lib
)

add_library(tools OBJECT 
    exiter.cpp
    extended_kalman_filter.cpp
    ransac_sine_fitter.cpp
    img_tools.cpp
    math_tools.cpp
    plotter.cpp
    trajectory.cpp
    recorder.cpp
    logger.cpp
    pid.cpp
)

add_library(trt_infer SHARED
    infer/trt_engine.cpp
)

target_link_libraries(trt_infer PRIVATE spdlog::spdlog)

if(TENSORRT_INCLUDE_DIR AND CUDA_RUNTIME_INCLUDE_DIR AND TENSORRT_LIB AND TENSORRT_ONNXPARSER_LIB AND CUDA_RUNTIME_LIB)
  set(TENSORRT_LIBRARIES ${TENSORRT_LIB} ${TENSORRT_ONNXPARSER_LIB})
  if(TENSORRT_PLUGIN_LIB)
    list(APPEND TENSORRT_LIBRARIES ${TENSORRT_PLUGIN_LIB})
  endif()

  target_include_directories(trt_infer PRIVATE ${TENSORRT_INCLUDE_DIR} ${CUDA_RUNTIME_INCLUDE_DIR})
  target_link_libraries(trt_infer PRIVATE ${CUDA_RUNTIME_LIB} ${TENSORRT_LIBRARIES})
else()
  message(WARNING "TensorRT/CUDA dev files not found. trt_infer will build a runtime stub.")
endif()
