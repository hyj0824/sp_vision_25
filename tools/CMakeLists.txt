cmake_minimum_required(VERSION 3.16)

find_package(CUDAToolkit REQUIRED)

find_path(
    TENSORRT_INCLUDE_DIR
    NvInfer.h
    PATHS
        /usr/include
        /usr/include/aarch64-linux-gnu
        /usr/include/x86_64-linux-gnu
)

find_library(TENSORRT_LIB nvinfer)
find_library(TENSORRT_PLUGIN_LIB nvinfer_plugin)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIB)
  message(FATAL_ERROR "TensorRT not found. Please install TensorRT dev packages on this platform.")
endif()

set(TENSORRT_LIBRARIES ${TENSORRT_LIB})
if(TENSORRT_PLUGIN_LIB)
  list(APPEND TENSORRT_LIBRARIES ${TENSORRT_PLUGIN_LIB})
endif()

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

target_include_directories(trt_infer PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(trt_infer PRIVATE spdlog::spdlog CUDA::cudart ${TENSORRT_LIBRARIES})
