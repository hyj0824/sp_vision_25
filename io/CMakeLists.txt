cmake_minimum_required(VERSION 3.16)

find_package(yaml-cpp REQUIRED)

add_subdirectory(serial)

# 创建目标 io
add_library(io STATIC 
    hikrobot/hikrobot.cpp    
    mindvision/mindvision.cpp  
    usbcamera/usbcamera.cpp  
    camera.cpp
    gimbal/gimbal.cpp
)

# HIK SDK
if (DEFINED ENV{MVCAM_SDK_PATH})
    target_include_directories(io PUBLIC $ENV{MVCAM_SDK_PATH}/include)
    if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        target_link_directories(io PUBLIC $ENV{MVCAM_SDK_PATH}/lib/aarch64)
    else ()
        target_link_directories(io PUBLIC $ENV{MVCAM_SDK_PATH}/lib/64)
    endif ()
    target_link_libraries(io MvCameraControl)
    message("Hik SDK found at $ENV{MVCAM_SDK_PATH}")
else ()
    message(WARNING "Hik SDK not found, skipping Hik camera support.")
endif ()

# mindvision
target_include_directories(io PUBLIC mindvision/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  target_link_directories(io PUBLIC mindvision/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC mindvision/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()
target_link_libraries(io MVSDK)

# other libs
target_link_libraries(io usb-1.0 yaml-cpp serial)
