cmake_minimum_required(VERSION 3.16)

find_package(Ceres REQUIRED)

add_library(auto_buff OBJECT 
    rune_detector.cpp
    ceres_rune_predictor.cpp
)

if(TARGET Ceres::ceres)
    set(AUTO_BUFF_CERES_TARGET Ceres::ceres)
else()
    set(AUTO_BUFF_CERES_TARGET ${CERES_LIBRARIES})
endif()

if(CERES_INCLUDE_DIRS)
    target_include_directories(auto_buff PUBLIC ${CERES_INCLUDE_DIRS})
endif()

target_link_libraries(auto_buff PUBLIC trt_infer ${AUTO_BUFF_CERES_TARGET})
