cmake_minimum_required(VERSION 3.16)

add_subdirectory(planner/tinympc)

add_library(auto_aim OBJECT 
    armor.cpp
    classifier.cpp
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
)

target_link_libraries(
    auto_aim
    PUBLIC
        ${OpenCV_LIBS}
        fmt::fmt
        yaml-cpp
        nlohmann_json::nlohmann_json
        io
        trt_infer
        tinympcstatic
)
