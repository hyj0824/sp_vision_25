cmake_minimum_required(VERSION 3.16)

add_library(omniperception OBJECT 
    decider.cpp
    perceptron.cpp
)
target_link_libraries(omniperception PUBLIC auto_aim trt_infer)
