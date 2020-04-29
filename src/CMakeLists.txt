cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(yolov3)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(/path/to/your/project/inc)

add_executable(yolov3 main.cpp)
target_link_libraries(yolov3 "${TORCH_LIBRARIES}")
target_link_libraries(yolov3 ${OpenCV_LIBS})
set_property(TARGET yolov3 PROPERTY CXX_STANDARD 11)
