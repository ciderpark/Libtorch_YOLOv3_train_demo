cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(yolov3-voc)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(OpenCV REQUIRED)

include_directories(/home/ciderpark/eclipse-workspace/YOLO-V3_voc2012/inc)

add_executable(yolov3-voc main.cpp)
target_link_libraries(yolov3-voc "${TORCH_LIBRARIES}")
target_link_libraries(yolov3-voc ${OpenCV_LIBS})
set_property(TARGET yolov3-voc PROPERTY CXX_STANDARD 14)
