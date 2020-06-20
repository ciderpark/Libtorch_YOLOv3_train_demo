cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(yolov3-coco)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
find_package(OpenCV REQUIRED)

include_directories(/path/to/YOLO-V3_coco/inc)

add_executable(yolov3-coco main.cpp)
target_link_libraries(yolov3-coco "${TORCH_LIBRARIES}")
target_link_libraries(yolov3-coco ${OpenCV_LIBS})
set_property(TARGET yolov3-coco PROPERTY CXX_STANDARD 14)
