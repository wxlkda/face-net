cmake_minimum_required(VERSION 3.10)
project(FaceNetCpp)

set(CMAKE_CXX_STANDARD 14)
set(TENSORFLOW_DIR "/path/to/tensorflow")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${TENSORFLOW_DIR}/include)
link_directories(${TENSORFLOW_DIR}/lib)

add_executable(FaceNetCpp main.cpp src/facenet.cpp src/dataset.cpp)
target_link_libraries(FaceNetCpp ${OpenCV_LIBS} tensorflow_cc tensorflow_framework)
