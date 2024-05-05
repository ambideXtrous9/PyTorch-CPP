cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(main)

set(CMAKE_PREFIX_PATH /home/ss/STUDY/PyTorch-CPP/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main main.cpp)

target_link_libraries(main "${TORCH_LIBRARIES}")


set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS} )

set_property(TARGET main PROPERTY CXX_STANDARD 20)
