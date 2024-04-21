cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(main)

set(CMAKE_PREFIX_PATH /home/ss/STUDY/PyTorch-CPP/libtorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")



option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

add_executable(main CNNMNIST.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}")


set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS} )

set_property(TARGET main PROPERTY CXX_STANDARD 20)
