# PyTorch-CPP

Neural Network using PyTorch C++

## Installation

1. Install g++, cmake

```
sudo apt install g++ cmake
```

2. Download the Pytorch ***libtorch*** package and unzip it : ![libtorch]('https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.2.2%2Bcpu.zip')

3. create a file : CMakelists.txt

```c++
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(main)
set(CMAKE_PREFIX_PATH /home/ss/STUDY/PyTorch-CPP/libtorch) // path to torchlib after unzip
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(main net.cpp)                                 // filepath of of .cpp file 
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)          // standard to 17

```

4. make a folder ***build*** and inside build folder : call ``` cmake ..```

5. then inside build folder : ```make```

6. run the generated executable : ```./main```

