cmake_minimum_required(VERSION 3.22)
project(gpu_fft LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_BUILD_TYPE Debug)

find_package(CUDAToolkit REQUIRED)
find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG)

add_executable(gpu_fft 
    src/reference.cu
    src/algorithm.cu
    src/main.cpp)

target_include_directories(gpu_fft PUBLIC includes/)
target_link_libraries(gpu_fft mathdx::cufftdx)

target_link_libraries(gpu_fft CUDA::cufft CUDA::cudart)
