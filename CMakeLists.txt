cmake_minimum_required(VERSION 3.22)
project(gpu_fft LANGUAGES CUDA CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

set(CMAKE_BUILD_TYPE Release)


find_package(CUDAToolkit REQUIRED)
find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG)

add_executable(gpu_fft 
    src/perf_test.cu
    src/main.cpp)

target_include_directories(gpu_fft PUBLIC includes/)
target_include_directories(gpu_fft PUBLIC thirdparty/cutlass/include)
target_link_libraries(gpu_fft mathdx::cufftdx)
target_link_libraries(gpu_fft CUDA::cufft CUDA::cudart)
target_compile_options(gpu_fft PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info>)



file(GLOB BENCH_SOURCES ${PROJECT_SOURCE_DIR}/src/bench/*.cu)
add_executable(bench_fft ${BENCH_SOURCES} src/bench.cu)
target_include_directories(bench_fft PUBLIC includes/)
target_link_libraries(bench_fft mathdx::cufftdx)
