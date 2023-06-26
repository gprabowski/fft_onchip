#pragma once

#include <vector>

#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <common.cuh>
#include <config.hpp>

namespace testing {
namespace cg = cooperative_groups;

template <typename CT, typename FFTExec, int Size, int FFTsPerBlock>

__launch_bounds__(FFTExec::max_threads_per_block) __global__
    void fft_tester(int inner_repeats, CT *data) {
  extern __shared__ __align__(sizeof(CT)) char shared[];
  CT *shared_data = reinterpret_cast<CT *>(shared);

  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  // 1. copy data
  cg::memcpy_async(block, shared_data,
                   data + Size * FFTExec::ffts_per_unit *
                              FFTExec::units_per_block * grid.block_rank(),
                   sizeof(CT) * Size * FFTExec::units_per_block *
                       FFTExec::ffts_per_unit);
  cg::wait(block);

  FFTExec fft(shared_data);
#pragma unroll 1
  for (int i = 0; i < inner_repeats; ++i) {
    fft();
  }

  block.sync();

  const auto elems_per_t =
      (Size * FFTExec::units_per_block * FFTExec::ffts_per_unit) / block.size();
  const auto batch_size = elems_per_t * sizeof(CT);
  memcpy(&data[Size * FFTExec::units_per_block * FFTExec::ffts_per_unit *
                   grid.block_rank() +
               block.thread_rank() * elems_per_t],
         &shared_data[block.thread_rank() * elems_per_t], batch_size);
}

template <typename CT, int Size, typename FFTExec>
__forceinline__ double run_fft_kernel(int inner_runs, int shared_size, CT *data,
                                      size_t sm_count) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const auto blocks = (config::sm_multiplier * sm_count) /
                      (FFTExec::units_per_block * FFTExec::ffts_per_unit);
  const auto threads = dim3(FFTExec::threads, FFTExec::units_per_block, 1);

  const auto final_shared =
      shared_size * FFTExec::units_per_block * FFTExec::ffts_per_unit;

  gpuErrchk(cudaFuncSetAttribute(
      fft_tester<CT, FFTExec, Size, FFTExec::units_per_block>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, final_shared));

  gpuErrchk(cudaEventRecord(start));
  fft_tester<CT, FFTExec, Size, FFTExec::units_per_block>
      <<<blocks, threads, final_shared>>>(inner_runs,
                                          thrust::raw_pointer_cast(data));
  gpuErrchk(cudaEventRecord(stop));
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  float time_elapsed;
  cudaEventElapsedTime(&time_elapsed, start, stop);

  // return microseconds
  return time_elapsed * 1000.f;
}

template <typename CT, int Size, typename FFTExec>
double run_perf_tests(const std::vector<config::CT> &data) {
  constexpr auto sm_size = Size * sizeof(CT);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // get number of SMs
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  const auto sm_count = props.multiProcessorCount;

  thrust::host_vector<CT> h_data;
  thrust::device_vector<CT> d_data;

  for (int i = 0; i < data.size() * config::sm_multiplier * sm_count; ++i) {
    h_data.push_back({data[i % Size].real(), data[i % Size].imag()});
  }

  d_data = h_data;

  const auto time_100 = run_fft_kernel<CT, Size, FFTExec>(
      100, sm_size, thrust::raw_pointer_cast(d_data.data()), sm_count);

  const auto time_1100 = run_fft_kernel<CT, Size, FFTExec>(
      1100, sm_size, thrust::raw_pointer_cast(d_data.data()), sm_count);

  // Will return time in microseconds
  const double final_time = static_cast<double>(time_1100 - time_100) / 1000.0;

  return final_time;
}

template <typename CT, int Size, typename FFTExec>
void perf_test_printer(const std::vector<config::CT> &data) {
  const auto t = testing::run_perf_tests<CT, Size, FFTExec>(data);
  std::cout << t << std::endl;
}

} // namespace testing
