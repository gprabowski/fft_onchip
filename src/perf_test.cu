#include "config.hpp"
#include <perf_test.cuh>

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <common.cuh>
#include <legacy16_fft.cuh>
#include <legacy8_fft.cuh>
#include <reference.cuh>
#include <tensor_fft_4096.cuh>
#include <tensor_fft_64.cuh>
#include <tensor_fft_8.cuh>
#include <tester.cuh>

#include <chrono>

namespace testing {

template <int InnerRuns, int SharedSize, typename CT, int Size,
          typename FFTExec>
__forceinline__ double run_fft_kernel(CT *data, size_t sm_count) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  constexpr auto FinalShared =
      SharedSize * FFTExec::ffts_per_block * FFTExec::ffts_per_unit;

  gpuErrchk(cudaFuncSetAttribute(
      tester::fft_tester<InnerRuns, CT, FFTExec, Size, FFTExec::ffts_per_block>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, FinalShared));

  gpuErrchk(cudaEventRecord(start));
  // Running around 100 blocks / SM
  tester::fft_tester<InnerRuns, CT, FFTExec, Size, FFTExec::ffts_per_block>
      <<<(config::sm_multiplier * sm_count) /
             (FFTExec::ffts_per_block * FFTExec::ffts_per_unit),
         dim3(FFTExec::threads, FFTExec::ffts_per_block, 1), FinalShared>>>(
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
double run_perf_test(const std::vector<config::CT> &data,
                     std::vector<config::CT> &out) {
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

  constexpr auto sm_size = Size * sizeof(CT);

  // correctness check
  run_fft_kernel<1, sm_size, CT, Size, FFTExec>(
      thrust::raw_pointer_cast(d_data.data()), sm_count);

  h_data = d_data;
  out.resize(h_data.size());
  for (int i = 0; i < h_data.size(); ++i) {
    out[i] = config::CT{h_data[i].real(), h_data[i].imag()};
  }

  const auto time_100 = run_fft_kernel<100, sm_size, CT, Size, FFTExec>(
      thrust::raw_pointer_cast(d_data.data()), sm_count);

  const auto time_1100 = run_fft_kernel<1100, sm_size, CT, Size, FFTExec>(
      thrust::raw_pointer_cast(d_data.data()), sm_count);

  // Will return time in microseconds
  const double final_time = static_cast<double>(time_1100 - time_100) / 1000.0;

  return final_time;
}

void test(std::vector<config::CT> &data) {
  std::vector<config::CT> out_algorithm, out_reference;

  using customExec = fft::tensor_fft_64<config::CT, config::N>;
  using refExec = fft::reference_fft<config::N>;

  auto alg_data = data;
  auto ref_data = data;

  const auto alg_run =
      run_perf_test<config::CT, config::N, customExec>(alg_data, out_algorithm);

  const auto ref_run =
      run_perf_test<refExec::VT, config::N, refExec>(ref_data, out_reference);

  double mse{0.0};

  for (int i = 0; i < out_reference.size(); ++i) {
    const auto se = norm(out_reference[i] - out_algorithm[i]);
    mse += se;
  }

  if constexpr (config::print_results) {
    for (int i = 0; i < data.size() * 16; ++i) {
      std::cout << " Ref: " << out_reference[i].real() << " "
                << out_reference[i].imag()
                << " Ten: " << out_algorithm[i].real() << " "
                << out_algorithm[i].imag() << std::endl;
    }
  }

  mse /= data.size();

  std::cout << "Tensor FFT took: " << alg_run << " microseconds \n";
  std::cout << "Reference took: " << ref_run << " microseconds \n";
  std::cout << "MSE: " << mse << "\n";
}
} // namespace testing
