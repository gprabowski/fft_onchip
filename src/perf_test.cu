#include "config.hpp"
#include <perf_test.cuh>

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <common.cuh>
#include <legacy16_fft.cuh>
#include <legacy8_fft.cuh>
#include <reference.cuh>
#include <tensor_fft.cuh>
#include <tester.cuh>

#include <chrono>

namespace testing {

template <int InnerRuns, int SharedSize, typename CT, int Size, int Radix,
          typename FFTExec>
__forceinline__ double run_fft_kernel(CT *data) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  gpuErrchk(cudaFuncSetAttribute(
      tester::fft_tester<InnerRuns, CT, FFTExec, Size, Radix>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, SharedSize));

  gpuErrchk(cudaEventRecord(start));
  tester::fft_tester<InnerRuns, CT, FFTExec, Size, Radix>
      <<<2800, FFTExec::threads, SharedSize>>>(thrust::raw_pointer_cast(data));
  gpuErrchk(cudaEventRecord(stop));
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  float time_elapsed;
  cudaEventElapsedTime(&time_elapsed, start, stop);

  // return microseconds
  return time_elapsed * 1000.f;
}

template <typename CT, int Size, int Radix, typename FFTExec>
double run_perf_test(const std::vector<config::CT> &data,
                     std::vector<config::CT> &out) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  thrust::host_vector<CT> h_data;
  thrust::device_vector<CT> d_data;

  for (int i = 0; i < data.size(); ++i) {
    h_data.push_back({data[i].real(), data[i].imag()});
  }

  d_data = h_data;

  constexpr auto sm_size = Size * sizeof(CT);

  // correctness check
  run_fft_kernel<1, sm_size, CT, Size, Radix, FFTExec>(
      thrust::raw_pointer_cast(d_data.data()));

  h_data = d_data;
  for (int i = 0; i < data.size(); ++i) {
    out[i] = config::CT{h_data[i].real(), h_data[i].imag()};
  }

  const auto time_1000 =
      run_fft_kernel<1000, sm_size, CT, Size, Radix, FFTExec>(
          thrust::raw_pointer_cast(d_data.data()));

  const auto time_11000 =
      run_fft_kernel<11000, sm_size, CT, Size, Radix, FFTExec>(
          thrust::raw_pointer_cast(d_data.data()));

  // Will return time in microseconds
  const double final_time =
      static_cast<double>(time_11000 - time_1000) / 10000.0;

  return final_time;
}

void test(std::vector<config::CT> &data) {
  std::vector<config::CT> out_algorithm(data.size()),
      out_reference(data.size());

  using customExec = fft::simple_fft<config::CT, config::N, config::radix>;
  using refExec = fft::reference_fft<config::N>;

  auto alg_data = data;
  auto ref_data = data;

  const auto alg_run =
      run_perf_test<config::CT, config::N, config::radix, customExec>(
          alg_data, out_algorithm);

  const auto ref_run =
      run_perf_test<refExec::VT, config::N, config::radix, refExec>(
          ref_data, out_reference);

  double mse{0.0};

  for (int i = 0; i < data.size(); ++i) {
    const auto se = norm(out_reference[i] - out_algorithm[i]);
    if constexpr (config::print_results) {
      std::cout << " Ref: " << out_reference[i].real() << " "
                << out_reference[i].imag()
                << " Ten: " << out_algorithm[i].real() << " "
                << out_algorithm[i].imag() << std::endl;
    }
    mse += se;
  }
  mse /= data.size();

  std::cout << "Tensor FFT took: " << alg_run << " microseconds \n";
  std::cout << "Reference took: " << ref_run << " microseconds \n";
  std::cout << "MSE: " << mse << "\n";
}
} // namespace testing
