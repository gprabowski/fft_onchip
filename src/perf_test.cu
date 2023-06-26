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
#include <testing.cuh>

#include <chrono>

namespace testing {

template <typename CT, int Size, typename FFTExec>
double run_tests(const std::vector<config::CT> &data,
                 std::vector<config::CT> &out) {
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
  run_fft_kernel<CT, Size, FFTExec>(
      1, sm_size, thrust::raw_pointer_cast(d_data.data()), sm_count);

  h_data = d_data;
  out.resize(h_data.size());
  for (int i = 0; i < h_data.size(); ++i) {
    out[i] = config::CT{h_data[i].real(), h_data[i].imag()};
  }

  return run_perf_tests<CT, Size, FFTExec>(data);
}

void test(std::vector<config::CT> &data) {
  std::vector<config::CT> out_algorithm, out_reference;

  using customExec = fft::tensor_fft_64<config::CT, config::N, 4, 2>;
  using refExec = fft::reference_fft<config::N>;

  auto alg_data = data;
  auto ref_data = data;

  const auto alg_run =
      run_tests<config::CT, config::N, customExec>(alg_data, out_algorithm);

  const auto ref_run =
      run_tests<refExec::VT, config::N, refExec>(ref_data, out_reference);

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
