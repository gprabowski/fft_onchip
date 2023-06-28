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

void test(std::vector<config::CT> &data) {
  std::vector<config::CT> out_algorithm, out_reference;

  using customExec = fft::tensor_fft_64<config::CT, config::N, 4, 2>;
  using refExec = fft::reference_fft<config::N>;

  auto alg_data = data;
  auto ref_data = data;

  const auto alg_run =
      run_perf_and_corr_tests<config::CT, config::N, customExec>(alg_data,
                                                                 out_algorithm);

  const auto ref_run = run_perf_and_corr_tests<refExec::VT, config::N, refExec>(
      ref_data, out_reference);

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
