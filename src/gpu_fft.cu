#include <algorithm>
#include <random>
#include <vector>

#include <config.hpp>
#include <perf_test.cuh>

#include <common.cuh>
#include <reference.cuh>
#include <tensor_fft_64.cuh>
#include <testing.cuh>

int main() {
  using config::CT;
  using config::N;

  std::random_device rd;
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  std::vector<CT> data(N);

  // generate data
  std::transform(begin(data), end(data), begin(data), [&](auto) {
    return CT{dist(rd), dist(rd)};
  });

  // compare correctness
  std::vector<config::CT> out_algorithm, out_reference;

  using customExec = fft::tensor_fft_64<config::CT, config::N, 4, 2>;
  using refExec = fft::reference_fft<config::N>;

  auto alg_data = data;
  auto ref_data = data;

  const auto alg_run_transfers =
      testing::run_perf_and_corr_tests<config::CT, config::N, customExec, true>(
          alg_data, out_algorithm);

  const auto alg_run_no_transfers =
      testing::run_perf_and_corr_tests<config::CT, config::N, customExec,
                                       false>(alg_data, out_algorithm);

  const auto ref_run_transfers =
      testing::run_perf_and_corr_tests<refExec::VT, config::N, refExec, true>(
          ref_data, out_reference);

  const auto ref_run_no_transfers =
      testing::run_perf_and_corr_tests<refExec::VT, config::N, refExec, false>(
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

  std::cout << "Transfers Mode, Time Tensor, Time cuFFTDx, MSE" << std::endl;
  std::cout << "Included," << alg_run_transfers << "," << ref_run_transfers
            << "," << mse << std::endl;
  std::cout << "Excluded," << alg_run_no_transfers << ","
            << ref_run_no_transfers << ","
            << "N/A" << std::endl;
}
