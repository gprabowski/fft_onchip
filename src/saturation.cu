#include <algorithm>
#include <random>
#include <vector>

#include <config.hpp>
#include <testing.cuh>

#include <reference.cuh>
#include <tensor_fft_64.cuh>

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
  using customExec = fft::tensor_fft_64<config::CT, config::N, 4, 2>;
  using refExec = fft::reference_fft<config::N>;

  auto alg_data = data;
  auto ref_data = data;

  std::vector<config::CT> mock_out;

  std::cout << "Blocks/SM, Time Tensor, Time cuFFTDx" << std::endl;

  for (int i = 1; i < 256; i += 2) {
    const auto alg_run =
        testing::run_perf_tests<config::CT, config::N, customExec>(alg_data, i);

    const auto ref_run =
        testing::run_perf_tests<refExec::VT, config::N, refExec>(ref_data, i);

    std::cout << i << "," << alg_run << "," << ref_run << std::endl;
  }
}
