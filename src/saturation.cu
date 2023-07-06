#include <algorithm>
#include <random>
#include <vector>

#include <config.hpp>
#include <testing.cuh>

#include <reference.cuh>
#include <tensor_fft_128.cuh>

int main() {
  using config::CT;
  constexpr auto N = 128;

  std::random_device rd;
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  std::vector<CT> data(N);

  // generate data
  std::transform(begin(data), end(data), begin(data), [&](auto) {
    return CT{dist(rd), dist(rd)};
  });

  // compare correctness
  using refExec = fft::reference_fft<N>;

  auto alg_data = data;
  auto ref_data = data;

  std::vector<config::CT> mock_out;

  // Clock warmup
  for (int i = 0; i < 100; ++i) {
    testing::run_perf_tests<refExec::VT, N, refExec>(data);
  }

  std::cout << "Blocks/SM, Time Tensor, Time cuFFTDx" << std::endl;

  for (int i = 4; i <= 256; i += 4) {
    auto alg_run =
        testing::run_perf_tests<config::CT, N,
                                fft::tensor_fft_128<config::CT, N>>(data, i);

    auto ref_run = testing::run_perf_tests<refExec::VT, N, refExec>(data, i);

    std::cout << i << "," << alg_run << "," << ref_run << std::endl;
  }
}
