#include <algorithm>
#include <random>
#include <vector>

#include <config.hpp>
#include <testing.cuh>

#include <reference.cuh>
#include <tensor_fft_128.cuh>
#include <tensor_fft_256.cuh>
#include <tensor_fft_512.cuh>
#include <tensor_fft_64.cuh>
#include <tensor_fft_8.cuh>

template <typename FFTExec, int Size> void run_saturation_test() {

  std::random_device rd;
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  std::vector<config::CT> data(Size);

  // generate data
  std::transform(begin(data), end(data), begin(data), [&](auto) {
    return config::CT{dist(rd), dist(rd)};
  });

  // Clock warmup
  using refExec = fft::reference_fft<Size>;

  for (int i = 0; i < 100; ++i) {
    testing::run_perf_tests<typename refExec::VT, Size, refExec>(data);
  }

  std::cout << "Blocks/SM, Size, Time Tensor, Time cuFFTDx" << std::endl;

  for (int i = 4; i <= 1024; i += 4) {
    auto alg_run = testing::run_perf_tests<config::CT, Size, FFTExec>(data, i);

    auto ref_run =
        testing::run_perf_tests<typename refExec::VT, Size, refExec>(data, i);

    std::cout << i << "," << Size << "," << alg_run << "," << ref_run
              << std::endl;
  }
}

int main() {
  using config::CT;

  // compare correctness
  using t8 = fft::tensor_fft_8<config::CT, 8>;
  // using t64 = fft::tensor_fft_64<config::CT, 64>;
  // using t128 = fft::tensor_fft_128<config::CT, 128>;
  // using t256 = fft::tensor_fft_256<config::CT, 256>;
  // using t512 = fft::tensor_fft_512<config::CT, 512>;

  run_saturation_test<t8, 8>();
  // run_saturation_test<t64, 64>();
  // run_saturation_test<t128, 128>();
  // run_saturation_test<t256, 256>();
  // run_saturation_test<t512, 512>();
}
