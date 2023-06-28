
#include <config.hpp>
#include <generated_tests.cuh>
#include <tensor_fft_64.cuh>
#include <testing.cuh>
#include <vector>

namespace bench {
void test0(const std::vector<config::CT> &data) {
  testing::perf_test_printer<config::CT, config::N,
                             fft::tensor_fft_64<config::CT, config::N, 1, 1>>(
      data);
  testing::perf_test_printer<config::CT, config::N,
                             fft::tensor_fft_64<config::CT, config::N, 1, 2>>(
      data);
  testing::perf_test_printer<config::CT, config::N,
                             fft::tensor_fft_64<config::CT, config::N, 1, 4>>(
      data);
  testing::perf_test_printer<config::CT, config::N,
                             fft::tensor_fft_64<config::CT, config::N, 1, 6>>(
      data);
  testing::perf_test_printer<config::CT, config::N,
                             fft::tensor_fft_64<config::CT, config::N, 1, 8>>(
      data);
  testing::perf_test_printer<config::CT, config::N,
                             fft::tensor_fft_64<config::CT, config::N, 1, 10>>(
      data);
  testing::perf_test_printer<config::CT, config::N,
                             fft::tensor_fft_64<config::CT, config::N, 1, 12>>(
      data);
  testing::perf_test_printer<config::CT, config::N,
                             fft::tensor_fft_64<config::CT, config::N, 1, 14>>(
      data);
  testing::perf_test_printer<config::CT, config::N,
                             fft::tensor_fft_64<config::CT, config::N, 1, 16>>(
      data);
  testing::perf_test_printer<config::CT, config::N,
                             fft::tensor_fft_64<config::CT, config::N, 1, 18>>(
      data);
  testing::perf_test_printer<config::CT, config::N,
                             fft::tensor_fft_64<config::CT, config::N, 1, 20>>(
      data);
  testing::perf_test_printer<config::CT, config::N,
                             fft::tensor_fft_64<config::CT, config::N, 1, 22>>(
      data);
  testing::perf_test_printer<config::CT, config::N,
                             fft::tensor_fft_64<config::CT, config::N, 1, 24>>(
      data);
  testing::perf_test_printer<config::CT, config::N,
                             fft::tensor_fft_64<config::CT, config::N, 1, 26>>(
      data);
  testing::perf_test_printer<config::CT, config::N,
                             fft::tensor_fft_64<config::CT, config::N, 1, 28>>(
      data);
}
} // namespace bench
