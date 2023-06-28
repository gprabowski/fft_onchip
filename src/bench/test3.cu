
#include <config.hpp>
#include <tensor_fft_64.cuh>
#include <testing.cuh>
#include <vector>
#include <generated_tests.cuh>

namespace bench {
void test3(const std::vector<config::CT> &data){testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 24>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 26>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 28>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 30>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 32>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 4, 1>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 4, 2>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 4, 4>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 4, 6>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 4, 8>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 4, 10>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 4, 12>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 4, 14>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 4, 16>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 6, 1>>(data);
}}