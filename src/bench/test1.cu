
#include <config.hpp>
#include <tensor_fft_64.cuh>
#include <testing.cuh>
#include <vector>
#include <generated_tests.cuh>

namespace bench {
void test1(const std::vector<config::CT> &data){testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 30>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 32>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 34>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 36>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 38>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 40>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 42>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 44>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 46>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 48>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 50>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 52>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 54>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 56>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 58>>(data);
}}