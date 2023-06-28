
#include <config.hpp>
#include <tensor_fft_64.cuh>
#include <testing.cuh>
#include <vector>
#include <generated_tests.cuh>

namespace bench {
void test5(const std::vector<config::CT> &data){testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 12, 2>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 12, 4>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 14, 1>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 14, 2>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 14, 4>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 16, 1>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 16, 2>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 16, 4>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 18, 1>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 18, 2>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 20, 1>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 20, 2>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 22, 1>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 22, 2>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 24, 1>>(data);
}}