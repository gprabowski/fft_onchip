
#include <config.hpp>
#include <tensor_fft_64.cuh>
#include <testing.cuh>
#include <vector>
#include <generated_tests.cuh>

namespace bench {
void test4(const std::vector<config::CT> &data){testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 6, 2>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 6, 4>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 6, 6>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 6, 8>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 6, 10>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 8, 1>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 8, 2>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 8, 4>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 8, 6>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 8, 8>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 10, 1>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 10, 2>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 10, 4>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 10, 6>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 12, 1>>(data);
}}