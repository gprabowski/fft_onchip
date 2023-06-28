
#include <config.hpp>
#include <tensor_fft_64.cuh>
#include <testing.cuh>
#include <vector>
#include <generated_tests.cuh>

namespace bench {
void test2(const std::vector<config::CT> &data){testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 60>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 62>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 1, 64>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 1>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 2>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 4>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 6>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 8>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 10>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 12>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 14>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 16>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 18>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 20>>(data);
testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, 2, 22>>(data);
}}