#pragma once
#include <vector>
#include <config.hpp>
#include <tensor_fft_64.cuh>
#include <reference.cuh>
#include <testing.cuh>
#include <generated_tests.cuh>

namespace bench {
void test(const std::vector<config::CT> &data) {
    using baseline = fft::reference_fft<config::N>;
    for(int i = 0; i < 100; ++i) {
        testing::perf_test_printer<baseline::VT, config::N, baseline>(data);
    }
test0(data);test1(data);test2(data);test3(data);test4(data);test5(data);test6(data);}}