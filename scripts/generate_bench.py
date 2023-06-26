#!/usr/bin/env python3

import os
import sys


def main():
    header = open("../includes/bench.cuh", "w")

    header.write("""#pragma once
#include <config.hpp>
#include <tensor_fft_64.cuh>
#include <reference.cuh>
#include <testing.cuh>

namespace bench {
void test(const std::vector<config::CT> &data) {
    using baseline = fft::reference_fft<config::N>;
    testing::perf_test_printer<baseline::VT, config::N, baseline>(data);
""")

    units_range = [1] + [i for i in range(2, 33, 2)]
    ffts_range = [1] + [i for i in range(2, 65, 2)]
    for i in units_range:
        for j in ffts_range:
            if(i * j > 64):
                continue
            header.write(f"testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, {i}, {j}>>(data);\n")
    header.write("}}")
    header.close()

if __name__ == '__main__':
    sys.exit(main())
