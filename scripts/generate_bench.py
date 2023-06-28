#!/usr/bin/env python3

import os
import sys

gen_header_preambule = """#pragma once
#include <config.hpp>
#include <vector>
namespace bench {
"""

header_preambule = """#pragma once
#include <vector>
#include <config.hpp>
#include <tensor_fft_64.cuh>
#include <reference.cuh>
#include <testing.cuh>
#include <generated_tests.cuh>

namespace bench {
void test(const std::vector<config::CT> &data) {
    using baseline = fft::reference_fft<config::N>;
    testing::perf_test_printer<baseline::VT, config::N, baseline>(data);
"""

src_preambule = """
#include <config.hpp>
#include <tensor_fft_64.cuh>
#include <testing.cuh>
#include <vector>
#include <generated_tests.cuh>

namespace bench {
"""

def main():
    file_counter = 0
    test_counter = 0
    header = open("../includes/bench.cuh", "w")
    gen_header = open("../includes/generated_tests.cuh", "w")
    curr_file = open(f"../src/bench/test{file_counter}.cu", "w")

    gen_header.write(gen_header_preambule)

    gen_header.write(f"void test{file_counter}(const std::vector<config::CT> &data);")

    header.write(header_preambule)
    header.write(f"test{file_counter}(data);")

    curr_file.write(src_preambule)
    curr_file.write(f"void test{file_counter}(const std::vector<config::CT> &data)"+"{")
    

    units_range = [1] + [i for i in range(2, 33, 2)]
    ffts_range = [1] + [i for i in range(2, 65, 2)]
    for i in units_range:
        for j in ffts_range:
            if(i * j > 64):
                continue
            curr_file.write(f"testing::perf_test_printer<config::CT, config::N, fft::tensor_fft_64<config::CT, config::N, {i}, {j}>>(data);\n")
            test_counter += 1
            if(test_counter >= 15):
                test_counter = 0
                file_counter += 1
                gen_header.write(f"void test{file_counter}(const std::vector<config::CT> &data);")
                header.write(f"test{file_counter}(data);")
                curr_file.write("}}")
                curr_file.close()
                curr_file = open(f"../src/bench/test{file_counter}.cu", "w")
                curr_file.write(src_preambule)
                curr_file.write(f"void test{file_counter}(const std::vector<config::CT> &data)"+"{")

    header.write("}}")

    gen_header.write("}")
    header.close()
    gen_header.close()

    curr_file.write("}}")
    curr_file.close()

if __name__ == '__main__':
    sys.exit(main())
