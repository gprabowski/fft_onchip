#!/usr/bin/env python3

import os
import sys

gen_header_preambule = """#pragma once
#include <config.hpp>
#include <vector>
namespace bench {
"""

# takes size as format argument
header_preambule = """#pragma once
#include <vector>
#include <config.hpp>

#include <tensor_fft_64.cuh>
#include <tensor_fft_128.cuh>
#include <tensor_fft_256.cuh>
#include <tensor_fft_512.cuh>
#include <tensor_fft_4096.cuh>

#include <reference.cuh>
#include <testing.cuh>
#include <bench/generated_tests{0}.cuh>

namespace bench {{
void test_{0}(const std::vector<config::CT> &data) {{
    using baseline = fft::reference_fft<{0}>;
    for(int i = 0; i < 100; ++i) {{
        testing::perf_test_printer<baseline::VT, {0}, baseline>(data, config::sm_multiplier, false);
    }}

    testing::perf_test_printer<baseline::VT, {0}, baseline>(data);
"""

# takes size as format argument
src_preambule = """
#include <config.hpp>

#include <tensor_fft_64.cuh>
#include <tensor_fft_128.cuh>
#include <tensor_fft_256.cuh>
#include <tensor_fft_512.cuh>
#include <tensor_fft_4096.cuh>

#include <testing.cuh>
#include <vector>
#include <bench/generated_tests{0}.cuh>

namespace bench {{
"""

threads_necessary = {64: 32, 128: 64, 256: 128, 512: 256, 4096: 512}

def main():
    main_header = open(f"../includes/bench.cuh", "w")
    main_header.write("""
#pragma once

#include <vector>
#include <random>
#include <iostream>
#include <config.hpp>

using config::CT;
                      """)

    fft_sizes = [64, 128, 256, 512, 4096]

    for fft_size in fft_sizes: 
        file_counter = 0
        test_counter = 0
        header = open(f"../includes/bench/bench{fft_size}.cuh", "w")
        main_header.write(f"#include <bench/bench{fft_size}.cuh>\n")

        gen_header = open(f"../includes/bench/generated_tests{fft_size}.cuh", "w")
        curr_file = open(f"../src/bench/test_{fft_size}_{file_counter}.cu", "w")

        gen_header.write(gen_header_preambule + "\n")

        gen_header.write(f"void test_{fft_size}_{file_counter}(const std::vector<config::CT> &data);\n")

        header.write(header_preambule.format(fft_size) + "\n")
        header.write(f"test_{fft_size}_{file_counter}(data);\n")

        curr_file.write(src_preambule.format(fft_size) + "\n")
        curr_file.write(f"void test_{fft_size}_{file_counter}(const std::vector<config::CT> &data) "+"{\n")
        

        units_range = [1] + [i for i in range(2, 33, 2)]
        ffts_range = [1] + [i for i in range(2, 65, 2)]

        max_shared_b = 65536

        for i in units_range:
            for j in ffts_range:
                required_sm = fft_size * 2 * 8 * i * j
                required_threads = i * threads_necessary[fft_size] 
                if(required_sm >= max_shared_b or required_threads > 1024):
                    continue
                curr_file.write(f"testing::perf_test_printer<config::CT, {fft_size}, fft::tensor_fft_{fft_size}<config::CT, {fft_size}, {i}, {j}>>(data);\n")
                test_counter += 1
                if(test_counter >= 15):
                    test_counter = 0
                    file_counter += 1
                    gen_header.write(f"void test_{fft_size}_{file_counter}(const std::vector<config::CT> &data);\n")
                    header.write(f"test_{fft_size}_{file_counter}(data);\n")
                    curr_file.write("}}\n")
                    curr_file.close()
                    curr_file = open(f"../src/bench/test_{fft_size}_{file_counter}.cu", "w")
                    curr_file.write(src_preambule.format(fft_size) + "\n")
                    curr_file.write(f"void test_{fft_size}_{file_counter}(const std::vector<config::CT> &data) "+"{\n")

        header.write("}}\n")

        gen_header.write("}\n")
        header.close()
        gen_header.close()

        curr_file.write("}}\n")
        curr_file.close()

    main_header.write("""

namespace bench {
    void test() {
          std::random_device rd;
          std::uniform_real_distribution<float> dist(0.0, 1.0);

          
                      """)
    for fft_size in fft_sizes:
        main_header.write("""
          std::vector<CT> data_{0}({0});

          // generate data
          std::transform(begin(data_{0}), end(data_{0}), begin(data_{0}), [&](auto) {{
            return CT{{dist(rd), dist(rd)}};
          }});

          bench::test_{0}(data_{0});


                          """.format(fft_size))
    main_header.write("}}")
    main_header.close()

if __name__ == '__main__':
    sys.exit(main())
