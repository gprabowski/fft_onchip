#pragma once

#include <chrono>
#include <vector>
#include <complex>

#include <cuda_fp16.h>

namespace fft {
size_t run_reference(const std::vector<__half2> &data,
                     std::vector<__half2> &out);
}
