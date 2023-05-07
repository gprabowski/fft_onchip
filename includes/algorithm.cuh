#pragma once

#include <chrono>
#include <vector>
#include <complex>

namespace fft {
size_t run_algorithm(const std::vector<std::complex<double>> &data,
                     std::vector<std::complex<double>> &out);
}
