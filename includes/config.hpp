#pragma once
#include <thrust/complex.h>

namespace config {

using CT = thrust::complex<double>;

constexpr int N = 64 * 8;

constexpr bool print_results = true;
constexpr int sm_multiplier = 64;

} // namespace config
