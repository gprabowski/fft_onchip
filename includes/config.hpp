#pragma once
#include <thrust/complex.h>

namespace config {

using CT = thrust::complex<double>;

constexpr int N = 64 * 64;

constexpr bool print_results = true;
constexpr int sm_multiplier = 1;

} // namespace config
