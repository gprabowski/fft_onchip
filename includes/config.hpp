#pragma once
#include <thrust/complex.h>

namespace config {

using CT = thrust::complex<double>;

constexpr bool print_results = true;
constexpr int sm_multiplier = 32;

} // namespace config
