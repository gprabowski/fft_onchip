#pragma once
#include <thrust/complex.h>

namespace config {

using CT = thrust::complex<double>;

constexpr int radix = 8;
constexpr int N = radix * radix;

} // namespace config
