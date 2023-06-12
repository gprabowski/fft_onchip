#pragma once
#include <thrust/complex.h>

namespace config {

using CT = thrust::complex<double>;

constexpr int radix = 16;
constexpr int N = radix * radix;

} // namespace config
