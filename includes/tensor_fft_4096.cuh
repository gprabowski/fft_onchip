#pragma once

#include <cooperative_groups.h>
#include <tensor_utils.cuh>

namespace fft {

namespace cg = cooperative_groups;

template <typename CT, int Size> struct tensor_fft_4096 {
  using this_t = tensor_fft_4096<CT, Size>;

  static constexpr auto threads = 32;

  template <int N> inline __device__ CT pow_theta(int p) const {
    p = p % N;
    float s, c;
    const float ang = p * (-2.f / N);
    sincospif(ang, &s, &c);
    return {c, s};
  }

  CT *sh_d;

  __device__ tensor_fft_4096(CT *d) : sh_d(d) {}

  __device__ void operator()() {
    // 0. load rows into warps
    // 1. perform fft on rows
    // 2. twiddle
    // 3. save to shared
    // 4. load columns into warps
    // 5. perform fft on columns
    // 6. save results into shared
  }
};
} // namespace fft
