#pragma once

#include <tensor_utils.cuh>

#define FULL_MASK 0xffffffff

namespace fft {

template <typename CT, int Size, int Radix> struct simple_fft {
  using this_t = simple_fft<CT, Size, Radix>;
  static constexpr auto RadixSquared = Radix * Radix;

  static constexpr dim3 threads = 32;

  const int tid = threadIdx.x;
  const int laneIdx = tid % 32;

  template<int N>
  inline __device__ CT pow_theta(int p) const {
    p = p % N;
    float s, c;
    const float ang = p * (-2.f / N);
    sincospif(ang, &s, &c);
    return {c, s};
  }

  CT *sh_d;

  __device__ simple_fft(CT *d) : sh_d(d) {}

  __device__ void operator()() {
    // 0. Prepare result indices
    const auto crow = laneIdx >> 2;
    const auto ccol = ((laneIdx % 4) * 2); // or + 1

    // 1. Fill A "Matrix" with twiddles
    const auto arow = laneIdx >> 2;
    const auto acol = laneIdx % 4;
    const CT a1 = pow_theta<Radix>(arow * acol);
    const CT a2 = pow_theta<Radix>(arow * (acol + 4));

    // 2. Load B elements
    const auto brow = laneIdx % 4;
    const auto bcol = laneIdx >> 2;

    // in here we tranpose the matrix (as its naturally
    // set in memory in a column major fashion) 
    CT b1 = sh_d[brow * 8 + bcol];
    CT b2 = sh_d[(brow + 4) * 8 + bcol];
    
    // 3. prepare accumulators
    double c1r{0}, c2r{0}, c1i{0}, c2i{0};

    // 3. First GEMM
    complex_gemm8x8x8(a1, a2, b1, b2, c1r, c1i, c2r, c2i)

    // 4. Twiddle
    b1 = pow_theta<RadixSquared>(crow * ccol) * CT{c1r, c1i};
    b2 = pow_theta<RadixSquared>(crow * (ccol + 1)) * CT{c2r, c2i};

    // 5. Exchange elements
    // effectively we transpose the matrix here
    const auto needed_lane_first = bcol * 4 + brow / 2;
    const auto needed_lane_second = needed_lane_first + 2;

    // REUSING C REGISTERS HERE AS TEMP STORAGE, NO SEMANTIC MEANING
    c1r = __shfl_sync(FULL_MASK, b1.real(), needed_lane_first);
    c1i = __shfl_sync(FULL_MASK, b1.imag(), needed_lane_first);
    c2r = __shfl_sync(FULL_MASK, b2.real(), needed_lane_first);
    c2i = __shfl_sync(FULL_MASK, b2.imag(), needed_lane_first);
    const auto tmp = (brow & 1) ? CT{c2r, c2i} : CT{c1r, c1i};

    // REUSING C REGISTERS HERE AS TEMP STORAGE, NO SEMANTIC MEANING
    c1r = __shfl_sync(FULL_MASK, b1.real(), needed_lane_second);
    c1i = __shfl_sync(FULL_MASK, b1.imag(), needed_lane_second);
    c2r = __shfl_sync(FULL_MASK, b2.real(), needed_lane_second);
    c2i = __shfl_sync(FULL_MASK, b2.imag(), needed_lane_second);

    b1 = tmp;
    b2 = (brow & 1) ? CT{c2r, c2i} : CT{c1r, c1i};

    c1r = c1i = c2r = c2i = 0.0;

    // 6. Second GEMM
    complex_gemm8x8x8(a1, a2, b1, b2, c1r, c1i, c2r, c2i)

    // 7. Save results
    // perform digit reversal (that's why indexing is reversed)
    sh_d[crow * 8 + ccol] = CT{c1r, c1i};
    sh_d[crow * 8 + (ccol + 1)] = CT{c2r, c2i};
  }
};
}
