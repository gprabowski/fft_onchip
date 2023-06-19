#pragma once

#include <tensor_utils.cuh>
#include <cooperative_groups.h>

namespace fft {

namespace cg = cooperative_groups;

template <typename CT, int Size, int Radix> struct simple_fft {
  using this_t = simple_fft<CT, Size, Radix>;
  static constexpr auto RadixSquared = Radix * Radix;

  static constexpr dim3 threads = 32;

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
    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<32>(block);

    const auto lane_idx = warp.thread_rank();

    double s11{0}, s12{0}, s21{0}, s22{0}, s31{0}, s32{0};
    // 0. Prepare result indices
    const auto crow = lane_idx >> 2;
    const auto ccol = ((lane_idx % 4) * 2); // or + 1

    // 1. Fill A "Matrix" with twiddles
    const auto arow = lane_idx >> 2;
    const auto acol = lane_idx % 4;
    const CT a1 = pow_theta<Radix>(arow * acol);
    const CT a2 = pow_theta<Radix>(arow * (acol + 4));

    // 2. Load B elements
    const auto brow = lane_idx % 4;
    const auto bcol = lane_idx >> 2;

    // in here we tranpose the matrix (as its naturally
    // set in memory in a column major fashion) 
    CT b1 = sh_d[brow * 8 + bcol];
    CT b2 = sh_d[(brow + 4) * 8 + bcol];
    
    // 3. prepare accumulators
    double c1r{0}, c2r{0}, c1i{0}, c2i{0};

    // 3. First GEMM
    karatsuba_inline_mma_8x8x8(a1, a2, b1, b2, s11, s12, s21, s22, s31, s32);
    c1r = s11 - s21;
    c1i = s31 - s21 - s11;
    c2r = s12 - s22;
    c2i = s32 - s22 - s12;

    // 4. Twiddle
    b1 = pow_theta<RadixSquared>(crow * ccol) * CT{c1r, c1i};
    b2 = pow_theta<RadixSquared>(crow * (ccol + 1)) * CT{c2r, c2i};

    // 5. Exchange elements
    // effectively we transpose the matrix here
    const auto transpose_lane_b1 = bcol * 4 + brow / 2;
    const auto transpose_lane_b2 = transpose_lane_b1 + 2;

    // REUSING C REGISTERS HERE AS TEMP STORAGE, NO SEMANTIC MEANING
    c1r = warp.shfl(b1.real(), transpose_lane_b1);
    c1i = warp.shfl(b1.imag(), transpose_lane_b1);
    c2r = warp.shfl(b2.real(), transpose_lane_b1);
    c2i = warp.shfl(b2.imag(), transpose_lane_b1);
    const auto tmp = (brow & 1) ? CT{c2r, c2i} : CT{c1r, c1i};

    // REUSING C REGISTERS HERE AS TEMP STORAGE, NO SEMANTIC MEANING
    c1r = warp.shfl(b1.real(), transpose_lane_b2);
    c1i = warp.shfl(b1.imag(), transpose_lane_b2);
    c2r = warp.shfl(b2.real(), transpose_lane_b2);
    c2i = warp.shfl(b2.imag(), transpose_lane_b2);

    b1 = tmp;
    b2 = (brow & 1) ? CT{c2r, c2i} : CT{c1r, c1i};

    s11 = s12 = s21 = s22 = s31 = s32 = 0.0;

    // 6. Second GEMM
    karatsuba_inline_mma_8x8x8(a1, a2, b1, b2, s11, s12, s21, s22, s31, s32);
    c1r = s11 - s21;
    c1i = s31 - s21 - s11;
    c2r = s12 - s22;
    c2i = s32 - s22 - s12;

    // 7. Save results
    // perform digit reversal (that's why indexing is reversed)
    sh_d[crow * 8 + ccol] = CT{c1r, c1i};
    sh_d[crow * 8 + (ccol + 1)] = CT{c2r, c2i};
  }
};
}
