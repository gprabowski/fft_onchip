#pragma once

#include <config.hpp>
#include <cooperative_groups.h>
#include <tensor_utils.cuh>

namespace fft_kernels {
namespace cg = cooperative_groups;

template <typename DT>
__device__ __forceinline__ void
c64_fft64(const DT &a1, const DT &a2, DT &b1, DT &b2, const DT &twiddle1,
          const DT &twiddle2, const size_t &transpose_lane_b1,
          const size_t &transpose_lane_b2) {
  const auto block = cg::this_thread_block();
  const auto warp = cg::tiled_partition<32>(block);

  double c1r{0}, c1i{0}, c2r{0}, c2i{0};
  double s11{0}, s12{0}, s21{0}, s22{0}, s31{0}, s32{0};

  // 3. First GEMM
  karatsuba_inline_mma_8x8x8(a1, a2, b1, b2, s11, s12, s21, s22, s31, s32);
  c1r = s11 - s21;
  c1i = s31 - s21 - s11;
  c2r = s12 - s22;
  c2i = s32 - s22 - s12;

  // 4. Twiddle
  b1 = twiddle1 * DT{c1r, c1i};
  b2 = twiddle2 * DT{c2r, c2i};

  // 5. Exchange elements
  // effectively we transpose the matrix here
  // REUSING C REGISTERS HERE AS TEMP STORAGE, NO SEMANTIC MEANING
  c1r = warp.shfl(b1.real(), transpose_lane_b1);
  c1i = warp.shfl(b1.imag(), transpose_lane_b1);
  c2r = warp.shfl(b2.real(), transpose_lane_b1);
  c2i = warp.shfl(b2.imag(), transpose_lane_b1);
  const auto tmp = (warp.thread_rank() & 1) ? DT{c2r, c2i} : DT{c1r, c1i};

  // REUSING C REGISTERS HERE AS TEMP STORAGE, NO SEMANTIC MEANING
  c1r = warp.shfl(b1.real(), transpose_lane_b2);
  c1i = warp.shfl(b1.imag(), transpose_lane_b2);
  c2r = warp.shfl(b2.real(), transpose_lane_b2);
  c2i = warp.shfl(b2.imag(), transpose_lane_b2);

  b1 = tmp;
  b2 = (warp.thread_rank() & 1) ? DT{c2r, c2i} : DT{c1r, c1i};

  s11 = s12 = s21 = s22 = s31 = s32 = 0.0;

  // 6. Second GEMM
  karatsuba_inline_mma_8x8x8(a1, a2, b1, b2, s11, s12, s21, s22, s31, s32);
  b1.real(s11 - s21);
  b1.imag(s31 - s21 - s11);
  b2.real(s12 - s22);
  b2.imag(s32 - s22 - s12);
}
} // namespace fft_kernels
