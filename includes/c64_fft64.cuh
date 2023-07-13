#pragma once

#include <config.hpp>
#include <tensor_utils.cuh>

namespace fft_kernels {

template <typename DT>
__device__ __forceinline__ void c64_fft64(const DT &a1, const DT &a2, DT &b1,
                                          DT &b2, const DT &twiddle1,
                                          const DT &twiddle2) {
  double s11{0}, s12{0}, s21{0}, s22{0}, s31{0}, s32{0};

  // 3. First GEMM
  karatsuba_inline_mma_8x8x8(a1, a2, b1, b2, s11, s12, s21, s22, s31, s32);

  // 4. Twiddle
  b1 = twiddle1 * DT{s11 - s21, s31 - s21 - s11};
  b2 = twiddle2 * DT{s12 - s22, s32 - s22 - s12};

  s11 = s12 = s21 = s22 = s31 = s32 = 0.0;

  // 6. Second GEMM
  karatsuba_inline_mma_8x8x8(a1, a2, b1, b2, s11, s12, s21, s22, s31, s32);
  b1 = DT{s11 - s21, s31 - s21 - s11};
  b2 = DT{s12 - s22, s32 - s22 - s12};
}
} // namespace fft_kernels
