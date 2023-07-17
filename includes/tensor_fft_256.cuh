#pragma once

#include <c64_fft64.cuh>
#include <common.cuh>
#include <constants.cuh>
#include <tensor_utils.cuh>

namespace fft {

template <typename CT, int Size, int UPB = 2, int FPU = 1>
struct tensor_fft_256 {
  using this_t = tensor_fft_256<CT, Size>;

  static constexpr auto threads = 32;
  static constexpr auto units_per_block = UPB;
  static constexpr auto ffts_per_unit = FPU;
  static constexpr auto max_threads_per_block = units_per_block * threads;

  mma_fp64_884_indexes indexing;

  static constexpr char print_type[] = "MMA256";

  static_assert(Size == 256, "SIZE MUST BE 256");

  template <int N> __forceinline__ __device__ CT pow_theta(int p) const {
    return {constants::twiddles[2 * ((p & (N - 1)) * 256 / N)],
            constants::twiddles[2 * ((p & (N - 1)) * 256 / N) + 1]};
  }

  CT *sh_d;

  __device__ tensor_fft_256(CT *d) : sh_d(d) {}

  __device__ void operator()() {

    CT b[8];

    auto data = sh_d + Size * ffts_per_unit *
                           ((threadIdx.x + blockDim.x * threadIdx.y) / 32);

    for (int i = 0; i < ffts_per_unit; ++i) {
      const auto a1 = pow_theta<8>(indexing.arow * (2 * indexing.acol));
      const auto a2 = pow_theta<8>(indexing.arow * (2 * indexing.acol + 1));

      // radix-8
      for (int j = 0; j < 4; ++j) {
        b[2 * j] = data[8 * j + indexing.brow * 2 * 32 + indexing.bcol];
        b[2 * j + 1] =
            data[8 * j + (indexing.brow * 2 + 1) * 32 + indexing.bcol];

        double s11, s12, s21, s22, s31, s32;
        s11 = s12 = s22 = s21 = s31 = s32 = 0.0;
        karatsuba_inline_mma_8x8x8(a1, a2, b[2 * j], b[2 * j + 1], s11, s12,
                                   s21, s22, s31, s32);

        b[2 * j].real(s11 - s21);
        b[2 * j].imag(s31 - s21 - s11);
        b[2 * j + 1].real(s12 - s22);
        b[2 * j + 1].imag(s32 - s22 - s12);
      }

      // twiddle
      b[2] *= pow_theta<32>(indexing.crow);
      b[4] *= pow_theta<32>(2 * indexing.crow);
      b[6] *= pow_theta<32>(3 * indexing.crow);

      b[3] *= pow_theta<32>(indexing.crow);
      b[5] *= pow_theta<32>(2 * indexing.crow);
      b[7] *= pow_theta<32>(3 * indexing.crow);

      // radix-4

      const auto tmp1 = b[0] + b[4];
      const auto tmp2 = b[0] - b[4];
      const auto tmp3 = b[1] + b[5];
      const auto tmp4 = b[1] - b[5];

      const auto tmp5 = b[2] + b[6];
      const auto tmp6 = b[2] - b[6];
      const auto tmp7 = b[3] + b[7];
      const auto tmp8 = b[3] - b[7];

      const auto tmp9 = CT{-tmp6.imag(), tmp6.real()};
      const auto tmp10 = CT{-tmp8.imag(), tmp8.real()};

      b[0] = tmp1 + tmp5;
      b[1] = tmp3 + tmp7;

      b[2] = tmp2 - tmp9;
      b[3] = tmp4 - tmp10;

      b[4] = tmp1 - tmp5;
      b[5] = tmp3 - tmp7;

      b[6] = tmp2 + tmp9;
      b[7] = tmp4 + tmp10;

      // and radix-8 again

      for (int j = 0; j < 4; ++j) {
        // first twiddle
        b[2 * j] *= pow_theta<256>(indexing.brow * 2 * (indexing.bcol + j * 8));
        b[2 * j + 1] *=
            pow_theta<256>((indexing.brow * 2 + 1) * (indexing.bcol + j * 8));

        double s11, s12, s21, s22, s31, s32;
        s11 = s12 = s22 = s21 = s31 = s32 = 0.0;
        karatsuba_inline_mma_8x8x8(a1, a2, b[2 * j], b[2 * j + 1], s11, s12,
                                   s21, s22, s31, s32);

        b[2 * j].real(s11 - s21);
        b[2 * j].imag(s31 - s21 - s11);
        b[2 * j + 1].real(s12 - s22);
        b[2 * j + 1].imag(s32 - s22 - s12);

        data[(indexing.ccol + j * 8) + indexing.crow * 32] = b[2 * j];
        data[(indexing.ccol + 1 + j * 8) + indexing.crow * 32] = b[2 * j + 1];
      }
    }

    data += Size;
  }
};
} // namespace fft
