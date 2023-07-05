#pragma once

#include <c64_fft64.cuh>
#include <common.cuh>
#include <tensor_utils.cuh>

namespace fft {

template <typename CT, int Size, int UPB = 4, int FPU = 1>
struct tensor_fft_256 {
  using this_t = tensor_fft_256<CT, Size>;

  static constexpr auto threads = 32;
  static constexpr auto units_per_block = UPB;
  static constexpr auto ffts_per_unit = FPU;
  static constexpr auto max_threads_per_block = units_per_block * threads;

  mma_fp64_884_indexes indexing;

  const CT twiddle1 = pow_theta<64>(indexing.crow * indexing.ccol);
  const CT twiddle2 = pow_theta<64>(indexing.crow * (indexing.ccol + 1));

  const CT a1 = pow_theta<8>(indexing.arow * indexing.acol);
  const CT a2 = pow_theta<8>(indexing.arow * (indexing.acol + 4));

  static constexpr char print_type[] = "MMA256";

  static_assert(Size == 256, "SIZE MUST BE 256");

  template <int N> inline __device__ CT pow_theta(int p) const {
    p = p % N;
    float s, c;
    const float ang = p * (-2.f / N);
    sincospif(ang, &s, &c);
    return {c, s};
  }

  // this function multiplies complex number by i
  static __forceinline__ __device__ CT rev(const CT& v) {
    return CT{-v.imag(), v.real()};
  }

  CT *sh_d;

  __device__ tensor_fft_256(CT *d) : sh_d(d) {}

  __device__ void operator()() {

    // local storage for all FFT elements
    CT b[8];

    auto data = sh_d + Size * ffts_per_unit *
                           ((threadIdx.x + blockDim.x * threadIdx.y) / 32);

    const auto output_idx = indexing.crow * 8 + indexing.ccol;

    // 0. Prepare mma and transpose indices

    // 1. Pre-load b for 1st iter
    // in here we tranpose the matrix (as its naturally
    // set in memory in a column major fashion)

    const auto bpos = indexing.brow * 32 + indexing.bcol * 4;

    for (int i = 0; i < ffts_per_unit; ++i) {
      // 3. Compute FFT on 256 elements
      b[tm8] = data[pos];
      b[1] = data[bpos + 128];

      b[2] = data[1 + bpos];
      b[3] = data[1 + bpos + 128];

      b[4] = data[2 + bpos];
      b[5] = data[2 + bpos + 128];

      b[6] = data[3 + bpos];
      b[7] = data[3 + bpos + 128];

      fft_kernels::c64_fft64<CT>(a1, a2, b[0], b[1], twiddle1, twiddle2,
                                 indexing.transpose_lane_b1,
                                 indexing.transpose_lane_b2);
      fft_kernels::c64_fft64<CT>(a1, a2, b[2], b[3], twiddle1, twiddle2,
                                 indexing.transpose_lane_b1,
                                 indexing.transpose_lane_b2);
      fft_kernels::c64_fft64<CT>(a1, a2, b[4], b[5], twiddle1, twiddle2,
                                 indexing.transpose_lane_b1,
                                 indexing.transpose_lane_b2);
      fft_kernels::c64_fft64<CT>(a1, a2, b[6], b[7], twiddle1, twiddle2,
                                 indexing.transpose_lane_b1,
                                 indexing.transpose_lane_b2);

      // 5. perform radix-4 stage
      const CT tw3 = pow_theta<256>(output_idx);

      const CT tw4 = pow_theta<256>(output_idx + 1);

      // compiler optimizes this to only 2 mults total
      b[2] *= tw3;
      b[4] *= tw3 * tw3;
      b[6] *= tw3 * tw3 * tw3;

      data[output_idx] = b[0] + b[2] + b[4] + b[6];
      data[output_idx + 64] = (b[0] - b[4]) - rev(b[2] - b[6]);
      data[output_idx + 128] = (b[0] + b[4]) - (b[2] + b[6]);
      data[output_idx + 192] = (b[0] - b[4]) + rev(b[2] - b[6]);

      // compiler optimizes this to only 2 mults total
      b[3] *= tw4;
      b[5] *= tw4 * tw4;
      b[7] *= tw4 * tw4 * tw4;

      data[output_idx + 1] = b[1] + b[3] + b[5] + b[7];
      data[output_idx + 65] = (b[1] - b[5]) - rev(b[3] - b[7]);
      data[output_idx + 129] = (b[1] + b[5]) - (b[3] + b[7]);
      data[output_idx + 193] = (b[1] - b[5]) + rev(b[3] - b[7]);

      data += Size;
    }
  }
};
} // namespace fft
