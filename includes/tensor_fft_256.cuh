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

  CT *sh_d;

  __device__ tensor_fft_256(CT *d) : sh_d(d) {}

  __device__ void operator()() {

    // local storage for all FFT elements
    CT local_b[8 * ffts_per_unit];

    auto local_data =
        sh_d +
        Size * ffts_per_unit * ((threadIdx.x + blockDim.x * threadIdx.y) / 32);

    const auto output_idx = indexing.crow * 8 + indexing.ccol;

    // 0. Prepare mma and transpose indices

    // 1. Pre-load b for 1st iter
    // in here we tranpose the matrix (as its naturally
    // set in memory in a column major fashion)

    const auto bpos = indexing.brow * 32 + indexing.bcol * 4;
    const auto tm8 = threadIdx.x % 8;

    for (int i = 0; i < ffts_per_unit; ++i) {
      // 3. Compute FFT on 256 elements
      local_b[8 * i + tm8] = local_data[i * Size + tm8 / 2 + bpos];
      local_b[8 * i + (tm8 + 1) % 8] =
          local_data[i * Size + ((tm8 + 1) % 8) / 2 + bpos + 128];

      local_b[8 * i + (tm8 + 2) % 8] =
          local_data[i * Size + ((tm8 + 2) % 8) / 2 + bpos];
      local_b[8 * i + (tm8 + 3) % 8] =
          local_data[i * Size + ((tm8 + 3) % 8) / 2 + bpos + 128];

      local_b[8 * i + (tm8 + 4) % 8] =
          local_data[i * Size + ((tm8 + 4) % 8) / 2 + bpos];
      local_b[8 * i + (tm8 + 5) % 8] =
          local_data[i * Size + ((tm8 + 5) % 8) / 2 + bpos + 128];

      local_b[8 * i + (tm8 + 6) % 8] =
          local_data[i * Size + ((tm8 + 6) % 8) / 2 + bpos];
      local_b[8 * i + (tm8 + 7) % 8] =
          local_data[i * Size + ((tm8 + 7) % 8) / 2 + bpos + 128];

      fft_kernels::c64_fft64<CT>(a1, a2, local_b[8 * i], local_b[8 * i + 1],
                                 twiddle1, twiddle2, indexing.transpose_lane_b1,
                                 indexing.transpose_lane_b2);
      fft_kernels::c64_fft64<CT>(a1, a2, local_b[8 * i + 2], local_b[8 * i + 3],
                                 twiddle1, twiddle2, indexing.transpose_lane_b1,
                                 indexing.transpose_lane_b2);
      fft_kernels::c64_fft64<CT>(a1, a2, local_b[8 * i + 4], local_b[8 * i + 5],
                                 twiddle1, twiddle2, indexing.transpose_lane_b1,
                                 indexing.transpose_lane_b2);
      fft_kernels::c64_fft64<CT>(a1, a2, local_b[8 * i + 6], local_b[8 * i + 7],
                                 twiddle1, twiddle2, indexing.transpose_lane_b1,
                                 indexing.transpose_lane_b2);

      // 5. perform radix-4 stage
      const CT tw3 = pow_theta<256>(output_idx);

      const CT tw4 = pow_theta<256>(output_idx + 1);

      local_b[8 * i + 2] *= tw3;
      local_b[8 * i + 4] *= tw3 * tw3;
      local_b[8 * i + 6] *= tw3 * tw3 * tw3;

      local_data[i * Size + output_idx] = local_b[8 * i] + local_b[8 * i + 2] +
                                          local_b[8 * i + 4] +
                                          local_b[8 * i + 6];

      local_data[i * Size + output_idx + 64] =
          (local_b[8 * i] - local_b[8 * i + 4]) -
          CT{0, 1} * (local_b[8 * i + 2] - local_b[8 * i + 6]);
      local_data[i * Size + output_idx + 128] =
          (local_b[8 * i] + local_b[8 * i + 4]) -
          (local_b[8 * i + 2] + local_b[8 * i + 6]);
      local_data[i * Size + output_idx + 192] =
          (local_b[8 * i] - local_b[8 * i + 4]) +
          CT{0, 1} * (local_b[8 * i + 2] - local_b[8 * i + 6]);

      local_b[8 * i + 3] *= tw4;
      local_b[8 * i + 5] *= tw4 * tw4;
      local_b[8 * i + 7] *= tw4 * tw4 * tw4;

      local_data[i * Size + output_idx + 1] =
          local_b[8 * i + 1] + local_b[8 * i + 3] + local_b[8 * i + 5] +
          local_b[8 * i + 7];

      local_data[i * Size + output_idx + 65] =
          (local_b[8 * i + 1] - local_b[8 * i + 5]) -
          CT{0, 1} * (local_b[8 * i + 3] - local_b[8 * i + 7]);
      local_data[i * Size + output_idx + 129] =
          (local_b[8 * i + 1] + local_b[8 * i + 5]) -
          (local_b[8 * i + 3] + local_b[8 * i + 7]);
      local_data[i * Size + output_idx + 193] =
          (local_b[8 * i + 1] - local_b[8 * i + 5]) +
          CT{0, 1} * (local_b[8 * i + 3] - local_b[8 * i + 7]);
    }
  }
};
} // namespace fft
