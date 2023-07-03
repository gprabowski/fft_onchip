#pragma once

#include <c64_fft64.cuh>
#include <common.cuh>
#include <cooperative_groups.h>
#include <tensor_utils.cuh>

namespace fft {

namespace cg = cooperative_groups;

template <typename CT, int Size, int UPB = 4, int FPU = 1>
struct tensor_fft_256 {
  using this_t = tensor_fft_256<CT, Size>;

  static constexpr auto threads = 32;
  static constexpr auto units_per_block = UPB;
  static constexpr auto ffts_per_unit = FPU;
  static constexpr auto max_threads_per_block = units_per_block * threads;

  mma_fp64_884_indexes;
  const int output_idx = crow * 8 + ccol;

  const CT twiddle1 = pow_theta<64>(crow * ccol);
  const CT twiddle2 = pow_theta<64>(crow * (ccol + 1));

  const CT a1 = pow_theta<8>(arow * acol);
  const CT a2 = pow_theta<8>(arow * (acol + 4));

  static constexpr char print_type[] = "MMA256";

  static_assert(Size == 256, "SIZE MUST BE 256");

  template <int N> inline __device__ CT pow_theta(int p) const {
    p = p % N;
    float s, c;
    const float ang = p * (-2.f / N);
    sincospif(ang, &s, &c);
    return {c, s};
  }

  CT *local_data;
  CT local_b[8 * ffts_per_unit];

  __device__ tensor_fft_256(CT *d)
      : local_data(d + Size * ffts_per_unit *
                           ((threadIdx.x + blockDim.x * threadIdx.y) / 32)) {}

  __device__ void operator()() {
    // local storage for all FFT elements

    // 0. Prepare mma and transpose indices

    // 1. Pre-load b for 1st iter
    // in here we tranpose the matrix (as its naturally
    // set in memory in a column major fashion)
    local_b[0] = local_data[brow * 32 + bcol * 4];
    local_b[1] = local_data[(brow + 4) * 32 + bcol * 4];

    for (int i = 0; i < ffts_per_unit; ++i) {
      // 2. Pre-load B elements for next iteration
      if (i < ffts_per_unit - 1) {
        local_b[8 * (i + 1)] =
            local_data[(i + 1) * Size + brow * 32 + bcol * 4];
        local_b[8 * (i + 1) + 1] =
            local_data[(i + 1) * Size + (brow + 4) * 32 + bcol * 4];
      }

      // 3. Compute FFT on 256 elements
      local_b[2] = local_data[1 + brow * 32 + bcol * 4];
      local_b[3] = local_data[1 + (brow + 4) * 32 + bcol * 4];

      fft_kernels::c64_fft64<CT>(a1, a2, local_b[8 * i], local_b[8 * i + 1],
                                 twiddle1, twiddle2, transpose_lane_b1,
                                 transpose_lane_b1 + 2);

      local_b[4] = local_data[2 + brow * 32 + bcol * 4];
      local_b[5] = local_data[2 + (brow + 4) * 32 + bcol * 4];

      fft_kernels::c64_fft64<CT>(a1, a2, local_b[8 * i + 2], local_b[8 * i + 3],
                                 twiddle1, twiddle2, transpose_lane_b1,
                                 transpose_lane_b1 + 2);
      local_b[6] = local_data[3 + brow * 32 + bcol * 4];
      local_b[7] = local_data[3 + (brow + 4) * 32 + bcol * 4];

      fft_kernels::c64_fft64<CT>(a1, a2, local_b[8 * i + 4], local_b[8 * i + 5],
                                 twiddle1, twiddle2, transpose_lane_b1,
                                 transpose_lane_b1 + 2);

      fft_kernels::c64_fft64<CT>(a1, a2, local_b[8 * i + 6], local_b[8 * i + 7],
                                 twiddle1, twiddle2, transpose_lane_b1,
                                 transpose_lane_b1 + 2);

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
