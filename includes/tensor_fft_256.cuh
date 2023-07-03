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
    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<32>(block);

    // local storage for all FFT elements
    CT local_b[8 * ffts_per_unit];

    auto local_data = sh_d + Size * ffts_per_unit * (warp.meta_group_rank());
    const auto output_idx = indexing.crow * 8 + indexing.ccol;

    // 0. Prepare mma and transpose indices

    // 1. Pre-load b for 1st iter
    // in here we tranpose the matrix (as its naturally
    // set in memory in a column major fashion)
    local_b[0] = local_data[indexing.brow * 32 + indexing.bcol * 4];
    local_b[1] = local_data[(indexing.brow + 4) * 32 + indexing.bcol * 4];

    local_b[2] = local_data[1 + indexing.brow * 32 + indexing.bcol * 4];
    local_b[3] = local_data[1 + (indexing.brow + 4) * 32 + indexing.bcol * 4];

    local_b[4] = local_data[2 + indexing.brow * 32 + indexing.bcol * 4];
    local_b[5] = local_data[2 + (indexing.brow + 4) * 32 + indexing.bcol * 4];

    local_b[6] = local_data[3 + indexing.brow * 32 + indexing.bcol * 4];
    local_b[7] = local_data[3 + (indexing.brow + 4) * 32 + indexing.bcol * 4];

    for (int i = 0; i < ffts_per_unit; ++i) {
      // 2. Pre-load B elements for next iteration
      if (i < ffts_per_unit - 1) {
        local_b[8 * (i + 1)] =
            local_data[(i + 1) * Size + indexing.brow * 32 + indexing.bcol * 4];
        local_b[8 * (i + 1) + 1] =
            local_data[(i + 1) * Size + (indexing.brow + 4) * 32 +
                       indexing.bcol * 4];

        local_b[8 * (i + 1) + 2] =
            local_data[(i + 1) * Size + 1 + indexing.brow * 32 +
                       indexing.bcol * 4];
        local_b[8 * (i + 1) + 3] =
            local_data[(i + 1) * Size + 1 + (indexing.brow + 4) * 32 +
                       indexing.bcol * 4];

        local_b[8 * (i + 1) + 4] =
            local_data[(i + 1) * Size + 2 + indexing.brow * 32 +
                       indexing.bcol * 4];
        local_b[8 * (i + 1) + 5] =
            local_data[(i + 1) * Size + 2 + (indexing.brow + 4) * 32 +
                       indexing.bcol * 4];

        local_b[8 * (i + 1) + 6] =
            local_data[(i + 1) * Size + 3 + indexing.brow * 32 +
                       indexing.bcol * 4];
        local_b[8 * (i + 1) + 7] =
            local_data[(i + 1) * Size + 3 + (indexing.brow + 4) * 32 +
                       indexing.bcol * 4];
      }

      // 3. Compute FFT on 256 elements
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
      const CT twiddle5_1 = pow_theta<256>(output_idx);
      const CT twiddle6_1 = pow_theta<256>(2 * output_idx);
      const CT twiddle7_1 = pow_theta<256>(3 * output_idx);

      const CT twiddle5_2 = pow_theta<256>(output_idx + 1);
      const CT twiddle6_2 = pow_theta<256>(2 * output_idx + 2);
      const CT twiddle7_2 = pow_theta<256>(3 * output_idx + 3);

      local_b[8 * i + 2] *= twiddle5_1;
      local_b[8 * i + 4] *= twiddle6_1;
      local_b[8 * i + 6] *= twiddle7_1;

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

      local_b[8 * i + 3] *= twiddle5_2;
      local_b[8 * i + 5] *= twiddle6_2;
      local_b[8 * i + 7] *= twiddle7_2;

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
