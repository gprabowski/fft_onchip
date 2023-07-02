#pragma once

#include <c64_fft64.cuh>
#include <common.cuh>
#include <cooperative_groups.h>
#include <tensor_utils.cuh>

namespace fft {

namespace cg = cooperative_groups;

template <typename CT, int Size, int UPB = 4, int FPU = 1>
struct tensor_fft_128 {
  using this_t = tensor_fft_128<CT, Size>;

  static constexpr auto threads = 64;
  static constexpr auto units_per_block = UPB;
  static constexpr auto ffts_per_unit = FPU;
  static constexpr auto max_threads_per_block = units_per_block * threads;

  mma_fp64_884_indexes indexing;

  const CT twiddle1 = pow_theta<64>(indexing.crow * indexing.ccol);
  const CT twiddle2 = pow_theta<64>(indexing.crow * (indexing.ccol + 1));

  const CT a1 = pow_theta<8>(indexing.arow * indexing.acol);
  const CT a2 = pow_theta<8>(indexing.arow * (indexing.acol + 4));

  static constexpr char print_type[] = "MMA128";

  static_assert(Size == 128, "SIZE MUST BE 128");

  template <int N> inline __device__ CT pow_theta(int p) const {
    p = p % N;
    float s, c;
    const float ang = p * (-2.f / N);
    sincospif(ang, &s, &c);
    return {c, s};
  }

  CT *sh_d;

  __device__ tensor_fft_128(CT *d) : sh_d(d) {}

  __device__ void operator()() {
    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto fft_group = cg::tiled_partition<64>(block);
    const auto warp = cg::tiled_partition<32>(fft_group);

    const CT twiddle4 = pow_theta<128>(fft_group.thread_rank());

    const auto warp_local_idx = warp.meta_group_rank();

    // local storage for all FFT elements
    CT local_b[2 * ffts_per_unit];

    auto local_data =
        (sh_d + Size * ffts_per_unit * (fft_group.meta_group_rank())) +
        warp_local_idx;

    // 0. Prepare mma and transpose indices

    // 1. Pre-load b for 1st iter
    // in here we tranpose the matrix (as its naturally
    // set in memory in a column major fashion)
    local_b[0] = local_data[indexing.brow * 16 + indexing.bcol * 2];
    local_b[1] = local_data[(indexing.brow + 4) * 16 + indexing.bcol * 2];

#pragma unroll
    for (int i = 0; i < ffts_per_unit; ++i) {
      // 2. Pre-load B elements for next iteration
      if (i < ffts_per_unit - 1) {
        local_b[2 * (i + 1)] =
            local_data[(i + 1) * Size + indexing.brow * 16 + indexing.bcol * 2];
        local_b[2 * (i + 1) + 1] =
            local_data[(i + 1) * Size + (indexing.brow + 4) * 16 +
                       indexing.bcol * 2];
      }

      // 3. Compute FFT on 128 elements
      fft_kernels::c64_fft64<CT>(a1, a2, local_b[2 * i], local_b[2 * i + 1],
                                 twiddle1, twiddle2, indexing.transpose_lane_b1,
                                 indexing.transpose_lane_b2);

      block.sync();
      // 4. Save intermediate results to memory in correct order
      local_data[i * Size + indexing.crow * 16 + indexing.ccol * 2] =
          local_b[2 * i];
      local_data[i * Size + indexing.crow * 16 + (indexing.ccol + 1) * 2] =
          local_b[2 * i + 1];

      local_data -= warp_local_idx;
      block.sync();

      // 5. perform radix-2 stage
      local_b[2 * i] = local_data[i * Size + 2 * fft_group.thread_rank()];
      local_b[2 * i + 1] =
          local_data[i * Size + 2 * fft_group.thread_rank() + 1];

      block.sync();

      const auto tmp = local_b[2 * i + 1] * twiddle4;
      local_b[2 * i + 1] = local_b[2 * i] - tmp;
      local_b[2 * i] = local_b[2 * i] + tmp;

      local_data[i * Size + fft_group.thread_rank()] = local_b[2 * i];
      local_data[i * Size + fft_group.thread_rank() + fft_group.num_threads()] =
          local_b[2 * i + 1];
    }
  }
};
} // namespace fft
