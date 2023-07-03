#pragma once

#include <c64_fft64.cuh>
#include <common.cuh>
#include <cooperative_groups.h>
#include <tensor_utils.cuh>

namespace fft {

namespace cg = cooperative_groups;

template <typename CT, int Size, int UPB = 1, int FPU = 1>
struct tensor_fft_4096 {
  using this_t = tensor_fft_4096<CT, Size>;

  static constexpr auto threads = 512;
  static constexpr auto units_per_block = UPB;
  static constexpr auto ffts_per_unit = FPU;
  static constexpr auto max_threads_per_block = units_per_block * threads;

  mma_fp64_884_indexes;
  const int b_row_idx = brow * 8 + bcol;
  const int c_row_idx = crow * 8 + ccol;

  const CT twiddle1 = pow_theta<64>(crow * ccol);
  const CT twiddle2 = pow_theta<64>(crow * (ccol + 1));

  const CT a1 = pow_theta<8>(arow * acol);
  const CT a2 = pow_theta<8>(arow * (acol + 4));

  static constexpr char print_type[] = "MMA4096";

  static_assert(Size == 4096, "SIZE MUST BE 4096");

  template <int N> inline __device__ CT pow_theta(int p) const {
    p = p % N;
    float s, c;
    const float ang = p * (-2.f / N);
    sincospif(ang, &s, &c);
    return {c, s};
  }

  CT *sh_d;

  __device__ tensor_fft_4096(CT *d) : sh_d(d) {}

  __device__ void operator()() {
    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<32>(block);

    // local storage for all FFT elements
    CT b1, b2;
    CT local_b[4096 / threads];

    // 0. Prepare mma and transpose indices
    // 1. Pre-load b for 1st iter
    // in here we tranpose the matrix (as its naturally
    // set in memory in a column major fashion)

    for (int i = warp.meta_group_rank(); i < 64; i += warp.meta_group_size()) {

      b1 = sh_d[i + b_row_idx * 64];
      b2 = sh_d[i + (b_row_idx + 32) * 64];

      // 3. Compute FFT on 64 elements
      fft_kernels::c64_fft64<CT>(a1, a2, b1, b2, twiddle1, twiddle2,
                                 transpose_lane_b1, transpose_lane_b2);

      // 4. Save intermediate results to memory in correct order
      sh_d[i + c_row_idx * 64] = b1;
      sh_d[i + (c_row_idx + 1) * 64] = b2;
    }

    block.sync();

    // 5. load and twiddle
    for (int i = warp.meta_group_rank(); i < 64; i += warp.meta_group_size()) {
      const auto twiddle_3 = pow_theta<4096>(i * b_row_idx);
      const auto twiddle_4 = pow_theta<4096>(i * (b_row_idx + 32));
      const auto reg_idx = 2 * (i / warp.meta_group_size());

      local_b[reg_idx] = twiddle_3 * sh_d[i * 64 + b_row_idx];
      local_b[reg_idx + 1] = twiddle_4 * sh_d[i * 64 + (b_row_idx + 32)];

      // 3. Compute FFT on 64 elements
      fft_kernels::c64_fft64<CT>(a1, a2, local_b[reg_idx], local_b[reg_idx + 1],
                                 twiddle1, twiddle2, transpose_lane_b1,
                                 transpose_lane_b2);
    }

    block.sync();

    for (int i = warp.meta_group_rank(); i < 64; i += warp.meta_group_size()) {
      const auto reg_idx = 2 * (i / warp.meta_group_size());
      sh_d[i + c_row_idx * 64] = local_b[reg_idx];
      sh_d[i + (c_row_idx + 1) * 64] = local_b[reg_idx + 1];
    }
  }
};
} // namespace fft
