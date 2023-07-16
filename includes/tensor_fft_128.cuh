#pragma once

#include <c64_fft64.cuh>
#include <common.cuh>
#include <cooperative_groups.h>
#include <tensor_utils.cuh>

namespace fft {

namespace cg = cooperative_groups;

template <typename CT, int Size, int UPB = 2, int FPU = 1>
struct tensor_fft_128 {
  using this_t = tensor_fft_128<CT, Size>;

  static constexpr auto threads = 32;
  static constexpr auto units_per_block = UPB;
  static constexpr auto ffts_per_unit = FPU;
  static constexpr auto max_threads_per_block = units_per_block * threads;

  mma_fp64_884_indexes indexing;

  const CT twiddle1 = pow_theta<64>(indexing.crow * indexing.ccol);
  const CT twiddle2 = pow_theta<64>(indexing.crow * (indexing.ccol + 1));

  const CT a1 = pow_theta<8>(indexing.arow * (2 * indexing.acol));
  const CT a2 = pow_theta<8>(indexing.arow * (2 * indexing.acol + 1));

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
    const auto warp = cg::tiled_partition<32>(block);

    const auto output_idx = indexing.crow * 8 + indexing.ccol;

    const CT twiddle4 = pow_theta<128>(output_idx);
    const CT twiddle5 = pow_theta<128>(output_idx + 1);

    // local storage for all FFT elements
    CT local_b[ffts_per_unit * 4];

    auto local_data = sh_d + Size * ffts_per_unit * (warp.meta_group_rank());

    // 0. Prepare mma and transpose indices

    // 1. Pre-load b for 1st iter
    // in here we tranpose the matrix (as its naturally
    // set in memory in a column major fashion)
    local_b[0] = local_data[indexing.bpos / 2];
    local_b[1] = local_data[indexing.bpos / 2 + 16];
    local_b[2] = local_data[1 + indexing.bpos / 2];
    local_b[3] = local_data[1 + indexing.bpos / 2 + 16];

    for (int i = 0; i < ffts_per_unit; ++i) {
      if (i + 1 < ffts_per_unit) {
        local_b[4 * (i + 1)] = local_data[i * Size + indexing.bpos / 2];
        local_b[4 * (i + 1) + 1] =
            local_data[i * Size + indexing.bpos / 2 + 16];
        local_b[4 * (i + 1) + 2] = local_data[i * Size + 1 + indexing.bpos / 2];
        local_b[4 * (i + 1) + 3] =
            local_data[i * Size + 1 + indexing.bpos / 2 + 16];
      }
      // 3. Compute FFT on 128 elements
      fft_kernels::c64_fft64<CT>(a1, a2, local_b[4 * i], local_b[4 * i + 1],
                                 twiddle1, twiddle2);

      fft_kernels::c64_fft64<CT>(a1, a2, local_b[4 * i + 2], local_b[4 * i + 3],
                                 twiddle1, twiddle2);

      auto tmp = local_b[4 * i + 2] * twiddle4;
      local_b[4 * i + 2] = local_b[4 * i] - tmp;
      local_b[4 * i] = local_b[4 * i] + tmp;

      local_data[i * Size + output_idx] = local_b[4 * i];
      local_data[i * Size + output_idx + 64] = local_b[4 * i + 2];

      tmp = local_b[4 * i + 3] * twiddle5;
      local_b[4 * i + 3] = local_b[4 * i + 1] - tmp;
      local_b[4 * i + 1] = local_b[4 * i + 1] + tmp;

      local_data[i * Size + output_idx + 1] = local_b[4 * i + 1];
      local_data[i * Size + output_idx + 65] = local_b[4 * i + 3];
    }
  }
};
} // namespace fft
