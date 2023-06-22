#pragma once

#include <c64_fft64.cuh>
#include <common.cuh>
#include <cooperative_groups.h>
#include <tensor_utils.cuh>

namespace fft {

namespace cg = cooperative_groups;

template <typename CT, int Size> struct tensor_fft_64 {
  using this_t = tensor_fft_64<CT, Size>;

  static constexpr auto threads = 32;
  static constexpr auto ffts_per_block = 2;
  static constexpr auto ffts_per_unit = 8;

  static_assert(Size == 64, "SIZE MUST BE 64");

  template <int N> inline __device__ CT pow_theta(int p) const {
    p = p % N;
    float s, c;
    const float ang = p * (-2.f / N);
    sincospif(ang, &s, &c);
    return {c, s};
  }

  CT *sh_d;

  __device__ tensor_fft_64(CT *d) : sh_d(d) {}

  __device__ void operator()() {
    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<32>(block);

    // local storage for all FFT elements
    CT local_b[2 * ffts_per_unit];

    auto local_data =
        (sh_d + Size * ffts_per_unit * (block.thread_rank() / threads));

    // 0. Prepare mma and transpose indices
    mma_fp64_884_indexes indexing;

    // 1. Pre-load b for 1st iter
    // in here we tranpose the matrix (as its naturally
    // set in memory in a column major fashion)
    local_b[0] = local_data[indexing.brow * 8 + indexing.bcol];
    local_b[1] = local_data[(indexing.brow + 4) * 8 + indexing.bcol];

    // 2. Fill A "Matrix" with twiddles
    const CT a1 = pow_theta<8>(indexing.arow * indexing.acol);
    const CT a2 = pow_theta<8>(indexing.arow * (indexing.acol + 4));

    // 3. Pre-compute twiddles
    const CT twiddle1 = pow_theta<64>(indexing.crow * indexing.ccol);
    const CT twiddle2 = pow_theta<64>(indexing.crow * (indexing.ccol + 1));

#pragma unroll
    for (int i = 0; i < ffts_per_unit; ++i) {

      // 4. Pre-load B elements for next iteration
      if (i < ffts_per_unit - 1) {
        local_b[2 * (i + 1)] =
            local_data[(i + 1) * Size + indexing.brow * 8 + indexing.bcol];
        local_b[2 * (i + 1) + 1] =
            local_data[(i + 1) * Size + (indexing.brow + 4) * 8 +
                       indexing.bcol];
      }

      fft_kernels::c64_fft64<CT>(a1, a2, local_b[2 * i], local_b[2 * i + 1],
                                 twiddle1, twiddle2, indexing.transpose_lane_b1,
                                 indexing.transpose_lane_b2);
    }

#pragma unroll
    for (int i = 0; i < ffts_per_unit; ++i) {
      // 5. Save results
      // perform digit reversal (that's why indexing is reversed)
      local_data[i * Size + indexing.crow * 8 + indexing.ccol] = local_b[2 * i];
      local_data[i * Size + indexing.crow * 8 + (indexing.ccol + 1)] =
          local_b[2 * i + 1];
    }
  }
};
} // namespace fft