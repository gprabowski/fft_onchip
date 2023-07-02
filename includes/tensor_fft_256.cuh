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

  static constexpr auto threads = 128;
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
    const auto fft_group = cg::tiled_partition<128>(block);
    const auto radix4_group = cg::tiled_partition<64>(fft_group);
    const auto warp = cg::tiled_partition<32>(fft_group);

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
    local_b[0] = local_data[indexing.brow * 32 + indexing.bcol * 4];
    local_b[1] = local_data[(indexing.brow + 4) * 32 + indexing.bcol * 4];

#pragma unroll
    for (int i = 0; i < ffts_per_unit; ++i) {
      // 2. Pre-load B elements for next iteration
      if (i < ffts_per_unit - 1) {
        local_b[2 * (i + 1)] =
            local_data[(i + 1) * Size + indexing.brow * 32 + indexing.bcol * 4];
        local_b[2 * (i + 1) + 1] =
            local_data[(i + 1) * Size + (indexing.brow + 4) * 32 +
                       indexing.bcol * 4];
      }

      // 3. Compute FFT on 256 elements
      fft_kernels::c64_fft64<CT>(a1, a2, local_b[2 * i], local_b[2 * i + 1],
                                 twiddle1, twiddle2, indexing.transpose_lane_b1,
                                 indexing.transpose_lane_b2);

      fft_group.sync();
      // 4. Save intermediate results to memory in correct order
      local_data[i * Size + indexing.crow * 32 + indexing.ccol * 4] =
          local_b[2 * i];
      local_data[i * Size + indexing.crow * 32 + (indexing.ccol + 1) * 4] =
          local_b[2 * i + 1];

      local_data -= warp_local_idx;
      fft_group.sync();

      if (radix4_group.meta_group_rank() == 0) {
        // 5. perform radix-4 stage
        const CT twiddle5 = pow_theta<256>(1 * radix4_group.thread_rank());
        const CT twiddle6 = pow_theta<256>(2 * radix4_group.thread_rank());
        const CT twiddle7 = pow_theta<256>(3 * radix4_group.thread_rank());

        const auto r1 = local_data[i * Size + 4 * radix4_group.thread_rank()];
        const auto r2 =
            twiddle5 *
            local_data[i * Size + 4 * radix4_group.thread_rank() + 1];
        const auto r3 =
            twiddle6 *
            local_data[i * Size + 4 * radix4_group.thread_rank() + 2];
        const auto r4 =
            twiddle7 *
            local_data[i * Size + 4 * radix4_group.thread_rank() + 3];

        radix4_group.sync();

        local_data[i * Size + radix4_group.thread_rank()] = r1 + r2 + r3 + r4;
        local_data[i * Size + radix4_group.thread_rank() + 64] =
            r1 - r3 - CT{0, 1} * (r2 - r4);
        local_data[i * Size + radix4_group.thread_rank() + 128] =
            r1 + r3 - r2 - r4;
        local_data[i * Size + radix4_group.thread_rank() + 192] =
            r1 - r3 + CT{0, 1} * (r2 - r4);
      }
    }
  }
};
} // namespace fft
