#pragma once

#include <c64_fft64.cuh>
#include <common.cuh>
#include <cooperative_groups.h>
#include <tensor_utils.cuh>

namespace fft {

namespace cg = cooperative_groups;

template <typename CT, int Size, int UPB = 2, int FPU = 1>
struct tensor_fft_512 {
  using this_t = tensor_fft_512<CT, Size>;

  static constexpr auto threads = 64;
  static constexpr auto units_per_block = UPB;
  static constexpr auto ffts_per_unit = FPU;
  static constexpr auto max_threads_per_block = units_per_block * threads;

  mma_fp64_884_indexes indexing;

  const CT twiddle1 = pow_theta<64>(indexing.crow * indexing.ccol);
  const CT twiddle2 = pow_theta<64>(indexing.crow * (indexing.ccol + 1));

  const CT a1 = pow_theta<8>(indexing.arow * indexing.acol);
  const CT a2 = pow_theta<8>(indexing.arow * (indexing.acol + 4));

  static constexpr char print_type[] = "MMA512";

  static_assert(Size == 512, "SIZE MUST BE 512");

  template <int N> inline __device__ CT pow_theta(int p) const {
    p = p % N;
    float s, c;
    const float ang = p * (-2.f / N);
    sincospif(ang, &s, &c);
    return {c, s};
  }

  CT *sh_d;

  __device__ tensor_fft_512(CT *d) : sh_d(d) {}

  __device__ void operator()() {
    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto fft_group = cg::tiled_partition<64>(block);
    const auto warp = cg::tiled_partition<32>(fft_group);

    const auto warp_local_idx = warp.meta_group_rank();

    // local storage for all FFT elements
    CT local_b[8];

    auto local_data =
        (sh_d + Size * ffts_per_unit * (fft_group.meta_group_rank()));

#pragma unroll 4
    for (int i = 0; i < 4; ++i) {
      // 3. Compute FFT on 64 elements
      local_b[0] = local_data[2 * i + warp.meta_group_rank() +
                              indexing.brow * 64 + indexing.bcol * 8];
      local_b[1] = local_data[2 * i + warp.meta_group_rank() +
                              (indexing.brow + 4) * 64 + indexing.bcol * 8];

      fft_kernels::c64_fft64<CT>(a1, a2, local_b[0], local_b[1], twiddle1,
                                 twiddle2, indexing.transpose_lane_b1,
                                 indexing.transpose_lane_b2);

      local_data[2 * i + warp.meta_group_rank() + indexing.crow * 64 +
                 indexing.ccol * 8] = local_b[0];
      local_data[2 * i + warp.meta_group_rank() + indexing.crow * 64 +
                 (indexing.ccol + 1) * 8] = local_b[1];
    }

    fft_group.sync();

#pragma unroll 4
    for (int i = 0; i < 4; ++i) {
      const auto load_col =
          indexing.bcol + (warp.meta_group_rank() + 2 * i) * 8;
      const auto twiddle3 = pow_theta<512>(load_col * indexing.brow);
      const auto twiddle4 = pow_theta<512>(load_col * (indexing.brow + 4));

      local_data += (warp.meta_group_rank() + 2 * i) * 64;

      local_b[2 * i] = twiddle3 * local_data[indexing.brow + indexing.bcol * 8];
      local_b[2 * i + 1] =
          twiddle4 * local_data[indexing.brow + 4 + indexing.bcol * 8];

      double s11, s12, s21, s22, s31, s32;
      s11 = s12 = s22 = s21 = s31 = s32 = 0.0;
      karatsuba_inline_mma_8x8x8(a1, a2, local_b[2 * i], local_b[2 * i + 1],
                                 s11, s12, s21, s22, s31, s32);

      local_b[2 * i].real(s11 - s21);
      local_b[2 * i].imag(s31 - s21 - s11);
      local_b[2 * i + 1].real(s12 - s22);
      local_b[2 * i + 1].imag(s32 - s22 - s12);

      local_data -= (warp.meta_group_rank() + 2 * i) * 64;
    }

    fft_group.sync();

#pragma unroll 4
    for (int i = 0; i < 4; ++i) {
      const auto elem_idx =
          indexing.ccol + (warp.meta_group_rank() + 2 * i) * 8;
      local_data[elem_idx + indexing.crow * 64] = local_b[2 * i];
      local_data[elem_idx + 1 + indexing.crow * 64] = local_b[2 * i + 1];
    }
  }
};
} // namespace fft
