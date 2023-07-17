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

  const CT a1 = pow_theta<8>(indexing.arow * (2 * indexing.acol));
  const CT a2 = pow_theta<8>(indexing.arow * (2 * indexing.acol + 1));

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
    const auto fft_group = cg::tiled_partition<128>(block);
    const auto warp = cg::tiled_partition<32>(fft_group);

    const auto warp_local_idx = warp.meta_group_rank();

    // local storage for all FFT elements
    CT local_b[4];

    auto local_data =
        (sh_d + Size * ffts_per_unit * (fft_group.meta_group_rank()));

#pragma unroll ffts_per_unit
    for (int fs = 0; fs < ffts_per_unit; ++fs) {

      // perform the initial radix-8
#pragma unroll 2
      for (int i = 0; i < 2; ++i) {
        local_b[2 * i] = local_data[(indexing.brow * 2) * 64 + indexing.bcol +
                                    i * 8 + warp_local_idx * 16];
        local_b[2 * i + 1] =
            local_data[(indexing.brow * 2 + 1) * 64 + indexing.bcol + i * 8 +
                       warp_local_idx * 16];

        double s11, s12, s21, s22, s31, s32;
        s11 = s12 = s22 = s21 = s31 = s32 = 0.0;
        karatsuba_inline_mma_8x8x8(a1, a2, local_b[2 * i], local_b[2 * i + 1],
                                   s11, s12, s21, s22, s31, s32);

        local_data[indexing.crow * 64 + indexing.ccol + i * 8 +
                   warp_local_idx * 16]
            .real(s11 - s21);
        local_data[indexing.crow * 64 + indexing.ccol + i * 8 +
                   warp_local_idx * 16]
            .imag(s31 - s21 - s11);
        local_data[indexing.crow * 64 + indexing.ccol + 1 + i * 8 +
                   warp_local_idx * 16]
            .real(s12 - s22);
        local_data[indexing.crow * 64 + indexing.ccol + 1 + i * 8 +
                   warp_local_idx * 16]
            .imag(s32 - s22 - s12);
      }

      fft_group.sync();

      // perform the remaining radix-64
#pragma unroll 2
      for (int i = 0; i < 2; ++i) {
        const auto elem_idx = i + warp_local_idx * 2;
        // load and twiddle
        local_b[2 * i] = pow_theta<64>(elem_idx * ((indexing.brow * 2) * 8 +
                                                   indexing.bcol)) *
                         local_data[(indexing.brow * 2) * 8 + indexing.bcol +
                                    i * 64 + warp_local_idx * 128];
        local_b[2 * i + 1] =
            pow_theta<64>(elem_idx *
                          ((indexing.brow * 2 + 1) * 8 + indexing.bcol)) *
            local_data[(indexing.brow * 2 + 1) * 8 + indexing.bcol + i * 64 +
                       warp_local_idx * 128];

        double s11, s12, s21, s22, s31, s32;
        s11 = s12 = s22 = s21 = s31 = s32 = 0.0;
        karatsuba_inline_mma_8x8x8(a1, a2, local_b[2 * i], local_b[2 * i + 1],
                                   s11, s12, s21, s22, s31, s32);

        const auto n_elem_idx = indexing.crow * 8 + elem_idx;

        local_b[2 * i] = pow_theta<512>(n_elem_idx * indexing.brow * 2) *
                         CT{s11 - s21, s31 - s21 - s11};
        local_b[2 * i + 1] =
            pow_theta<512>(n_elem_idx * (indexing.brow * 2 + 1)) *
            CT{s12 - s22, s32 - s22 - s12};

        s11 = s12 = s22 = s21 = s31 = s32 = 0.0;
        karatsuba_inline_mma_8x8x8(a1, a2, local_b[2 * i], local_b[2 * i + 1],
                                   s11, s12, s21, s22, s31, s32);
        local_b[2 * i].real(s11 - s21);
        local_b[2 * i].imag(s31 - s21 - s11);
        local_b[2 * i + 1].real(s12 - s22);
        local_b[2 * i + 1].imag(s32 - s22 - s12);
      }

      fft_group.sync();

      for (int i = 0; i < 2; ++i) {
        const auto elem_idx = i + warp_local_idx * 2;
        const auto out_offset =
            elem_idx + 8 * indexing.ccol + indexing.crow * 64;
        local_data[out_offset] = local_b[2 * i];
        local_data[out_offset + 8] = local_b[2 * i + 1];
      }

      local_data += Size;
    }
  }
};
} // namespace fft
