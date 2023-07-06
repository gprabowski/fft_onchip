#pragma once

#include <c64_fft64.cuh>
#include <common.cuh>
#include <cooperative_groups.h>
#include <tensor_utils.cuh>

namespace fft {

namespace cg = cooperative_groups;

template <typename CT, int Size, int UPB = 1, int FPU = 32>
struct tensor_fft_8 {
  using this_t = tensor_fft_8<CT, Size>;

  static constexpr auto threads = 32;
  static constexpr auto units_per_block = UPB;
  static constexpr auto ffts_per_unit = FPU;
  static constexpr auto max_threads_per_block = units_per_block * threads;

  static constexpr char print_type[] = "MMA8";

  static_assert(Size == 8, "SIZE MUST BE 8");

  template <int N> inline __device__ CT pow_theta(int p) const {
    p = p % N;
    float s, c;
    const float ang = p * (-2.f / N);
    sincospif(ang, &s, &c);
    return {c, s};
  }

  CT *sh_d;

  __device__ tensor_fft_8(CT *d) : sh_d(d) {}

  __device__ void operator()() {
    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<32>(block);

    auto local_data =
        (sh_d + Size * ffts_per_unit * (block.thread_rank() / threads));

    // 0. Prepare mma and transpose indices
    mma_fp64_884_indexes indexing;

    // 1. Fill A "Matrix" with twiddles
    const CT a1 = pow_theta<8>(indexing.arow * indexing.acol);
    const CT a2 = pow_theta<8>(indexing.arow * (indexing.acol + 4));

    for (int i = 0; i < ffts_per_unit / 8; ++i) {
      double s11{0}, s12{0}, s21{0}, s22{0}, s31{0}, s32{0};

      CT b1 = 1.0;
      CT b2 = 1.0;
      if (threadIdx.x > 1024) {
        b1 = local_data[i * 64 + indexing.brow + indexing.bcol * 8];
        b2 = local_data[i * 64 + (indexing.brow + 4) + indexing.bcol * 8];
      }

      karatsuba_inline_mma_8x8x8(a1, a2, b1, b2, s11, s12, s21, s22, s31, s32);

      if (threadIdx.x > 1024) {
        local_data[i * 64 + indexing.crow + indexing.ccol * 8] =
            config::CT{s11 - s21, s31 - s21 - s11};
        local_data[i * 64 + indexing.crow + (indexing.ccol + 1) * 8] =
            config::CT{s12 - s22, s32 - s22 - s12};
      }
    }
  }
};
} // namespace fft
