#pragma once

#include <c64_fft64.cuh>
#include <common.cuh>
#include <cooperative_groups.h>
#include <tensor_utils.cuh>

namespace fft {

namespace cg = cooperative_groups;

template <typename CT, int Size> struct tensor_fft_4096 {
  using this_t = tensor_fft_4096<CT, Size>;

  static constexpr auto num_warps = 8;
  static constexpr auto threads = num_warps * 32;
  static constexpr auto ffts_per_block = 1;
  static constexpr auto ffts_per_unit = 1;

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
    // THIS KERNEL WAS CREATED FOR PROFILING PURPOSES
    // AND DOES NOT COMPUTE A CORRECT FFT
    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto warp = cg::tiled_partition<32>(block);

    auto local_data =
        (sh_d + Size * ffts_per_unit * (block.thread_rank() / threads));

    mma_fp64_884_indexes indexing;

    auto b1 = CT{};
    auto b2 = CT{};

    const CT a1 = CT{};
    const CT a2 = CT{};

    double s11, s12, s21, s22, s31, s32;

    // Simulate Row FFT
#pragma unroll
    for (int i = 0; i < 64; i += num_warps) {
      // to perform a 64 element fft we need a column and row fft of size 8
      // that's why two here
      karatsuba_inline_mma_8x8x8(a1, a2, b1, b2, s11, s12, s21, s22, s31, s32);
      karatsuba_inline_mma_8x8x8(a1, a2, b1, b2, s11, s12, s21, s22, s31, s32);
    }

    // Transpose
    block.sync();

#pragma unroll
    // Simulate Column fft
    for (int i = 0; i < 64; i += num_warps) {
      // to perform a 64 element fft we need a column and row fft of size 8
      // that's why two here
      karatsuba_inline_mma_8x8x8(a1, a2, b1, b2, s11, s12, s21, s22, s31, s32);
      karatsuba_inline_mma_8x8x8(a1, a2, b1, b2, s11, s12, s21, s22, s31, s32);
    }

    b1 = CT{s11 - s21, s31 - s21 - s11};
    b2 = CT{s12 - s22, s32 - s22 - s12};

    local_data[indexing.crow * 8 + indexing.ccol] = b1;
    local_data[indexing.crow * 8 + (indexing.ccol + 1)] = b2;
  }
};
} // namespace fft
