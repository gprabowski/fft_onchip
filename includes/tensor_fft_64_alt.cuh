#pragma once

#include <c64_fft64.cuh>
#include <common.cuh>
#include <cooperative_groups.h>
#include <tensor_utils.cuh>

namespace fft {

namespace cg = cooperative_groups;

template <typename CT, int Size, int UPB = 4, int FPU = 1>
struct tensor_fft_64_alt {
  using this_t = tensor_fft_64_alt<CT, Size>;

  static constexpr auto threads = 64;
  static constexpr auto units_per_block = UPB;
  static constexpr auto ffts_per_unit = FPU;
  static constexpr auto max_threads_per_block = units_per_block * threads;

  mma_fp64_884_indexes indexing;

  // compute A matrix elements
  double a[4];
  const bool acol_even = (indexing.acol % 2 == 0);
  const bool brow_even = (indexing.brow % 2 == 0),
             bcol_even = (indexing.bcol % 2 == 0);

  static constexpr char print_type[] = "MMA64";

  static_assert(Size == 64, "SIZE MUST BE 64");

  template <int N> inline __device__ CT pow_theta(int p) const {
    p = p % N;
    float s, c;
    const float ang = p * (-2.f / N);
    sincospif(ang, &s, &c);
    return {c, s};
  }

  CT *sh_d;

  __device__ tensor_fft_64_alt(CT *d) : sh_d(d) {
    auto tmp = pow_theta<8>(indexing.arow * (indexing.acol / 2));
    a[0] = acol_even ? tmp.real() : tmp.imag();
    tmp = pow_theta<8>(indexing.arow * (indexing.acol / 2 + 2));
    a[1] = acol_even ? tmp.real() : tmp.imag();
    tmp = pow_theta<8>(indexing.arow * (indexing.acol / 2 + 4));
    a[2] = acol_even ? tmp.real() : tmp.imag();
    tmp = pow_theta<8>(indexing.arow * (indexing.acol / 2 + 6));
    a[3] = acol_even ? tmp.real() : tmp.imag();
  }

  __device__ void operator()() {
    const auto grid = cg::this_grid();
    const auto block = cg::this_thread_block();
    const auto fft_group = cg::tiled_partition<64>(block);
    const auto warp = cg::tiled_partition<32>(fft_group);

    const auto cpx_bcol = indexing.bcol / 2 + warp.meta_group_rank() * 4;

    auto local_data =
        (sh_d + Size * ffts_per_unit * (block.thread_rank() / threads));

    double acc1{0.}, acc2{0.};

    // 1. perform first DFT
    for (int i = 0; i < 4; ++i) {
      const auto cpx_brow = indexing.brow / 2 + i * 2;

      CT elem = local_data[cpx_brow * 8 + cpx_bcol];
      double b = (brow_even && bcol_even)
                     ? elem.real()
                     : (brow_even ? elem.imag()
                                  : (bcol_even ? -elem.imag() : elem.real()));
      mma8x8x4(a[i], b, acc1, acc2);
    }

    // 2. twiddle
    CT res = pow_theta<64>(indexing.crow *
                           (indexing.ccol / 2 + warp.meta_group_rank() * 4)) *
             CT{acc1, acc2};

    fft_group.sync();

    // 3. Save to memory
    local_data[indexing.crow +
               (indexing.ccol / 2 + warp.meta_group_rank() * 4) * 8] = res;

    fft_group.sync();

    // 3. repeat
    acc1 = acc2 = 0.0;

    for (int i = 0; i < 4; ++i) {
      const auto cpx_brow = indexing.brow / 2 + i * 2;

      CT elem = local_data[cpx_brow + cpx_bcol * 8];
      double b = (brow_even && bcol_even)
                     ? elem.real()
                     : (brow_even ? elem.imag()
                                  : (bcol_even ? -elem.imag() : elem.real()));
      mma8x8x4(a[i], b, acc1, acc2);
    }

    local_data[indexing.crow * 8 + indexing.ccol / 2 +
               warp.meta_group_rank() * 4] = CT{acc1, acc2};
  }
};
} // namespace fft
