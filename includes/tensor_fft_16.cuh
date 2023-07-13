#pragma once

#include <c64_fft64.cuh>
#include <common.cuh>
#include <tensor_utils.cuh>

namespace fft {

template <typename CT, int Size, int UPB = 2, int FPU = 2>
struct tensor_fft_16 {
  using this_t = tensor_fft_16<CT, Size>;

  static constexpr auto threads = 32;
  static constexpr auto units_per_block = UPB;
  static constexpr auto ffts_per_unit = FPU;
  static constexpr auto max_threads_per_block = units_per_block * threads;

  mma_fp64_884_indexes indexing;

  const CT a = pow_theta<4>((indexing.arow % 4) * indexing.acol);
  const CT tw = pow_theta<16>((indexing.bcol % 4) * indexing.brow);

  static constexpr char print_type[] = "MMA16";

  static_assert(Size == 16, "SIZE MUST BE 16");

  template <int N> inline __device__ CT pow_theta(int p) const {
    p = p % N;
    float s, c;
    const float ang = p * (-2.f / N);
    sincospif(ang, &s, &c);
    return {c, s};
  }

  CT *sh_d;

  __device__ tensor_fft_16(CT *d) : sh_d(d) {}

  __device__ void operator()() {

    auto data = sh_d + Size * ffts_per_unit *
                           ((threadIdx.x + blockDim.x * threadIdx.y) / 32);

    // 4
    constexpr auto rstride0 = 4;
    constexpr auto cstride0 = 1;
    const int half = (threadIdx.x >= 16) ? 1 : 0;
    double s[6] = {0, 0, 0, 0, 0, 0};

    auto b = data[(indexing.bcol & 1) * 16 + (indexing.bcol % 4) * cstride0 +
                  indexing.brow * rstride0];
    karatsuba_inline_mma_8x8x4(a, b, s[0], s[1], s[2], s[3], s[4], s[5]);
    // twiddle and transpose
    b = tw *
        CT{s[0 + half] - s[2 + half], s[4 + half] - s[2 + half] - s[0 + half]};

    s[0] = s[1] = s[2] = s[3] = s[4] = s[5] = 0.0;
    karatsuba_inline_mma_8x8x4(a, b, s[0], s[1], s[2], s[3], s[4], s[5]);
    const int offset = indexing.crow >= 4 ? 16 : 0;
    data[offset + half + (indexing.ccol % 4) + (indexing.crow % 4) * 4] =
        CT{s[0 + half] - s[2 + half], s[4 + half] - s[2 + half] - s[0 + half]};
  }
};
} // namespace fft
