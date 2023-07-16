#pragma once

#include <c64_fft64.cuh>
#include <common.cuh>
#include <tensor_utils.cuh>

namespace fft {

template <typename CT, int Size, int UPB = 8, int FPU = 1>
struct tensor_fft_256 {
  using this_t = tensor_fft_256<CT, Size>;

  static constexpr auto threads = 32;
  static constexpr auto units_per_block = UPB;
  static constexpr auto ffts_per_unit = FPU;
  static constexpr auto max_threads_per_block = units_per_block * threads;

  mma_fp64_884_indexes indexing;

  const CT twiddle1 = pow_theta<64>(indexing.crow * indexing.ccol);
  const CT twiddle2 = pow_theta<64>(indexing.crow * (indexing.ccol + 1));

  const CT a1 = pow_theta<8>(indexing.arow * (2 * indexing.acol));
  const CT a2 = pow_theta<8>(indexing.arow * (2 * indexing.acol + 1));

  const CT tw3 = pow_theta<256>(indexing.cpos);
  const CT tw4 = pow_theta<256>(indexing.cpos + 1);

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

    CT b[8];

    auto data = sh_d + Size * ffts_per_unit *
                           ((threadIdx.x + blockDim.x * threadIdx.y) / 32);

    for (int i = 0; i < ffts_per_unit; ++i) {
#pragma unroll 4
      for (int load_i = 0; load_i < 4; ++load_i) {
        // preloading doesn't give anything
        b[2 * load_i] = data[load_i + indexing.bpos];
        b[2 * load_i + 1] = data[load_i + indexing.bpos + 32];
      }

      fft_kernels::c64_fft64<CT>(a1, a2, b[0], b[1], twiddle1, twiddle2);
      fft_kernels::c64_fft64<CT>(a1, a2, b[2], b[3], twiddle1, twiddle2);
      fft_kernels::c64_fft64<CT>(a1, a2, b[4], b[5], twiddle1, twiddle2);
      fft_kernels::c64_fft64<CT>(a1, a2, b[6], b[7], twiddle1, twiddle2);

      b[2] *= tw3;
      b[4] *= tw3 * tw3;
      b[6] *= tw3 * tw3 * tw3;

      b[3] *= tw4;
      b[5] *= tw4 * tw4;
      b[7] *= tw4 * tw4 * tw4;

      data[indexing.cpos] = b[0] + b[2] + b[4] + b[6];
      data[indexing.cpos + 64] =
          b[0] - b[4] -
          CT{-b[2].imag() + b[6].imag(), b[2].real() - b[6].real()};
      data[indexing.cpos + 128] = b[0] + b[4] - (b[2] + b[6]);
      data[indexing.cpos + 192] =
          b[0] - b[4] +
          CT{-b[2].imag() + b[6].imag(), b[2].real() - b[6].real()};

      data[indexing.cpos + 1] = b[1] + b[3] + b[5] + b[7];
      data[indexing.cpos + 65] =
          b[1] - b[5] -
          CT{-b[3].imag() + b[7].imag(), b[3].real() - b[7].real()};
      data[indexing.cpos + 129] = b[1] + b[5] - (b[3] + b[7]);
      data[indexing.cpos + 193] =
          b[1] - b[5] +
          CT{-b[3].imag() + b[7].imag(), b[3].real() - b[7].real()};

      data += Size;
    }
  }
};
} // namespace fft
