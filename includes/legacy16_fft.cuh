#pragma once

#include <chrono>
#include <vector>
#include <complex>

#include <thrust/complex.h>

#include <common.cuh>

namespace fft {

template <int Num, int Base> inline constexpr int static_log3() {
  if constexpr (Num == Base)
    return 1;
  return 1 + static_log3<Num / Base, Base>();
}

__device__ constexpr double SQRT_1_2 = 0.707106781188;
__device__ constexpr double COS_8 = 0.923879532511;
__device__ constexpr double SIN_8 = 0.382683432365;

template <typename CT, int Size> struct legacy16_fft {
  using this_t = legacy16_fft<CT, Size>;


  static constexpr auto BlockSize = Size / 16;
  static constexpr dim3 threads = {BlockSize, 1, 1};
  static constexpr int depth = static_log3<Size, 16>();

  const int tid = threadIdx.x;
  const int warpIdx = tid / 32;
  const int laneIdx = tid % 32;

  inline __device__ CT pow_theta(int p, int N) const {
    p = p & (N - 1);
    double s, c;
    const double ang = p * (-2.0 / N);
    sincospi(ang, &s, &c);
    return {c, s};
  }

  CT *sh_d, *sh_f;

  __device__ legacy16_fft(CT *d, CT *f) : sh_d(d), sh_f(f) {}


    static inline __device__ CT sqr_1(const CT &a) {
      return {a.real() * a.real() - a.imag() * a.imag(), 2.0f * a.real() * a.imag()};
    }

    static __device__ inline CT b02(const CT &a) {
        return a;
    }

    static __device__ inline CT b12(const CT &a) {
      return {a.imag(), -a.real()};
    }


    static __device__ inline CT b04(const CT &a) {
        return a;
    }
    static __device__ inline CT b14(const CT &a) {
      return CT{SQRT_1_2 * (a.real() + a.imag()), SQRT_1_2 * (-a.real() + a.imag())};
    }

    static __device__ inline CT b24(const CT &a) {
      return {a.imag(), -a.real()};
    }

    static __device__ inline CT b34(const CT &a) {
      return CT{SQRT_1_2 * (-a.real() + a.imag()), SQRT_1_2 * (-a.real() - a.imag())};
    }

    static __device__ inline CT b08(const CT &a) {
      return a;
    }

    static __device__ inline CT b18(const CT &a) {
      return CT{COS_8, -SIN_8} * a;
    }

    static __device__ inline CT b28(const CT &a) {
      return CT{SQRT_1_2 * (a.real() + a.imag()), SQRT_1_2 * (-a.real() + a.imag())};
    }

    static __device__ inline CT b38(const CT &a) {
      return CT{SIN_8, -COS_8} * a;
    }

    static __device__ inline CT b48(const CT &a) {
      return {a.imag(), -a.real()};
    }

    static __device__ inline CT b58(const CT &a) {
      return CT{-SIN_8, -COS_8} * a;
    }

    static __device__ inline CT b68(const CT &a) {
      return CT{SQRT_1_2 * (-a.real() + a.imag()), SQRT_1_2 * (-a.real() - a.imag())};
    }

    static __device__ inline CT b78(const CT &a) {
      return CT{-COS_8, -SIN_8} * a;
    }

    using mul_op = CT (*)(const CT &);

    template <mul_op op>
    __device__ inline void dft2_twiddle(CT &a, CT &b) {
      CT tmp = op(a - b);
      a = a + b;
      b = tmp;
    }


  __device__ void operator()() {
      int p = 1;
      for(int i = 0; i < depth; ++i) {
          int k = tid & (p - 1);

          const auto local_data = sh_d + tid;
          auto local_out = sh_d + ((tid - k) << 4) + k;

          CT u[16];

          __syncthreads();
          // load
          for (int m = 0; m < 16; ++m) {
            u[m] = local_data[m * BlockSize];
          }
          __syncthreads();

          // twiddle
          for (int m = 1; m < 16; ++m) {
            u[m] = pow_theta(m * k, 16 * p) * u[m];
          }
          // load

          // 8x in place dft2 and twiddle (1)
          dft2_twiddle<b08>(u[0], u[8]);
          dft2_twiddle<b18>(u[1], u[9]);
          dft2_twiddle<b28>(u[2], u[10]);
          dft2_twiddle<b38>(u[3], u[11]);
          dft2_twiddle<b48>(u[4], u[12]);
          dft2_twiddle<b58>(u[5], u[13]);
          dft2_twiddle<b68>(u[6], u[14]);
          dft2_twiddle<b78>(u[7], u[15]);

          // 8x in place dft2 and twiddle (2)
          dft2_twiddle<b04>(u[0], u[4]);
          dft2_twiddle<b14>(u[1], u[5]);
          dft2_twiddle<b24>(u[2], u[6]);
          dft2_twiddle<b34>(u[3], u[7]);
          dft2_twiddle<b04>(u[8], u[12]);
          dft2_twiddle<b14>(u[9], u[13]);
          dft2_twiddle<b24>(u[10], u[14]);
          dft2_twiddle<b34>(u[11], u[15]);

          // 8x in place dft2 and twiddle (3)
          dft2_twiddle<b02>(u[0], u[2]);
          dft2_twiddle<b12>(u[1], u[3]);
          dft2_twiddle<b02>(u[4], u[6]);
          dft2_twiddle<b12>(u[5], u[7]);
          dft2_twiddle<b02>(u[8], u[10]);
          dft2_twiddle<b12>(u[9], u[11]);
          dft2_twiddle<b02>(u[12], u[14]);
          dft2_twiddle<b12>(u[13], u[15]);

          // Final 8x dft2 and store
          local_out[0] = u[0] + u[1];
          local_out[p] = u[8] + u[9];
          local_out[2 * p] = u[4] + u[5];
          local_out[3 * p] = u[12] + u[13];
          local_out[4 * p] = u[2] + u[3];
          local_out[5 * p] = u[10] + u[11];
          local_out[6 * p] = u[6] + u[7];
          local_out[7 * p] = u[14] + u[15];
          local_out[8 * p] = u[0] - u[1];
          local_out[9 * p] = u[8] - u[9];
          local_out[10 * p] = u[4] - u[5];
          local_out[11 * p] = u[12] - u[13];
          local_out[12 * p] = u[2] - u[3];
          local_out[13 * p] = u[10] - u[11];
          local_out[14 * p] = u[6] - u[7];
          local_out[15 * p] = u[14] - u[15];

          p *= 16;
    }
  }

};
}
