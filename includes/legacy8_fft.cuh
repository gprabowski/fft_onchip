#pragma once

namespace fft {

template <int Num, int Base> inline constexpr int static_log2() {
  if constexpr (Num == Base)
    return 1;
  return 1 + static_log<Num / Base, Base>();
}

template <typename CT, int Size> struct legacy_fft {
  using this_t = legacy_fft<CT, Size>;

  static constexpr auto BlockSize = Size / 8;
  static constexpr auto threads = BlockSize;
  static constexpr int depth = static_log2<Size, 8>();
  static constexpr auto ffts_per_block = 1;
  static constexpr auto ffts_per_unit = 1;

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

  CT *sh_d;

  __device__ legacy_fft(CT *d) : sh_d(d) {}

  inline __device__ CT sqr_1(const CT &a) {
    return {a.real() * a.real() - a.imag() * a.imag(),
            2.0f * a.real() * a.imag()};
  }

  __device__ inline CT b12(const CT &a) { return {a.imag(), -a.real()}; }

  static constexpr double SQRT_1_2 = 0.707106781188;
  __device__ inline CT b14(const CT &a) {
    return CT{SQRT_1_2 * (a.real() + a.imag()),
              SQRT_1_2 * (-a.real() + a.imag())};
  }

  __device__ inline CT b24(const CT &a) { return {a.imag(), -a.real()}; }

  __device__ inline CT b34(const CT &a) {
    return CT{SQRT_1_2 * (-a.real() + a.imag()),
              SQRT_1_2 * (-a.real() - a.imag())};
  }

  __device__ void operator()() {
    int p = 1;
    for (int i = 0; i < depth; ++i) {
      int k = tid & (p - 1);
      const auto local_data = sh_d + tid;
      auto local_out = sh_d + ((tid - k) << 3) + k;

      CT twiddle = pow_theta(k, 8 * p);

      __syncthreads();

      CT u0 = local_data[0];
      CT u1 = twiddle * local_data[BlockSize];
      CT u2 = local_data[2 * BlockSize];
      CT u3 = twiddle * local_data[3 * BlockSize];
      CT u4 = local_data[4 * BlockSize];
      CT u5 = twiddle * local_data[5 * BlockSize];
      CT u6 = local_data[6 * BlockSize];
      CT u7 = twiddle * local_data[7 * BlockSize];

      __syncthreads();

      twiddle = sqr_1(twiddle);
      u2 = twiddle * u2;
      u3 = twiddle * u3;
      u6 = twiddle * u6;
      u7 = twiddle * u7;
      twiddle = sqr_1(twiddle);
      u4 = twiddle * u4;
      u5 = twiddle * u5;
      u6 = twiddle * u6;
      u7 = twiddle * u7;

      const CT v0 = u0 + u4;
      const CT v4 = u0 - u4;
      const CT v1 = u1 + u5;
      const CT v5 = b14(u1 - u5);
      const CT v2 = u2 + u6;
      const CT v6 = b24(u2 - u6);
      const CT v3 = u3 + u7;
      const CT v7 = b34(u3 - u7);

      u0 = v0 + v2;
      u2 = v0 - v2;
      u1 = v1 + v3;
      u3 = b12(v1 - v3);
      u4 = v4 + v6;
      u6 = v4 - v6;
      u5 = v5 + v7;
      u7 = b12(v5 - v7);

      local_out[0] = u0 + u1;
      local_out[p] = u4 + u5;
      local_out[2 * p] = u2 + u3;
      local_out[3 * p] = u6 + u7;
      local_out[4 * p] = u0 - u1;
      local_out[5 * p] = u4 - u5;
      local_out[6 * p] = u2 - u3;
      local_out[7 * p] = u6 - u7;

      p *= 8;
    }
  }
};
} // namespace fft
