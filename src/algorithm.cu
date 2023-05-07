#include <algorithm.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuComplex.h>

namespace fft {

constexpr double PI = 3.14159265359;

__device__ __host__ inline cuDoubleComplex operator+(const cuDoubleComplex &a,
                                                     const cuDoubleComplex &b) {
  return cuCadd(a, b);
}

__device__ __host__ inline cuDoubleComplex operator-(const cuDoubleComplex &a,
                                                     const cuDoubleComplex &b) {
  return cuCsub(a, b);
}

__device__ __host__ inline cuDoubleComplex operator*(const cuDoubleComplex &a,
                                                     const cuDoubleComplex &b) {
  return cuCmul(a, b);
}

struct complex4 {
  cuDoubleComplex v[4];
  __device__ __host__ inline cuDoubleComplex &operator[](int i) { return v[i]; }
  __device__ __host__ inline cuDoubleComplex operator[](int i) const {
    return v[i];
  }
};

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

__device__ __host__ constexpr static inline int ilog2(unsigned int n) {
  return 31 - __builtin_clz(n);
}

__device__ double2 scale(const double &s, const cuDoubleComplex &v) {
  return {s * v.x, s * v.y};
}

inline __device__ void dft2(cuDoubleComplex &a, cuDoubleComplex &b) {
  const auto tmp = cuCsub(a, b);
  a = cuCadd(a, b);
  b = tmp;
}

inline __device__ cuDoubleComplex exp_alpha(double alpha) {
  cuDoubleComplex ret;
  sincos(alpha, &ret.y, &ret.x);
  return ret;
}

inline __device__ cuDoubleComplex pow_theta(const int p, const int q) {
  return exp_alpha((-PI * p) / q);
}

inline __device__ cuDoubleComplex mul_plq(const cuDoubleComplex &a) {
  return {a.y, -a.x};
}

inline __device__ cuDoubleComplex sqr_1(const cuDoubleComplex &a) {
  return {a.x * a.x - a.y * a.y, 2.0f * a.x * a.y};
}

inline __device__ complex4 dft2_4(const complex4 &a) {
  return {a[0] + a[2], a[1] + a[3], a[0] - a[2], a[1] - a[3]};
}

inline __device__ complex4 dft4_4(const complex4 &a) {
  complex4 x = dft2_4(a);
  return dft2_4({x[0], x[2], x[1], mul_plq(x[3])});
}

inline __device__ complex4 mul_4(const complex4 &a, const complex4 &b) {
  return {cuCmul(a.v[0], b.v[0]), cuCmul(a.v[1], b.v[1]),
          cuCmul(a.v[2], b.v[2]), cuCmul(a.v[3], b.v[3])};
}

__global__ void fft_radix4_longer(const cuDoubleComplex *data,
                                  cuDoubleComplex *out, int p) {
  const auto threads = blockDim.x * gridDim.x;
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  int k = tid & (p - 1);

  const auto local_data = data + tid;
  const auto local_out = out + ((tid - k) << 2) + k;

  double alpha = -PI * static_cast<double>(k) / (2 * p);

  // load and twiddle
  complex4 u =
      dft4_4({local_data[0], cuCmul(exp_alpha(alpha), local_data[threads]),
              cuCmul(exp_alpha(2 * alpha), local_data[2 * threads]),
              cuCmul(exp_alpha(3 * alpha), local_data[3 * threads])});
  local_out[0] = u[0];
  local_out[p] = u[1];
  local_out[2 * p] = u[2];
  local_out[3 * p] = u[3];
}
__global__ void fft_radix4(const cuDoubleComplex *data, cuDoubleComplex *out,
                           int p) {
  const auto threads = blockDim.x * gridDim.x;
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  int k = tid & (p - 1);

  const auto local_data = data + tid;
  const auto local_out = out + ((tid - k) << 2) + k;

  double alpha = -PI * static_cast<double>(k) / (2 * p);

  cuDoubleComplex twiddle = exp_alpha(alpha);

  // load and twiddle
  cuDoubleComplex u0 = local_data[0];
  cuDoubleComplex u1 = twiddle * local_data[threads];
  cuDoubleComplex u2 = local_data[2 * threads];
  cuDoubleComplex u3 = twiddle * local_data[3 * threads];

  twiddle = sqr_1(twiddle);
  u2 = twiddle * u2;
  u3 = twiddle * u3;

  // 2x DFT and twiddle
  cuDoubleComplex v0 = u0 + u2;
  cuDoubleComplex v1 = u0 - u2;
  cuDoubleComplex v2 = u1 + u3;
  cuDoubleComplex v3 = mul_plq(u1 - u3);

  // 2x DFT and store
  local_out[0] = v0 + v2;
  local_out[p] = v1 + v3;
  local_out[2 * p] = v0 - v2;
  local_out[3 * p] = v1 - v3;
}

// mul_p*q* returns a * EXP(-I * PI * P / Q)
__device__ inline cuDoubleComplex mul_p0q1(const cuDoubleComplex &a) {
  return a;
}
__device__ inline cuDoubleComplex mul_p0q2(const cuDoubleComplex &a) {
  return a;
}
__device__ inline cuDoubleComplex mul_p1q2(const cuDoubleComplex &a) {
  return {a.y, -a.x};
}

// cos (pi / 4)
constexpr double SQRT_1_2 = 0.707106781188;
__device__ inline cuDoubleComplex mul_p0q4(const cuDoubleComplex &a) {
  return a;
}

__device__ inline cuDoubleComplex mul_p1q4(const cuDoubleComplex &a) {
  return cuDoubleComplex{SQRT_1_2 * (a.x + a.y), SQRT_1_2 * (-a.x + a.y)};
}

__device__ inline cuDoubleComplex mul_p2q4(const cuDoubleComplex &a) {
  return {a.y, -a.x};
}

__device__ inline cuDoubleComplex mul_p3q4(const cuDoubleComplex &a) {
  return cuDoubleComplex{SQRT_1_2 * (-a.x + a.y), SQRT_1_2 * (-a.x - a.y)};
}

__global__ void fft_radix8(const cuDoubleComplex *data, cuDoubleComplex *out,
                           int p) {
  const auto threads = blockDim.x * gridDim.x;
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  int k = tid & (p - 1);

  const auto local_data = data + tid;
  auto local_out = out + ((tid - k) << 3) + k;

  double alpha = -PI * static_cast<double>(k) / (4 * p);

  cuDoubleComplex twiddle = exp_alpha(alpha);

  cuDoubleComplex u0 = local_data[0];
  cuDoubleComplex u1 = twiddle * local_data[threads];
  cuDoubleComplex u2 = local_data[2 * threads];
  cuDoubleComplex u3 = twiddle * local_data[3 * threads];
  cuDoubleComplex u4 = local_data[4 * threads];
  cuDoubleComplex u5 = twiddle * local_data[5 * threads];
  cuDoubleComplex u6 = local_data[6 * threads];
  cuDoubleComplex u7 = twiddle * local_data[7 * threads];

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

  const cuDoubleComplex v0 = u0 + u4;
  const cuDoubleComplex v4 = mul_p0q4(u0 - u4);
  const cuDoubleComplex v1 = u1 + u5;
  const cuDoubleComplex v5 = mul_p1q4(u1 - u5);
  const cuDoubleComplex v2 = u2 + u6;
  const cuDoubleComplex v6 = mul_p2q4(u2 - u6);
  const cuDoubleComplex v3 = u3 + u7;
  const cuDoubleComplex v7 = mul_p3q4(u3 - u7);

  u0 = v0 + v2;
  u2 = mul_p0q2(v0 - v2);
  u1 = v1 + v3;
  u3 = mul_p1q2(v1 - v3);
  u4 = v4 + v6;
  u6 = mul_p0q2(v4 - v6);
  u5 = v5 + v7;
  u7 = mul_p1q2(v5 - v7);

  local_out[0] = u0 + u1;
  local_out[p] = u4 + u5;
  local_out[2 * p] = u2 + u3;
  local_out[3 * p] = u6 + u7;
  local_out[4 * p] = u0 - u1;
  local_out[5 * p] = u4 - u5;
  local_out[6 * p] = u2 - u3;
  local_out[7 * p] = u6 - u7;
}

constexpr double COS_8 = 0.923879532511f; // cos(Pi/8)
constexpr double SIN_8 = 0.382683432365f; // sin(Pi/8)

__device__ inline cuDoubleComplex mul_p0q8(const cuDoubleComplex &a) {
  return a;
}

__device__ inline cuDoubleComplex mul_p1q8(const cuDoubleComplex &a) {
  return cuDoubleComplex{COS_8, -SIN_8} * a;
}

// assumes that there's always N / 2 threads
__global__ void fft_radix2(cuDoubleComplex *data, cuDoubleComplex *out, int p) {
  const auto threads = blockDim.x * gridDim.x;
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  int k = tid & (p - 1);

  cuDoubleComplex u0 = data[tid];
  cuDoubleComplex u1 = data[tid + threads];

  cuDoubleComplex twiddle = pow_theta(k, p);
  const auto tmp = u1 * twiddle;
  dft2(u0, u1);
  int j = ((tid - k) << 1) + k;

  out[j] = u0;
  out[j + p] = u1;
};

size_t run_algorithm(const std::vector<std::complex<double>> &data,
                     std::vector<std::complex<double>> &out) {
  thrust::host_vector<cuDoubleComplex> h_data(data.size());
  thrust::device_vector<cuDoubleComplex> d_data1;
  thrust::device_vector<cuDoubleComplex> d_data2(data.size());

  for (int i = 0; i < data.size(); ++i) {
    h_data[i] = {data[i].real(), data[i].imag()};
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  d_data1 = h_data;

  bool odd = true;
  const auto log_n = ilog2(data.size());
  int p = 1;

  if (log_n % 3 == 1) {
    fft_radix2<<<data.size() / 512, 256>>>(
        thrust::raw_pointer_cast(odd ? d_data1.data() : d_data2.data()),
        thrust::raw_pointer_cast(odd ? d_data2.data() : d_data1.data()), p);
    p *= 2;
    odd = !odd;
  } else if (log_n % 3 == 2) {
    fft_radix4<<<data.size() / 1024, 256>>>(
        thrust::raw_pointer_cast(odd ? d_data1.data() : d_data2.data()),
        thrust::raw_pointer_cast(odd ? d_data2.data() : d_data1.data()), p);
    p *= 4;
    odd = !odd;
  }

  for (; p < data.size(); p *= 8) {
    fft_radix8<<<data.size() / 2048, 256>>>(
        thrust::raw_pointer_cast(odd ? d_data1.data() : d_data2.data()),
        thrust::raw_pointer_cast(odd ? d_data2.data() : d_data1.data()), p);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaGetLastError());
    odd = !odd;
  }

  h_data = odd ? d_data1 : d_data2;

  auto t2 = std::chrono::high_resolution_clock::now();
  const auto res_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

  for (int i = 0; i < data.size(); ++i) {
    out[i] = {h_data[i].x, h_data[i].y};
  }
  return res_ms.count();
}
} // namespace fft
