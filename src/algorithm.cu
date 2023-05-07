#include <algorithm.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuComplex.h>

namespace fft {

constexpr double PI = 3.14159265359;

struct complex4 {
  cuDoubleComplex v[4];
  cuDoubleComplex &operator[](const int i) { return v[i]; }
};

struct complex2 {
  cuDoubleComplex v[2];
  cuDoubleComplex &operator[](const int i) { return v[i]; }
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
  auto tmp = a;
  dft2(tmp.v[0], tmp.v[2]);
  dft2(tmp.v[1], tmp.v[3]);
  return tmp;
}

inline __device__ complex4 dft4_4(const complex4 &a) {
  complex4 x = dft2_4(a);
  return dft2_4({x.v[0], x.v[2], x.v[1], mul_plq(x.v[1])});
}

inline __device__ complex4 mul_4(const complex4 &a, const complex4 &b) {
  return {cuCmul(a.v[0], b.v[0]), cuCmul(a.v[1], b.v[1]),
          cuCmul(a.v[2], b.v[2]), cuCmul(a.v[3], b.v[3])};
}

__global__ void fft_radix4(const cuDoubleComplex *data, cuDoubleComplex *out,
                           int p) {
  const auto threads = blockDim.x * gridDim.x;
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  int k = tid & (p - 1);

  const auto local_data = data + tid;
  const auto local_out = out + ((tid - k) << 2) + k;

  double alpha = -PI * static_cast<double>(k) / (2 * p);

  // load and twiddle
  cuDoubleComplex u0 = local_data[0];
  cuDoubleComplex u1 = cuCmul(exp_alpha(alpha), local_data[threads]);
  cuDoubleComplex u2 = cuCmul(exp_alpha(2 * alpha), local_data[2 * threads]);
  cuDoubleComplex u3 = cuCmul(exp_alpha(3 * alpha), local_data[3 * threads]);

  // 2x DFT and twiddle
  cuDoubleComplex v0 = cuCadd(u0, u2);
  cuDoubleComplex v1 = cuCsub(u0, u2);
  cuDoubleComplex v2 = cuCadd(u1, u3);
  cuDoubleComplex v3 = mul_plq(cuCsub(u1, u3)); // twiddle

  // 2x DFT and store
  local_out[0] = cuCadd(v0, v2);
  local_out[p] = cuCadd(v1, v3);
  local_out[2 * p] = cuCsub(v0, v2);
  local_out[3 * p] = cuCsub(v1, v3);
}

// assumes that there's always N / 2 threads
__global__ void fft_radix2(cuDoubleComplex *data, cuDoubleComplex *out, int p) {
  const auto threads = blockDim.x * gridDim.x;
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  int k = tid & (p - 1);

  cuDoubleComplex u0 = data[tid];
  cuDoubleComplex u1 = data[tid + threads];

  cuDoubleComplex twiddle = pow_theta(k, p);
  const auto tmp = cuCmul(u1, twiddle);
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
  if (log_n & 1) {
    fft_radix2<<<data.size() / 512, 256>>>(
        thrust::raw_pointer_cast(odd ? d_data1.data() : d_data2.data()),
        thrust::raw_pointer_cast(odd ? d_data2.data() : d_data1.data()), p);
    p = 2;
  } else {
    fft_radix4<<<data.size() / 1024, 256>>>(
        thrust::raw_pointer_cast(odd ? d_data1.data() : d_data2.data()),
        thrust::raw_pointer_cast(odd ? d_data2.data() : d_data1.data()), p);
    p = 4;
  }

  odd = !odd;

  for (; p < data.size(); p *= 4) {
    fft_radix4<<<data.size() / 1024, 256>>>(
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
