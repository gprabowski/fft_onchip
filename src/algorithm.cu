#include <algorithm.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda_fp16.h>

#include <config.hpp>

#include "cuComplex.h"

namespace fft {

__device__ static constexpr float PI = 3.14159265359;

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

inline __device__ __half2 hcmul(const __half2 &a, const __half2 &b) {
  return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

__device__ __host__ constexpr static inline int ilog2(unsigned int n) {
  return 31 - __builtin_clz(n);
}

inline __device__ __half2 exp_alpha(__half alpha) {
  __half2 ret;
  ret.x = hcos(alpha);
  ret.x = hsin(alpha);
  return ret;
}

inline __device__ cuComplex pow_theta(int p, int q) {
  p = p % q;
  const auto ang = (-2.f * PI * p) / q;
  float s, c;
  sincosf(ang, &s, &c);
  return {c, s};
}

__device__ int reverseDigits(int number, int base) {
  int reversedNumber = 0;

  while (number > 0) {
    reversedNumber = reversedNumber * base + number % base;
    number /= base;
  }

  return reversedNumber;
}

__global__ void onchip_reference(cuComplex *data) {
  extern __shared__ cuComplex shared_data[];

  const auto tid = threadIdx.x;

  // 1. copy data
  shared_data[tid] = data[tid];
  __syncthreads();

  // perform FFT
  const auto warpIdx = tid / 32;
  const auto laneIdx = tid % 32;
  const auto k = laneIdx;
  // perform two radix-32 iterations
  float local_r = shared_data[warpIdx + k * 32].x;
  float local_i = shared_data[warpIdx + k * 32].y;
  cuComplex result = {__half(0), __half(0)};
  __syncwarp();
// perform warp local butterfly
#define FULL_MASK 0xffffffff
  for (int m = 0; m < 32; ++m) {
    result = cuCaddf(result, cuCmulf(pow_theta(m * k, 32),
                                     {__shfl_sync(FULL_MASK, local_r, m),
                                      __shfl_sync(FULL_MASK, local_i, m)}));
  }
  shared_data[warpIdx + k * 32] = result;
  __syncthreads();

  local_r = shared_data[warpIdx * 32 + k].x;
  local_i = shared_data[warpIdx * 32 + k].y;
  result = cuComplex{__half(0), __half(0)};

  __syncwarp();
  for (int m = 0; m < 32; ++m) {
    result = cuCaddf(
        result, cuCmulf(pow_theta(m * reverseDigits(warpIdx * 32 + k, 32), N),
                        {__shfl_sync(FULL_MASK, local_r, m),
                         __shfl_sync(FULL_MASK, local_i, m)}));
  }

  data[reverseDigits(warpIdx * 32 + k, 32)] = result;
}

size_t run_algorithm(const std::vector<__half2> &data,
                     std::vector<__half2> &out) {
  thrust::host_vector<cuComplex> h_data;
  thrust::device_vector<cuComplex> d_data;

  for (int i = 0; i < data.size(); ++i) {
    h_data.push_back({data[i].x, data[i].y});
  }

  d_data = h_data;

  auto t1 = std::chrono::high_resolution_clock::now();
  onchip_reference<<<1, N, N * sizeof(cuComplex)>>>(
      thrust::raw_pointer_cast(d_data.data()));
  cudaDeviceSynchronize();
  h_data = d_data;
  auto t2 = std::chrono::high_resolution_clock::now();

  const auto res_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

  for (int i = 0; i < data.size(); ++i) {
    out[i] = __half2{h_data[i].x, h_data[i].y};
  }

  return res_ms.count();
}
} // namespace fft
