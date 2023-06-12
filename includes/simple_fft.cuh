#pragma once

#include <chrono>
#include <vector>
#include <complex>

#include <cuda_fp16.h>

#include <thrust/complex.h>

#include <register_cute.cuh>

namespace fft {
using namespace cute;

template <typename CT, int Size, int Radix> struct simple_fft {
  using this_t = simple_fft<CT, Size, Radix>;
  static constexpr auto RadixSquared = Radix * Radix;

  using TensorShape = Shape<_1, _1, _1>;
  using TiledMma =
      cute::TiledMMA<cute::MMA_Atom<cute::SM80_8x8x4_C64C64C64C64_TN>, // Atom
                     Layout<TensorShape>>;

  static constexpr dim3 threads = 32;

  const int tid = threadIdx.x;
  const int laneIdx = tid % 32;

  template<int N>
  inline __device__ CT pow_theta(int p) const {
    p = p % N;
    double s, c;
    const double ang = p * (-2.0 / N);
    sincospi(ang, &s, &c);
    return {c, s};
  }

  CT *sh_d, *sh_f;

  __device__ simple_fft(CT *d, CT *f) : sh_d(d), sh_f(f) {}

  __device__ void operator()() {
    auto thr_vmnk =
        typename TiledMma::ThrLayoutVMNK{}.get_flat_coord(laneIdx);
    auto thrmma = cute::ThrMMA<TiledMma, decltype(thr_vmnk)>(thr_vmnk);

    // Define CuTe tensors
    auto dft_matrix = cute::make_tensor(
        cute::make_smem_ptr(sh_f),
        cute::make_shape(cute::Int<Radix>{}, cute::Int<Radix>{}),
        cute::make_stride(cute::Int<1>{}, cute::Int<Radix>{}));

    Tensor tCsA = thrmma.partition_A(dft_matrix);                    

    Tensor tCrA = thrmma.make_fragment_A(tCsA);              

    const auto first = (threadIdx.x / 4) + (threadIdx.x % 4) * 8;
    const auto row = first % 8;
    const auto col = first / 8;

    tCrA(0) = pow_theta<Radix>(row * col);
    tCrA(1) = pow_theta<Radix>(row *(col + 4));

    auto data = cute::make_tensor(
        cute::make_smem_ptr(sh_d),
        cute::make_shape(cute::Int<Radix>{}, cute::Int<Radix>{}),
        cute::make_stride(cute::Int<Radix>{}, cute::Int<1>{}));

    auto data_transposed = cute::make_tensor(
        cute::make_smem_ptr(sh_d),
        cute::make_shape(cute::Int<Radix>{}, cute::Int<Radix>{}),
        cute::make_stride(cute::Int<1>{}, cute::Int<Radix>{}));

    __syncthreads();

    register_gemm(thrmma, tCrA, data_transposed, data, true);

    __syncthreads();

    register_gemm(thrmma, tCrA, data, data_transposed);

    __syncthreads();
  }

};
}
