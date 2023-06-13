#pragma once

#include <chrono>
#include <vector>
#include <complex>

#include <cuda_fp16.h>

#include <thrust/complex.h>

template<int N>
inline __device__ config::CT pow_theta(int p) {
    p = p % N;
    double s, c;
    const double ang = p * (-2.0 / N);
    sincospi(ang, &s, &c);
    return {c, s};
}

__device__ __forceinline__ int descramble(int a) {
    a = a & (63 ^ 8);
    return (((a & 7) << 3) | (a & 48)>>4);
}

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

    Tensor tCsB = thrmma.partition_B(data_transposed);
    Tensor tCsC = thrmma.partition_C(data);
    Tensor tCrB = thrmma.make_fragment_B(tCsB);
    Tensor tCrC = thrmma.make_fragment_C(tCsC);

    cute::copy(tCsB, tCrB);

    clear(tCrC);

    constexpr int K_BLOCK_MAX = size<2>(tCrA);

    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      gemm(thrmma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    }

    const auto firstc = threadIdx.x * 2;
    const auto rowc = firstc % 8;
    const auto colc = firstc / 8;

    const auto firsts = firstc + 1;
    const auto rows = firsts % 8;
    const auto cols = firsts / 8;

    const auto needed0 = (threadIdx.x >> 2) * 8 + (threadIdx.x % 4);
    const auto needed1 = needed0 + 4;

    const bool idx1 = needed0 & 1;
    const auto add1 = needed0 >> 1;
    const bool idx2 = needed1 & 1;
    const auto add2 = needed1 >> 1;

    Tensor tCsBf = thrmma.partition_B(data);
    Tensor tCsCf = thrmma.partition_C(data_transposed);

    tCrC(0) *= pow_theta<64>(rowc * colc);
    tCrC(1) *= pow_theta<64>(rows * cols);

    #define FULL_MASK 0xffffffff
    tCrB(0)= CT{__shfl_sync(FULL_MASK, tCrC(idx1).real(), add1), __shfl_sync(FULL_MASK, tCrC(idx1).imag(), add1)};
    tCrB(1)= CT{__shfl_sync(FULL_MASK, tCrC(idx2).real(), add2), __shfl_sync(FULL_MASK, tCrC(idx2).imag(), add2)};

    clear(tCrC);

    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      gemm(thrmma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    }

    tCsCf(0) = tCrC(0);
    tCsCf(1) = tCrC(1);

    __syncthreads();
  }
};
}
