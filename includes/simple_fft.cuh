#pragma once

#include <chrono>
#include <vector>
#include <complex>

#include <cuda_fp16.h>

#include <thrust/complex.h>

#include <custom_cute.cuh>

namespace fft {

using namespace cute;


template <int Num, int Base> inline constexpr int static_log2() {
  if constexpr (Num == Base)
    return 1;
  return 1 + static_log<Num / Base, Base>();
}

template <typename CT, int Size, int Radix> struct simple_fft {
  using this_t = simple_fft<CT, Size, Radix>;

  using TensorTiledMma =
      cute::TiledMMA<cute::MMA_Atom<cute::SM80_8x8x4_C64C64C64C64_TN>, // Atom
                     Layout<Shape<_1, _1, _1>>>;

  using UniTiledMma =
      cute::TiledMMA<cute::MMA_Atom<cute::UniversalFMA<CT, CT, CT, CT>>, // Atom
                     Layout<Shape<_8, _4, _1>>>;

  using TiledMma = TensorTiledMma;

  static constexpr auto warps_in_group = 1;
  static constexpr auto BlockSize = (32 * warps_in_group) * (Size / Radix);
  static constexpr dim3 threads = BlockSize;
  static constexpr int RadixSquared = Radix * Radix;
  static constexpr int depth = static_log2<Size, Radix>();

  const int tid = threadIdx.x;
  const int local_idx = tid % (warps_in_group * 32);
  const int warpIdx = tid / 32;
  const int warp_group = warpIdx / (warps_in_group);
  const int warp_group_idx = warpIdx % (warps_in_group);
  const int laneIdx = tid % 32;

  inline __device__ CT pow_theta(int p, int q) {
    p = p % q;
    const auto ang = (-2.f * PI * p) / q;
    return {cos(ang), sin(ang)};
  }

  CT *sh_d, *sh_f;

  __device__ simple_fft(CT *d, CT *f) : sh_d(d), sh_f(f) {}

  __device__ void operator()() {
    // 1. Create a DFT matrix for atomic FFTs
    for (int id = tid; id < RadixSquared; id += BlockSize) {
      const auto column = id / Radix;
      const auto row = id % Radix;
      sh_f[id] = pow_theta(row * column, Radix);
    }

    // Define CuTe tensors
    auto dft_matrix = cute::make_tensor(
        cute::make_smem_ptr(sh_f),
        cute::make_shape(cute::Int<Radix>{}, cute::Int<Radix>{}),
        cute::make_stride(cute::Int<1>{}, cute::Int<Radix>{}));

    auto out_matrix = cute::make_tensor(
        cute::make_smem_ptr(sh_d),
        cute::make_shape(cute::Int<Radix>{}, cute::Int<1>{}),
        cute::make_stride(cute::Int<1>{}, cute::Int<Radix>{}));

    auto data_first = cute::make_tensor(
        cute::make_smem_ptr(sh_d),
        cute::make_shape(cute::Int<1>{}, cute::Int<Radix>{}),
        cute::make_stride(cute::Int<Radix>{}, cute::Int<1>{}));

    auto thr_vmnk =
        typename TiledMma::ThrLayoutVMNK{}.get_flat_coord(laneIdx);
    auto thrmma = cute::ThrMMA<TiledMma, decltype(thr_vmnk)>(thr_vmnk);

    __syncthreads();

    // first iteration doesn't require twiddling
    custom_gemm(thrmma, dft_matrix, data_first, out_matrix);

    __syncthreads();

  }

};
}
