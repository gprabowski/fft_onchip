#pragma once

#include <chrono>
#include <vector>
#include <complex>

#include <cuda_fp16.h>

#include <thrust/complex.h>

#include <custom_cute.cuh>

namespace fft {
__device__ static constexpr double PI = 3.1415926535897931;
using namespace cute;


template <int Num, int Base> constexpr int static_log() {
  if constexpr (Num == Base)
    return 1;
  return 1 + static_log<Num / Base, Base>();
}

template <typename CT, int Size, int Radix> struct fft_executor {
  using this_t = fft_executor<CT, Size, Radix>;

  using TensorShape = Shape<_2, _2, _1>;
  using TensorTiledMma =
      cute::TiledMMA<cute::MMA_Atom<cute::SM80_8x8x4_C64C64C64C64_TN>, // Atom
                     Layout<TensorShape>>;

  using UniTiledMma =
      cute::TiledMMA<cute::MMA_Atom<cute::UniversalFMA<CT, CT, CT, CT>>, // Atom
                     Layout<Shape<_16, _16, _1>>>;

  using TiledMma = TensorTiledMma;

  static constexpr auto warps_in_group = size(TensorShape{});
  static constexpr dim3 threads = (32 * warps_in_group)* (Size / (Radix * Radix));
  static constexpr int RadixSquared = Radix * Radix;
  static constexpr int depth = static_log<Size, Radix>();

  const int tid = threadIdx.x;
  const int local_idx = tid % (warps_in_group * 32);
  const int warpIdx = tid / 32;
  const int warp_group = warpIdx / (warps_in_group);
  const int warp_group_idx = warpIdx % (warps_in_group);
  const int laneIdx = tid % 32;

  template<int N>
  inline __device__ CT pow_theta(int p) const {
    p = p % N;
    float s, c;
    const float ang = p * (-2.0 / N);
    sincospi(ang, &s, &c);
    return {c, s};
  }

  CT *sh_d, *sh_f;

  __device__ fft_executor(CT *d, CT *f) : sh_d(d), sh_f(f) {}

  __device__ void operator()() {
    // 1. Create a DFT matrix for atomic FFTs
    for (int id = tid; id < RadixSquared; id += threads.x) {
      const auto column = id / Radix;
      const auto row = id % Radix;
      sh_f[id] = column * row == 0 ? 1 : pow_theta<Radix>(row * column);
    }

    // Define CuTe tensors
    auto dft_matrix = cute::make_tensor(
        cute::make_smem_ptr(sh_f),
        cute::make_shape(cute::Int<Radix>{}, cute::Int<Radix>{}),
        cute::make_stride(cute::Int<1>{}, cute::Int<Radix>{}));

    auto data_first = cute::make_tensor(
        cute::make_smem_ptr(sh_d + warp_group * Radix),
        cute::make_shape(cute::Int<Radix>{}, cute::Int<Radix>{}),
        cute::make_stride(cute::Int<1>{}, cute::Int<Size / Radix>{}));

    auto thr_vmnk =
        typename TiledMma::ThrLayoutVMNK{}.get_flat_coord(local_idx);
    auto thrmma = cute::ThrMMA<TiledMma, decltype(thr_vmnk)>(thr_vmnk);

    __syncthreads();

    // first iteration doesn't require twiddling
    custom_gemm(thrmma, dft_matrix, data_first, data_first);

    __syncthreads();

    repeat_fft<32 * warps_in_group, depth - 1, Size / RadixSquared, Size / Radix, Radix>(thrmma);
  }

  template <int GroupSize, int Repeats, int ColDist, int RowDist, int SeqLen, typename ThrMma>
  __forceinline__ __device__ void repeat_fft(ThrMma mma) {

    auto dft_matrix = cute::make_tensor(
        cute::make_smem_ptr(sh_f),
        cute::make_shape(cute::Int<Radix>{}, cute::Int<Radix>{}),
        cute::make_stride(cute::Int<1>{}, cute::Int<Radix>{}));

    auto data = cute::make_tensor(
        cute::make_smem_ptr(sh_d + warp_group),
        cute::make_shape(cute::Int<Radix>{}, cute::Int<Radix>{}),
        cute::make_stride(cute::Int<ColDist>{}, cute::Int<RowDist>{}));

    // twiddle scale
    for (int id = local_idx; id < SeqLen * Radix; id += GroupSize) {
      const auto column = id / Radix;
      const auto row = id % Radix;
      if(row < column) continue;
      const auto tw = pow_theta<SeqLen * Radix>(row * column);
      data[id] *= tw;
      data[column + row * Radix] *= tw;
    }

    __syncwarp();

      // perform fft
    if constexpr(SeqLen == Radix) {
      auto sq_data = cute::make_tensor(
          cute::make_smem_ptr(sh_d + warp_group),
          cute::make_shape(cute::Int<Radix>{}, cute::Int<Radix>{}),
          cute::make_stride(cute::Int<ColDist>{}, cute::Int<RowDist>{}));
      custom_gemm(mma, dft_matrix, sq_data, sq_data);
    } else if constexpr(SeqLen > Radix)  {
      auto sq_data = cute::make_tensor(
          cute::make_smem_ptr(sh_d + Radix * RowDist * warp_group),
          cute::make_shape(cute::Int<Radix>{}, cute::Int<Radix>{}),
          cute::make_stride(cute::Int<ColDist>{}, cute::Int<RowDist>{}));
      custom_gemm(mma, dft_matrix, sq_data, sq_data);
    }

    __syncthreads();

    if constexpr (Repeats > 1) {
      repeat_fft<GroupSize * Radix, Repeats - 1, ColDist / Radix, ColDist, SeqLen * Radix>(mma);
    }
  }
};
}
