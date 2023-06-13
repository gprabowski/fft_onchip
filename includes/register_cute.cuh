#pragma once

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>

#include <config.hpp>

using namespace cute;

template<int N>
inline __device__ config::CT pow_theta(int p) {
    p = p % N;
    double s, c;
    const double ang = p * (-2.0 / N);
    sincospi(ang, &s, &c);
    return {c, s};
}

template <class... Args,
          class TA, class ALayout, class TB, class BLayout,
          class TC, class CLayout,
          __CUTE_REQUIRES(ALayout::rank == 3 && is_rmem<TA>::value &&
                          BLayout::rank == 2 && is_smem<TB>::value &&
                          CLayout::rank == 2 && is_smem<TC>::value)>
__device__
__forceinline__
void
register_gemm(ThrMMA<Args...> const& thr_mma,
     Tensor<TA, ALayout> tCrA,
     Tensor<TB, BLayout> sB,
     Tensor<TC, CLayout> sC, 
     bool twiddle = false)
{
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCsC = thr_mma.partition_C(sC);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor tCrC = thr_mma.make_fragment_C(tCsC);

  cute::copy(tCsB, tCrB);

  clear(tCrC);

  constexpr int K_BLOCK_MAX = size<2>(tCrA);

  CUTE_UNROLL
  for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
  {
    gemm(thr_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
  }

  const auto first = (threadIdx.x / 4) + (threadIdx.x % 4) * 16;
  const auto row = first % 8;
  const auto col = first / 8;

  tCsC(0) = twiddle ? pow_theta<64>(row * col) * tCrC(0) : tCrC(0);
  tCsC(1) = twiddle ? pow_theta<64>(row *(col + 1)) * tCrC(1) : tCrC(1);
}
