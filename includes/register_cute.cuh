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
void
register_gemm(ThrMMA<Args...> const& thr_mma,
     Tensor<TA, ALayout> tCrA,
     Tensor<TB, BLayout> sB,
     Tensor<TC, CLayout> sC, 
     bool twiddle = false)
{
  CUTE_STATIC_ASSERT_V(size<0>(sB) == size<1>(sC));  // BN == CN

  using TypeA = typename TA::value_type;
  using TypeB = typename TB::value_type;
  using TypeC = typename TC::value_type;

  // Original, static size of the problem
  auto M = size<0>(sC);
  auto N = size<1>(sC);
  auto K = size<1>(sC);

  // Block size of the compute tile
  auto BLK_M = tile_size<0>(thr_mma);
  auto BLK_N = tile_size<1>(thr_mma);
  auto BLK_K = tile_size<2>(thr_mma);

  // Partition the sA and sB tiles across the threads for the MMA
  Tensor tCsB = thr_mma.partition_B(sB);                    // (MMA,MMA_N,MMA_K)
  Tensor tCsC = thr_mma.partition_C(sC);                    // (MMA,MMA_M,MMA_N)
  // Create register tensors for the MMA to operate on
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                      // (MMA,MMA_N,MMA_K)
  Tensor tCrC = thr_mma.make_fragment_C(tCsC);                      // (MMA,MMA_M,MMA_N)

  //
  // PREFETCH k_block = 0 (with k-predication)
  //

  CUTE_UNROLL
  for (int i = 0; i < size<0>(tCsB); ++i) {                // Copy MMA_I
      CUTE_UNROLL
      for (int n = 0; n < size<1>(tCsB); ++n) {            // Copy MMA_N, predicated on n
        tCrB(i,n,0) = tCsB(i,n,0);
    }
  }

  //
  // MAINLOOP
  //

  // Clear accumulators
  clear(tCrC);

  constexpr int K_BLOCK_MAX = size<2>(tCrA);

  CUTE_UNROLL
  for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
  {
    // static-if load the next k_block. No k-predication required on these loads.
    if (k_block < K_BLOCK_MAX-1)
    {
      // Load the next k_block
      int k_next = k_block + 1;

      CUTE_UNROLL
      for (int n = 0; n < size<1>(tCsB); ++n) {            // Copy MMA_N
        CUTE_UNROLL
        for (int i = 0; i < size<0>(tCsB); ++i) {          // Copy MMA_I predicated on n
          tCrB(i,n,k_next) = tCsB(i,n,k_next);
        }
      }
    }

    // GEMM on k_block in registers
    gemm(thr_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
  }

  const auto first = (threadIdx.x / 4) + (threadIdx.x % 4) * 16;
  const auto row = first % 8;
  const auto col = first / 8;

  tCsC(0) = twiddle ? pow_theta<64>(row * col) * tCrC(0) : tCrC(0);
  tCsC(1) = twiddle ? pow_theta<64>(row *(col + 1)) * tCrC(1) : tCrC(1);
}

