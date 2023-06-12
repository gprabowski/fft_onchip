#pragma once

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>

#include <config.hpp>

using namespace cute;


template <class... Args,
          class TA, class ALayout, class TB, class BLayout,
          class TC, class CLayout,
          __CUTE_REQUIRES(ALayout::rank == 2 && is_smem<TA>::value &&
                          BLayout::rank == 2 && is_smem<TB>::value &&
                          CLayout::rank == 2 && is_smem<TC>::value)>
__device__
void
custom_gemm(ThrMMA<Args...> const& thr_mma,
     Tensor<TA, ALayout> sA,
     Tensor<TB, BLayout> sB,
     Tensor<TC, CLayout> sC)
{
  CUTE_STATIC_ASSERT_V(size<0>(sA) == size<0>(sC));  // AM == CM
  CUTE_STATIC_ASSERT_V(size<0>(sB) == size<1>(sC));  // BN == CN
  CUTE_STATIC_ASSERT_V(size<1>(sA) == size<1>(sB));  // AK == BK

  using TypeA = typename TA::value_type;
  using TypeB = typename TB::value_type;
  using TypeC = typename TC::value_type;

  // Original, static size of the problem
  auto M = size<0>(sC);
  auto N = size<1>(sC);
  auto K = size<1>(sA);

  // Block size of the compute tile
  auto BLK_M = tile_size<0>(thr_mma);
  auto BLK_N = tile_size<1>(thr_mma);
  auto BLK_K = tile_size<2>(thr_mma);

  // Compute the "residues"
  auto m_residue = M - BLK_M * (ceil_div(M, BLK_M) - Int<1>{});  //  (0,BLK_M]
  auto n_residue = N - BLK_N * (ceil_div(N, BLK_N) - Int<1>{});  //  (0,BLK_N]
  auto k_residue = K - BLK_K * (ceil_div(K, BLK_K)           );  // (-BLK_K,0]

  // Shift the origin so k_residue is zeroth tile
  sA.data() = &sA(0,k_residue);
  sB.data() = &sB(0,k_residue);

  //
  // MMA Partitioning
  //

  // Round the layout extents up to BLK_X
  Tensor rounded_sA = sA.compose(make_shape(ceil_div(M, BLK_M) * BLK_M, ceil_div(K, BLK_K) * BLK_K));
  Tensor rounded_sB = sB.compose(make_shape(ceil_div(N, BLK_N) * BLK_N, ceil_div(K, BLK_K) * BLK_K));
  Tensor rounded_sC = sC.compose(make_shape(ceil_div(M, BLK_M) * BLK_M, ceil_div(N, BLK_N) * BLK_N));

  // Partition the sA and sB tiles across the threads for the MMA
  Tensor tCsA = thr_mma.partition_A(rounded_sA);                    // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(rounded_sB);                    // (MMA,MMA_N,MMA_K)
  Tensor tCsC = thr_mma.partition_C(rounded_sC);                    // (MMA,MMA_M,MMA_N)
  // Create register tensors for the MMA to operate on
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                      // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                      // (MMA,MMA_N,MMA_K)
  Tensor tCrC = thr_mma.make_fragment_C(tCsC);                      // (MMA,MMA_M,MMA_N)

  //
  // PREDICATION
  //

  // Allocate the preds for only the MMA-mode of tCsA and tCsB
  Tensor tCpA = make_tensor<bool>(size<0>(tCsA));
  Tensor tCpB = make_tensor<bool>(size<0>(tCsB));

  // Create coordinate tensors on a single compute block for predication
  Tensor cA = make_identity_tensor(make_shape(BLK_M, BLK_K));        // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor cB = make_identity_tensor(make_shape(BLK_N, BLK_K));        // (BLK_M,BLK_K) -> (blk_n,blk_k)

  // Repeat partitioning with thr_mma
  Tensor tCcA = thr_mma.partition_A(cA);                             // (MMA,1,1) -> (blk_m,blk_k)
  Tensor tCcB = thr_mma.partition_B(cB);                             // (MMA,1,1) -> (blk_n,blk_k)

  // Populate the m and n predicates
  CUTE_UNROLL
  for (int i = 0; i < size(tCpA); ++i) {
    tCpA(i) = elem_less(get<0>(tCcA(i)), m_residue);
  }
  CUTE_UNROLL
  for (int i = 0; i < size(tCpB); ++i) {
    tCpB(i) = elem_less(get<0>(tCcB(i)), n_residue);
  }

  //
  // PREFETCH k_block = 0 (with k-predication)
  //

  CUTE_UNROLL
  for (int i = 0; i < size<0>(tCsA); ++i) {                // Copy MMA_I
    if (k_residue == 0 || get<1>(tCcA(i)) >= -k_residue) { // k_block = 0, predicated on k
      CUTE_UNROLL
      for (int m = 0; m < size<1>(tCsA); ++m) {            // Copy MMA_M, predicated on m
        tCrA(i,m,0) = ((m_residue == BLK_M || m < size<1>(tCsA)-1 || tCpA(i)) ? (tCsA(i,m,0)) : TypeA{});
      }
    }
  }

  CUTE_UNROLL
  for (int i = 0; i < size<0>(tCsB); ++i) {                // Copy MMA_I
    if (k_residue == 0 || get<1>(tCcB(i)) >= -k_residue) { // k_block = 0, predicated on k
      CUTE_UNROLL
      for (int n = 0; n < size<1>(tCsB); ++n) {            // Copy MMA_N, predicated on n
        tCrB(i,n,0) = ((n_residue == BLK_N || n < size<1>(tCsB)-1 || tCpB(i)) ? (tCsB(i,n,0)) : TypeB{});
      }
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
      for (int m = 0; m < size<1>(tCsA); ++m) {            // Copy MMA_M
        CUTE_UNROLL
        for (int i = 0; i < size<0>(tCsA); ++i) {          // Copy_if MMA_I predicated on m
          tCrA(i,m,k_next) = (m_residue == BLK_M || m < size<1>(tCsA)-1 || tCpA(i)) ? tCsA(i,m,k_next) : TypeA{};
        }
      }

      CUTE_UNROLL
      for (int n = 0; n < size<1>(tCsB); ++n) {            // Copy MMA_N
        CUTE_UNROLL
        for (int i = 0; i < size<0>(tCsB); ++i) {          // Copy MMA_I predicated on n
          tCrB(i,n,k_next) = (n_residue == BLK_N || n < size<1>(tCsB)-1 || tCpB(i)) ? tCsB(i,n,k_next) : TypeB{};
        }
      }
    }

    // GEMM on k_block in registers
    gemm(thr_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
  }

  //
  // Epilogue
  //

  Tensor cC   = make_identity_tensor(make_shape(BLK_M, BLK_N));      // (BLK_M,BLK_N) -> (blk_m,blk_n)
  Tensor tCcC = thr_mma.partition_C(cC);                             // (MMA, 1, 1)   -> (blk_m,blk_n)

  CUTE_UNROLL
  for (int m = 0; m < size<1>(tCsC); ++m)
  {
    CUTE_UNROLL
    for (int n = 0; n < size<2>(tCsC); ++n)
    {
      CUTE_UNROLL
      for (int i = 0; i < size<0>(tCsC); ++i)
      {
        if ((m_residue == BLK_M || m < size<1>(tCrC)-1 || get<0>(tCcC(i)) < m_residue) &&
            (n_residue == BLK_N || n < size<2>(tCrC)-1 || get<1>(tCcC(i)) < n_residue))
        {
          tCsC(i,m,n) = tCrC(i,m,n);
        }
      }
    }
  }
}

