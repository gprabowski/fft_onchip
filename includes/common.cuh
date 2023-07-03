#pragma once
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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

template <int Num, int Base> constexpr int static_log() {
  if constexpr (Num == Base)
    return 1;
  return 1 + static_log<Num / Base, Base>();
}

#define mma_fp64_884_indexes                                                   \
  const int lane_id = (threadIdx.x + blockDim.x * threadIdx.y) % 32;           \
  const int arow = lane_id >> 2;                                               \
  const int &bcol = arow;                                                      \
  const int &crow = arow;                                                      \
  const int acol = lane_id % 4;                                                \
  const int &brow = acol;                                                      \
  const int ccol = (acol * 2);                                                 \
  const int transpose_lane_b1 = bcol * 4 + brow / 2;                           \
  const int transpose_lane_b2 = transpose_lane_b1 + 2;
