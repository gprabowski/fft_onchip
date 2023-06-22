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

struct mma_fp64_884_indexes {
  const size_t lane_id =
      cg::tiled_partition<32>(cg::this_thread_block()).thread_rank();
  const int arow = lane_id >> 2;
  const int acol = lane_id % 4; // or + 4
  const int brow = lane_id % 4; // or + 4
  const int bcol = lane_id >> 2;
  const int crow = lane_id >> 2;
  const int ccol = ((lane_id % 4) * 2); // or + 1

  const int transpose_lane_b1 = bcol * 4 + brow / 2;
  const int transpose_lane_b2 = transpose_lane_b1 + 2;
};
