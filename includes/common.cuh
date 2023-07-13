#pragma once

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
  const int lane_id = (threadIdx.x + blockDim.x * threadIdx.y) % 32;
  const int arow = lane_id >> 2;
  const int acol = lane_id % 4; // or + 4
  const int ccol = (acol * 2);  // or + 1

#define brow acol
#define bcol arow
#define crow arow

  const int cpos = crow * 8 + ccol;
  const int bpos = (brow * 2) * 32 + bcol * 4;
};
