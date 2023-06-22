#pragma once
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace tester {

namespace cg = cooperative_groups;

template <int InnerRepeats, typename CT, typename FFTExec, int Size,
          int FFTsPerBlock>
__global__ void fft_tester(CT *data) {
  extern __shared__ __align__(sizeof(CT)) char shared[];
  CT *shared_data = reinterpret_cast<CT *>(shared);

  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  constexpr int elements_per_block =
      Size * FFTExec::ffts_per_block * FFTExec::ffts_per_unit;

  // 1. copy data
  if constexpr (false) {
    cg::memcpy_async(block, shared_data,
                     data + elements_per_block * grid.block_rank(),
                     sizeof(CT) * elements_per_block);
    cg::wait(block);
  }

  block.sync();

  // First iteration is set for 11000
  // Second iteration is set for 1000
  // The result is time difference
  // divided by 10000

  FFTExec fft(shared_data);
  for (int i = 0; i < InnerRepeats; ++i) {
    fft();
  }

  block.sync();

  const auto elems_per_thread =
      (Size * FFTExec::ffts_per_block * FFTExec::ffts_per_unit) / block.size();
  const auto batch_size = elems_per_thread * sizeof(CT);
  if (false) {
    memcpy(&data[elements_per_block * grid.block_rank() +
                 block.thread_rank() * elems_per_thread],
           &shared_data[block.thread_rank() * elems_per_thread], batch_size);
  }
}
} // namespace tester
