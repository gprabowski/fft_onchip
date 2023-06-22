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

  // 1. copy data
  cg::memcpy_async(block, shared_data,
                   data + Size * FFTExec::ffts_per_unit *
                              FFTExec::ffts_per_block * grid.block_rank(),
                   sizeof(CT) * Size * FFTExec::ffts_per_block *
                       FFTExec::ffts_per_unit);
  cg::wait(block);

  // First iteration is set for 11000
  // Second iteration is set for 1000
  // The result is time difference
  // divided by 10000

  FFTExec fft(shared_data);
  for (int i = 0; i < InnerRepeats; ++i) {
    fft();
  }

  block.sync();

  const auto elems_per_t =
      (Size * FFTExec::ffts_per_block * FFTExec::ffts_per_unit) / block.size();
  const auto batch_size = elems_per_t * sizeof(CT);
  memcpy(&data[Size * FFTExec::ffts_per_block * FFTExec::ffts_per_unit *
                   grid.block_rank() +
               block.thread_rank() * elems_per_t],
         &shared_data[block.thread_rank() * elems_per_t], batch_size);
}
} // namespace tester
