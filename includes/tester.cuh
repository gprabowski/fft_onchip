#pragma once
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace tester {
    template <int InnerRepeats, typename CT, typename FFTExec, int Size, 
             int Radix>
    __global__ void fft_tester(CT *data) {
      extern __shared__ __align__(sizeof(CT)) char shared[];
      CT* shared_data = reinterpret_cast<CT*>(shared);

      auto grid = cooperative_groups::this_grid();
      auto block = cooperative_groups::this_thread_block();

      // 1. copy data
      cooperative_groups::memcpy_async(block, shared_data, data + Size * grid.block_rank(), sizeof(CT) * Size);
      cooperative_groups::wait(block);

      // First iteration is set for 11000
      // Second iteration is set for 1000
      // The result is time difference 
      // divided by 10000

      FFTExec fft(shared_data);
      for (int i = 0; i < InnerRepeats; ++i) {
        fft();
      }

      block.sync();

      const auto elems_per_t = Size / block.size();
      const auto batch_size = elems_per_t * sizeof(CT);
      memcpy(&data[Size * grid.block_rank() + block.thread_rank() * elems_per_t], &shared_data[block.thread_rank() * elems_per_t], batch_size);
    }
}
