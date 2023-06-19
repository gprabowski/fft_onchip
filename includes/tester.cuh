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

      const auto tid = threadIdx.x;

      // 1. copy data
      cooperative_groups::memcpy_async(block, shared_data, data, sizeof(CT) * Size);
      cooperative_groups::wait(block);

      // First iteration is set for 11000
      // Second iteration is set for 1000
      // The result is time difference 
      // divided by 10000

      FFTExec fft(shared_data);
      for (int i = 0; i < InnerRepeats; ++i) {
        fft();
      }

      __syncthreads();

      for (int id = tid; id < Size; id += blockDim.x) {
        data[id] = shared_data[id];
      }
    }
}
