#pragma once
#include "device_launch_parameters.h"

namespace tester {
    template <int InnerRepeats, typename CT, typename FFTExec, int Size, 
             int Radix>
    __global__ void fft_tester(CT *data) {
      extern __shared__ __align__(sizeof(CT)) char shared[];
      CT* shared_data = reinterpret_cast<CT*>(shared);
      CT *shared_F = shared_data + Size;

      const auto tid = threadIdx.x;

      // 1. copy data
      for (int id = tid; id < Size; id += blockDim.x) {
        shared_data[id] = data[id];
      }

      // First iteration is set for 11000
      // Second iteration is set for 1000
      // The result is time difference 
      // divided by 10000

      __syncthreads();
      FFTExec fft(shared_data, shared_F);
      for (int i = 0; i < InnerRepeats; ++i) {
        fft();
      }

      __syncthreads();

      for (int id = tid; id < Size; id += blockDim.x) {
        data[id] = shared_data[id];
      }
    }
}
