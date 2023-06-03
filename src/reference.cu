#include <cuComplex.h>
#include <cufft.h>
#include <reference.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cufftdx.hpp>

#include <config.hpp>

namespace fft {

using namespace cufftdx;

template <typename FFT>
__global__ void onchip_reference(typename FFT::value_type *data) {
  using complex_type = typename FFT::value_type;
  extern __shared__ typename FFT::value_type shared_data[];

  const auto tid = threadIdx.x;

  // 1. copy data
  for (int id = tid; id < size_of<FFT>::value; id += blockDim.x) {
    shared_data[id] = data[id];
  }

  FFT().execute(shared_data);

  // 2. copy data back
  for (int id = tid; id < size_of<FFT>::value; id += blockDim.x) {
    data[id] = shared_data[id];
  }
}

size_t run_reference(const std::vector<__half2> &data,
                     std::vector<__half2> &out) {

  using FFT = decltype(Size<N>() + Precision<float>() + Type<fft_type::c2c>() +
                       Direction<fft_direction::forward>() + FFTsPerBlock<1>() +
                       SM<750>() + Block());

  cudaError_t error_code = cudaSuccess;

  using complex_type = typename FFT::value_type;
  thrust::host_vector<complex_type> h_data;
  thrust::device_vector<complex_type> d_data;

  for (int i = 0; i < data.size(); ++i) {
    h_data.push_back({data[i].x, data[i].y});
  }

  d_data = h_data;

  auto t1 = std::chrono::high_resolution_clock::now();
  onchip_reference<FFT>
      <<<1, FFT::block_dim,
         data.size() * sizeof(float) * 2 + FFT::shared_memory_size>>>(
          thrust::raw_pointer_cast(d_data.data()));
  h_data = d_data;
  auto t2 = std::chrono::high_resolution_clock::now();

  const auto res_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

  for (int i = 0; i < data.size(); ++i) {
    out[i] = __half2{h_data[i].x, h_data[i].y};
  }

  return res_ms.count();
}
} // namespace fft
