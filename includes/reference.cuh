#pragma once

#include <chrono>
#include <vector>
#include <complex>

#include <cuda_fp16.h>

#include <thrust/complex.h> 

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cufftdx.hpp>

namespace fft {
using namespace cufftdx;

template <int FftSize> struct reference_fft {
  using FFT =
      decltype(Size<FftSize>() + Precision<double>() + Type<fft_type::c2c>() +
               Direction<fft_direction::forward>() + FFTsPerBlock<1>() +
               SM<860>() + Block());
  using VT = typename FFT::value_type;
  static constexpr auto threads = FFT::block_dim;
  VT *sh_d;
  __device__ reference_fft(VT *d, VT*s) : sh_d(d) {}
  __device__ void operator()() { FFT().execute(sh_d); };
};
}
