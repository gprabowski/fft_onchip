#pragma once

#include <cufftdx.hpp>
#include <operators/block_operators.hpp>

namespace fft {
using namespace cufftdx;

template <int FftSize> struct reference_fft {
  using iFFT =
      decltype(Size<FftSize>() + Precision<double>() + Type<fft_type::c2c>() +
               Direction<fft_direction::forward>() + SM<860>() + Block());
  using FFT = decltype(iFFT() + FFTsPerBlock<iFFT::suggested_ffts_per_block>());

  using VT = typename FFT::value_type;
  static constexpr auto threads = FFT::block_dim.x;
  static constexpr auto ffts_per_block = FFT::ffts_per_block;
  static constexpr auto ffts_per_unit = 1;

  VT *sh_d;
  __device__ reference_fft(VT *d) : sh_d(d) {}
  __device__ void operator()() { FFT().execute(sh_d); };
};
} // namespace fft
