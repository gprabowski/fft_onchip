#include <perf_test.cuh>

#include <iostream>

#include <algorithm.cuh>
#include <cuda_fp16.h>
#include <reference.cuh>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <legacy16_fft.cuh>
#include <legacy_fft.cuh>
#include <simple_fft.cuh>
#include <tester.cuh>

namespace testing {

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

template <typename CT, int Size, int Radix, typename FFTExec>
size_t run_perf_test(const std::vector<config::CT> &data,
                     std::vector<config::CT> &out) {
  thrust::host_vector<CT> h_data;
  thrust::device_vector<CT> d_data;

  for (int i = 0; i < data.size(); ++i) {
    h_data.push_back({data[i].real(), data[i].imag()});
  }

  d_data = h_data;

  constexpr auto sm_size = (Size + Radix * Radix + 32) * sizeof(CT);

  gpuErrchk(cudaFuncSetAttribute(
      tester::fft_tester<1, CT, FFTExec, Size, Radix>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, sm_size));

  gpuErrchk(cudaFuncSetAttribute(
      tester::fft_tester<1000, CT, FFTExec, Size, Radix>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, sm_size));

  gpuErrchk(cudaFuncSetAttribute(
      tester::fft_tester<11000, CT, FFTExec, Size, Radix>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, sm_size));

  // correctness check
  tester::fft_tester<1, CT, FFTExec, Size, Radix>
      <<<1, FFTExec::threads, sm_size>>>(
          thrust::raw_pointer_cast(d_data.data()));
  std::cout << FFTExec::threads << std::endl;
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  h_data = d_data;

  for (int i = 0; i < data.size(); ++i) {
    out[i] = config::CT{h_data[i].real(), h_data[i].imag()};
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  tester::fft_tester<1000, CT, FFTExec, Size, Radix>
      <<<1, FFTExec::threads, sm_size>>>(
          thrust::raw_pointer_cast(d_data.data()));
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  auto t2 = std::chrono::high_resolution_clock::now();

  const auto run100_t =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

  t1 = std::chrono::high_resolution_clock::now();
  tester::fft_tester<11000, CT, FFTExec, Size, Radix>
      <<<1, FFTExec::threads, sm_size>>>(
          thrust::raw_pointer_cast(d_data.data()));
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  h_data = d_data;
  t2 = std::chrono::high_resolution_clock::now();

  const auto run1100_t =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

  const size_t final_time =
      static_cast<double>(run1100_t.count() - run100_t.count()) / 10000.0;

  return final_time;
}

__host__ __device__ int reverseDigits(int number, int base) {
  int reversedNumber = 0;

  while (number > 0) {
    reversedNumber = reversedNumber * base + number % base;
    number /= base;
  }

  return reversedNumber;
}

void test(std::vector<config::CT> &data) {
  std::size_t alg_run = 0, ref_run = 0;
  std::vector<config::CT> out_algorithm(data.size()),
      out_reference(data.size());

  using customExec = fft::simple_fft<config::CT, config::N, config::radix>;
  using refExec = fft::reference_fft<config::N>;

  auto alg_data = data;
  auto ref_data = data;
  alg_run = run_perf_test<config::CT, config::N, config::radix, customExec>(
      alg_data, out_algorithm);
  ref_run = run_perf_test<refExec::VT, config::N, config::radix, refExec>(
      ref_data, out_reference);

  std::cout << "Algorithm took: " << alg_run << " microseconds \n";
  std::cout << "Reference took: " << ref_run << " microseconds \n";

  double mse{0.0};

  for (int i = 0; i < data.size(); ++i) {
    const auto se =
        norm(out_reference[i] - out_algorithm[reverseDigits(i, config::radix)]);
    if constexpr (true) {
      std::cout << "R: " << out_reference[i].real() << " "
                << out_reference[i].imag() << " A: "
                << out_algorithm[reverseDigits(i, config::radix)].real() << " "
                << out_algorithm[reverseDigits(i, config::radix)].imag()
                << std::endl;
    }
    mse += se;
  }
  mse /= data.size();

  std::cout << "MSE: " << mse << "\n";
}
} // namespace testing
