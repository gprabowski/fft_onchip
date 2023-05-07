#include <cuComplex.h>
#include <cufft.h>
#include <reference.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace fft {
size_t run_reference(const std::vector<std::complex<double>> &data,
                     std::vector<std::complex<double>> &out) {
  thrust::host_vector<cufftDoubleComplex> h_data;
  thrust::device_vector<cufftDoubleComplex> d_data;

  for (int i = 0; i < data.size(); ++i) {
    h_data.push_back({data[i].real(), data[i].imag()});
  }

  d_data = h_data;
  cufftHandle plan;
  cufftPlan1d(&plan, data.size(), CUFFT_Z2Z, 1);
  auto t1 = std::chrono::high_resolution_clock::now();
  cufftExecZ2Z(plan, thrust::raw_pointer_cast(d_data.data()),
               thrust::raw_pointer_cast(d_data.data()), CUFFT_FORWARD);
  h_data = d_data;
  auto t2 = std::chrono::high_resolution_clock::now();

  const auto res_ms =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

  for (int i = 0; i < data.size(); ++i) {
    out[i] = {h_data[i].x, h_data[i].y};
  }

  return res_ms.count();
}
} // namespace fft
