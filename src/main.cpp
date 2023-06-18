#include <algorithm>
#include <complex>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>

#include <config.hpp>

#include <perf_test.cuh>
#include <thrust/complex.h>

using CT = thrust::complex<double>;
thrust::complex<double> operator-(const CT &v1, const CT &v2) {
  return {v1.real() - v2.real(), v1.imag() - v2.imag()};
}

int main() {
  using config::CT;
  using config::N;

  std::random_device rd;
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  std::vector<thrust::complex<double>> data(N);
  std::vector<thrust::complex<double>> out_reference(N);
  std::vector<thrust::complex<double>> out_algorithm(N);

  // generate data
  int i = 0;
  std::transform(begin(data), end(data), begin(data), [&](auto) {
    return thrust::complex<double>{dist(rd), dist(rd)};
    // return thrust::complex<double>{i++, 0.0};
  });

  // compare correctness
  testing::test(data);
}
