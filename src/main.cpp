#include <algorithm>
#include <complex>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>

#include <cuda_fp16.h>

#include <algorithm.cuh>
#include <reference.cuh>

#include <config.hpp>

float norm(const half2 &v) { return v.x * v.x + v.y * v.y; }
__half2 operator-(const half2 &v1, const half2 &v2) {
  return {v1.x - v2.x, v1.y - v2.y};
}

int main() {
  std::random_device rd;
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  std::vector<__half2> data(N);
  std::vector<__half2> out_reference(N);
  std::vector<__half2> out_algorithm(N);

  // generate data
  std::transform(begin(data), end(data), begin(data), [&](auto) {
    return __half2{__float2half(dist(rd)), __float2half(dist(rd))};
  });

  // compare correctness
  std::size_t alg_run = 0, ref_run = 0;
  for (int i = 0; i < 1000; ++i) {
    alg_run += fft::run_algorithm(data, out_algorithm);
    ref_run += fft::run_reference(data, out_reference);
  }
  std::cout << "Algorithm took: " << alg_run / 1000.f << " microseconds \n";
  std::cout << "Reference took: " << ref_run / 1000.f << " microseconds \n";

  float mse{0.0};

  for (int i = 0; i < data.size(); ++i) {
    const auto se = norm(out_reference[i] - out_algorithm[i]);
    if constexpr (false) {
      std::cout << "R: " << out_reference[i].x << " " << out_reference[i].y
                << " I: " << out_algorithm[i].x << " " << out_algorithm[i].y
                << std::endl;
    }
    mse += se;
  }
  mse /= data.size();

  std::cout << "MSE: " << mse << "\n";
}
