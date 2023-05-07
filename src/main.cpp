#include <algorithm>
#include <complex>
#include <iostream>
#include <random>
#include <vector>

#include <algorithm.cuh>
#include <reference.cuh>

int main() {
  std::random_device rd;
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  constexpr int N = 1 << 18;
  std::vector<std::complex<double>> data(N);
  std::vector<std::complex<double>> out_reference(N);
  std::vector<std::complex<double>> out_algorithm(N);

  // generate data
  std::transform(begin(data), end(data), begin(data), [&](auto) {
    return 3.5;
    std::complex<double>(dist(rd), dist(rd));
  });

  // compare correctness
  std::cout << "Reference took: " << fft::run_reference(data, out_reference)
            << "\n";
  std::cout << "Algorithm took: " << fft::run_algorithm(data, out_algorithm)
            << "\n";
  double mse{0.0};

  for (int i = 0; i < data.size(); ++i) {
    const auto se = norm(out_reference[i] - out_algorithm[i]);
    mse += se;
  }
  mse /= data.size();

  std::cout << "MSE: " << mse << "\n";
}
