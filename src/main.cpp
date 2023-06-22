#include <algorithm>
#include <random>
#include <vector>

#include <config.hpp>
#include <perf_test.cuh>

int main() {
  using config::CT;
  using config::N;

  std::random_device rd;
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  std::vector<CT> data(N);

  // generate data
  std::transform(begin(data), end(data), begin(data), [&](auto) {
    return CT{10, 10};
    // return CT{dist(rd), dist(rd)};
  });

  // compare correctness
  testing::test(data);
}
