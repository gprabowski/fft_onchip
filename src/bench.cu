#include <bench.cuh>

int main() {
  std::cout << "FFT Type, Size, FPU, UPB, Time [microseconds]" << std::endl;

  bench::test();
  return 0;
}
