# fft_onchip

to compile 

```
cd fft_onchip
mkdir build
cd build
cmake ../ -DCMAKE_CUDA_ARCHITECTURES=86 -Dmathdx_ROOT=/home/path/to/mathdx/nvidia/mathdx/22.11 -GNinja
ninja
./gpu_fft
```

currently inside one can find 4 kernels: 
1. using cuFFTDx `include/reference.cuh`
2. custom using tensor cores `include/tensor_fft.cuh` for 64point fft
3. custom using radix-8 DIF kernel `include/legacy8_fft.cuh`
4. custom using radix-16 DIF kernel `include/legacy16_fft.cuh`
