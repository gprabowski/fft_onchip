# fft_onchip

to compile 

```
cd fft_onchip
mkdir build
cd build
cmake ../ -DCMAKE_CUDA_ARCHITECTURES=86 -Dmathdx_ROOT=/home/path/to/mathdx/nvidia/mathdx/22.11 -GNinja
ninja
```

currently inside one can find 5 kernels: 
1. using cuFFTDx
2. custom using tensor cores
3. custom using fma
4. custom using radix-8 DIF kernel
5. custom using radix-16 DIF kernel
