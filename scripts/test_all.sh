#!/usr/bin/env bash

# 1. Perform bench testing for the best config
cd ../build/
if [ -f "bench_fft" ]; then
    ./bench_fft > ../scripts/results/configs.csv
fi

# 2. Generate cuobjdump
if cuobjdump --version &> /dev/null; then
    cuobjdump --dump-resource-usage gpu_fft > ../scripts/results/resource_dump.txt
fi

# 3. Generate data for increasing GPU saturation
# (REMARK: it's best to test this with manipulating
# FPU and UPB parameters of kernels)
cd ../build/
if [ -f "saturation_fft" ]; then
    ./saturation_fft > ../scripts/results/saturation.csv
fi

# 4. Generate results for the best configuration with and without data transfers
cd ../build/
if [ -f "gpu_fft" ]; then
    ./gpu_fft > ../scripts/results/results.csv
fi
