from nvidia/cuda:12.2.0-devel-ubuntu20.04

RUN apt-get update && apt-get -y install wget git ninja-build python3
RUN mkdir /home/pg && mkdir /home/pg/soft

# Download CMAKE and put in /home/pg/soft/cmake
RUN cd /home/pg/soft && wget https://github.com/Kitware/CMake/releases/download/v3.27.0-rc4/cmake-3.27.0-rc4-linux-x86_64.sh \
&& chmod +x cmake-3.27.0-rc4-linux-x86_64.sh && mkdir cmake && ./cmake-3.27.0-rc4-linux-x86_64.sh --skip-license --prefix=cmake 

# Setup mathdx in /home/pg/soft/mathdx
RUN cd /home/pg/soft && wget https://developer.download.nvidia.com/compute/mathdx/redist/mathdx/linux-x86_64/nvidia-mathdx-22.11.0-Linux.tar.gz \
&& tar -xvf nvidia-mathdx-22.11.0-Linux.tar.gz && rm *.tar.gz && mv nvidia-mathdx-22.11.0-Linux/nvidia/mathdx/ mathdx \
&& rm -rf nvidia-mathdx-22.11.0-Linux

# Clone repo and put in /home/pg/fft_onchip
RUN cd /home/pg && git clone https://github.com/gprabowski/fft_onchip.git
