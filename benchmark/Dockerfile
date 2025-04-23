FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    libssl-dev \              
    libgl1 \             
    libgl1-mesa-glx \    
    libgtk2.0-dev \     
    libboost-all-dev \
    clang-15 \   
    libomp-15-dev \
    ninja-build \
    python3-dev \        
    python3-pip \        
    git \                
    wget \               
    unzip \
    && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /deps

ENV OPENSSL_ROOT_DIR=/usr/include/openssl
RUN wget https://cmake.org/files/v3.29/cmake-3.29.2.tar.gz
RUN tar -xzvf cmake-3.29.2.tar.gz && cd cmake-3.29.2
RUN cd cmake-3.29.2 \
    && ./bootstrap \
    && make -j$(nproc) \
    && make install

WORKDIR /deps/eigen
RUN git clone https://gitlab.com/libeigen/eigen.git .
RUN git checkout 3.4.0
RUN cmake -Bbuild -GNinja -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
RUN cmake --install build

WORKDIR /sota
COPY ACMH/ /sota/ACMH
COPY eth3d/ /sota/eth3d

RUN python3 -m pip install -r /sota/eth3d/requirements.txt
