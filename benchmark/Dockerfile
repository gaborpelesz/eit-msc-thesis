FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libqt5opengl5-dev \
    libvtk9-dev \
    cmake \
    clang-18 \
    libomp-18-dev \
    # PCL is required for HPM-MVS only
    libpcl-dev \
    ninja-build \
    python3-dev \
    python3-venv \
    python3-pip \
    git \
    wget \
    unzip \
    && apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /deps

WORKDIR /deps/eigen
RUN git clone https://gitlab.com/libeigen/eigen.git .
RUN git checkout 3.4.0
RUN cmake -Bbuild -GNinja -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
RUN cmake --install build

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv venv

COPY eth3d/ /sota/eth3d
RUN uv pip install -r /sota/eth3d/requirements.txt
# TODO: opencv needs VTK

WORKDIR /deps/opencv_contrib
RUN git clone https://github.com/opencv/opencv_contrib.git . \
 && git checkout 4.12.0
WORKDIR /deps/opencv
RUN git clone https://github.com/opencv/opencv.git . \
 && git checkout 4.12.0
RUN cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_C_COMPILER=clang-18 \
   -DCMAKE_CXX_COMPILER=clang++-18 \
   -DOPENCV_EXTRA_MODULES_PATH=/deps/opencv_contrib/modules \
   -DBUILD_LIST=core,imgcodecs,imgproc,calib3d,cudev,highgui,viz \
   -DCMAKE_CUDA_ARCHITECTURES=75 \
   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
   -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
   -DBUILD_SHARED_LIBS=OFF \
   -DBUILD_TESTS=OFF \
   -DWITH_CUDA=ON \
   -DWITH_VTK=ON \
   -DBUILD_PERF_TESTS=OFF \
   -DBUILD_EXAMPLES=OFF \
   -DBUILD_DOCS=OFF \
   -DBUILD_opencv_apps=OFF \
   -DBUILD_PROTOBUF=OFF \
   -DWITH_ADE=OFF \
 && cmake --build build \
 && cmake --install build

WORKDIR /sota

COPY cuda-multi-view-stereo/ /sota/cuda-multi-view-stereo/
RUN cd /sota/cuda-multi-view-stereo \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
    && cmake --build build


COPY ACMH/ /sota/ACMH
RUN cd /sota/ACMH \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
    && cmake --build build

COPY ACMM/ /sota/ACMM/
RUN cd /sota/ACMM \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
    && cmake --build build

COPY ACMP/ /sota/ACMP/
RUN cd /sota/ACMP \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
    && cmake --build build

COPY ACMMP/ /sota/ACMMP/
RUN cd /sota/ACMMP \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
    && cmake --build build

COPY APD-MVS/ /sota/APD-MVS/
RUN cd /sota/APD-MVS \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
    && cmake --build build

COPY HPM-MVS/ /sota/HPM-MVS/
RUN cd /sota/HPM-MVS/HPM-MVS \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
    && cmake --build build

COPY HPM-MVS_plusplus/ /sota/HPM-MVS_plusplus/
RUN cd /sota/HPM-MVS_plusplus \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
    && cmake --build build

COPY MP-MVS/ /sota/MP-MVS/
RUN cd /sota/MP-MVS \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
    && cmake --build build

COPY cuda-multi-view-stereo/ /sota/cuda-multi-view-stereo/
RUN cd /sota/cuda-multi-view-stereo \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
    && cmake --build build
