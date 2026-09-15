# The batch image: the ten campaign forks, the shared converter and the ETH3D
# evaluator, built for one CUDA architecture.
#
# `uv run bench image-manifest` MUST be run in benchmark/ before `docker build`.
# It writes image-manifest.json (fork SHAs, converter hashes, this file's hash)
# from the working tree; the build copies it to /sota/image-manifest.json and
# `bench run` refuses to start when the image's copy differs from a freshly
# computed one, because a run record's fork SHAs and converter hashes are read
# from the working tree and would otherwise name code the image never contained
# (D24.1). A missing image-manifest.json fails this build at the final COPY.
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

FROM base AS deps

WORKDIR /deps

WORKDIR /deps/eigen
RUN git clone https://gitlab.com/libeigen/eigen.git .
RUN git checkout 3.4.0
RUN cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/deps/install \
  -DCMAKE_C_COMPILER=clang-18 \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DBUILD_TESTING=OFF
RUN cmake --install build

WORKDIR /deps/opencv_contrib
RUN git clone https://github.com/opencv/opencv_contrib.git . \
 && git checkout 4.12.0
WORKDIR /deps/opencv
RUN git clone https://github.com/opencv/opencv.git . \
 && git checkout 4.12.0
RUN cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_INSTALL_PREFIX=/deps/install \
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

FROM base AS eval

# CUDA target for every method build. The campaign fleet is Blackwell (120);
# override on a development machine, e.g. --build-arg CUDA_ARCH=75.
ARG CUDA_ARCH=120

# Phase-timer instrumentation (bench_timer.h, vendored into all ten forks).
# 1 keeps the timer compiled in; it stays silent unless MVS_BENCH_PHASES is set
# at run time. --build-arg BENCH_PHASES=0 expands every BENCH_* macro to
# nothing, which is what the instrumented-vs-compiled-out byte-identity check
# (SPEC R-TIM-09) compares against. It has to reach the host and the device
# compiler separately: the forks include bench_timer.h from both .cpp and .cu
# translation units, and CMAKE_CXX_FLAGS does not feed nvcc.
ARG BENCH_PHASES=1

WORKDIR /

COPY --link --from=deps /deps/install /deps/install
ENV Eigen3_DIR=/deps/install/share/eigen3/cmake/
ENV OpenCV_DIR=/deps/install/lib/cmake/opencv4/

COPY methods/cuda-multi-view-stereo/ /sota/cuda-multi-view-stereo/
RUN cd /sota/cuda-multi-view-stereo \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
      -DCMAKE_CXX_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
      -DCMAKE_CUDA_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
    && cmake --build build


COPY methods/ACMH/ /sota/ACMH
RUN cd /sota/ACMH \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
      -DCMAKE_CXX_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
      -DCMAKE_CUDA_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
    && cmake --build build

COPY methods/ACMM/ /sota/ACMM/
RUN cd /sota/ACMM \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
      -DCMAKE_CXX_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
      -DCMAKE_CUDA_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
    && cmake --build build

COPY methods/ACMP/ /sota/ACMP/
RUN cd /sota/ACMP \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
      -DCMAKE_CXX_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
      -DCMAKE_CUDA_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
    && cmake --build build

COPY methods/ACMMP/ /sota/ACMMP/
RUN cd /sota/ACMMP \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
      -DCMAKE_CXX_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
      -DCMAKE_CUDA_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
    && cmake --build build

COPY methods/APD-MVS/ /sota/APD-MVS/
RUN cd /sota/APD-MVS \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
      -DCMAKE_CXX_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
      -DCMAKE_CUDA_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
    && cmake --build build

COPY methods/HPM-MVS/ /sota/HPM-MVS/
RUN cd /sota/HPM-MVS/HPM-MVS \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
      -DCMAKE_CXX_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
      -DCMAKE_CUDA_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
    && cmake --build build

COPY methods/HPM-MVS_plusplus/ /sota/HPM-MVS_plusplus/
RUN cd /sota/HPM-MVS_plusplus \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
      -DCMAKE_CXX_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
      -DCMAKE_CUDA_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
    && cmake --build build

COPY methods/MP-MVS/ /sota/MP-MVS/
RUN cd /sota/MP-MVS \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
      -DCMAKE_CXX_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
      -DCMAKE_CUDA_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
    && cmake --build build

COPY methods/DPE-MVS/ /sota/DPE-MVS/
RUN cd /sota/DPE-MVS/DPE-MVS \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
      -DCMAKE_CXX_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
      -DCMAKE_CUDA_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
    && cmake --build build

# Quorum MVS, the thesis author's own method (`provenance: own`). Not a fork, so
# it has no upstream stanza above it; it builds exactly like the forks do, and
# deliberately with the same BENCH_PHASES switch. Its src/timing/bench_timer.h is
# md5-identical to ACMH's and ACMMP's copies, so omitting the flag here would
# leave the own method carrying the phase timer while the ten forks honour the
# switch -- a timing asymmetry in its own favour, in exactly the comparison this
# thesis is built on.
#
# The submodule is PRIVATE. `docker build` reads the checked-out working tree, so
# this layer only builds for someone who could clone it; for anyone else the COPY
# fails loudly rather than silently producing an image with thirteen methods.
COPY methods/quorum-mvs/ /sota/quorum-mvs/
RUN cd /sota/quorum-mvs \
    && cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=clang++-18 \
      -DCMAKE_CXX_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
      -DCMAKE_CUDA_FLAGS=-DBENCH_PHASES=${BENCH_PHASES} \
    && cmake --build build

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ETH3D evaluator, the gaborpelesz/multi-view-evaluation fork of upstream
# ETH3D/multi-view-evaluation, pinned by the submodule. The fork's own
# CMakeLists carries what multi-view-evaluation-compat.patch used to add, so the
# patch is no longer applied; it is kept in the tree because the pre-fork
# submodule SHAs still need it.
#
# MVE_PCL_IO_FALLBACK=ON. fast_ply is still tried first and still handles every
# binary little-endian float32 cloud directly, which is nine of the ten methods
# that emit one; PCL's general reader is reached only for what it declines. The
# tenth is CUMVS, which writes ASCII PLY through VTK's writer (F-074) and was
# unscoreable while this was OFF -- that is how its four outdoor runs failed.
#
# The earlier OFF setting conflated two independent things, and this comment
# used to repeat the conflation: the ~60 ms of VTK loading that fec59c4 measured
# away, and a cmake failure that made ON impossible to select here at all. The
# second is fixed in the fork (c703dfb); the first is real but is startup cost
# on a program whose evaluation takes ~15 s, it is not part of any method's
# measured time, and it buys back a method.
#
# Equivalence is measured, not assumed: the ON build reproduces all fifteen
# printed cells of the upstream evaluator's recorded CUMVS scores, and matches
# the OFF build cell for cell on a binary cloud.
# MVE_LTO=OFF is explicit because this image cannot do ThinLTO: it carries no
# llvm-ar/llvm-ranlib, so CMake's IPO probe fails and silently skips it. Left at
# its ON default the build requests an optimisation it does not get, which is
# the kind of gap that turns into a wrong claim later. MVE_NATIVE_ARCH keeps its
# ON default -- that is the binary every equivalence check here was run against
# -- but it means this layer is valid only on the ISA it was compiled on, the
# fork's CMakeLists documenting that this program is not invariant to FMA
# contraction. The campaign's hardware fingerprint already pins that.
# Not for MPI itself, and deliberately not in the base stage: Ubuntu's VTK 9.1
# names MPI::MPI_C in VTK::mpi's link interface and ships nothing that defines
# it, so PCL's `io` component will not configure without MPI's cmake package
# present. Only the evaluator below needs it, and putting it in `base` would
# invalidate the Eigen, OpenCV and fourteen method layers on a change that
# concerns none of them.
RUN apt update && apt install -y --no-install-recommends mpi-default-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /multi-view-evaluation
ADD eth3d/multi-view-evaluation /multi-view-evaluation
RUN cmake -Bbuild -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang-18 \
      -DCMAKE_CXX_COMPILER=clang++-18 \
      -DMVE_PCL_IO_FALLBACK=ON \
      -DMVE_LTO=OFF \
    && cmake --build build \
    && cp build/ETH3DMultiViewEvaluation /usr/local/bin/ETH3DMultiViewEvaluation

WORKDIR /sota

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=eth3d,target=eth3d \
    uv sync --locked --no-install-project

ADD ./ /sota

WORKDIR /sota
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

ENV SOTA_MVS_METHOD_ROOT /sota

# Last, so that regenerating the manifest invalidates nothing above it. The
# build fails here if `bench image-manifest` was not run first.
COPY image-manifest.json /sota/image-manifest.json

ENTRYPOINT [ "bash" ]