FROM nvidia/cuda:13.3.1-cudnn-devel-ubuntu26.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git wget unzip ca-certificates gnupg \
    libyaml-cpp-dev libzstd-dev liblmdb-dev \
    python3.14 python3.14-venv python3.14-dev python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Install TensorRT from NVIDIA's network apt repo (pulls TRT 11.x for CUDA 13.3)
WORKDIR /tmp_deps
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i cuda-keyring_1.1-1_all.deb \
 && apt-get update \
 && apt-get install -y --no-install-recommends tensorrt tensorrt-dev \
 && rm -rf /var/lib/apt/lists/* /tmp_deps

WORKDIR /opt/talbot_deps

# Fetch LibTorch, Fathom, and chess-library
RUN wget -q https://download.pytorch.org/libtorch/cu132/libtorch-shared-with-deps-2.14.0%2Bcu132.zip \
 && unzip -q libtorch-*.zip \
 && rm libtorch-*.zip \
 && git clone --depth 1 https://github.com/jdart1/Fathom.git \
&& git clone --depth 1 https://github.com/Disservin/chess-library.git \
 && git clone --depth 1 https://github.com/cameron314/concurrentqueue.git

# Build config
ENV TALBOT_DEP_ROOT=/opt/talbot_deps \
    TALBOT_CUDA_PATH=/usr/local/cuda-13.3 \
    TALBOT_TRT_DIR=/usr

ENV LD_LIBRARY_PATH="${TALBOT_DEP_ROOT}/libtorch/lib:${TALBOT_CUDA_PATH}/lib64:${TALBOT_TRT_DIR}/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

WORKDIR /app
COPY requirements.txt .
RUN python3.14 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j$(nproc)