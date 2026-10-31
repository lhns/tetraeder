# Tetraeder GPU box image (ghcr.io/lhns/tetraeder-box), used on rented Vast.ai boxes.
# Contains the toolchain plus llama.cpp built with CUDA at the repo pin (vendor/PIN) for baselines.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG LLAMA_TAG
ARG LLAMA_COMMIT
ARG RUST_VERSION=1.95.0
ARG SCCACHE_VERSION=0.10.0
# Ampere (8.6), Ada (8.9), Hopper (9.0), Blackwell (12.0)
ARG CUDA_ARCHS="86;89;90;120"

# GGML_NATIVE=OFF (llama.cpp below and box-side Rust builds): the image is built on a CI runner and runs on any
# Vast host, so ggml CPU code targets its portable AVX2 baseline instead of the build machine's ISA.
ENV DEBIAN_FRONTEND=noninteractive \
    TETRAEDER_GGML_NATIVE=OFF \
    NVIDIA_DRIVER_CAPABILITIES=all \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    SCCACHE_DIR=/opt/sccache \
    SCCACHE_CACHE_SIZE=10G \
    RUSTC_WRAPPER=sccache \
    CMAKE_C_COMPILER_LAUNCHER=sccache \
    CMAKE_CXX_COMPILER_LAUNCHER=sccache \
    CMAKE_CUDA_COMPILER_LAUNCHER=sccache \
    PATH=/opt/cargo/bin:/usr/local/cuda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential pkg-config cmake ninja-build git curl ca-certificates \
        fio rsync jq unzip python3 libclang-dev \
        libvulkan1 libvulkan-dev mesa-vulkan-drivers vulkan-tools glslc \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://github.com/mozilla/sccache/releases/download/v${SCCACHE_VERSION}/sccache-v${SCCACHE_VERSION}-$(uname -m)-unknown-linux-musl.tar.gz" \
      | tar -xz --strip-components=1 -C /usr/local/bin "sccache-v${SCCACHE_VERSION}-$(uname -m)-unknown-linux-musl/sccache"

# Same version as rust-toolchain.toml (passed in by the workflow).
RUN curl -fsSL https://sh.rustup.rs | RUSTC_WRAPPER= sh -s -- -y --profile minimal \
        --default-toolchain "${RUST_VERSION}" -c rustfmt -c clippy

# llama.cpp at the pin, fetched by commit (the build context does not carry the submodule).
# Static libs, tools (llama-bench, llama-cli, llama-server, llama-simple for the parity check) and headers go to /opt/llama.cpp.
RUN test -n "${LLAMA_COMMIT}" && \
    git init /tmp/llama.cpp && cd /tmp/llama.cpp && \
    git remote add origin https://github.com/ggml-org/llama.cpp && \
    git fetch --depth 1 origin "${LLAMA_COMMIT}" && git checkout -q FETCH_HEAD && \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/llama.cpp \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_OPENMP=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF && \
    cmake --build build -j"$(nproc)" && \
    cmake --install build && \
    for b in llama-bench llama-cli llama-server llama-simple; do \
        test -x "build/bin/$b" && install -D "build/bin/$b" "/opt/llama.cpp/bin/$b" || echo "note: $b not built"; \
    done && \
    echo "tag=${LLAMA_TAG} commit=${LLAMA_COMMIT}" > /opt/llama.cpp/PIN && \
    sccache --show-stats && rm -rf /tmp/llama.cpp

ENV PATH=/opt/llama.cpp/bin:$PATH \
    LLAMA_CPP_DIR=/opt/llama.cpp

WORKDIR /root
