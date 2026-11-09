# Tetraeder GPU box image (ghcr.io/lhns/tetraeder-box), used on rented Vast.ai boxes.
# Contains the toolchain plus llama.cpp and ik_llama.cpp built with CUDA at the repo pins (vendor/PIN) for baselines.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG LLAMA_TAG
ARG LLAMA_COMMIT
ARG IK_COMMIT
ARG LLAMA_BASELINE_TAG
ARG LLAMA_BASELINE_COMMIT
ARG RUST_VERSION=1.95.0
ARG SCCACHE_VERSION=0.10.0
# Ampere (8.6), Ada (8.9), Hopper (9.0), Blackwell (12.0)
ARG CUDA_ARCHS="86;89;90;120"
# ik_llama.cpp compiles many more CUDA kernels: for all four archs the image build ran past 3 h on the CI runner.
# The boxes are RTX 3090 (taxameter.toml offers), so that baseline is built for Ampere only.
ARG IK_CUDA_ARCHS="86"
# The llama.cpp baseline build (a commit after the MoE cache, #30966) is Ampere only for the same reason.
ARG BASELINE_CUDA_ARCHS="86"

# Our Rust builds on a box use the native ISA ggml (ADR 0052 addendum 2026-11-09): one CPU variant instead of 14, so a
# fresh box does not pay for 14 variants plus CUDA. CI and releases build the DL default. The baselines below keep
# their own -DGGML_NATIVE=OFF (portable AVX2).
ENV DEBIAN_FRONTEND=noninteractive \
    TETRAEDER_GGML_NATIVE=ON \
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

# ik_llama.cpp at its pin (ik_commit in vendor/PIN), same portable CPU and CUDA arch settings. Only the baseline
# tools are installed (no libs or headers). Its option names match upstream's here: GGML_CUDA, GGML_NATIVE=OFF
# (turns on AVX2/FMA/F16C, which its IQK kernels need), GGML_OPENMP, LLAMA_BUILD_EXAMPLES (llama-bench and
# llama-cli live in examples/). BUILD_SHARED_LIBS=OFF so the binaries need no libs from the build tree.
RUN test -n "${IK_COMMIT}" && \
    git init /tmp/ik_llama.cpp && cd /tmp/ik_llama.cpp && \
    git remote add origin https://github.com/ikawrakow/ik_llama.cpp && \
    git fetch --depth 1 origin "${IK_COMMIT}" && git checkout -q FETCH_HEAD && \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_OPENMP=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${IK_CUDA_ARCHS}" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_CURL=OFF && \
    cmake --build build -j"$(nproc)" --target llama-bench llama-cli && \
    for b in llama-bench llama-cli; do \
        install -D "build/bin/$b" "/opt/ik_llama/bin/$b"; \
    done && \
    echo "commit=${IK_COMMIT}" > /opt/ik_llama/PIN && \
    sccache --show-stats && rm -rf /tmp/ik_llama.cpp

# llama.cpp baseline (baseline_commit in vendor/PIN): a newer commit with the GPU expert cache and CPU share of misses
# (llama-server --moe-cache-mib, --moe-cache-cpu-share), only for the exit pairing. The engine, ggml-sys and parity
# stay on the pin above. Tools only, plus the cache tuner and its docs, in /opt/llama.cpp-baseline (not on PATH).
RUN test -n "${LLAMA_BASELINE_COMMIT}" && \
    git init /tmp/llama.cpp-baseline && cd /tmp/llama.cpp-baseline && \
    git remote add origin https://github.com/ggml-org/llama.cpp && \
    git fetch --depth 1 origin "${LLAMA_BASELINE_COMMIT}" && git checkout -q FETCH_HEAD && \
    cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DGGML_OPENMP=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${BASELINE_CUDA_ARCHS}" \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF && \
    cmake --build build -j"$(nproc)" --target llama-server llama-bench llama-cli && \
    for b in llama-server llama-bench llama-cli; do \
        install -D "build/bin/$b" "/opt/llama.cpp-baseline/bin/$b"; \
    done && \
    install -D scripts/moe-cache-tune.py /opt/llama.cpp-baseline/scripts/moe-cache-tune.py && \
    install -D docs/moe-cache.md /opt/llama.cpp-baseline/scripts/moe-cache.md && \
    echo "tag=${LLAMA_BASELINE_TAG} commit=${LLAMA_BASELINE_COMMIT}" > /opt/llama.cpp-baseline/PIN && \
    sccache --show-stats && rm -rf /tmp/llama.cpp-baseline

# ik_llama's binaries are not on PATH: its names collide with llama.cpp's; the harness uses /opt/ik_llama/bin.
ENV PATH=/opt/llama.cpp/bin:$PATH \
    LLAMA_CPP_DIR=/opt/llama.cpp

WORKDIR /root
