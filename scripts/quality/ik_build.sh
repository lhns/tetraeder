#!/usr/bin/env bash
# ik_llama.cpp's llama-quantize and llama-perplexity at the pin (vendor/PIN ik_commit), CUDA for this box's arch,
# into /root/ik/bin (the image installs only its llama-bench and llama-cli). Ends with IK_BUILD_DONE rc=N.
exec 7> /root/ik_build.lock; flock -n 7 || exit 0
cd "$(dirname "$0")/../.." || exit 1
commit=$(sed -n 's/^ik_commit=//p' vendor/PIN)
arch=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)
command -v sccache > /dev/null 2>&1 && export CMAKE_C_COMPILER_LAUNCHER=sccache CMAKE_CXX_COMPILER_LAUNCHER=sccache \
  CMAKE_CUDA_COMPILER_LAUNCHER=sccache
rc=0
{
  [ -d /root/ik_src ] || { git init -q /root/ik_src && git -C /root/ik_src remote add origin https://github.com/ikawrakow/ik_llama.cpp; }
  git -C /root/ik_src fetch -q --depth 1 origin "$commit" && git -C /root/ik_src checkout -q FETCH_HEAD &&
  cmake -S /root/ik_src -B /root/ik_src/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DCMAKE_CUDA_ARCHITECTURES="$arch" \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF > /dev/null &&
  cmake --build /root/ik_src/build -j "${IK_JOBS:-4}" --target llama-quantize llama-perplexity &&
  mkdir -p /root/ik/bin && cp /root/ik_src/build/bin/llama-quantize /root/ik_src/build/bin/llama-perplexity /root/ik/bin/
} || rc=$?
echo "IK_BUILD_DONE rc=$rc $(date +%T)"
