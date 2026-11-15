#!/usr/bin/env bash
# Rerun of gpu_smoke2.sh step 1 with --no-fail-fast (the first run stopped at the gpu_dense failure and never ran
# gpu_multi_seq). Same env, build cached. Ends with GPU_SMOKE2B_DONE in runs/vast/gpu-smoke2b.log.
exec 8> /root/tetraeder_gpu_smoke2b.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
log=runs/vast/gpu-smoke2b.log
exec >> "$log" 2>&1
out="bench/results/${RESULTS_DATE:-$(date +%Y%m%d)}/gpu-smoke2"
mkdir -p "$out"
export TETRAEDER_REQUIRE_GPU=1 TETRAEDER_MODELS_DIR="$PWD/.hf_cache"
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
[ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
timeout -k 30 30m cargo test --release --no-fail-fast -p tetraeder-model --features tetraeder-ggml-sys/cuda \
  --test gpu_dense --test gpu_multi_seq -- --nocapture --test-threads 1 > "$out/gpu-tests-nofailfast.log" 2>&1 8>&-
echo "GPU_SMOKE2B_DONE rc=$?"
