#!/usr/bin/env bash
# First CUDA run of the own builders: gpu_dense + gpu_multi_seq, then step_shapes on Qwen3-8B (llama, llama_streams,
# dense). Records only. Started detached: `taxameter run --id <id> scripts/gpu_smoke2.sh --env RESULTS_DATE=<date>`.
# Needs the 8B at .hf_cache/Qwen__Qwen3-8B-GGUF/ (provision --models). Ends with GPU_SMOKE2_DONE in
# runs/vast/gpu-smoke2.log; results in bench/results/<RESULTS_DATE>/gpu-smoke2/.
exec 8> /root/tetraeder_gpu_smoke2.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
log=runs/vast/gpu-smoke2.log
exec >> "$log" 2>&1
echo "=== RUN $(date +%FT%T) gpu_smoke2.sh ==="
out="bench/results/${RESULTS_DATE:-$(date +%Y%m%d)}/gpu-smoke2"
mkdir -p "$out"
export TETRAEDER_REQUIRE_GPU=1 TETRAEDER_MODELS_DIR="$PWD/.hf_cache"
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
echo "[$(date +%T)] cuda: $CUDA_ENV_STATUS"
case "$CUDA_ENV_STATUS" in ok | ok_preload) ;; *) echo "GPU_SMOKE2_DONE rc=9 no cuda"; exit 9 ;; esac
bash scripts/ensure_patches.sh 8>&- || { echo "GPU_SMOKE2_DONE rc=8 patches"; exit 8; }
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv > "$out/gpu.txt" 2>&1
f8="$TETRAEDER_MODELS_DIR/Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
[ -s "$f8" ] || { echo "GPU_SMOKE2_DONE rc=7 no 8B at $f8"; exit 7; }

echo "[$(date +%T)] start tests"
timeout -k 30 40m cargo test --release --no-fail-fast -p tetraeder-model --features tetraeder-ggml-sys/cuda --test gpu_dense --test gpu_multi_seq \
  -- --nocapture --test-threads 1 > "$out/gpu-tests.log" 2>&1 8>&-
rc1=$?
echo "[$(date +%T)] tests rc=$rc1"
echo "[$(date +%T)] start shapes"
TETRAEDER_SHAPES_GGUF="$f8" TETRAEDER_SHAPES_NGL=99 TETRAEDER_SHAPES_ENGINES=llama,llama_streams,dense \
  TETRAEDER_SHAPES_THREADS="$(nproc 2> /dev/null || echo 4)" timeout -k 30 30m \
  cargo test --release -p tetraeder-scheduler --features cuda --test step_shapes -- --nocapture --test-threads 1 \
  > "$out/gpu-shapes.txt" 2>&1 8>&-
rc2=$?
echo "[$(date +%T)] shapes rc=$rc2"
echo "{\"tests_rc\": $rc1, \"shapes_rc\": $rc2}" > "$out/rc.json"
echo "GPU_SMOKE2_DONE rc=$rc1,$rc2"
