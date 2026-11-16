#!/usr/bin/env bash
# Re-run of crates/model/tests/gpu_real_batched.rs on CUDA after the D2a parity failure (bench/results/20261116d2a/
# phase4/): Qwen3-8B (dense builder) and Qwen3-30B-A3B (MoE builder) Q4_K_M, three sequences, against libllama batched,
# with the ADR 0071 anchor on the real model (libllama batched against split and against shifted KV cells) and
# sequence 0 alone as the strict control (same cells and shapes in both engines, plain GPU_TOL).
# Provision with the two models in .hf_cache:
#   --models Qwen/Qwen3-8B-GGUF:Qwen3-8B-Q4_K_M.gguf,Qwen/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf
# Started detached: `taxameter run --id <id> scripts/gpu_real_batched.sh [--env RESULTS_DATE=<date>]`.
# Logs bench/results/<date>/real-batched/{q8b,q30b}.log, rc.json; ends with GPU_REAL_BATCHED_DONE in
# runs/vast/gpu-real-batched.log and real-batched/DONE. About 15 min after the build: 8B, then 30B, seven runs each,
# never two on the GPU at once (each test drops libllama's model before the builder loads).
exec 8> /root/tetraeder_gpu_real_batched.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
exec >> runs/vast/gpu-real-batched.log 2>&1
echo "=== RUN $(date +%FT%T) gpu_real_batched.sh ==="
out="bench/results/${RESULTS_DATE:-$(date +%Y%m%d)}/real-batched"
mkdir -p "$out"
rm -f "$out/DONE"
bash scripts/ensure_patches.sh 8>&- || { echo "rc=1" > "$out/DONE"; echo "GPU_REAL_BATCHED_DONE rc=1"; exit 1; }
export TETRAEDER_MODELS_DIR="$PWD/.hf_cache" HF_HUB_OFFLINE=1
# the test skips (green) without a model file: fail here instead
for f in Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf; do
  [ -f "$TETRAEDER_MODELS_DIR/$f" ] || { echo "missing $f"; echo "rc=1" > "$out/DONE"; echo "GPU_REAL_BATCHED_DONE rc=1"; exit 1; }
done
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
[ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem,clocks_throttle_reasons.active --format=csv > "$out/gpu.csv" 2>&1
test_cmd="cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda --test gpu_real_batched"
declare -A rc
for m in q8b q30b; do
  case $m in
    q8b) filter=gpu_qwen3_8b ;;
    q30b) filter=gpu_qwen3_30b ;;
  esac
  echo "[$(date +%T)] start $m"
  # cargo with fd 8 closed: the sccache server it starts would otherwise keep the lock
  TETRAEDER_REQUIRE_GPU=1 timeout -k 30 30m $test_cmd "$filter" -- --nocapture --test-threads 1 \
    > "$out/$m.log" 2>&1 8>&-
  rc[$m]=$?
  echo "[$(date +%T)] $m rc=${rc[$m]}"
  grep -E "worst .* of the largest logit|bound max|argmax .* against|panicked|differs" "$out/$m.log" | head -40
done
echo "{\"q8b_rc\": ${rc[q8b]}, \"q30b_rc\": ${rc[q30b]}}" > "$out/rc.json"
r=0
[ "${rc[q8b]}" = 0 ] && [ "${rc[q30b]}" = 0 ] || r=1
echo "rc=$r" > "$out/DONE"
echo "GPU_REAL_BATCHED_DONE rc=$r"
exit "$r"
