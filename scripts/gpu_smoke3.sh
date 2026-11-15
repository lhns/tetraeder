#!/usr/bin/env bash
# CUDA parity re-run after gpu-smoke2 (ADR 0071): gpu_dense and gpu_multi_seq with --no-fail-fast and the per-row
# prints, including the libllama CUDA batched-vs-split anchors (gpu_libllama_batched_vs_split_anchor in both suites).
# Records only. Started detached: `taxameter run --id <id> scripts/gpu_smoke3.sh --env RESULTS_DATE=<date>`.
# Qwen3-8B at .hf_cache/Qwen__Qwen3-8B-GGUF/ is optional (gpu_qwen3_8b_matches_libllama skips without it).
# Ends with GPU_SMOKE3_DONE in runs/vast/gpu-smoke3.log and bench/results/<RESULTS_DATE>/DONE.
exec 8> /root/tetraeder_gpu_smoke3.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
log=runs/vast/gpu-smoke3.log
exec >> "$log" 2>&1
echo "=== RUN $(date +%FT%T) gpu_smoke3.sh ==="
date_dir="bench/results/${RESULTS_DATE:-$(date +%Y%m%d)}"
out="$date_dir/gpu-smoke3"
mkdir -p "$out"
rm -f "$date_dir/DONE"
finish() {
  echo "rc=$1" > "$date_dir/DONE"
  echo "GPU_SMOKE3_DONE rc=$1"
  exit "$1"
}
export TETRAEDER_REQUIRE_GPU=1 TETRAEDER_MODELS_DIR="$PWD/.hf_cache"
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
echo "[$(date +%T)] cuda: $CUDA_ENV_STATUS"
case "$CUDA_ENV_STATUS" in ok | ok_preload) ;; *) echo "no cuda"; finish 9 ;; esac
bash scripts/ensure_patches.sh 8>&- || { echo "patches"; finish 8; }
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,clocks.sm,clocks_throttle_reasons.active --format=csv \
  > "$out/gpu.log" 2>&1

echo "[$(date +%T)] build"
timeout -k 30 40m cargo test --release --no-run -p tetraeder-model --features tetraeder-ggml-sys/cuda \
  --test gpu_dense --test gpu_multi_seq > "$out/build.log" 2>&1 8>&-
rcb=$?
echo "[$(date +%T)] build rc=$rcb"
[ "$rcb" = 0 ] || finish 7
rc=0
for t in gpu_dense gpu_multi_seq; do
  echo "[$(date +%T)] start $t"
  timeout -k 30 15m cargo test --release --no-fail-fast -p tetraeder-model --features tetraeder-ggml-sys/cuda --test "$t" \
    -- --nocapture --test-threads 1 > "$out/$t.log" 2>&1 8>&-
  r=$?
  echo "[$(date +%T)] $t rc=$r"
  [ "$r" = 0 ] || rc=$r
done
echo "{\"build_rc\": $rcb, \"tests_rc\": $rc}" > "$out/rc.json"
finish "$rc"
