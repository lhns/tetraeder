#!/usr/bin/env bash
# gpu_smoke5.sh plus the Phase 5 E5 single-GPU executor tests (gpu_ in crates/executor/tests/gpu.rs).
# Started detached: `taxameter run --id <id> scripts/gpu_smoke6.sh --env RESULTS_DATE=<date>`.
# Ends with GPU_SMOKE6_DONE in runs/vast/gpu-smoke6.log and bench/results/<RESULTS_DATE>/DONE6.
exec 8> /root/tetraeder_gpu_smoke6.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
log=runs/vast/gpu-smoke6.log
exec >> "$log" 2>&1
echo "=== RUN $(date +%FT%T) gpu_smoke6.sh ==="
date_dir="bench/results/${RESULTS_DATE:-$(date +%Y%m%d)}"
out="$date_dir/gpu-smoke5"
mkdir -p "$out"
rm -f "$date_dir/DONE6"
export RESULTS_DATE
bash scripts/gpu_smoke5.sh 8>&-
rc5=$?
echo "[$(date +%T)] smoke5 rc=$rc5"
export TETRAEDER_MODELS_DIR="$PWD/.hf_cache"
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
[ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
echo "[$(date +%T)] start executor_gpu"
TETRAEDER_REQUIRE_GPUS=1 GGML_CUDA_GRAPH_DEBUG=1 timeout -k 30 25m cargo test --release --no-fail-fast \
  -p tetraeder-executor --features cuda --test gpu gpu_ -- --nocapture --test-threads=1 \
  > "$out/executor_gpu.log" 2>&1 8>&-
r=$?
echo "[$(date +%T)] executor_gpu rc=$r"
rc=$rc5
[ "$r" = 0 ] || rc=$r
echo "{\"smoke5_rc\": $rc5, \"executor_gpu_rc\": $r}" > "$out/rc6.json"
echo "rc=$rc" > "$date_dir/DONE6"
echo "GPU_SMOKE6_DONE rc=$rc"
exit "$rc"
