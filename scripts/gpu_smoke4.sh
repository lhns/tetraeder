#!/usr/bin/env bash
# gpu-smoke3 plus the single-GPU Phase 5 tests (S1): mul_mat_id_skip gpu_ with timing JSON, exec_bindings gpu_.
# Started detached: `taxameter run --id <id> scripts/gpu_smoke4.sh --env RESULTS_DATE=<date>`.
# Ends with GPU_SMOKE4_DONE in runs/vast/gpu-smoke4.log and bench/results/<RESULTS_DATE>/DONE4.
exec 8> /root/tetraeder_gpu_smoke4.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
log=runs/vast/gpu-smoke4.log
exec >> "$log" 2>&1
echo "=== RUN $(date +%FT%T) gpu_smoke4.sh ==="
date_dir="bench/results/${RESULTS_DATE:-$(date +%Y%m%d)}"
out="$date_dir/s1"
mkdir -p "$out"
rm -f "$date_dir/DONE4"
export RESULTS_DATE
bash scripts/gpu_smoke3.sh 8>&-
rc3=$?
echo "[$(date +%T)] smoke3 rc=$rc3"
export TETRAEDER_REQUIRE_GPU=1 TETRAEDER_MODELS_DIR="$PWD/.hf_cache"
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
. scripts/box_libs.sh
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
[ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
rc=0
echo "[$(date +%T)] start mul_mat_id_skip"
TETRAEDER_MMID_JSON="$PWD/$out/mul_mat_id_skip.json" timeout -k 30 25m cargo test --release -p tetraeder-ggml-sys \
  --features cuda --test mul_mat_id_skip gpu_ -- --nocapture --test-threads=1 > "$out/mul_mat_id_skip.log" 2>&1 8>&-
r=$?; echo "[$(date +%T)] mul_mat_id_skip rc=$r"; [ "$r" = 0 ] || rc=$r
echo "[$(date +%T)] start exec_bindings"
timeout -k 30 25m cargo test --release --no-fail-fast -p tetraeder-ggml-sys --features cuda --test exec_bindings gpu_ \
  -- --nocapture --test-threads=1 > "$out/exec_bindings.log" 2>&1 8>&-
r=$?; echo "[$(date +%T)] exec_bindings rc=$r"; [ "$r" = 0 ] || rc=$r
echo "{\"smoke3_rc\": $rc3, \"s1_rc\": $rc}" > "$out/rc.json"
echo "rc=$rc" > "$date_dir/DONE4"
echo "GPU_SMOKE4_DONE rc=$rc"
