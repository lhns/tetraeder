#!/usr/bin/env bash
# Re-run after gpu-smoke3/S1 (bench/results/20261116gpu3): the move test with the host-restored reference
# (gpu_multi_seq, multi_seq's CPU moves at the box's thread count), the CPU anchor with libllama's op offload off
# (gpu_dense), and patch 0004 with the mm_ids_helper fix: every mul_mat_id_skip case printed, the EP timing JSON.
# Records only. Started detached: `taxameter run --id <id> scripts/gpu_smoke5.sh --env RESULTS_DATE=<date>`.
# Ends with GPU_SMOKE5_DONE in runs/vast/gpu-smoke5.log and bench/results/<RESULTS_DATE>/DONE5.
exec 8> /root/tetraeder_gpu_smoke5.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
log=runs/vast/gpu-smoke5.log
exec >> "$log" 2>&1
echo "=== RUN $(date +%FT%T) gpu_smoke5.sh ==="
date_dir="bench/results/${RESULTS_DATE:-$(date +%Y%m%d)}"
out="$date_dir/gpu-smoke5"
mkdir -p "$out"
rm -f "$date_dir/DONE5"
finish() {
  echo "rc=$1" > "$date_dir/DONE5"
  echo "GPU_SMOKE5_DONE rc=$1"
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
nproc > "$out/nproc.txt" 2>&1
grep -m1 'model name' /proc/cpuinfo >> "$out/nproc.txt" 2>&1
grep -m1 -o 'avx512[a-z_]*' /proc/cpuinfo | head -1 >> "$out/nproc.txt" 2>&1

echo "[$(date +%T)] build"
timeout -k 30 40m cargo test --release --no-run -p tetraeder-model --features tetraeder-ggml-sys/cuda \
  --test gpu_dense --test gpu_multi_seq --test multi_seq > "$out/build.log" 2>&1 8>&-
rcb=$?
timeout -k 30 20m cargo test --release --no-run -p tetraeder-ggml-sys --features cuda --test mul_mat_id_skip \
  >> "$out/build.log" 2>&1 8>&-
rcb2=$?
echo "[$(date +%T)] build rc=$rcb/$rcb2"
[ "$rcb" = 0 ] && [ "$rcb2" = 0 ] || finish 7
rc=0
run() { # name, cargo test args...
  local name=$1
  shift
  echo "[$(date +%T)] start $name"
  timeout -k 30 15m cargo test --release --no-fail-fast "$@" > "$out/$name.log" 2>&1 8>&-
  local r=$?
  echo "[$(date +%T)] $name rc=$r"
  [ "$r" = 0 ] || rc=$r
}
run mul_mat_id_skip -p tetraeder-ggml-sys --features cuda --test mul_mat_id_skip gpu_ -- --nocapture --test-threads=1
grep -h '^MUL_MAT_ID_SKIP ' "$out/mul_mat_id_skip.log" | sed 's/^MUL_MAT_ID_SKIP //' > "$out/mul_mat_id_skip.json"
run gpu_multi_seq -p tetraeder-model --features tetraeder-ggml-sys/cuda --test gpu_multi_seq \
  -- --nocapture --test-threads 1
run gpu_dense -p tetraeder-model --features tetraeder-ggml-sys/cuda --test gpu_dense -- --nocapture --test-threads 1
run multi_seq_move -p tetraeder-model --features tetraeder-ggml-sys/cuda --test multi_seq cpu_chunked_move \
  -- --nocapture --test-threads 1
echo "{\"build_rc\": $rcb, \"tests_rc\": $rc}" > "$out/rc.json"
finish "$rc"
