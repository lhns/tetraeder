#!/usr/bin/env bash
# CUDA graph replay in decode (ADR 0036): tests/gpu_graphs.rs on Qwen3-30B-A3B in three processes
# (graphs on, graphs off, graphs on with ggml's CUDA graph log), then on and off once more for
# pairing. Started detached by `taxameter run scripts/graphs_check.sh`; ends with
# GRAPHS_CHECK_DONE rc=N in its log.
exec 8> /root/tetraeder_graphs_check.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
bash scripts/ensure_patches.sh 8>&- || exit 1  # ggml patch queue (ADR 0052)
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
date_dir="$PWD/bench/results/$(date +%Y%m%d)"
mkdir -p "$date_dir" runs/vast
log=runs/vast/graphs_check.log
{
  if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
    archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
    [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
  fi
  models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
  export TETRAEDER_MOE_GGUF="${MOE_GGUF:-$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf}"
  echo "[$(date +%T)] start graphs-check"
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active --format=csv
  # Build once; each run below is a separate process, since ggml reads the CUDA graph variables
  # once. GGML_CUDA_DISABLE_GRAPHS counts when present, even empty: never set it for "on".
  cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda --test gpu_graphs --no-run
  rc=$?
  run() { # name, env prefix...
    name=$1
    shift
    [ "$rc" -eq 0 ] || return
    echo "[$(date +%T)] run $name"
    TETRAEDER_REQUIRE_GPU=1 TETRAEDER_GRAPHS_OUT="$date_dir/cuda-graphs-$name.json" env -u GGML_CUDA_DISABLE_GRAPHS -u GGML_CUDA_GRAPH_DEBUG "$@" \
      timeout -k 30 "${GRAPHS_MIN:-15}m" cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda \
      --test gpu_graphs -- --nocapture
    rc=$?
  }
  run on
  run off GGML_CUDA_DISABLE_GRAPHS=1
  run debug GGML_CUDA_GRAPH_DEBUG=1
  run on2
  run off2 GGML_CUDA_DISABLE_GRAPHS=1
  echo "GRAPHS_CHECK_DONE rc=$rc"
} >> "$log" 2>&1
# sccache inherits fd 8 and would keep the flock after this script exits (as in nightly.sh)
command -v sccache > /dev/null 2>&1 && timeout 30 sccache --stop-server > /dev/null 2>&1
