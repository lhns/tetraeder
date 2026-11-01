#!/usr/bin/env bash
# Runs only nightly.sh's gpu-moe stage (tests/real_moe.rs on Qwen3-30B-A3B), for a rerun on a box that already ran
# the nightly. Started detached by `taxameter run scripts/moe_check.sh`; ends with MOE_CHECK_DONE rc=N in its log.
exec 8> /root/tetraeder_moe_check.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
date_dir="$PWD/bench/results/$(date +%Y%m%d)"
mkdir -p "$date_dir" runs/vast
log=runs/vast/moe_check.log
{
  if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
    archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
    [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
  fi
  models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
  echo "[$(date +%T)] start gpu-moe"
  TETRAEDER_MOE_GGUF="${MOE_GGUF:-$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf}" \
    TETRAEDER_REAL_OUT="$date_dir/builder-real.json" \
    timeout -k 30 "${MOE_MIN:-15}m" cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda \
    --test quantized --test real_moe -- --nocapture
  rc=$?
  echo "MOE_CHECK_DONE rc=$rc"
} >> "$log" 2>&1
# sccache inherits fd 8 and would keep the flock after this script exits (as in nightly.sh)
command -v sccache > /dev/null 2>&1 && timeout 30 sccache --stop-server > /dev/null 2>&1
