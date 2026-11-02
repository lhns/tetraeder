#!/usr/bin/env bash
# CPU speed of the builder against libllama (tests/cpu_bench.rs, ADR 0036) on the synthetic MoE and on
# Qwen3-30B-A3B, GPU hidden. Started detached by `taxameter run scripts/cpu_bench.sh`; ends with
# CPU_BENCH_DONE rc=N in its log. THREADS (comma list) overrides the thread counts.
exec 8> /root/tetraeder_cpu_bench.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
date_dir="$PWD/bench/results/$(date +%Y%m%d)"
mkdir -p "$date_dir" runs/vast
log=runs/vast/cpu_bench.log
{
  if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
    archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
    [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
  fi
  models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
  # CPU only on both sides: libllama would otherwise offload large ops to the GPU.
  export CUDA_VISIBLE_DEVICES='' GGML_VK_VISIBLE_DEVICES=''
  # Default: 8 and the container's CPU quota rounded down (ADR 0021), else the available parallelism.
  quota=$(bash scripts/cpu_quota.sh)
  THREADS="${THREADS:-${quota:+8,$quota}}"
  [ -n "$THREADS" ] && export TETRAEDER_BENCH_THREADS="$THREADS"
  export TETRAEDER_BENCH_MACHINE="vast host $(hostname), $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
  cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda --test cpu_bench --no-run
  for which in synth real; do
    echo "[$(date +%T)] start $which"
    if [ "$which" = real ]; then
      export TETRAEDER_MOE_GGUF="${MOE_GGUF:-$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf}"
    fi
    TETRAEDER_BENCH_OUT="$date_dir/cpu-bench-$which-$(date +%H%M%S).json" timeout -k 30 "${BENCH_MIN:-40}m" \
      cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda --test cpu_bench -- --ignored --nocapture
    rc=$?
    [ $rc -ne 0 ] && break
  done
  echo "CPU_BENCH_DONE rc=$rc"
} >> "$log" 2>&1
# sccache inherits fd 8 and would keep the flock after this script exits (as in nightly.sh)
command -v sccache > /dev/null 2>&1 && timeout 30 sccache --stop-server > /dev/null 2>&1
