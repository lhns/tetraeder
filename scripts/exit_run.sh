#!/usr/bin/env bash
# Phase 2 exit run (ADR 0035, 0037): Qwen3-235B-A22B Q4_K_M, larger than the box RAM, builder with MissPolicy::Cpu
# (tests/exit_regime.rs) against llama.cpp with expert offload (-ot) on the same box and threads.
# Waits for scripts/fetch_exit_model.sh. Started detached by `taxameter run scripts/exit_run.sh`; ends with
# EXIT_RUN_DONE rc=N in runs/vast/exit_run.log. Every stage has a time limit; a failed smoke gate stops the run.
exec 8> /root/tetraeder_exit_run.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
date_dir="$PWD/bench/results/$(date +%Y%m%d)"
mkdir -p "$date_dir" runs/vast
log=runs/vast/exit_run.log
stamp=$(date +%H%M%S)
# this box's GPU only, as moe_check.sh: another arch list rebuilds ggml-cuda for every arch (an hour, no cache hit)
if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
model="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}/Qwen__Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf"
threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"
# RAM the container may use: the cgroup limit if set, else MemTotal. The exit test needs a model larger than RAM, but
# Vast's cgroup limit can exceed the offer's RAM (183 GB on a "126 GB" offer, model 142 GB). So a balloon process
# holds the difference to EMULATE_RAM_GB (default 128) as touched anonymous memory (no swap: it stays resident), and
# both engines, page cache included, run within the rest. The RAM tier gets the emulated RAM minus RAM_RESERVE_GB.
mem=$(cat /sys/fs/cgroup/memory.max 2> /dev/null)
case "$mem" in '' | max) mem=$(awk '/MemTotal/ { print $2 * 1024 }' /proc/meminfo) ;; esac
emulated=$((${EMULATE_RAM_GB:-128} * 1000000000))
balloon=$((mem - emulated))
if [ "$balloon" -gt 0 ]; then
  python3 -c "
import sys, time
n = int(sys.argv[1])
b = bytearray(n)
b[::4096] = b'\x01' * len(range(0, n, 4096))
print('balloon holds', n, 'B', flush=True)
time.sleep(10**9)
" "$balloon" >> runs/vast/exit_run.log 2>&1 &
  balloon_pid=$!
  trap 'kill $balloon_pid 2> /dev/null' EXIT
  mem=$emulated
  sleep 60  # touching the pages takes a while; the run must not start before
fi
ram_bytes=$((mem - ${RAM_RESERVE_GB:-20} * 1000000000))
vram_bytes="${VRAM_SLOT_BYTES:-5500000000}"
bench="${LLAMA_BENCH:-/opt/llama.cpp/bin/llama-bench}"
# natural text for the prompt (random token ids spread routing pessimistically)
prompt=/root/exit_prompt.txt
cat AGENTS.md docs/adr/*.md > "$prompt"
rc=0
{
  echo "[$(date +%T)] start: threads $threads, memory $mem B (balloon ${balloon} B), ram tier $ram_bytes B, vram slots $vram_bytes B"
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active --format=csv
  echo "[$(date +%T)] waiting for the model"
  for _ in $(seq 1 1440); do grep -q FETCH_DONE runs/vast/fetch_exit.log 2> /dev/null && break; sleep 5; done
  grep FETCH_DONE runs/vast/fetch_exit.log | tail -1
  grep -q "FETCH_DONE rc=0" runs/vast/fetch_exit.log || { echo "model fetch failed"; rc=20; }
  cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda --test exit_regime --no-run || rc=21
} >> "$log" 2>&1

# STAGES (space list of stage names, default all) lets a relaunch skip stages that already ran
want() { [ -z "${STAGES:-}" ] || [[ " $STAGES " == *" $1 "* ]]; }

# exit_regime <name> <minutes> [ENV=VALUE...]: one harness run, JSON to exit-<name>-<stamp>.json
exit_regime() {
  local name=$1 min=$2
  shift 2
  want "$name" || return 0
  echo "[$(date +%T)] start $name"
  env TETRAEDER_EXIT_GGUF="$model" TETRAEDER_EXIT_GPU=1 TETRAEDER_EXIT_THREADS="$threads" \
    TETRAEDER_EXIT_VRAM_SLOT_BYTES="$vram_bytes" TETRAEDER_EXIT_RAM_SLOT_BYTES="$ram_bytes" \
    TETRAEDER_EXIT_PROMPT_FILE="$prompt" TETRAEDER_EXIT_OUT="$date_dir/exit-$name-$stamp.json" \
    TETRAEDER_EXIT_MACHINE="vast $(hostname)" "$@" \
    timeout -k 30 "${min}m" cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda \
    --test exit_regime -- --ignored --nocapture
  local r=$?
  echo "[$(date +%T)] done $name rc=$r"
  return $r
}

# baseline <engine-bin> <name> <minutes> <args...>: llama-bench JSON to <name>-<stamp>.json
baseline() {
  local bin=$1 name=$2 min=$3
  shift 3
  want "$name" || return 0
  [ -x "$bin" ] || { echo "skip $name: $bin missing"; return 0; }
  echo "[$(date +%T)] start $name"
  timeout -k 30 "${min}m" "$bin" -m "$model" -ngl 99 -fa 1 -t "$threads" -r 1 -o json "$@" > "$date_dir/$name-$stamp.json"
  local r=$?
  echo "[$(date +%T)] done $name rc=$r"
}

if [ $rc = 0 ]; then
  {
    # smoke gate: real header, the full KV (a VRAM OOM shows here), direct I/O, the backend split, memory
    exit_regime smoke 20 TETRAEDER_EXIT_PREFILL=8 TETRAEDER_EXIT_UBATCH=8 TETRAEDER_EXIT_DECODE=2 \
      TETRAEDER_EXIT_CTX=33024 TETRAEDER_EXIT_OP_OFFLOAD=1 || rc=22
    python3 - "$date_dir/exit-smoke-$stamp.json" << 'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    sys.exit(f"smoke json: {e}")
print("smoke:", {k: d.get(k) for k in ("io_backend", "slot_bytes_all_layers", "options")})
for r in d.get("runs") or []:
    print("  run:", {k: r.get(k) for k in ("max_splits", "rss_bytes", "peak_rss_bytes", "decode_tok_s")})
    for m in r.get("memory") or []:
        print("  memory:", m)
EOF
    if [ $rc = 0 ]; then
      # ubatch 2048: a pair group passes one row per (token, expert) pair to CUDA's mul_mat_id, whose id helper
      # needs 4 B of shared memory per row (about 25k rows on Ampere); ubatch 4096 at top-8 is 32k (abort, 2026-11-02)
      exit_regime main 60 TETRAEDER_EXIT_PREFILL=32768 TETRAEDER_EXIT_UBATCH=2048 TETRAEDER_EXIT_DECODE=64 \
        TETRAEDER_EXIT_CTX=33024 TETRAEDER_EXIT_OP_OFFLOAD=1 || rc=23
      # baselines at depth 0, same box and threads, before the variants so a short box still has the comparison:
      # all experts on the CPU, and the same expert VRAM as the builder (layers 0-3 of about 1.46 GB each on the GPU;
      # llama-bench takes ',' as alternative configs, so overrides in one config are joined with ';')
      # (llama.cpp only: ik_llama was OOM-killed at 128 GB)
      baseline "$bench" exit-llama-ot-all 30 -ub 2048 -p 2048 -n 32 -ot 'exps=CPU'
      baseline "$bench" exit-llama-ot-samevram 30 -ub 2048 -p 2048 -n 32 -ot 'blk\.[0-3]\.ffn_.*_exps=CUDA0;exps=CPU'
      exit_regime refresh-hot 30 TETRAEDER_EXIT_PREFILL=8192 TETRAEDER_EXIT_UBATCH=2048 TETRAEDER_EXIT_DECODE=64 \
        TETRAEDER_EXIT_OP_OFFLOAD=1 TETRAEDER_EXIT_REFRESH_HOT=1 || rc=24
      exit_regime cpu-prefill 30 TETRAEDER_EXIT_PREFILL=2048 TETRAEDER_EXIT_UBATCH=2048 TETRAEDER_EXIT_DECODE=32 \
        TETRAEDER_EXIT_OP_OFFLOAD=0 || rc=25
      # the deep prompt last: on a page-faulting 235B it may take long
      baseline "$bench" exit-llama-ot-all-d32k 40 -ub 2048 -p 2048 -n 32 -d 32768 -ot 'exps=CPU'
      baseline "$bench" exit-llama-ot-samevram-d32k 40 -ub 2048 -p 2048 -n 32 -d 32768 -ot 'blk\.[0-3]\.ffn_.*_exps=CUDA0;exps=CPU'
    fi
  } >> "$log" 2>&1
fi
touch "$date_dir/DONE"
echo "EXIT_RUN_DONE rc=$rc" >> "$log"
command -v sccache > /dev/null 2>&1 && timeout 30 sccache --stop-server > /dev/null 2>&1
