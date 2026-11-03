#!/usr/bin/env bash
# ADR 0043 measurement: speculative decoding in the Phase 2 exit regime, builder against llama.cpp on the same box.
# Builder: crates/scheduler/tests/spec_regime.rs, one prefill, then one greedy decode whose blocks rotate between plain,
# n-gram and draft-model speculation, then a plain replay that checks the tokens. llama.cpp: llama-server with the same
# -ot placement, without and with -md (same draft GGUF), one completion each (scripts/spec_llama.py).
# Same model, threads and memory setup as scripts/exit_run.sh (balloon to EMULATE_RAM_GB). Waits for
# scripts/fetch_exit_model.sh and scripts/fetch_spec_models.sh. Started detached by `taxameter run scripts/spec_run.sh`;
# ends with SPEC_RUN_DONE rc=N in runs/vast/spec_run.log. STAGES picks stages (default: the main and code ones).
exec 8> /root/tetraeder_spec_run.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
date_dir="$PWD/bench/results/$(date +%Y%m%d)"
mkdir -p "$date_dir" runs/vast
log=runs/vast/spec_run.log
stamp=$(date +%H%M%S)
models="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
big="$models/Qwen__Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf"
m30="$models/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
draft="${DRAFT_GGUF:-$models/Qwen__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf}"
server="${LLAMA_SERVER:-/opt/llama.cpp/bin/llama-server}"
threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"
STAGES="${STAGES:-main-builder main-llama-off main-llama-on code-builder code-llama-off code-llama-on}"
want() { [[ " $STAGES " == *" $1 "* ]]; }

# memory as scripts/exit_run.sh: the container limit, or EMULATE_RAM_GB with a balloon holding the rest
total=$(awk '/MemTotal/ { print $2 * 1024 }' /proc/meminfo)
mem=$(cat /sys/fs/cgroup/memory.max 2> /dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2> /dev/null)
case "$mem" in '' | max) mem=$total ;; esac
[ "$mem" -gt "$total" ] && mem=$total
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
" "$balloon" >> "$log" 2>&1 &
  balloon_pid=$!
  trap 'kill $balloon_pid 2> /dev/null' EXIT
  mem=$emulated
  sleep 60
fi
ram_bytes=$((mem - ${RAM_RESERVE_GB:-20} * 1000000000))
vram_bytes="${VRAM_SLOT_BYTES:-5500000000}"
natural=/root/exit_prompt.txt
[ -f "$natural" ] || cat AGENTS.md docs/adr/*.md > "$natural"
code=/root/spec_code_prompt.txt
[ -f "$code" ] || cat crates/model/src/*.rs crates/memory/src/*.rs > "$code"
variants="${SPEC_VARIANTS:-plain,ngram:a,draft:a,draft:3}"
rc=0

# builder <name> <minutes> [ENV=VALUE...]: spec_regime.rs, JSON to spec-<name>-<stamp>.json
builder() {
  local name=$1 min=$2
  shift 2
  want "$name" || return 0
  echo "[$(date +%T)] start $name"
  env TETRAEDER_EXIT_GGUF="$big" TETRAEDER_EXIT_GPU=1 TETRAEDER_EXIT_THREADS="$threads" \
    TETRAEDER_EXIT_VRAM_SLOT_BYTES="$vram_bytes" TETRAEDER_EXIT_RAM_SLOT_BYTES="$ram_bytes" \
    TETRAEDER_EXIT_OP_OFFLOAD=1 TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_PROMPT_FILE="$natural" \
    TETRAEDER_EXIT_SPEC_VARIANTS="$variants" TETRAEDER_EXIT_SPEC_DRAFT_GGUF="$draft" \
    TETRAEDER_EXIT_OUT="$date_dir/spec-$name-$stamp.json" TETRAEDER_EXIT_MACHINE="vast $(hostname)" "$@" \
    timeout -k 30 "${min}m" cargo test --release -p tetraeder-scheduler --features cuda --test spec_regime \
    -- --ignored --nocapture
  local r=$?
  echo "[$(date +%T)] done $name rc=$r"
  return $r
}

# llama <name> <minutes> <model> <prompt file> <prompt tokens> <ctx> <-ot> [server args...]: spec-<name>-<stamp>.json
# A start that fails with a draft (VRAM) is retried once with a q8_0 draft KV cache.
llama() {
  local name=$1 min=$2 model=$3 prompt=$4 n=$5 ctx=$6 ot=$7
  shift 7
  want "$name" || return 0
  [ -x "$server" ] || { echo "skip $name: $server missing"; return 0; }
  echo "[$(date +%T)] start $name"
  local out="$date_dir/spec-$name-$stamp.json" r
  for extra in "" "-ctkd q8_0 -ctvd q8_0"; do
    # shellcheck disable=SC2086 # extra is a word list
    timeout -k 30 "${min}m" python3 scripts/spec_llama.py --server "$server" --prompt-file "$prompt" \
      --prompt-tokens "$n" --decode "${LLAMA_DECODE:-128}" --out "$out" --note "threads $threads, vast $(hostname)" -- \
      -m "$model" -ngl 99 -fa on -t "$threads" -c "$ctx" -b 2048 -ub 2048 -ot "$ot" "$@" $extra
    r=$?
    [ $r = 0 ] && break
    [[ " $* " == *" -md "* ]] || break
    echo "[$(date +%T)] $name failed rc=$r, retry with a q8_0 draft KV cache"
  done
  echo "[$(date +%T)] done $name rc=$r"
}

{
  echo "[$(date +%T)] start: stages '$STAGES', threads $threads, memory $mem B (balloon $balloon B), ram tier $ram_bytes B, vram slots $vram_bytes B"
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active --format=csv
  for _ in $(seq 1 1440); do grep -q SPEC_FETCH_DONE runs/vast/fetch_spec.log 2> /dev/null && break; sleep 5; done
  grep -q "SPEC_FETCH_DONE rc=0" runs/vast/fetch_spec.log || { echo "draft fetch failed"; rc=20; }
  if [[ "$STAGES" == *main-* || "$STAGES" == *code-* ]]; then
    for _ in $(seq 1 1440); do grep -q FETCH_DONE runs/vast/fetch_exit.log 2> /dev/null && break; sleep 5; done
    grep -q "FETCH_DONE rc=0" runs/vast/fetch_exit.log || { echo "235B fetch failed"; rc=20; }
  fi
  if [[ "$STAGES" == *builder* ]]; then
    cargo test --release -p tetraeder-scheduler --features cuda --test spec_regime --no-run || rc=21
  fi
} >> "$log" 2>&1

samevram='blk\.[0-3]\.ffn_.*_exps=CUDA0,exps=CPU'
if [ $rc = 0 ]; then
  {
    # 30B quick iteration: experts streamed through small tiers, llama.cpp with every expert on the CPU
    builder s30-builder 20 TETRAEDER_EXIT_GGUF="$m30" TETRAEDER_EXIT_PREFILL=4096 TETRAEDER_EXIT_UBATCH=2048 \
      TETRAEDER_EXIT_CTX=4864 TETRAEDER_EXIT_VRAM_SLOT_BYTES=2000000000 TETRAEDER_EXIT_RAM_SLOT_BYTES=6000000000
    llama s30-llama-off 15 "$m30" "$natural" 4096 4864 'exps=CPU'
    llama s30-llama-on 15 "$m30" "$natural" 4096 4864 'exps=CPU' -md "$draft" -ngld 99
    # the exit regime: 32K natural-text prefill, then 4 rounds of 32-token blocks per variant
    builder main-builder 60 TETRAEDER_EXIT_PREFILL=32768 TETRAEDER_EXIT_UBATCH=2048 TETRAEDER_EXIT_CTX=33280 || rc=23
    llama main-llama-off 45 "$big" "$natural" 32768 33280 "$samevram"
    llama main-llama-on 45 "$big" "$natural" 32768 33280 "$samevram" -md "$draft" -ngld 99
    # code-like text at 4K
    builder code-builder 25 TETRAEDER_EXIT_PROMPT_FILE="$code" TETRAEDER_EXIT_PREFILL=4096 TETRAEDER_EXIT_UBATCH=2048 \
      TETRAEDER_EXIT_CTX=4864 || rc=24
    llama code-llama-off 20 "$big" "$code" 4096 4864 "$samevram"
    llama code-llama-on 20 "$big" "$code" 4096 4864 "$samevram" -md "$draft" -ngld 99
  } >> "$log" 2>&1
fi
touch "$date_dir/DONE"
echo "SPEC_RUN_DONE rc=$rc" >> "$log"
command -v sccache > /dev/null 2>&1 && timeout 30 sccache --stop-server > /dev/null 2>&1
