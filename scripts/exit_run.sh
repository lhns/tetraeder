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
# (cgroup v2 memory.max, else v1 limit_in_bytes, which reads as a huge number when unlimited; at most MemTotal)
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
" "$balloon" >> runs/vast/exit_run.log 2>&1 &
  balloon_pid=$!
  trap 'kill $balloon_pid 2> /dev/null' EXIT
  mem=$emulated
  sleep 60  # touching the pages takes a while; the run must not start before
fi
# The builder reads with direct I/O (no page cache): besides the tiers it holds the bounce buffers (1.3 GB) and about
# 1 GB of process memory (peak RSS 72.75 GB with a 70.8 GB RAM tier, 2026-11-02), so 8 GB is left for the rest.
ram_bytes=$((mem - ${RAM_RESERVE_GB:-8} * 1000000000))
# expert VRAM: VRAM_SLOT_BYTES if set, else what llama.cpp's tuned -ot puts on the GPU (stage llama-fit), else 5.5 GB
vram_bytes="${VRAM_SLOT_BYTES:-5500000000}"
# OT_FIT: the tuned llama.cpp -ot when the llama-fit stage is skipped
ot_fit="${OT_FIT:-blk\.[0-3]\.ffn_.*_exps=CUDA0;exps=CPU}"
ik_bench="${IK_BENCH:-/opt/ik_llama/bin/llama-bench}"
bench="${LLAMA_BENCH:-/opt/llama.cpp/bin/llama-bench}"
# natural text for the prompt (random token ids spread routing pessimistically), frozen: it was built from the
# checkout's AGENTS.md and ADRs, so every ADR edit changed the 32K prompt (ADR 0038, Results). Copied for the scripts
# that read /root/exit_prompt.txt (Colibri, KTransformers).
prompt=/root/exit_prompt.txt
cp bench/prompts/exit_prompt.txt "$prompt"
rc=0
# STAGES (space list of stage names, default all) lets a relaunch skip stages that already ran
want() { [ -z "${STAGES:-}" ] || [[ " $STAGES " == *" $1 "* ]]; }
# KTransformers installs while the model downloads (its own log and lock)
if want ktransformers; then
  bash scripts/ktransformers_build.sh 8>&- &
  kt_build_pid=$!
fi
# Colibri builds while the model downloads (its own log and lock; fd 8 is this script's lock)
if want colibri-short || want colibri-deep; then
  bash scripts/colibri_build.sh 8>&- &
  colibri_build_pid=$!
fi
{
  echo "[$(date +%T)] start: threads $threads, memory $mem B (balloon ${balloon} B), ram tier $ram_bytes B, vram slots $vram_bytes B"
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active --format=csv
  # build while the model downloads
  cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda --test exit_regime --no-run || rc=21
  if want gpu-tests; then
    # GPU checks on the synthetic fixtures (no model needed): logits against RAM slots and prefetch (ADR 0045)
    echo "[$(date +%T)] start gpu-tests"
    timeout -k 30 20m cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda --test cpu_experts       -- gpu_ --nocapture --test-threads 1 > "$date_dir/gpu-tests-$stamp.log" 2>&1
    echo "[$(date +%T)] done gpu-tests rc=$?"
    tail -20 "$date_dir/gpu-tests-$stamp.log"
  fi
  echo "[$(date +%T)] waiting for the model"
  for _ in $(seq 1 1440); do grep -q FETCH_DONE runs/vast/fetch_exit.log 2> /dev/null && break; sleep 5; done
  grep FETCH_DONE runs/vast/fetch_exit.log | tail -1
  grep -q "FETCH_DONE rc=0" runs/vast/fetch_exit.log || { echo "model fetch failed"; rc=20; }
} >> "$log" 2>&1

# exit_regime <name> <minutes> [ENV=VALUE...]: one harness run, JSON to exit-<name>-<stamp>.json
exit_regime() {
  local name=$1 min=$2
  shift 2
  want "$name" || return 0
  # RANDOM_PROMPT=1: pseudo-random token ids instead of the prompt text, as llama-bench uses
  local text=(TETRAEDER_EXIT_PROMPT_FILE="$prompt")
  [[ " $* " == *" RANDOM_PROMPT=1 "* ]] && text=()
  echo "[$(date +%T)] start $name"
  env "${text[@]}" TETRAEDER_EXIT_GGUF="$model" TETRAEDER_EXIT_GPU=1 TETRAEDER_EXIT_THREADS="$threads" \
    TETRAEDER_EXIT_VRAM_SLOT_BYTES="$vram_bytes" TETRAEDER_EXIT_RAM_SLOT_BYTES="$ram_bytes" \
    TETRAEDER_EXIT_OUT="$date_dir/exit-$name-$stamp.json" \
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

# colibri <short|deep>: the Colibri baseline (scripts/colibri_exit.sh), same model, threads and RAM as the builder
colibri() {
  want "colibri-$1" || return 0
  [ -n "${colibri_build_pid:-}" ] && wait "$colibri_build_pid"
  colibri_build_pid=
  echo "[$(date +%T)] start colibri-$1"
  bash scripts/colibri_exit.sh "$model" "$threads" "$ram_bytes" "$date_dir/exit-colibri-$1-$stamp.json" "$1" 8>&-
  echo "[$(date +%T)] done colibri-$1 rc=$?"
}

# ktransformers: the KTransformers baseline (scripts/ktransformers_exit.sh), deep profile, same threads and RAM
ktransformers() {
  want ktransformers || return 0
  [ -n "${kt_build_pid:-}" ] && wait "$kt_build_pid"
  kt_build_pid=
  echo "[$(date +%T)] start ktransformers"
  bash scripts/ktransformers_exit.sh "$model" "$threads" "$ram_bytes" "$date_dir/exit-ktransformers-$stamp.json" deep 8>&-
  echo "[$(date +%T)] done ktransformers rc=$?"
}

# llama_fit: the most layers' experts llama.cpp keeps on the GPU at ubatch 2048 and llama-bench's context for
# -p 2048 -n 32 -d 32768 (34848; at 33024 eight layers fit llama-server but not llama-bench) (scripts/llama_fit.sh); the
# builder then gets the same expert bytes as VRAM slots (from the smoke run's per-layer expert sizes), so both engines
# use the same VRAM for experts
llama_fit() {
  want llama-fit || return 0
  echo "[$(date +%T)] start llama-fit"
  local f="$date_dir/exit-llama-fit-$stamp.json"
  FIT_MAX="${FIT_MAX:-9}" bash scripts/llama_fit.sh "$model" "$threads" 2048 34848 "$f" || return 0
  local n
  n=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['layers'])" "$f") || return 0
  [ "$n" -ge 0 ] || return 0
  ot_fit=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['ot'])" "$f")
  if [ -z "${VRAM_SLOT_BYTES:-}" ] && [ "$n" -gt 0 ]; then
    vram_bytes=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(sum(d['expert_bytes_per_layer'][:int(sys.argv[2])]) * int(sys.argv[3]))" "$(ls -t "$date_dir"/exit-smoke-*.json | head -1)" "$n" 128)
  fi
  echo "[$(date +%T)] done llama-fit: $n layers, -ot $ot_fit, builder vram slots $vram_bytes B"
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
      llama_fit
      # ubatch 2048: a pair group passes one row per (token, expert) pair to CUDA's mul_mat_id, whose id helper
      # needs 4 B of shared memory per row (about 25k rows on Ampere); ubatch 4096 at top-8 is 32k (abort, 2026-11-02)
      deep=(TETRAEDER_EXIT_PREFILL=32768 TETRAEDER_EXIT_UBATCH=2048 TETRAEDER_EXIT_DECODE=64 TETRAEDER_EXIT_CTX=33024
        TETRAEDER_EXIT_OP_OFFLOAD=1)
      # logit drift against libllama at 2K (ADR 0045): prefetch off and on, a smaller RAM tier, CUDA fusion off
      short=(TETRAEDER_EXIT_PREFILL=2048 TETRAEDER_EXIT_UBATCH=2048 TETRAEDER_EXIT_DECODE=8 TETRAEDER_EXIT_CTX=4096
        TETRAEDER_EXIT_OP_OFFLOAD=1)
      exit_regime drift-ref 15 "${short[@]}" TETRAEDER_EXIT_ENGINE=libllama TETRAEDER_EXIT_REF_OT="$ot_fit"
      exit_regime drift-off 15 "${short[@]}" TETRAEDER_EXIT_PREFETCH=0
      exit_regime drift-on 15 "${short[@]}" TETRAEDER_EXIT_PREFETCH=1
      exit_regime drift-small 15 "${short[@]}" TETRAEDER_EXIT_PREFETCH=0 TETRAEDER_EXIT_RAM_SLOT_BYTES=30000000000
      exit_regime drift-nooffload 15 "${short[@]}" TETRAEDER_EXIT_PREFETCH=0 TETRAEDER_EXIT_OP_OFFLOAD=0
      exit_regime drift-nofuse 15 "${short[@]}" TETRAEDER_EXIT_PREFETCH=0 GGML_CUDA_DISABLE_FUSION=1
      # libllama's own sensitivity to how the same tokens are batched: the yardstick for the builder's drift
      exit_regime drift-ref-ub512 15 "${short[@]}" TETRAEDER_EXIT_UBATCH=512 TETRAEDER_EXIT_ENGINE=libllama         TETRAEDER_EXIT_REF_OT="$ot_fit"
      exit_regime drift-ref-nofuse 15 "${short[@]}" TETRAEDER_EXIT_ENGINE=libllama TETRAEDER_EXIT_REF_OT="$ot_fit"         GGML_CUDA_DISABLE_FUSION=1
      # prefetch off and on (ADR 0038), paired
      exit_regime main 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=0 || rc=23
      exit_regime main-prefetch 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 || rc=26
      # the same on random token ids, the pair of llama-bench (random ids route differently from text)
      exit_regime main-prefetch-random 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 RANDOM_PROMPT=1 || rc=26
      # VARIANTS: extra builder runs at the same depth, "name:ENV=V,ENV=V name2:..." (stage names exit-<name>)
      for v in ${VARIANTS:-}; do
        IFS=',' read -ra extra <<< "${v#*:}"
        exit_regime "${v%%:*}" 60 "${deep[@]}" "${extra[@]}" || rc=27
      done
      # llama.cpp on the same prompt tokens through libllama (exit_regime's reference engine): logit parity of the
      # builder at 32K (greedy, first-token top-10, margins) and the tuned -ot baseline on real text
      exit_regime reference 60 "${deep[@]}" TETRAEDER_EXIT_ENGINE=libllama TETRAEDER_EXIT_REF_OT="$ot_fit" || rc=28
      # KTransformers, then the default builder again: a second pair for the noise band
      ktransformers
      exit_regime main-repeat 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 || rc=26
      # llama-bench, tuned -ot (random tokens; llama-bench takes ',' as alternative configs, so overrides in one config
      # are joined with ';'), then ik_llama.cpp with the same overrides (it was OOM-killed at 128 GB with mmap)
      baseline "$bench" exit-llama-ot-fit-d32k 40 -ub 2048 -p 2048 -n 32 -d 32768 -ot "$ot_fit"
      # (ik's llama-bench has no -d: -gp 32768,32 is decode after a 32K prompt)
      # (and its -ot takes ',' between overrides)
      baseline "$ik_bench" exit-ik-ot-fit-d32k 40 -ub 2048 -p 2048 -n 0 -gp 32768,32 -ot "${ot_fit//;/,}"
      # other tracks' stages (speculative decoding) go here
      # Colibri last; 32K deep is out of reach (2046 tokens took 895 s), so only the short profile by default
      colibri short
      [ -n "${STAGES:-}" ] && colibri deep
    fi
  } >> "$log" 2>&1
fi
touch "$date_dir/DONE"
echo "EXIT_RUN_DONE rc=$rc" >> "$log"
command -v sccache > /dev/null 2>&1 && timeout 30 sccache --stop-server > /dev/null 2>&1
