#!/usr/bin/env bash
# Is the ADR 0038 prefetch output-neutral on the 235B? Runs tests/exit_regime.rs with prefetch off and on, in pairs:
# an 8-token prefill without op offload (one CPU group per layer, no GPU matmuls of RAM experts), then a 2048-token
# prefill with op offload. Greedy tokens must match within each pair. Waits for scripts/exit_box.sh to finish; same
# model, threads and tiers as scripts/exit_run.sh (no balloon: for boxes whose container limit is the RAM to test).
# Started detached by `taxameter run scripts/prefetch_eq.sh`; ends with PREFETCH_EQ_DONE in runs/vast/prefetch_eq.log.
exec 8> /root/tetraeder_prefetch_eq.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
bash scripts/ensure_patches.sh 8>&- || exit 1  # ggml patch queue (ADR 0052)
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
# this box's GPU only, as exit_run.sh (same build, no ggml-cuda rebuild for every arch)
if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
log=runs/vast/prefetch_eq.log
date_dir="$PWD/bench/results/$(date +%Y%m%d)"
mkdir -p "$date_dir" runs/vast
stamp=$(date +%H%M%S)
model="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}/Qwen__Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf"
threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"
total=$(awk '/MemTotal/ { print $2 * 1024 }' /proc/meminfo)
mem=$(cat /sys/fs/cgroup/memory.max 2> /dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2> /dev/null)
case "$mem" in '' | max) mem=$total ;; esac
[ "$mem" -gt "$total" ] && mem=$total
ram_bytes=$((mem - ${RAM_RESERVE_GB:-20} * 1000000000))
prompt=/root/exit_prompt.txt
cp bench/prompts/exit_prompt.txt "$prompt"  # frozen, as in exit_run.sh

run() {
  local name=$1
  shift
  echo "[$(date +%T)] start $name"
  env TETRAEDER_EXIT_GGUF="$model" TETRAEDER_EXIT_GPU=1 TETRAEDER_EXIT_THREADS="$threads" \
    TETRAEDER_EXIT_VRAM_SLOT_BYTES="${VRAM_SLOT_BYTES:-5500000000}" TETRAEDER_EXIT_RAM_SLOT_BYTES="$ram_bytes" \
    TETRAEDER_EXIT_PROMPT_FILE="$prompt" TETRAEDER_EXIT_OUT="$date_dir/eq-$name-$stamp.json" \
    TETRAEDER_EXIT_MACHINE="vast $(hostname)" "$@" \
    timeout -k 30 15m cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda \
    --test exit_regime -- --ignored --nocapture
  echo "[$(date +%T)] done $name rc=$?"
}

{
  for _ in $(seq 1 2160); do grep -q EXIT_BOX_DONE runs/vast/exit_box.log 2> /dev/null && break; sleep 5; done
  if [ -n "${EQ_NAME:-}" ]; then
    # one extra pair: EQ_NAME, EQ_PREFILL, EQ_UBATCH, EQ_DECODE, EQ_CTX, EQ_OFFLOAD
    for p in 0 1; do
      run "$EQ_NAME-p$p" TETRAEDER_EXIT_PREFILL="$EQ_PREFILL" TETRAEDER_EXIT_UBATCH="$EQ_UBATCH"         TETRAEDER_EXIT_DECODE="$EQ_DECODE" TETRAEDER_EXIT_CTX="$EQ_CTX" TETRAEDER_EXIT_OP_OFFLOAD="$EQ_OFFLOAD"         TETRAEDER_EXIT_PREFETCH=$p
    done
    pairs=$EQ_NAME
  else
  pairs="short ub"
  for p in 0 1; do
    run "short-p$p" TETRAEDER_EXIT_PREFILL=8 TETRAEDER_EXIT_UBATCH=8 TETRAEDER_EXIT_DECODE=32 \
      TETRAEDER_EXIT_CTX=2048 TETRAEDER_EXIT_OP_OFFLOAD=0 TETRAEDER_EXIT_PREFETCH=$p
  done
  for p in 0 1; do
    run "ub-p$p" TETRAEDER_EXIT_PREFILL=2048 TETRAEDER_EXIT_UBATCH=2048 TETRAEDER_EXIT_DECODE=8 \
      TETRAEDER_EXIT_CTX=4096 TETRAEDER_EXIT_OP_OFFLOAD=1 TETRAEDER_EXIT_PREFETCH=$p
  done
  fi
  # shellcheck disable=SC2086 # one argument per pair
  python3 - "$date_dir" "$stamp" $pairs << 'EOF'
import json, sys
d, s = sys.argv[1:3]
for pair in sys.argv[3:]:
    g = []
    for p in (0, 1):
        try:
            g.append(json.load(open(f"{d}/eq-{pair}-p{p}-{s}.json"))["runs"][0]["greedy"])
        except Exception as e:
            g.append(f"missing: {e}")
    print(pair, "EQUAL" if g[0] == g[1] else "DIFFERENT", g[0], g[1])
EOF
  touch "$date_dir/DONE"
  echo "PREFETCH_EQ_DONE"
} >> "$log" 2>&1
