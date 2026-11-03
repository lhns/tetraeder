#!/usr/bin/env bash
# Phase 3a planner exit-test measurements for the own MoE builder (ADR 0040), runs ON the box, started detached
# (taxameter run). Qwen3-30B-A3B Q4_K_M with part of the VRAM held by a balloon process. The planner enumerates builder
# candidates (expert slot counts per tier, threads, ubatch, op offload) and the builder anchors; each anchor and candidate
# runs REPS (2) times through the exit_regime test binary, repeats interleaved (rep 1 of every run, then rep 2), with
# prompt 2048 and 128 decoded tokens. Anchors come first in each rep.
# Results: bench/results/<date>/planner-moe-builder/ (layout.json, candidates.json, runs.txt, <variant>.r<k>.json,
# log-<variant>.r<k>.txt, fixture.txt, env.txt, DONE). A run that fails (e.g. out of memory) leaves no JSON: the scorer
# counts it as a planner miss.
# Log runs/vast/planner_builder_campaign.log, ends with PLANNER_BUILDER_DONE rc=N.
# START (default 0) resumes at a global run index (a finished run with a result file is skipped anyway);
# a relaunch reuses candidates.json. OUT_DIR overrides the result directory (resume after midnight).
exec 8> /root/tetraeder_planner_builder_campaign.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
mkdir -p runs/vast
log=runs/vast/planner_builder_campaign.log
out="${OUT_DIR:-$PWD/bench/results/$(date +%Y%m%d)/planner-moe-builder}"
mkdir -p "$out"
models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
hf="Qwen/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf"
gguf="$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"
reps="${REPS:-2}"
rc=0
hold_pid=
finish() {
  [ -n "$hold_pid" ] && kill "$hold_pid" 2> /dev/null
  echo "rc=$rc" > "$out/DONE"
  echo "[$(date +%T)] PLANNER_BUILDER_DONE rc=$rc"
  command -v sccache > /dev/null 2>&1 && timeout 30 sccache --stop-server > /dev/null 2>&1
}
trap finish EXIT
exec >> "$log" 2>&1

if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
echo "[$(date +%T)] start: cuda $CUDA_ENV_STATUS, threads $threads, reps $reps, archs ${TETRAEDER_CUDA_ARCHS:-default}"
[ "$CUDA_ENV_STATUS" = fail ] && { rc=13; exit; }
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active --format=csv

if [ ! -s "$gguf" ]; then
  (TAXAMETER_DIR="$PWD" bash "${TAXAMETER_HOME:-/root/.taxameter}/box.sh" models "$hf"; echo "[$(date +%T)] fetch rc=$?") 8>&- &
  fetch_pid=$!
fi
echo "[$(date +%T)] start build"
timeout -k 30 40m cargo build --release -p tetraeder -p tetraeder-planner --features tetraeder-ggml-sys/cuda 8>&- \
  || { echo "build failed"; rc=10; exit; }
bin=$(timeout -k 30 40m cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda --test exit_regime \
  --no-run --message-format=json 8>&- | python3 -c '
import json, sys
for l in sys.stdin:
    try: m = json.loads(l)
    except ValueError: continue
    if m.get("reason") == "compiler-artifact" and m.get("target", {}).get("name") == "exit_regime" and m.get("executable"):
        print(m["executable"])')
[ -x "$bin" ] || { echo "test build failed"; rc=10; exit; }
echo "[$(date +%T)] done build: $bin"
[ -n "${fetch_pid:-}" ] && wait "$fetch_pid"
[ -s "$gguf" ] || { echo "no model"; rc=20; exit; }

# VRAM balloon: the engine gets the fixture's free VRAM
free_target="${FREE_MOE_MIB:-14336}"
python3 scripts/vram_hold.py --leave-free-mib "$free_target" > "$out/vram_hold.log" 2>&1 8>&- &
hold_pid=$!
for _ in $(seq 1 60); do grep -q 'holding' "$out/vram_hold.log" && break; sleep 2; done
cat "$out/vram_hold.log"
free_mib=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
mem=$(cat /sys/fs/cgroup/memory.max 2> /dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2> /dev/null)
total=$(awk '/MemTotal/ { print $2 * 1024 }' /proc/meminfo)
case "$mem" in '' | max) mem=$total ;; esac
[ "$mem" -gt "$total" ] && mem=$total
ram_bytes=$((mem - 8000000000))
echo "free_mib=$free_mib threads=$threads ram_bytes=$ram_bytes reps=$reps" > "$out/fixture.txt"
cat "$out/fixture.txt"
nvidia-smi -q -d CLOCK,PERFORMANCE > "$out/env.txt" 2>&1
uname -a >> "$out/env.txt"

if [ ! -s "$out/candidates.json" ]; then
  target/release/tetraeder plan layout "$gguf" --out "$out/layout.json" || { rc=21; exit; }
  target/release/tetraeder-plan candidates --engine builder --layout "$out/layout.json" \
    --gpu-bytes $((free_mib * 1048576)) --ram-bytes "$ram_bytes" --threads "$threads" --prompt 2048 --gen 128 \
    --model-key moe --hf "$hf" --out-candidates "$out/candidates.json" || { rc=22; exit; }
fi
# runs.txt: <variant id> <TAB> <KEY=VALUE ...>, anchors first
python3 - "$out/candidates.json" > "$out/runs.txt" << 'EOF'
import json, sys
f = json.load(open(sys.argv[1]))
vs = f["anchors"] + [c["variant"] for c in f["candidates"]]
for v in vs:
    print(v["id"] + "\t" + " ".join(f"{k}={x}" for k, x in sorted(v["env"].items())))
EOF
total_runs=$(wc -l < "$out/runs.txt")
echo "[$(date +%T)] $total_runs variants x $reps reps; about 4 min per run, $(( total_runs * reps * 4 / 60 )) h"
cat AGENTS.md docs/adr/*.md > /root/builder_prompt.txt  # prompt text, routed like real text

n=0
for rep in $(seq 1 "$reps"); do
  while IFS=$'\t' read -r id kv; do
    idx=$n
    n=$((n + 1))
    [ "$idx" -lt "${START:-0}" ] && continue
    res="$out/$id.r$rep.json"
    [ -s "$res" ] && { echo "[$(date +%T)] run $idx $id r$rep: have result"; continue; }
    echo "[$(date +%T)] run $idx/$((total_runs * reps)) $id r$rep: $kv"
    # shellcheck disable=SC2086  # kv is a space separated KEY=VALUE list without spaces inside
    env TETRAEDER_EXIT_GGUF="$gguf" TETRAEDER_EXIT_PROMPT_FILE=/root/builder_prompt.txt \
      TETRAEDER_EXIT_PREFILL=2048 TETRAEDER_EXIT_DECODE=128 $kv TETRAEDER_EXIT_OUT="$res" \
      timeout -k 10 "${RUN_TIMEOUT:-900}" "$bin" --ignored --exact exit_regime --nocapture \
      < /dev/null > "$out/log-$id.r$rep.txt" 2>&1 8>&-
    r=$?
    if [ "$r" -ne 0 ] || [ ! -s "$res" ]; then
      echo "[$(date +%T)] run $idx $id r$rep failed rc=$r"; tail -3 "$out/log-$id.r$rep.txt"
      rm -f "$res"; rc=2
    else
      echo "[$(date +%T)] run $idx $id r$rep done: $(python3 -c "import json,sys; r=json.load(open(sys.argv[1]))['runs'][-1]; print('prefill', round(r['prefill_tok_s'], 1), 'decode', round(r['decode_tok_s'], 2), 'tok/s, hit', round(r['decode_layers']['full_hit_rate_mean'], 3))" "$res")"
    fi
  done < "$out/runs.txt"
done
