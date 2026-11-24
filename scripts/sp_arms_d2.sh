#!/usr/bin/env bash
# Session stream-predict: the measured runs of arm D2 (arm D re-priced after its first pick was OOM-killed; scripts/sp_predict_d2.sh), same harness as sp_arms.sh (a copy: never edit a running script)
# Session stream-predict: the measured runs of both arms (tree node bench-235b-stream-predict). Runs only after the
# predictions are committed: FROZEN_SHA must equal the sha256 in predictions/FROZEN.sha256 and every frozen file must match
# SHA256SUMS. The config of an arm is the planner's pick in predictions/pred-<arm>-uniform.json (`exit_regime_env`).
#   arm D  RAM budget 64 GB through the balloon (EMULATE_RAM_GB=64, as scripts/exit_run.sh), experts streamed from NVMe
#   arm R  no balloon, every expert in the RAM tier, a warm prefill first (TETRAEDER_EXIT_PREFILL_WARM=1) so the timed
#          prefill and decode read nothing from disk
# Workload as ADR 0091 Case S: pp2048 (the frozen exit prompt) then tg128, ROUNDS (3) separate processes per arm.
# Env: RESULTS_NAME, FROZEN_SHA, ARMS ("D R"), ROUNDS, THREADS, EMULATE_RAM_GB (64). Ends with SP_ARMS_D2_DONE in
# runs/vast/sp_arms.log. Result files: bench/results/<name>/arms/exit-<arm>-r<k>.json (+ host block, balloon_ok).
exec 8> /root/tetraeder_sp_arms_d2.lock
flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
root=$PWD
name="${RESULTS_NAME:?set RESULTS_NAME}"
p="$root/bench/results/$name/predictions/d2"
res="$root/bench/results/$name/arms-d2"
mkdir -p "$res" runs/vast
log=runs/vast/sp_arms_d2.log
say() { echo "[$(date +%T)] $*"; }
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:/root/.local/bin:$PATH"
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
[ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
quota=$(bash scripts/cpu_quota.sh 2> /dev/null || nproc)
threads=$(bash scripts/clamp_threads.sh "${THREADS:-$quota}")
rounds="${ROUNDS:-3}"
arms="${ARMS:-D2}"
model="$root/.hf_cache/Qwen__Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf"
prompt=/root/exit_prompt.txt
cp bench/prompts/exit_prompt.txt "$prompt"
total=$(awk '/MemTotal/ { print $2 * 1024 }' /proc/meminfo)
mem=$(cat /sys/fs/cgroup/memory.max 2> /dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2> /dev/null)
case "$mem" in '' | max) mem=$total ;; esac
[ "$mem" -gt "$total" ] && mem=$total
emulated=$((${EMULATE_RAM_GB:-64} * 1000000000))
balloon_pid=""
sampler_pid=""
die() { say "FATAL: $*"; stop_balloon; stop_gpu_sampler; echo "SP_ARMS_D2_DONE rc=1"; exit 1; }

# ---- the freeze gate -----------------------------------------------------------------------------------------------
gate() {
  [ -s "$p/FROZEN.sha256" ] || die "no $p/FROZEN.sha256: the predictions are not written"
  local want
  want=$(awk '{print $1}' "$p/FROZEN.sha256")
  [ "${FROZEN_SHA:-}" = "$want" ] || die "FROZEN_SHA '${FROZEN_SHA:-}' is not the sha256 of SHA256SUMS ($want): commit the predictions first, then pass it"
  (cd "$p" && sha256sum -c FROZEN.sha256 && sha256sum -c SHA256SUMS && cd .. && sha256sum -c FROZEN.sha256 && sha256sum -c SHA256SUMS) > "$res/freeze-check.txt" 2>&1 || die "a frozen file changed (see $res/freeze-check.txt)"
  say "freeze gate ok ($want)"
}

# ---- balloon (as scripts/exit_run.sh) -----------------------------------------------------------------------------
start_balloon() {
  local n=$1
  python3 -c "
import sys, time
n = int(sys.argv[1])
b = bytearray(n)
b[::4096] = b'\x01' * len(range(0, n, 4096))
print('balloon holds', n, 'B', flush=True)
time.sleep(10**9)
" "$n" >> runs/vast/sp_arms_d2.balloon.log 2>&1 8>&- &
  balloon_pid=$!
  balloon_size=$n
  if echo -1000 > /proc/$balloon_pid/oom_score_adj 2> /dev/null; then say "balloon pid $balloon_pid oom_score_adj -1000"; else say "balloon pid $balloon_pid: oom_score_adj -1000 not allowed"; fi
  # touching the pages takes minutes for a balloon of hundreds of GB: wait until it holds 90% of its size
  local i
  for i in $(seq 1 90); do
    sleep 10
    python3 -I scripts/balloon_guard.py check --pid "$balloon_pid" --target "$n" > /dev/null 2>&1 && return 0
  done
  die "the balloon did not reach its size in 15 minutes"
}
stop_balloon() {
  [ -n "$balloon_pid" ] || return 0
  kill "$balloon_pid" 2> /dev/null
  wait "$balloon_pid" 2> /dev/null
  balloon_pid=""
  sleep 5
}
stop_gpu_sampler() { touch "$res/gpu-sampler.stop"; }

# ---- one round ------------------------------------------------------------------------------------------------------
run_round() { # run_round ARM ROUND
  local arm=$1 r=$2 json="$res/exit-$1-r$2.json" samples="$root/runs/vast/samples-sp-$1-r$2.jsonl" t0=$SECONDS
  [ -s "$json" ] && { say "arm $arm round $r exists, skipped"; return 0; }
  local envs=() line warm=()
  while IFS= read -r line; do [ -n "$line" ] && envs+=("$line"); done < <(python3 -I -c "
import json, sys
for k, v in sorted(json.load(open(sys.argv[1]))['exit_regime_env'].items()):
    print(f'{k}={v}')
" "$p/pred-$arm-uniform.json")
    [ -n "${HOST_DISK_DEV:-}" ] || HOST_DISK_DEV=$(python3 scripts/engine_sample.py probe-disk "$model")
  say "arm $arm round $r start: ${envs[*]} ${warm[*]} threads $threads"
  local bpre="ok"
  if [ -n "$balloon_pid" ]; then
    python3 -I scripts/balloon_guard.py check --pid "$balloon_pid" --target "$balloon_size" > "$res/balloon-pre-$arm-r$r.txt" 2>&1 || bpre="dead before the round"
  fi
  nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>&1 | sed 's/^/before: /' > "$res/gpu-$arm-r$r.txt"
  env "${envs[@]}" "${warm[@]}" TETRAEDER_EXIT_GGUF="$model" TETRAEDER_EXIT_PROMPT_FILE="$prompt" \
    TETRAEDER_EXIT_THREADS="$threads" TETRAEDER_EXIT_OUT="$json" TETRAEDER_EXIT_MACHINE="vast $(hostname) stream-predict arm $arm" \
    bash scripts/oom_engine.sh timeout -k 30 40m cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda \
    --test exit_regime -- --ignored --nocapture > "$res/exit-$arm-r$r.log" 2>&1 8>&- &
  local pid=$!
  TETRAEDER_STAGE="sp-$arm-r$r" python3 scripts/engine_sample.py sample --pid "$pid" --disk "$HOST_DISK_DEV" --out "$samples" 8>&- &
  local spid=$!
  wait "$pid"
  local rc=$?
  wait "$spid"
  nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>&1 | sed 's/^/after: /' >> "$res/gpu-$arm-r$r.txt"
  if [ -s "$json" ]; then
    python3 scripts/engine_sample.py summarize --samples "$samples" --threads "$threads" --builder-json "$json" \
      --out "$samples.summary" --merge-into "$json" > /dev/null || say "host summary failed for $arm r$r"
    python3 -I scripts/cotenant_guard.py merge --json "$json" || true
    if [ -n "$balloon_pid" ]; then
      local why ok=true
      python3 -I scripts/balloon_guard.py check --pid "$balloon_pid" --target "$balloon_size" > /dev/null 2>&1 || { ok=false; why="dead or shrunk after the round"; }
      [ "$bpre" = ok ] || { ok=false; why="${why:+$why; }$bpre"; }
      python3 -I scripts/balloon_guard.py merge --json "$json" --ok "$ok" --reason "${why:-}" || true
    fi
    cp "$samples" "$res/" 2> /dev/null
  else
    say "arm $arm round $r: no result JSON (rc $rc); last log lines:"
    tail -n 12 "$res/exit-$arm-r$r.log"
  fi
  say "arm $arm round $r done rc=$rc in $((SECONDS - t0)) s"
}

main() {
  say "sp_arms start arms='$arms' rounds=$rounds threads=$threads mem=$mem"
  gate
  OUT="$res" SAMPLER_PARENT=$$ setsid nohup bash scripts/gpu_sampler.sh 8>&- > /dev/null 2>&1 &
  local arm r
  for arm in $arms; do
    if [ "$arm" = D2 ]; then
      [ "$mem" -gt "$emulated" ] || die "container memory $mem is not above the emulated $emulated"
      start_balloon $((mem - emulated))
    fi
    for r in $(seq 1 "$rounds"); do run_round "$arm" "$r"; done
    stop_balloon
  done
  stop_gpu_sampler
  touch "$res/DONE"
  say "SP_ARMS_D2_DONE rc=0"
}
trap 'stop_balloon' EXIT
main >> "$log" 2>&1
