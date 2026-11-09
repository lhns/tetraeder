#!/usr/bin/env bash
# Phase 3a detector box runs (ADR 0041), runs ON the box, started detached (taxameter run).
# The own MoE builder (tests/exit_regime.rs) on Qwen3-30B-A3B Q4_K_M with VRAM slots, a small RAM tier and misses read
# from disk, decode in windows, a host sampler next to it (scripts/detector_sample.py). Runs:
#   3 calibration runs without a fault (expected rates and band), then ROUNDS (10) rounds of no fault, CPU co-tenant,
#   disk contention or throttle, VRAM pressure in a shuffled order per round, then PAIR_ROUNDS (3) rounds of the three
#   two-fault pairs. The fault label goes only to manifest.jsonl, never into an observation.
# Results: bench/results/<date>/detector/ (obs-NN.json, harness-NN.json, samples-NN.jsonl, calib.json, manifest.jsonl).
# Log runs/vast/detector_campaign.log, ends with DETECTOR_DONE rc=N. START resumes at a run number (default: after the
# last run with an observation, so a relaunch resumes). DEADLINE (epoch seconds, scripts/session.sh): no run starts
# later than RUN_MIN (default 3) minutes before it.
exec 8> /root/tetraeder_detector_campaign.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
mkdir -p runs/vast
log=runs/vast/detector_campaign.log
out="$PWD/bench/results/${RESULTS_DATE:-$(date +%Y%m%d)}/detector"
mkdir -p "$out"
models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
gguf="$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"
planned_slot_bytes="${SLOT_BYTES:-8000000000}"
rc=0
fault_pids=()
stop_faults() {
  for p in "${fault_pids[@]}"; do kill "$p" 2> /dev/null; done
  for p in "${fault_pids[@]}"; do wait "$p" 2> /dev/null; done
  fault_pids=()
  [ -n "${io_max_set:-}" ] && echo "$disk_dev rbps=max" > /sys/fs/cgroup/io.max 2> /dev/null
  io_max_set=
}
finish() {
  stop_faults
  echo "[$(date +%T)] DETECTOR_DONE rc=$rc"
  echo "rc=$rc" > "$out/../DONE"
  command -v sccache > /dev/null 2>&1 && timeout 30 sccache --stop-server > /dev/null 2>&1
}
trap finish EXIT
exec >> "$log" 2>&1

if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
echo "[$(date +%T)] start: cuda $CUDA_ENV_STATUS, threads $threads"
[ "$CUDA_ENV_STATUS" = fail ] && { rc=13; exit; }
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,clocks.sm,clocks_throttle_reasons.active --format=csv

if [ ! -s "$gguf" ]; then
  (TAXAMETER_DIR="$PWD" bash "${TAXAMETER_HOME:-/root/.taxameter}/box.sh" models \
    "Qwen/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf"; echo "[$(date +%T)] fetch rc=$?") 8>&- &
  fetch_pid=$!
fi
echo "[$(date +%T)] start build"
bin=$(timeout -k 30 40m cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda --test exit_regime \
  --no-run --message-format=json 8>&- | python3 -c '
import json, sys
for l in sys.stdin:
    try: m = json.loads(l)
    except ValueError: continue
    if m.get("reason") == "compiler-artifact" and m.get("target", {}).get("name") == "exit_regime" and m.get("executable"):
        print(m["executable"])')
[ -x "$bin" ] || { echo "build failed"; rc=10; exit; }
echo "[$(date +%T)] done build: $bin"
[ -n "${fetch_pid:-}" ] && wait "$fetch_pid"
[ -s "$gguf" ] || { echo "no model"; rc=20; exit; }

# the block device the model's direct reads hit: the one whose read counter moves most while reading 2 GB of it
python3 - "$gguf" > "$out/disk_dev.txt" << 'EOF'
import subprocess, sys
def snap():
    return {f"{l.split()[0]}:{l.split()[1]}": int(l.split()[5]) for l in open("/proc/diskstats") if len(l.split()) > 5}
a = snap()
subprocess.run(["dd", f"if={sys.argv[1]}", "of=/dev/null", "bs=8M", "count=256", "iflag=direct"], capture_output=True)
b = snap()
print(max(b, key=lambda k: b[k] - a.get(k, 0)))
EOF
disk_dev=$(cat "$out/disk_dev.txt")
echo "[$(date +%T)] disk $disk_dev, io.max writable: $( [ -w /sys/fs/cgroup/io.max ] && echo yes || echo no)"
cat AGENTS.md docs/adr/*.md > /root/detector_prompt.txt

# start_fault <name>: cpu, disk or vram
start_fault() {
  case "$1" in
    cpu)
      for _ in $(seq 1 $(( (threads + 1) / 2 ))); do python3 -c 'while True: pass' 8>&- & fault_pids+=($!); done ;;
    disk)
      rate=$(python3 -c "import json; print(int(json.load(open('$out/calib.json'))['rates']['disk:nvme0'] * 0.3))" 2> /dev/null)
      if [ -n "$rate" ] && echo "$disk_dev rbps=$rate" > /sys/fs/cgroup/io.max 2> /dev/null; then
        io_max_set=1
      else
        size=$(stat -c %s "$gguf")
        for k in 0 1 2 3; do
          (while true; do dd if="$gguf" of=/dev/null bs=8M iflag=direct skip=$(( (size / 8388608) * k / 4 )) \
            count=200 2> /dev/null; done) 8>&- &
          fault_pids+=($!)
        done
      fi ;;
    vram)
      python3 scripts/vram_hold.py --leave-free-mib "${PRESSURE_FREE_MIB:-10240}" > "$out/vram_hold.log" 2>&1 8>&- &
      fault_pids+=($!)
      for _ in $(seq 1 60); do grep -q holding "$out/vram_hold.log" && break; sleep 1; done ;;
  esac
}

# run_one <n> <label> [faults...]
run_one() {
  local n=$1 label=$2
  shift 2
  stop_faults
  for f in "$@"; do start_fault "$f"; done
  sleep 3
  local used_mib free_mib slot
  used_mib=$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  free_mib=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  # the slot budget the engine would size from free VRAM: at most the planned budget, 6 GB kept for the rest
  slot=$(( free_mib * 1048576 - 6000000000 ))
  [ "$slot" -gt "$planned_slot_bytes" ] && slot=$planned_slot_bytes
  echo "[$(date +%T)] run $n $label: vram used ${used_mib} MiB, slot bytes $slot"
  env TETRAEDER_EXIT_GGUF="$gguf" TETRAEDER_EXIT_GPU=1 TETRAEDER_EXIT_THREADS="$threads" \
    TETRAEDER_EXIT_VRAM_SLOT_BYTES="$slot" TETRAEDER_EXIT_RAM_SLOT_BYTES="${RAM_SLOT_BYTES:-2000000000}" \
    TETRAEDER_EXIT_PREFILL=256 TETRAEDER_EXIT_UBATCH=256 TETRAEDER_EXIT_DECODE="${DECODE:-96}" \
    TETRAEDER_EXIT_DECODE_WINDOWS=8 TETRAEDER_EXIT_OP_OFFLOAD=1 TETRAEDER_EXIT_PROMPT_FILE=/root/detector_prompt.txt \
    TETRAEDER_EXIT_OUT="$out/harness-$n.json" \
    timeout -k 10 10m "$bin" --ignored --exact exit_regime --nocapture > "$out/run-$n.log" 2>&1 8>&- &
  local pid=$!
  python3 scripts/detector_sample.py sample --pid "$pid" --disk "$disk_dev" --out "$out/samples-$n.jsonl" 8>&- &
  local spid=$!
  wait "$pid"
  local r=$?
  wait "$spid"
  stop_faults
  if [ "$r" -ne 0 ] || [ ! -s "$out/harness-$n.json" ]; then
    echo "[$(date +%T)] run $n failed rc=$r"; tail -5 "$out/run-$n.log"; return 1
  fi
  local calib_arg=()
  [ -s "$out/calib.json" ] && calib_arg=(--calib "$out/calib.json")
  python3 scripts/detector_sample.py observe --harness "$out/harness-$n.json" --samples "$out/samples-$n.jsonl" \
    --threads "$threads" --quota "$threads" --vram-other-bytes $(( used_mib * 1048576 )) \
    --planned-slot-bytes "$planned_slot_bytes" "${calib_arg[@]}" --id "run-$n" --out "$out/obs-$n.json" || return 1
  python3 -c "import json,sys; print(json.dumps({'id': 'run-$n', 'label': '$label', 'faults': sys.argv[1:]}))" "$@" \
    >> "$out/manifest.jsonl"
  echo "[$(date +%T)] run $n done: $(python3 -c "import json; h=json.load(open('$out/harness-$n.json'))['runs'][-1]; print('decode', round(h['decode_tok_s'], 2), 'tok/s, hit', round(h['decode_layers']['full_hit_rate_mean'], 3))")"
}

# deterministic schedule
python3 - "${ROUNDS:-10}" "${PAIR_ROUNDS:-3}" > "$out/schedule.txt" << 'EOF'
import random, sys
r = random.Random(20261103)
print("calib"); print("calib"); print("calib")
for _ in range(int(sys.argv[1])):
    s = ["none", "cpu", "disk", "vram"]; r.shuffle(s); print("\n".join(s))
for _ in range(int(sys.argv[2])):
    s = ["cpu+disk", "cpu+vram", "disk+vram"]; r.shuffle(s); print("\n".join(s))
EOF
if [ -z "${START:-}" ]; then
  last=$(find "$out" -maxdepth 1 -name 'obs-[0-9]*.json' -printf '%f\n' 2> /dev/null | tr -dc '0-9\n' | sort -n | tail -1)
  START=$(( ${last:+10#$last + 1} + 0 ))
  [ "$START" -gt 0 ] && echo "[$(date +%T)] resume at run $START"
fi
n=0
calib_obs=()
while read -r label; do
  if [ -n "${DEADLINE:-}" ] && [ $(($(date +%s) + ${RUN_MIN:-3} * 60)) -gt "$DEADLINE" ]; then
    echo "[$(date +%T)] deadline: stop before run $n"; break
  fi
  if [ "$n" -lt "$START" ]; then
    [ "$label" = calib ] && calib_obs+=("$out/obs-$(printf %02d "$n").json")
    n=$((n + 1)); continue
  fi
  case "$label" in
    calib | none) faults=() ;;
    *) IFS=+ read -ra faults <<< "$label" ;;
  esac
  run_one "$(printf %02d "$n")" "$label" "${faults[@]}" || rc=2
  [ "$label" = calib ] && calib_obs+=("$out/obs-$(printf %02d "$n").json")
  n=$((n + 1))
  if [ "$label" = calib ] && [ "${#calib_obs[@]}" -eq 3 ] && [ ! -s "$out/calib.json" ]; then
    python3 scripts/detector_sample.py calib --obs "${calib_obs[@]}" --out "$out/calib.json" && cat "$out/calib.json"
  fi
done < "$out/schedule.txt"
