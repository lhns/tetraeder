#!/usr/bin/env bash
# Nightly fault testbed on a GitHub CPU runner (ADR 0041 addendum, docs/testing.md section "Fault testbed").
# The real engine (tests/exit_regime.rs, CPU backend, synthetic MoE, small RAM tier so misses read from disk) in decode
# windows, scripts/detector_sample.py sampling next to it, and three real faults:
#   cpu   thread-capped busy loops next to the engine               -> FaultClass::CpuCotenant
#   disk  systemd-run scope with IOReadBandwidthMax (cgroup io.max) -> FaultClass::DiskSlow
#   mem   systemd-run scope with MemoryMax below the engine's RSS   -> FaultClass::Swapping (manifest name "mem")
# Same schedule shape and manifest format as scripts/detector_campaign.sh, so `tetraeder detector-score` reads the output.
# Needs Linux with sudo, systemd, cgroup v2 and GNU time. VRAM pressure, PCIe and GPU throttle stay box-only.
# Env: OUT (results dir), ROUNDS (6) rounds of none,cpu,disk,mem, PAIR_ROUNDS (2) rounds of the three pairs,
#      MEM_FRAC (0.4) MemoryMax as a fraction of the calibration peak RSS, DECODE (64), THREADS (nproc),
#      BIN (a built exit_regime test binary; default: build it).
# Results: $OUT/{obs,harness,samples,run,scope,rss}-NN, calib.json, manifest.jsonl, schedule.txt.
# Exit code: 0 unless the build, the setup or a run failed (the detector bars are not this script's concern).
cd "$(dirname "$0")/.." || exit 1
out="${OUT:-$PWD/bench/results/$(date +%Y%m%d)/detector-testbed}"
mkdir -p "$out"
work="${WORK:-$(mktemp -d)}"
threads="${THREADS:-$(nproc)}"
rc=0
fault_pids=()
stop_faults() {
  for p in "${fault_pids[@]}"; do kill "$p" 2> /dev/null; done
  for p in "${fault_pids[@]}"; do wait "$p" 2> /dev/null; done
  fault_pids=()
}
trap stop_faults EXIT
log() { echo "[$(date +%T)] $*"; }

bin="${BIN:-}"
if [ -z "$bin" ]; then
  log "build"
  bin=$(cargo test --release -p tetraeder-model --test exit_regime --no-run --message-format=json | python3 -c '
import json, sys
for l in sys.stdin:
    try: m = json.loads(l)
    except ValueError: continue
    if m.get("reason") == "compiler-artifact" and m.get("target", {}).get("name") == "exit_regime" and m.get("executable"):
        print(m["executable"])')
fi
[ -x "$bin" ] || { log "no exit_regime binary"; exit 10; }

# Generate the synthetic model once (unprivileged, in $work); every run reads the same file.
log "model"
TMPDIR="$work" TETRAEDER_EXIT_THREADS="$threads" TETRAEDER_EXIT_PREFILL=8 TETRAEDER_EXIT_UBATCH=8 TETRAEDER_EXIT_DECODE=2 \
  "$bin" --ignored --exact exit_regime --nocapture > "$out/model-gen.log" 2>&1 \
  || { log "model generation failed"; tail -5 "$out/model-gen.log"; exit 11; }
gguf=$(find "$work" -name "tetraeder-exit-*.gguf" | head -1)
[ -s "$gguf" ] || { log "no model file"; exit 11; }

# The whole disk behind the model (its read counter moves most while reading the file): io.max works on whole disks.
read -r disk_mm disk_node < <(python3 - "$gguf" << 'EOF'
import os, subprocess, sys
def snap():
    return {f"{l.split()[0]}:{l.split()[1]}": int(l.split()[5]) for l in open("/proc/diskstats") if len(l.split()) > 5}
a = snap()
subprocess.run(["dd", f"if={sys.argv[1]}", "of=/dev/null", "bs=8M", "iflag=direct"], capture_output=True)
b = snap()
mm = max(b, key=lambda k: b[k] - a.get(k, 0))
p = os.path.realpath(f"/sys/dev/block/{mm}")
if os.path.exists(p + "/partition"):
    p = os.path.dirname(p)
print(open(p + "/dev").read().strip(), "/dev/" + os.path.basename(p))
EOF
)
[ -n "$disk_node" ] || { log "no disk device"; exit 12; }
echo "$disk_mm $disk_node" > "$out/disk_dev.txt"
log "disk $disk_node ($disk_mm), threads $threads"
if [ -z "$(swapon --noheadings --show 2> /dev/null)" ]; then
  sudo fallocate -l 2G /swapfile-testbed && sudo chmod 600 /swapfile-testbed && sudo mkswap /swapfile-testbed > /dev/null \
    && sudo swapon /swapfile-testbed && log "swap added"
fi
log "swap: $(swapon --noheadings --show 2> /dev/null | tr -s ' ' | tr '\n' ';')"
cat AGENTS.md docs/adr/*.md > "$work/prompt.txt"

# Runs inside the scope: records the scope's limits, then the engine under GNU time (peak RSS in KB).
cat > "$work/scope_wrap.sh" << 'EOF'
#!/usr/bin/env bash
# scope_wrap.sh <info-file> <rss-file> cmd...
cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
{ echo "io.max: $(cat "$cg/io.max" 2> /dev/null)"; echo "memory.max: $(cat "$cg/memory.max" 2> /dev/null)"; } > "$1"
rss=$2
shift 2
exec /usr/bin/time -f %M -o "$rss" timeout -k 10 10m "$@"
EOF
chmod +x "$work/scope_wrap.sh"

# Does the runner enforce IOReadBandwidthMax? Otherwise the disk fault is concurrent direct readers, as on a box.
disk_mode=readers
# shellcheck disable=SC2024  # the redirect is meant to run as the invoking user
sudo systemd-run --scope -q -p "IOReadBandwidthMax=$disk_node 100000000" bash -c \
  'cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/io.max' > "$work/probe.txt" 2>&1
grep -q rbps=100000000 "$work/probe.txt" && disk_mode=scope
log "disk fault via $disk_mode (probe: $(tr '\n' ' ' < "$work/probe.txt"))"

# run_one <n> <label> [faults...]
run_one() {
  local n=$1 label=$2 f k rate memmax props=()
  shift 2
  stop_faults
  for f in "$@"; do
    case "$f" in
      cpu)
        for _ in $(seq 1 $(((threads + 1) / 2))); do python3 -c 'while True: pass' & fault_pids+=($!); done ;;
      disk)
        rate=$(python3 -c "import json; print(int(json.load(open('$out/calib.json'))['rates']['disk:nvme0'] * 0.3))")
        if [ "$disk_mode" = scope ]; then
          props+=(-p "IOReadBandwidthMax=$disk_node $rate")
        else
          for k in 0 1 2 3; do
            (while true; do dd if="$gguf" of=/dev/null bs=8M iflag=direct skip=$((k * 10)) count=10 2> /dev/null; done) &
            fault_pids+=($!)
          done
        fi ;;
      mem)
        memmax=$(python3 -c "
import glob, statistics
rss = [int(open(p).read().split()[-1]) * 1024 for p in glob.glob('$out/rss-*-calib.txt')]
print(int(statistics.median(rss) * ${MEM_FRAC:-0.4}))")
        props+=(-p "MemoryMax=$memmax") ;;
    esac
  done
  sleep 2
  log "run $n $label: props [${props[*]}]"
  # Every run, faulted or not, goes through a scope, so the measurement path is the same.
  # shellcheck disable=SC2024
  sudo env TETRAEDER_EXIT_GGUF="$gguf" TETRAEDER_EXIT_THREADS="$threads" TETRAEDER_EXIT_RAM_SLOTS=4 \
    TETRAEDER_EXIT_PREFILL=256 TETRAEDER_EXIT_UBATCH=256 TETRAEDER_EXIT_DECODE="${DECODE:-64}" \
    TETRAEDER_EXIT_DECODE_WINDOWS=8 TETRAEDER_EXIT_PROMPT_FILE="$work/prompt.txt" \
    TETRAEDER_EXIT_OUT="$out/harness-$n.json" \
    systemd-run --scope -q "${props[@]}" -- "$work/scope_wrap.sh" "$out/scope-$n.txt" "$out/rss-$n-$label.txt" \
    "$bin" --ignored --exact exit_regime --nocapture > "$out/run-$n.log" 2>&1 &
  local pid=$!
  # As root: the engine runs as root and /proc/PID/io of its tree is not readable otherwise.
  sudo python3 scripts/detector_sample.py sample --pid "$pid" --disk "$disk_mm" --out "$out/samples-$n.jsonl" &
  local spid=$!
  wait "$pid"
  local r=$?
  wait "$spid"
  stop_faults
  if [ "$r" -ne 0 ] || [ ! -s "$out/harness-$n.json" ]; then
    log "run $n failed rc=$r"; tail -5 "$out/run-$n.log"; return 1
  fi
  # The io.max the engine ran under comes from its scope (the root cgroup has none on a runner).
  local io_arg=() io_max calib_arg=()
  io_max=$(sed -n 's/.*rbps=\([0-9]*\).*/\1/p' "$out/scope-$n.txt" | head -1)
  [ -n "$io_max" ] && io_arg=(--io-max-bytes "$io_max")
  [ -s "$out/calib.json" ] && calib_arg=(--calib "$out/calib.json")
  python3 scripts/detector_sample.py observe --harness "$out/harness-$n.json" --samples "$out/samples-$n.jsonl" \
    --threads "$threads" --quota "$threads" "${calib_arg[@]}" "${io_arg[@]}" --id "run-$n" --out "$out/obs-$n.json" || return 1
  python3 -c "import json,sys; print(json.dumps({'id': 'run-$n', 'label': '$label', 'faults': sys.argv[1:]}))" "$@" \
    >> "$out/manifest.jsonl"
  log "run $n done: $(python3 -c "import json; h=json.load(open('$out/harness-$n.json'))['runs'][-1]; print('decode', round(h['decode_tok_s'], 2), 'tok/s, hit', round(h['decode_layers']['full_hit_rate_mean'], 3))")"
}

# Deterministic schedule, the shape of detector_campaign.sh with mem in place of vram.
python3 - "${ROUNDS:-6}" "${PAIR_ROUNDS:-2}" > "$out/schedule.txt" << 'EOF'
import random, sys
r = random.Random(20261109)
print("calib"); print("calib"); print("calib")
for _ in range(int(sys.argv[1])):
    s = ["none", "cpu", "disk", "mem"]; r.shuffle(s); print("\n".join(s))
for _ in range(int(sys.argv[2])):
    s = ["cpu+disk", "cpu+mem", "disk+mem"]; r.shuffle(s); print("\n".join(s))
EOF
rm -f "$out/manifest.jsonl"
n=0
calib_obs=()
while read -r label; do
  case "$label" in
    calib | none) faults=() ;;
    *) IFS=+ read -ra faults <<< "$label" ;;
  esac
  run_one "$(printf %02d "$n")" "$label" "${faults[@]}" || rc=2
  [ "$label" = calib ] && calib_obs+=("$out/obs-$(printf %02d "$n").json")
  n=$((n + 1))
  if [ "$label" = calib ] && [ "${#calib_obs[@]}" -eq 3 ]; then
    python3 scripts/detector_sample.py calib --obs "${calib_obs[@]}" --out "$out/calib.json" && cat "$out/calib.json" || rc=3
  fi
done < "$out/schedule.txt"
log "TESTBED_DONE rc=$rc"
exit "$rc"
