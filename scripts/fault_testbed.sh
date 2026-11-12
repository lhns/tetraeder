#!/usr/bin/env bash
# Nightly fault testbed on a GitHub CPU runner (ADR 0041 addendum, docs/testing.md section "Fault testbed").
# The real engine (tests/exit_regime.rs, CPU backend, synthetic MoE, small RAM tier so misses read from disk) in decode
# windows, scripts/detector_sample.py sampling next to it, and four real faults:
#   cpu   busy loops next to the engine (half the threads; 1 in pairs) -> FaultClass::CpuCotenant
#   disk  systemd-run scope with IOReadBandwidthMax (cgroup io.max) -> FaultClass::DiskSlow
#         (in a pair the cap shrinks with the partner fault's pace, so it still binds)
#   mem   systemd-run scope with MemoryMax below the engine's RSS   -> FaultClass::Swapping (manifest name "mem")
#   quota systemd-run scope with CPUQuota= at half the threads (min 1 core), ggml polling explicit at 50
#         (TETRAEDER_EXIT_CPU_POLL=50; auto polling is 0 under a quota since ADR 0061, so the fault needs it set)
#         -> FaultClass::CpuQuotaThrottle (ADR 0062; manifest name "quota"). Fallback when the runner does not
#         enforce CPUQuota: a cgroup v2 child with cpu.max; if neither works the quota runs are dropped from the
#         schedule with a logged reason. Needs THREADS >= 2. Single faults only (a pair partner would change the
#         throttled share). QUOTA_FAULT=0 drops it.
# Same schedule shape and manifest format as scripts/detector_campaign.sh, so `tetraeder detector-score` reads the output.
# Needs Linux with sudo, systemd, cgroup v2 and GNU time. VRAM pressure, PCIe and GPU throttle stay box-only.
# Env: OUT (results dir), ROUNDS (6) rounds of none,cpu,disk,mem,quota, PAIR_ROUNDS (2) rounds of the three pairs,
#      MEM_FRAC (0.4) MemoryMax as a fraction of the calibration peak RSS, DECODE (64), THREADS (nproc),
#      QUOTA_FAULT (1) 0 drops the quota fault,
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
# scope_wrap.sh <info-file> <rss-file> cmd...   (QUOTA_CG: cgroup v2 child with cpu.max to move into)
[ -n "$QUOTA_CG" ] && echo $$ > "$QUOTA_CG/cgroup.procs"
cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
{ echo "io.max: $(cat "$cg/io.max" 2> /dev/null)"; echo "memory.max: $(cat "$cg/memory.max" 2> /dev/null)"
  echo "cpu.max: $(cat "$cg/cpu.max" 2> /dev/null)"; } > "$1"
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

# Does the runner enforce CPUQuota on a scope? Else a cgroup v2 child with cpu.max (when writable), else no quota fault.
quota_cores=$((threads / 2 > 0 ? threads / 2 : 1))
quota_mode=skip
quota_why=""
if [ "${QUOTA_FAULT:-1}" = 0 ]; then
  quota_why="QUOTA_FAULT=0"
elif [ "$threads" -lt 2 ]; then
  quota_why="THREADS=$threads: no thread count above a whole-core quota"
else
  # shellcheck disable=SC2024
  sudo systemd-run --scope -q -p "CPUQuota=$((quota_cores * 100))%" bash -c \
    'cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/cpu.max' > "$work/probe-quota.txt" 2>&1
  want="$((quota_cores * 100000)) 100000"
  if grep -q "^$want" "$work/probe-quota.txt"; then
    quota_mode=scope
  elif sudo sh -c "echo +cpu > /sys/fs/cgroup/cgroup.subtree_control; mkdir -p /sys/fs/cgroup/tetraeder-quota; echo '$want' > /sys/fs/cgroup/tetraeder-quota/cpu.max" 2> /dev/null \
    && grep -q "^$want" /sys/fs/cgroup/tetraeder-quota/cpu.max 2> /dev/null; then
    quota_mode=cgroup
  else
    quota_why="neither systemd CPUQuota nor a writable cgroup v2 cpu.max (probe: $(tr '\n' ' ' < "$work/probe-quota.txt"))"
  fi
fi
log "quota fault via $quota_mode, $quota_cores of $threads cores${quota_why:+ ($quota_why)}"

# run_one <n> <label> [faults...]
run_one() {
  local n=$1 label=$2 f k rate memmax props=() env_extra=() obs_quota=$threads
  shift 2
  stop_faults
  for f in "$@"; do
    case "$f" in
      cpu)
        # Half the threads alone; one loop in a pair, or the engine turns so CPU-bound that the partner costs nothing.
        for _ in $(seq 1 $(($# > 1 ? 1 : (threads + 1) / 2))); do python3 -c 'while True: pass' & fault_pids+=($!); done ;;
      disk)
        # 0.3 x the calibrated rate. In a pair, also times the partner's pace over the calibration pace (its single
        # runs): a partner that slows decode 5x cuts the read demand 5x, and a fixed cap no longer binds (first run:
        # cpu+disk 1.34 s/token, cpu alone 1.36).
        rate=$(python3 - "$out" "$@" << 'PY'
import json, statistics, sys
out, faults = sys.argv[1], sys.argv[2:]
rate = json.load(open(f"{out}/calib.json"))["rates"]["disk:nvme0"] * 0.3
partner = [f for f in faults if f != "disk"]
if partner:
    tok = {}
    for line in open(f"{out}/manifest.jsonl"):
        m = json.loads(line)
        h = json.load(open(f"{out}/harness-{m['id'][4:]}.json"))["runs"][-1]
        tok.setdefault(m["label"], []).append(h["decode_tok_s"])
    if tok.get(partner[0]) and tok.get("calib"):
        rate *= min(1.0, statistics.median(tok[partner[0]]) / statistics.median(tok["calib"]))
print(int(rate))
PY
)
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
      quota)
        # Threads above the quota, polling explicit: the idle ggml workers spin away the quota the compute waits for.
        env_extra+=(TETRAEDER_EXIT_CPU_POLL=50)
        obs_quota=$quota_cores
        if [ "$quota_mode" = scope ]; then
          props+=(-p "CPUQuota=$((quota_cores * 100))%")
        else
          env_extra+=(QUOTA_CG=/sys/fs/cgroup/tetraeder-quota)
        fi ;;
    esac
  done
  sleep 2
  log "run $n $label: props [${props[*]}]"
  # Every run, faulted or not, goes through a scope, so the measurement path is the same.
  # shellcheck disable=SC2024
  sudo env TETRAEDER_EXIT_GGUF="$gguf" TETRAEDER_EXIT_THREADS="$threads" TETRAEDER_EXIT_RAM_SLOTS=4 \
    TETRAEDER_EXIT_PREFILL=256 TETRAEDER_EXIT_UBATCH=256 TETRAEDER_EXIT_DECODE="${DECODE:-64}" \
    TETRAEDER_EXIT_DECODE_WINDOWS=8 TETRAEDER_EXIT_PROMPT_FILE="$work/prompt.txt" \
    TETRAEDER_EXIT_OUT="$out/harness-$n.json" "${env_extra[@]}" \
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
    --threads "$threads" --quota "$obs_quota" "${calib_arg[@]}" "${io_arg[@]}" --id "run-$n" --out "$out/obs-$n.json" || return 1
  python3 -c "import json,sys; print(json.dumps({'id': 'run-$n', 'label': '$label', 'faults': sys.argv[1:]}))" "$@" \
    >> "$out/manifest.jsonl"
  log "run $n done: $(python3 -c "import json; h=json.load(open('$out/harness-$n.json'))['runs'][-1]; print('decode', round(h['decode_tok_s'], 2), 'tok/s, hit', round(h['decode_layers']['full_hit_rate_mean'], 3))")"
}

# Deterministic schedule, the shape of detector_campaign.sh with mem in place of vram, plus quota (singles only).
python3 - "${ROUNDS:-6}" "${PAIR_ROUNDS:-2}" > "$out/schedule.txt" << 'EOF'
import random, sys
r = random.Random(20261109)
print("calib"); print("calib"); print("calib")
for _ in range(int(sys.argv[1])):
    s = ["none", "cpu", "disk", "mem", "quota"]; r.shuffle(s); print("\n".join(s))
for _ in range(int(sys.argv[2])):
    s = ["cpu+disk", "cpu+mem", "disk+mem"]; r.shuffle(s); print("\n".join(s))
EOF
if [ "$quota_mode" = skip ]; then
  grep -vx quota "$out/schedule.txt" > "$work/schedule.tmp"
  mv "$work/schedule.tmp" "$out/schedule.txt"
  log "quota fault skipped: $quota_why"
fi
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
