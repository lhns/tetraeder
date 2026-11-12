#!/usr/bin/env bash
# ggml polling arms under a CPU quota on a GitHub CPU runner (ADR 0061, steps (a) to (c)).
# The real engine (tests/exit_regime.rs, CPU backend, synthetic MoE as in scripts/fault_testbed.sh) in decode windows,
# threads = all cores, CPU quota = half the cores (so it binds), three arms interleaved per round:
#   poll50  TETRAEDER_EXIT_CPU_POLL=50
#   poll0   TETRAEDER_EXIT_CPU_POLL=0
#   pause   TETRAEDER_EXIT_CPU_POLL=50 TETRAEDER_EXIT_POOL_PAUSE=1
# ROUNDS quota rounds (A B C A B C ...), then one round per arm without a quota. Every run goes through a systemd scope.
# The quota is a systemd CPUQuota on the scope, else a cgroup v2 child with cpu.max. If neither is enforced the script
# says so and exits 20 before any run.
# Env: OUT (results dir), ROUNDS (8), RAM_SLOTS (4; 16 holds every expert of the synthetic model, so decode reads nothing), DECODE (64), THREADS (nproc), BIN (built exit_regime test binary; default: build).
# Results: $OUT/{harness,cg,time}-NN, manifest.jsonl, quota.txt; summary: scripts/poll_bench_summary.py.
cd "$(dirname "$0")/.." || exit 1
out="${OUT:-$PWD/bench/results/$(date +%Y%m%d)/ci-poll}"
mkdir -p "$out"
work="${WORK:-$(mktemp -d)}"
threads="${THREADS:-$(nproc)}"
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

# Quota first: no point building a model if it cannot be enforced.
quota_cores=$((threads / 2 > 0 ? threads / 2 : 1))
want="$((quota_cores * 100000)) 100000"
quota_mode=none
if [ "$threads" -lt 2 ]; then
  log "QUOTA NOT ENFORCEABLE: THREADS=$threads leaves no thread count above a whole-core quota"
  exit 20
fi
# shellcheck disable=SC2024
sudo systemd-run --scope -q -p "CPUQuota=$((quota_cores * 100))%" bash -c \
  'cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/cpu.max' > "$work/probe-quota.txt" 2>&1
if grep -q "^$want" "$work/probe-quota.txt"; then
  quota_mode=scope
elif sudo sh -c "echo +cpu > /sys/fs/cgroup/cgroup.subtree_control; mkdir -p /sys/fs/cgroup/tetraeder-quota; echo '$want' > /sys/fs/cgroup/tetraeder-quota/cpu.max" 2> /dev/null \
  && grep -q "^$want" /sys/fs/cgroup/tetraeder-quota/cpu.max 2> /dev/null; then
  quota_mode=cgroup
fi
echo "mode=$quota_mode cores=$quota_cores of $threads want='$want' probe='$(tr '\n' ' ' < "$work/probe-quota.txt")'" | tee "$out/quota.txt"
if [ "$quota_mode" = none ]; then
  log "QUOTA NOT ENFORCEABLE: neither systemd CPUQuota nor a writable cgroup v2 cpu.max; the arms would run unthrottled"
  exit 20
fi
log "quota via $quota_mode, $quota_cores of $threads cores"

log "model"
TMPDIR="$work" TETRAEDER_EXIT_THREADS="$threads" TETRAEDER_EXIT_PREFILL=8 TETRAEDER_EXIT_UBATCH=8 TETRAEDER_EXIT_DECODE=2 \
  "$bin" --ignored --exact exit_regime --nocapture > "$out/model-gen.log" 2>&1 \
  || { log "model generation failed"; tail -5 "$out/model-gen.log"; exit 11; }
gguf=$(find "$work" -name "tetraeder-exit-*.gguf" | head -1)
[ -s "$gguf" ] || { log "no model file"; exit 11; }
cat AGENTS.md docs/adr/*.md > "$work/prompt.txt"

# Runs inside the scope: optionally moves into the cpu.max child, snapshots cpu.stat around the engine, which runs
# under GNU time (voluntary and involuntary context switches, wall seconds).
cat > "$work/scope_wrap.sh" << 'WRAP'
#!/usr/bin/env bash
# scope_wrap.sh <cg-file> <time-file> cmd...   (QUOTA_CG: cgroup v2 child with cpu.max to move into)
[ -n "$QUOTA_CG" ] && echo $$ > "$QUOTA_CG/cgroup.procs"
cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
snap() { sed "s/^/$1 /" "$cg/cpu.stat" 2> /dev/null; }
cgf=$1
tf=$2
{ echo "cpu.max $(cat "$cg/cpu.max" 2> /dev/null)"; snap pre; } > "$cgf"
shift 2
/usr/bin/time -f '%w %c %e' -o "$tf" timeout -k 10 15m "$@"
r=$?
snap post >> "$cgf"
exit $r
WRAP
chmod +x "$work/scope_wrap.sh"

# run_one <n> <round> <arm> <quota 0|1>
run_one() {
  local n=$1 round=$2 arm=$3 q=$4 props=() env_extra=()
  case "$arm" in
    poll50) env_extra+=(TETRAEDER_EXIT_CPU_POLL=50) ;;
    poll0) env_extra+=(TETRAEDER_EXIT_CPU_POLL=0) ;;
    pause) env_extra+=(TETRAEDER_EXIT_CPU_POLL=50 TETRAEDER_EXIT_POOL_PAUSE=1) ;;
  esac
  if [ "$q" = 1 ]; then
    if [ "$quota_mode" = scope ]; then props+=(-p "CPUQuota=$((quota_cores * 100))%"); else env_extra+=(QUOTA_CG=/sys/fs/cgroup/tetraeder-quota); fi
  fi
  log "run $n round $round $arm quota=$q"
  # shellcheck disable=SC2024
  sudo env TETRAEDER_EXIT_GGUF="$gguf" TETRAEDER_EXIT_THREADS="$threads" TETRAEDER_EXIT_RAM_SLOTS="${RAM_SLOTS:-4}" \
    TETRAEDER_EXIT_PREFILL=256 TETRAEDER_EXIT_UBATCH=256 TETRAEDER_EXIT_DECODE="${DECODE:-64}" \
    TETRAEDER_EXIT_DECODE_WINDOWS=8 TETRAEDER_EXIT_PROMPT_FILE="$work/prompt.txt" \
    TETRAEDER_EXIT_OUT="$out/harness-$n.json" "${env_extra[@]}" \
    systemd-run --scope -q "${props[@]}" -- "$work/scope_wrap.sh" "$out/cg-$n.txt" "$out/time-$n.txt" \
    "$bin" --ignored --exact exit_regime --nocapture > "$out/run-$n.log" 2>&1
  local r=$?
  if [ "$r" -ne 0 ] || [ ! -s "$out/harness-$n.json" ]; then
    log "run $n failed rc=$r"; tail -5 "$out/run-$n.log"; return 1
  fi
  python3 -c "import json; print(json.dumps({'id': '$n', 'round': $round, 'arm': '$arm', 'quota': $q == 1}))" >> "$out/manifest.jsonl"
  log "run $n done: $(python3 -c "import json; print(round(json.load(open('$out/harness-$n.json'))['runs'][-1]['decode_tok_s'], 3), 'tok/s')")"
}

rm -f "$out/manifest.jsonl"
rc=0
n=0
for round in $(seq 1 "${ROUNDS:-8}"); do
  for arm in poll50 poll0 pause; do
    run_one "$(printf %02d "$n")" "$round" "$arm" 1 || rc=2
    n=$((n + 1))
  done
done
for arm in poll50 poll0 pause; do
  run_one "$(printf %02d "$n")" 0 "$arm" 0 || rc=2
  n=$((n + 1))
done
log "POLL_BENCH_DONE rc=$rc"
exit "$rc"
