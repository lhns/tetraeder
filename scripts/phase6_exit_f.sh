#!/usr/bin/env bash
# Phase 6 exit, Case F (ADR 0091): two boxes, two replicas of Qwen3-30B-A3B (R_A led by node a, the head; R_B by node b),
# a closed loop of 8 streaming requests through the head, three fault modes on node b, three trials each, then the verdict
# (scripts/phase6_exit_f_summary.py, every pass rule as code). Runs on the DRIVER machine and works on the boxes through
# `taxameter ssh --id` (scripts/phase6_box.sh and phase6_fault.sh are the box side). Needs no Rust, no cargo locally.
#
# Stages (STAGES, in order; each is safe to repeat):
#   gates    both boxes, before any download: power, idle, one NUMA node, co-tenant by engine_sample.py (phase6_box.sh gates).
#            A failed gate ends the session with nothing downloaded: destroy the box
#   setup    certificates on the boxes, cluster.toml AFTER create (per-node files from the boxes' public endpoints), build,
#            the model on both boxes (same sha256), replicas.toml, the nodes and the head, both replicas in service
#   trials   the nine trials: for each mode and trial, B in service (waits, never restarts anything), the clock offset between
#            the boxes, the load on A, the fault on B fired at a target epoch, the restore, the load carried on until
#            restore + 60 s, then the trial's files into $out/trials/<mode>-<n>/. A finished trial is skipped (RESUME)
#   summary  collects the head's request log and the box logs, writes session.json and runs the verdict
# note TEXT  (not a stage) records a manual action at this moment: `bash scripts/phase6_exit_f.sh note "restarted the head"`.
#            Trial 1 of a mode fails its rule when one falls between its fault and the rejoin deadline. The script itself never
#            restarts a node or the head: when B is not in service before a trial it stops and says so.
#
# Env: RESULTS_NAME (required; results in bench/results/<name>), ID_A and ID_B (taxameter ids; not with LOOPBACK), T_CREATED_A and
# T_CREATED_B (epoch seconds of the two `create` calls, recorded for the "cluster.toml after create" rule), STAGES, MODES
# (worker daemon stop), TRIALS (3), TRIAL_S (180), FAULT_AT (60), STOP_FOR (60), RESTORE_AFTER (40: how long the killed daemon
# stays dead), REJOIN_S (60), CONC (8), PROMPT_TOKENS (512), MAX_TOKENS (256), NODE_DOWN_MS (2000), CAP (4 slots per replica),
# B_UP_WAIT_S (180), RESULTS_ROOT.
# LOOPBACK=1: the dry run. Both "boxes" are sets of processes on this machine (scripts/phase6_loopback.py in place of tetraeder,
# work dirs runs/phase6-loopback/{a,b}), no taxameter, no GPU, short times (TRIAL_S 14, FAULT_AT 4, STOP_FOR 3, RESTORE_AFTER 2,
# REJOIN_S 12, NODE_DOWN_MS 1000, 40 tokens). The loop and the verdict are the real ones; the verdict fails only the load-shape
# rule there, because 14 s is not 180 s. Needs a POSIX userland with pgrep, setsid and flock.
cd "$(dirname "$0")/.." || exit 1
repo=$PWD
export MSYS_NO_PATHCONV=1
LOOPBACK="${LOOPBACK:-0}"
say() { echo "[$(date +%T)] $*"; }
die() { say "FATAL: $*"; exit 1; }

if [ "${1:-}" = note ]; then
  name="${RESULTS_NAME:?set RESULTS_NAME}"
  mkdir -p "${RESULTS_ROOT:-$repo/bench/results}/$name"
  python3 -I -c 'import json, sys, time; print(json.dumps({"t": time.time(), "what": sys.argv[1]}))' "${2:?text}" \
    >> "${RESULTS_ROOT:-$repo/bench/results}/$name/manual.jsonl"
  echo "noted"
  exit 0
fi

name="${RESULTS_NAME:?set RESULTS_NAME}"
out="${RESULTS_ROOT:-$repo/bench/results}/$name"
stages="${STAGES:-gates setup trials summary}"
modes="${MODES:-worker daemon stop}"
trials="${TRIALS:-3}"
if [ "$LOOPBACK" = 1 ]; then
  lb="${LB_DIR:-$repo/runs/phase6-loopback}"
  port_a="${LB_PORT_A:-17301}" port_b="${LB_PORT_B:-17302}" head_port="${LB_HEAD_PORT:-18380}"
  trial_s="${TRIAL_S:-14}" fault_at="${FAULT_AT:-4}" stop_for="${STOP_FOR:-3}" restore_after="${RESTORE_AFTER:-2}"
  rejoin_s="${REJOIN_S:-12}" down_ms="${NODE_DOWN_MS:-1000}" prompt_tokens="${PROMPT_TOKENS:-512}" max_tokens="${MAX_TOKENS:-256}"
  b_up_wait="${B_UP_WAIT_S:-30}" poll_s=0.1 wait_step=0.5 warm_s=4
else
  trial_s="${TRIAL_S:-180}" fault_at="${FAULT_AT:-60}" stop_for="${STOP_FOR:-60}" restore_after="${RESTORE_AFTER:-40}"
  rejoin_s="${REJOIN_S:-60}" down_ms="${NODE_DOWN_MS:-2000}" prompt_tokens="${PROMPT_TOKENS:-512}" max_tokens="${MAX_TOKENS:-256}"
  b_up_wait="${B_UP_WAIT_S:-180}" head_port=8080 poll_s=0.2 wait_step=5 warm_s=10
fi
conc="${CONC:-8}" cap="${CAP:-4}"
bound=$(awk -v d="$down_ms" 'BEGIN { printf "%.3f", d / 1000 + 1 }') # node_down_after + 1 s (3 s on the real boxes)
mkdir -p "$out"/{gates,certs,trials,cluster}

# the two ways to reach a box (taxameter ssh, or the loopback), and the session file
# shellcheck source=scripts/phase6_exit_lib.sh
. "$repo/scripts/phase6_exit_lib.sh"

# ---- stages -----------------------------------------------------------------------------------------------------------------
st_gates() {
  local b rc=0
  for b in a b; do
    say "gates on box $b"
    bx "$b" 400 "bash scripts/phase6_box.sh gates" | tail -n 8
    for f in lscpu.txt cotenant.json gates.json power-gate.txt idle-gate.txt; do get "$b" "gates/$f" "$out/gates/$b/$f" || rc=1; done
    # the same checks again on the driver, from the files that are committed
    python3 -I scripts/sc_gates.py numa "$out/gates/$b/lscpu.txt" || rc=1
    python3 -I scripts/sc_gates.py cotenant "$out/gates/$b/cotenant.json" || rc=1
    python3 -I -c 'import json, sys; g = json.load(open(sys.argv[1])); sys.exit(0 if g.get("power_ok") is True and g.get("idle_ok") is True else 1)' \
      "$out/gates/$b/gates.json" || { echo "GATE FAILED $b: power or idle"; rc=1; }
  done
  [ "$rc" = 0 ] || die "a gate failed on a box: destroy it (nothing was downloaded) and try the next offer"
  say "gates done: both boxes pass"
}

write_tetraeder_shim() { # the loopback binary: the stand-in of scripts/phase6_loopback.py behind the name `tetraeder`
  mkdir -p "$lb"
  printf '#!/usr/bin/env bash\nexec python3 "%s/scripts/phase6_loopback.py" "$@"\n' "$(cd "$repo" && { pwd -W 2> /dev/null || pwd; })" > "$lb/tetraeder"
  chmod +x "$lb/tetraeder"
}

st_setup() {
  local b
  [ "$LOOPBACK" = 1 ] && { rm -rf "$lb"; write_tetraeder_shim; }
  say "certificates"
  for b in a b; do
    bx "$b" 120 "bash scripts/phase6_box.sh cert $b" | tail -n 2
    get "$b" keys/cert.pem "$out/certs/$b.pem"
  done
  sess certs "{\"a\": \"$(sha256sum "$out/certs/a.pem" | cut -d' ' -f1)\", \"b\": \"$(sha256sum "$out/certs/b.pem" | cut -d' ' -f1)\", \"pinned_on\": []}"
  say "endpoints and cluster.toml (after create)"
  local ea eb
  ea=$(bx a 120 "bash scripts/phase6_box.sh endpoint" | tail -n 1)
  eb=$(bx b 120 "bash scripts/phase6_box.sh endpoint" | tail -n 1)
  echo "a $ea b $eb"
  local extra=()
  [ "$LOOPBACK" = 1 ] && extra=(--node-listen "a=$port_a" --node-listen "b=$port_b")
  python3 -I scripts/phase6_cluster_toml.py --cluster phase6-exit-f --head a --node "a=$ea" --node "b=$eb" --down-ms "$down_ms" \
    --out-dir "$out/cluster" "${extra[@]}"
  for b in a b; do
    put "$b" "$out/cluster/cluster-$b.toml" cluster.toml
    put "$b" "$out/certs/a.pem" a.pem
    put "$b" "$out/certs/b.pem" b.pem
  done
  sess certs.pinned_on '["a", "b"]'
  sess t_cluster_toml "$(python3 -I -c 'import time; print(time.time())')"
  if [ "$LOOPBACK" = 1 ]; then
    now=$(python3 -I -c 'import time; print(time.time())')
    sess t_created "{\"a\": $(awk -v n="$now" 'BEGIN { printf "%.3f", n - 30 }'), \"b\": $(awk -v n="$now" 'BEGIN { printf "%.3f", n - 20 }')}"
  else
    sess t_created "{\"a\": ${T_CREATED_A:?set T_CREATED_A}, \"b\": ${T_CREATED_B:?set T_CREATED_B}}"
  fi
  if [ "$LOOPBACK" != 1 ]; then
    say "build and model on both boxes (parallel)"
    for b in a b; do
      bx "$b" 120 "bash scripts/phase6_box.sh build" | tail -n 1
      bx "$b" 120 "bash scripts/phase6_box.sh model" | tail -n 1
    done
    for b in a b; do
      wait_line "$b" logs/build.log '^P6_BUILD_DONE rc=0' 3600 || die "build on $b did not finish ok"
      wait_line "$b" logs/model.log '^P6_MODEL_DONE rc=0' 3600 || die "model on $b did not finish ok"
    done
  else
    for b in a b; do printf '%064d\n' 0 > "$lb/$b/model.sha256"; done
  fi
  local sha_a sha_b
  sha_a=$(bx a 60 "cat $(p6dir a)/model.sha256" | tail -n 1)
  sha_b=$(bx b 60 "cat $(p6dir b)/model.sha256" | tail -n 1)
  sess model_sha256 "\"$sha_a\""
  sess boxes "{\"a\": {\"id\": \"${ID_A:-loopback}\", \"model_sha256\": \"$sha_a\"}, \"b\": {\"id\": \"${ID_B:-loopback}\", \"model_sha256\": \"$sha_b\"}}"
  [ "$sha_a" = "$sha_b" ] || die "the two boxes hold different model files ($sha_a, $sha_b)"
  # the head's replicas: R_A led by a (in process), R_B led by b; the head's API key stays on box a
  printf 'model_sha256 = "%s"\n\n[[replica]]\nleader = "a"\ncap = %s\n\n[[replica]]\nleader = "b"\ncap = %s\n' "$sha_a" "$cap" "$cap" > "$out/cluster/replicas.toml"
  put a "$out/cluster/replicas.toml" replicas.toml
  bx a 60 "head -c 24 /dev/urandom | base64 > $(p6dir a)/api-key.txt"
  say "nodes, then the head"
  bx b 60 "bash scripts/phase6_box.sh node"
  bx a 60 "bash scripts/phase6_box.sh node"
  bx a 60 "bash scripts/phase6_box.sh serve --replicas $(p6dir a)/replicas.toml"
  sess replica_b '"replica 1"'
  sess bound_s "$bound"
  sess rejoin_limit_s "$rejoin_s"
  sess trials_per_mode "$trials"
  say "setup done"
}

clock_offset() { # prints the offset JSON of box b's clock to box a's (three exchanges, the best one kept)
  local t0 a t1 t2 b t3
  for _ in 1 2 3; do
    t0=$(python3 -I -c 'import time; print(time.time())')
    a=$(epoch_of a)
    t1=$(python3 -I -c 'import time; print(time.time())')
    t2=$(python3 -I -c 'import time; print(time.time())')
    b=$(epoch_of b)
    t3=$(python3 -I -c 'import time; print(time.time())')
    echo "$t0 $a $t1 $t2 $b $t3"
  done | python3 -I scripts/phase6_exit_f_summary.py clock
}

head_url() { echo "http://127.0.0.1:$head_port"; }
load_cmd() { # load_cmd LABEL SECONDS OUTFILE
  local A
  A=$(p6dir a)
  echo "python3 scripts/phase6_load.py $(head_url) --key-file $A/api-key.txt --concurrency $conc --duration $2 --prompt-tokens $prompt_tokens --chars-per-token $cpt --max-tokens $max_tokens --label $1 --out $3"
}

b_admits_since() { # b_admits_since EPOCH_B: how many admit lines of node b are at or after the epoch (b's clock)
  bx b 120 "bash scripts/phase6_box.sh admits $(p6dir b)/admits.txt > /dev/null; awk -v t=$1 '\$1 >= t { n++ } END { print n + 0 }' $(p6dir b)/admits.txt" | tail -n 1
}

ensure_b_serving() { # ensure_b_serving TAG: warm load on a, until b admits a request; never restarts anything
  local tag=$1 t0b n=0 t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "$b_up_wait" ]; do
    t0b=$(epoch_of b)
    bx a 60 "bash scripts/phase6_box.sh launch warm-$tag-$n $(load_cmd "warm-$tag-$n" "$warm_s" "$(p6dir a)/warm-$tag-$n.jsonl")" > /dev/null
    wait_line a "logs/launch-warm-$tag-$n.log" '^phase6_load ' 120 > /dev/null || true
    [ "$(b_admits_since "$t0b")" -gt 0 ] && { say "B serves ($tag)"; return 0; }
    n=$((n + 1))
  done
  die "B is not in service after $b_up_wait s before $tag. Nothing was restarted. Fix it by hand, record it with: bash scripts/phase6_exit_f.sh note '<what you did>', and run the stage again (finished trials are kept)"
}

st_trials() {
  [ "$LOOPBACK" = 1 ] && [ ! -x "$lb/tetraeder" ] && write_tetraeder_shim
  local A B mode n d label
  A=$(p6dir a)
  B=$(p6dir b)
  say "prompt size through the head's tokenizer"
  cal=$(bx a 120 "python3 scripts/phase6_load.py $(head_url) --key-file $A/api-key.txt --calibrate --prompt-tokens $prompt_tokens" | tail -n 1)
  cpt=$(python3 -I -c 'import json, sys; print(json.loads(sys.argv[1])["chars_per_token"])' "$cal")
  ptok=$(python3 -I -c 'import json, sys; print(json.loads(sys.argv[1])["prompt_tokens"])' "$cal")
  echo "chars per token $cpt, prompt $ptok tokens"
  sess load "{\"concurrency\": $conc, \"duration_s\": $trial_s, \"prompt_tokens\": $ptok, \"max_tokens\": $max_tokens, \"stream\": true}"
  for mode in $modes; do
    for n in $(seq 1 "$trials"); do
      d="$out/trials/$mode-$n"
      label="$mode-t$n"
      [ -s "$d/meta.json" ] && { say "$mode trial $n exists, skipped"; continue; }
      mkdir -p "$d"
      say "=== $mode trial $n"
      ensure_b_serving "$label"
      clk=$(clock_offset)
      sess clock "$clk"
      off=$(python3 -I -c 'import json, sys; print(json.loads(sys.argv[1])["b_minus_a_s"])' "$clk")
      echo "clock b minus a: $clk"
      # the load on a, its start epoch is on a's clock; the fault is armed on b for that epoch + FAULT_AT on b's clock
      t0a=$(bx a 60 "bash scripts/phase6_box.sh launch load-$label $(load_cmd "$label" "$trial_s" "$A/trials-$label.jsonl")" | head -n 1)
      case "$t0a" in '' | *[!0-9.]*) die "the load of $label was not started by this call ('$t0a'): a launch of the same tag already ran. Remove $A/pids/launch-load-$label.done on the box or rerun with a new RESULTS_NAME" ;; esac
      target_b=$(awk -v t="$t0a" -v f="$fault_at" -v o="$off" 'BEGIN { printf "%.3f", t + f + o }')
      bx b 60 "bash scripts/phase6_box.sh launch fault-$label env TARGET_EPOCH=$target_b FAULT_LOG=$B/logs/fault-$label.log STOP_FOR=$stop_for RESTORE_AFTER=$restore_after POLL_S=$poll_s P6=$B bash scripts/phase6_fault.sh $mode 0" > /dev/null
      say "load and fault started (fault at b epoch $target_b)"
      wait_line a "logs/launch-load-$label.log" '^phase6_load ' $((trial_s + 120)) > /dev/null || die "the load of $label did not finish"
      case "$mode" in
        worker) want='restore child-up' ;;
        daemon) want='restore start daemon' ;;
        stop) want='restore SIGCONT' ;;
      esac
      # the restore line is in the fault log; for the daemon mode the daemon is started by the fault script itself
      wait_line b "logs/fault-$label.log" "$want|restore done" $((restore_after + stop_for + 90)) > /dev/null || say "WARNING: no restore line in the fault log of $label"
      get b "logs/fault-$label.log" "$d/fault.log"
      t_fault_b=$(awk '/ fault / && !/ fault done/ { print $1; exit }' "$d/fault.log")
      t_restore_b=$(awk -v w="$want" 'index($0, w) { print $1; exit }' "$d/fault.log")
      [ -n "$t_fault_b" ] || die "no fault line in the log of $label"
      t_fault=$(awk -v t="$t_fault_b" -v o="$off" 'BEGIN { printf "%.3f", t - o }')
      t_restore="" tailed=0
      [ -n "$t_restore_b" ] && t_restore=$(awk -v t="$t_restore_b" -v o="$off" 'BEGIN { printf "%.3f", t - o }')
      # B must serve within rejoin_s of the restore: the load goes on until then (a trial of 180 s may end sooner)
      if [ -n "$t_restore" ]; then
        deadline=$(awk -v t="$t_restore" -v r="$rejoin_s" 'BEGIN { printf "%.3f", t + r }')
        have=$(b_admits_since "$t_restore_b")
        now_a=$(epoch_of a)
        left=$(awk -v d="$deadline" -v n="$now_a" 'BEGIN { x = d - n + 1; if (x < 0) x = 0; printf "%d", x + 0.999 }')
        if [ "$have" = 0 ] && [ "$left" -gt 0 ]; then
          say "no admit on B yet: the load goes on for $left s, to restore + $rejoin_s s"
          tailed=1
          bx a 60 "bash scripts/phase6_box.sh launch tail-$label $(load_cmd "tail-$label" "$left" "$A/tail-$label.jsonl")" > /dev/null
          wait_line a "logs/launch-tail-$label.log" '^phase6_load ' $((left + 120)) > /dev/null || true
        fi
      fi
      get a "trials-$label.jsonl" "$d/load.jsonl"
      [ "$tailed" = 1 ] && get a "tail-$label.jsonl" "$d/tail.jsonl"
      bx b 120 "bash scripts/phase6_box.sh admits $B/admits.txt > /dev/null"
      get b admits.txt "$d/b-admits.txt"
      load_end=$(python3 -I -c 'import json, sys; print(max((json.loads(x)["t_end"] for x in open(sys.argv[1]) if x.strip()), default=0))' "$d/load.jsonl")
      [ -s "$d/tail.jsonl" ] && load_end=$(cat "$d/load.jsonl" "$d/tail.jsonl" | python3 -I -c 'import json, sys; print(max(json.loads(x)["t_end"] for x in sys.stdin if x.strip()))')
      python3 -I - "$d/meta.json" "$mode" "$n" "$t_fault" "${t_restore:-null}" "$t_fault_b" "$label" "$load_end" "$want" "$off" << 'EOF'
import json, sys
p, mode, n, tf, tr, tfb, label, end, src, off = sys.argv[1:11]
json.dump({"mode": mode, "trial": int(n), "t_fault": float(tf), "t_restore": None if tr == "null" else float(tr), "t_fault_b_clock": float(tfb),
           "t_restore_source": src, "load_label": label, "load_end": float(end), "clock_b_minus_a_s": float(off)},
          open(p, "w"), indent=2)
EOF
      say "$mode trial $n recorded"
    done
  done
  say "trials done"
}

st_summary() {
  say "the head's request log and the box logs"
  get a logs/head-requests.jsonl "$out/head-requests.jsonl"
  gzip -f "$out/head-requests.jsonl"
  local b
  for b in a b; do
    bx "$b" 120 "cd $(p6dir "$b") && tar czf logs-$b.tgz logs gates cluster.toml 2> /dev/null; ls -l logs-$b.tgz" > /dev/null
    get "$b" "logs-$b.tgz" "$out/$b-logs.tgz" 2> /dev/null || true
  done
  python3 -I scripts/phase6_exit_f_summary.py "$out"
  rc=$?
  say "verdict in $out/verdict-f.json"
  return $rc
}

for s in $stages; do
  declare -F "st_$s" > /dev/null || die "unknown stage $s"
done
rc=0
for s in $stages; do
  say "STAGE $s"
  "st_$s" || rc=$?
done
exit "$rc"
