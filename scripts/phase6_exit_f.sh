#!/usr/bin/env bash
# Phase 6 exit, Case F (ADR 0091): two boxes, two replicas of Qwen3-30B-A3B (R_A led by node a, the head; R_B by node b),
# a closed loop of 8 streaming requests through the head, three fault modes on node b, three trials each, then the verdict
# (scripts/phase6_exit_f_summary.py, every pass rule as code). Runs on the DRIVER machine and works on the boxes through
# `taxameter ssh --id` (scripts/phase6_box.sh and phase6_fault.sh are the box side). Needs no Rust, no cargo locally.
#
# Stages (STAGES, in order; each is safe to repeat):
#   gates    both boxes, before any download: power, idle, one NUMA node, co-tenant by engine_sample.py (phase6_box.sh gates).
#            A failed gate ends the session with nothing downloaded: destroy the box
#   setup    build on both boxes, then certificates (they need the binary), cluster.toml AFTER create (per-node files from the
#            boxes' public endpoints), the model on both boxes (same sha256), replicas.toml, the nodes and the head, the fault
#            trigger on B (second mapped port 7001, a token), both replicas in service
#   trials   the trials: for each mode and attempt, B in service (waits, never restarts anything), the clock offset between the
#            boxes by direct exchanges with the trigger, the load on A; FAULT_AT seconds into it, once B's own log shows requests in
#            flight on B, the load fires the fault over a direct connection to the trigger on B (fault time = middle of send and ack,
#            uncertain by half that interval); the restore, the load carried on until restore + REJOIN_S, then the attempt's files
#            into $out/trials/<mode>-<attempt>/. An attempt whose fault found no request in flight on B is repeated, at most twice
#            per mode (MAX_REPEATS), every attempt kept and reported. A finished attempt is skipped (RESUME)
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
# MAX_REPEATS (2). On real boxes create both with `--ports 7000,7001`: the trigger listens on 7001.
# LOOPBACK=1: the dry run with the REAL binaries. Both "boxes" are sets of processes on this machine: a real `tetraeder serve --cluster`
# head and two real `tetraeder node` daemons (TETRAEDER_BIN, default target/debug/tetraeder; certificates from `tetraeder cluster cert`,
# cluster.toml from the writer) on the tiny MoE fixture, started by the same phase6_box.sh commands a box uses. Work dirs
# runs/phase6-loopback/{a,b}, no taxameter, no GPU, short times (TRIAL_S 20, FAULT_AT 6, STOP_FOR 6, RESTORE_AFTER 4, REJOIN_S 30,
# NODE_DOWN_MS 2000, 96-token prompt, 120 output tokens). The loop and the verdict are the real ones; the verdict fails only the load-shape
# rule there, because 20 s is not 180 s.
# LOOPBACK=fake: the same with scripts/phase6_loopback.py in place of tetraeder (shorter times; LB_* variables break one rule on purpose),
# for the summary-rule fixtures. Both need a POSIX userland with pgrep, setsid and flock.
cd "$(dirname "$0")/.." || exit 1
repo=$(pwd -W 2> /dev/null || pwd) # a Windows driver needs C:/ paths for python3
export MSYS_NO_PATHCONV=1
LOOPBACK="${LOOPBACK:-0}"
lb_fake=0
[ "$LOOPBACK" = fake ] && { LOOPBACK=1; lb_fake=1; }
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
trigger_port=7001
serve_extra=""
if [ "$LOOPBACK" = 1 ]; then
  lb="${LB_DIR:-$repo/runs/phase6-loopback}"
  port_a="${LB_PORT_A:-17301}" port_b="${LB_PORT_B:-17302}" head_port="${LB_HEAD_PORT:-18380}" trigger_lb="${LB_TRIGGER_PORT:-17311}"
  poll_s=0.1 wait_step=0.5 warm_s=4
  LB_BOX_ENV="TRIGGER_PORT=$trigger_lb TRIGGER_HOST=127.0.0.1"
  if [ "$lb_fake" = 1 ]; then
    trial_s="${TRIAL_S:-14}" fault_at="${FAULT_AT:-4}" stop_for="${STOP_FOR:-3}" restore_after="${RESTORE_AFTER:-2}"
    rejoin_s="${REJOIN_S:-12}" down_ms="${NODE_DOWN_MS:-1000}" prompt_tokens="${PROMPT_TOKENS:-512}" max_tokens="${MAX_TOKENS:-256}"
    b_up_wait="${B_UP_WAIT_S:-30}"
  else
    # the tiny MoE fixture: 256 training positions, byte-level vocabulary (one character is one token), as in cluster_integration.rs
    trial_s="${TRIAL_S:-20}" fault_at="${FAULT_AT:-6}" stop_for="${STOP_FOR:-6}" restore_after="${RESTORE_AFTER:-4}"
    rejoin_s="${REJOIN_S:-30}" down_ms="${NODE_DOWN_MS:-2000}" prompt_tokens="${PROMPT_TOKENS:-96}" max_tokens="${MAX_TOKENS:-120}"
    load_extra="--ignore-eos" # the tiny model emits EOS early at temperature 0.7: a fixed length keeps requests in flight across the fault window
    b_up_wait="${B_UP_WAIT_S:-90}"
    LB_BIN="${TETRAEDER_BIN:-$repo/target/debug/tetraeder}"
    case "$LB_BIN" in /* | ?:*) ;; *) LB_BIN="$repo/$LB_BIN" ;; esac
    export LB_BIN
    LB_BOX_ENV="$LB_BOX_ENV THREADS=${THREADS:-2} PAR=4 CTX=2048"
    serve_extra="--chat-template-file $repo/tests/fixtures/chatml.jinja"
  fi
  export LB_BOX_ENV
else
  trial_s="${TRIAL_S:-180}" fault_at="${FAULT_AT:-60}" stop_for="${STOP_FOR:-60}" restore_after="${RESTORE_AFTER:-40}"
  rejoin_s="${REJOIN_S:-60}" down_ms="${NODE_DOWN_MS:-2000}" prompt_tokens="${PROMPT_TOKENS:-512}" max_tokens="${MAX_TOKENS:-256}"
  b_up_wait="${B_UP_WAIT_S:-180}" head_port=8080 poll_s=0.2 wait_step=5 warm_s=10
fi
max_repeats="${MAX_REPEATS:-2}"
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
  if [ "$LOOPBACK" = 1 ]; then
    rm -rf "$lb"
    if [ "$lb_fake" = 1 ]; then write_tetraeder_shim; else [ -x "$LB_BIN" ] || die "no tetraeder binary at $LB_BIN (cargo build -p tetraeder, or set TETRAEDER_BIN)"; fi
  fi
  # `cluster cert` is the tetraeder binary: on a box that has not built yet it fails silently and the certificate stays empty. The
  # model download starts with the build and is awaited after the certificates (loopback: the build only checks the binary)
  say "build and model on both boxes (the certificates need the binary)"
  for b in a b; do
    bx "$b" 120 "bash scripts/phase6_box.sh build" | tail -n 1
    bx "$b" 120 "bash scripts/phase6_box.sh model" | tail -n 1
  done
  for b in a b; do wait_line "$b" logs/build.log '^P6_BUILD_DONE rc=0' 3600 || die "build on $b did not finish ok"; done
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
  for b in a b; do wait_line "$b" logs/model.log '^P6_MODEL_DONE rc=0' 3600 || die "model on $b did not finish ok"; done
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
  bx a 60 "bash scripts/phase6_box.sh serve --replicas $(p6dir a)/replicas.toml $serve_extra"
  start_trigger
  sess replica_b '"replica 1"'
  sess bound_s "$bound"
  sess rejoin_limit_s "$rejoin_s"
  sess trials_per_mode "$trials"
  say "setup done"
}

start_trigger() { # the fault trigger on box b (own process, port 7001), its token on both boxes, reachable from box a
  local tok peer t0=$SECONDS
  say "the fault trigger on box b"
  mkdir -p "$repo/runs"
  tok="$repo/runs/p6-trigger-token.$$"
  python3 -I -c 'import secrets; print(secrets.token_hex(16))' > "$tok"
  put a "$tok" trigger-token.txt
  put b "$tok" trigger-token.txt
  rm -f "$tok"
  bx b 60 "bash scripts/phase6_box.sh trigger"
  peer=$(bx b 60 "bash scripts/phase6_box.sh endpoint $trigger_port" | tail -n 1)
  until bx a 60 "python3 -I scripts/phase6_trigger.py clock --peer $peer --token-file $(p6dir a)/trigger-token.txt --samples 3" > /dev/null 2>&1; do
    [ $((SECONDS - t0)) -lt 60 ] || die "box a cannot reach the trigger on box b at $peer: create both boxes with --ports 7000,7001"
    sleep 2
  done
  say "trigger reachable from box a at $peer"
}

clock_offset() { # prints the offset JSON of box b's clock to box a's: direct exchanges with the trigger (not ssh), the narrowest kept
  bx a 120 "python3 -I scripts/phase6_trigger.py clock --peer $peer_b --token-file $(p6dir a)/trigger-token.txt --samples 20" | tail -n 1
}

head_url() { echo "http://127.0.0.1:$head_port"; }
load_cmd() { # load_cmd LABEL SECONDS OUTFILE [FIRE_MODE]: with a fire mode the load fires that fault on b (scripts/phase6_trigger.py)
  local A fire=""
  A=$(p6dir a)
  [ -n "${4:-}" ] && fire="--fire-peer $peer_b --fire-token-file $A/trigger-token.txt --fire-mode $4 --fire-after $fault_at --fire-out $A/fire-$1.json --fire-stop-for $stop_for --fire-restore-after $restore_after --fire-poll-s $poll_s"
  echo "python3 scripts/phase6_load.py $(head_url) --key-file $A/api-key.txt --concurrency $conc --duration $2 --prompt-tokens $prompt_tokens --chars-per-token $cpt --max-tokens $max_tokens ${load_extra:-} --label $1 --out $3 $fire"
}

b_admits_since() { # b_admits_since EPOCH_B: how many admit lines of node b are at or after the epoch (b's clock)
  bx b 120 "bash scripts/phase6_box.sh admits $(p6dir b)/admits.txt > /dev/null; awk -v t=$1 '\$1 >= t { n++ } END { print n + 0 }' $(p6dir b)/admits.txt" | tail -n 1
}

ensure_b_serving() { # ensure_b_serving TAG: warm load on a, until b admits a request; never restarts anything
  local tag=$1 t0b n=0 t0=$SECONDS
  if [ "$lb_fake" = 1 ]; then # the fake head logs when B's replica joins: wait for that signal, then give the warm loads their time
    bx a $((b_up_wait + 30)) "for i in \$(seq 1 $((b_up_wait * 2))); do grep -E 'replica 1 (joined|left)' $(p6dir a)/logs/serve.log 2> /dev/null | tail -n 1 | grep -q joined && exit 0; sleep 0.5; done; exit 1" > /dev/null       || die "B is not in service: its replica did not join the fake head within $b_up_wait s before $tag. Nothing was restarted"
    t0=$SECONDS
  fi
  while [ $((SECONDS - t0)) -lt "$b_up_wait" ]; do
    t0b=$(epoch_of b)
    bx a 60 "bash scripts/phase6_box.sh launch warm-$tag-$n $(load_cmd "warm-$tag-$n" "$warm_s" "$(p6dir a)/warm-$tag-$n.jsonl")" > /dev/null
    wait_line a "logs/launch-warm-$tag-$n.log" '^phase6_load ' 120 > /dev/null || true
    [ "$(b_admits_since "$t0b")" -gt 0 ] && { say "B serves ($tag)"; return 0; }
    n=$((n + 1))
  done
  die "B is not in service after $b_up_wait s before $tag. Nothing was restarted. Fix it by hand, record it with: bash scripts/phase6_exit_f.sh note '<what you did>', and run the stage again (finished trials are kept)"
}

trial_valid() { # trial_valid DIR: the trigger saw requests in flight on B at the fault (B's own evidence, not the client's outcome)
  python3 -I scripts/phase6_exit_f_summary.py valid "$1" > /dev/null
}

run_attempt() { # run_attempt MODE ATTEMPT: one attempt into $out/trials/MODE-ATTEMPT
  local mode=$1 n=$2 A B d label clk off want t_fault_b t_restore_b t_fault t_restore tailed=0 deadline have now_a left load_end
  A=$(p6dir a)
  B=$(p6dir b)
  d="$out/trials/$mode-$n"
  label="$mode-t$n-$(date +%s)" # one attempt: its launches, its load nonces and its files carry the stamp, so a rerun starts clean
  mkdir -p "$d"
  say "=== $mode attempt $n"
  ensure_b_serving "$label"
  clk=$(clock_offset)
  sess clock "$clk"
  off=$(python3 -I -c 'import json, sys; print(json.loads(sys.argv[1])["b_minus_a_s"])' "$clk")
  echo "clock b minus a: $clk"
  # the load on a fires the fault on b over the direct connection, once requests are in flight on b
  t0a=$(bx a 60 "bash scripts/phase6_box.sh launch load-$label $(load_cmd "$label" "$trial_s" "$A/trials-$label.jsonl" "$mode")" | sed -n 1p)
  case "$t0a" in '' | *[!0-9.]*) die "the load of $label was not started by this call ('$t0a'): a launch of the same tag already ran. Remove $A/pids/launch-load-$label.done on the box or rerun with a new RESULTS_NAME" ;; esac
  say "load started on a; it fires the $mode fault $fault_at s in, once b has requests in flight"
  wait_line a "logs/launch-load-$label.log" '^phase6_load ' $((trial_s + 120)) > /dev/null || die "the load of $label did not finish"
  get a "fire-$label.json" "$d/fire.json" || die "no fire record of $label on box a: the load did not reach the trigger"
  python3 -I -c 'import json, sys; d = json.load(open(sys.argv[1])); sys.exit(0 if "error" not in d else 1)' "$d/fire.json" \
    || die "the fault of $label was not fired: $(cat "$d/fire.json")"
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
  t_fault=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["t_fault"])' "$d/fire.json") # the middle of send and ack, A's clock
  t_restore=""
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
  python3 -I - "$d" "$mode" "$n" "${t_restore:-null}" "$t_fault_b" "$label" "$load_end" "$want" "$clk" << 'PYEOF'
import json, sys
d, mode, n, tr, tfb, label, end, src, clk = sys.argv[1:10]
fire, clk = json.load(open(d + "/fire.json")), json.loads(clk)
json.dump({"mode": mode, "trial": int(n), "t_fault": fire["t_fault"], "t_fault_source": "trigger send/ack midpoint, A's clock",
           "fault": fire, "t_restore": None if tr == "null" else float(tr), "t_fault_b_clock": float(tfb),
           "t_restore_source": src, "load_label": label, "load_end": float(end),
           "clock_b_minus_a_s": clk["b_minus_a_s"], "clock_uncertainty_s": clk["uncertainty_s"]},
          open(d + "/meta.json", "w"), indent=2)
PYEOF
  say "$mode attempt $n recorded"
}

st_trials() {
  [ "$LOOPBACK" = 1 ] && [ "$lb_fake" = 1 ] && [ ! -x "$lb/tetraeder" ] && write_tetraeder_shim
  local A mode n valid
  A=$(p6dir a)
  peer_b=$(bx b 60 "bash scripts/phase6_box.sh endpoint $trigger_port" | tail -n 1)
  say "the head answers /health (a real head loads its engine first)"
  local t0=$SECONDS
  until bx a 30 "python3 -I -c 'import http.client, sys; c = http.client.HTTPConnection(\"127.0.0.1\", $head_port, timeout=5); c.request(\"GET\", \"/health\"); sys.exit(0 if c.getresponse().status == 200 else 1)'" > /dev/null 2>&1; do
    [ $((SECONDS - t0)) -lt "${HEAD_WAIT_S:-300}" ] || die "the head does not answer on port $head_port after ${HEAD_WAIT_S:-300} s: look at logs/serve.log on box a"
    sleep 1
  done
  say "prompt size through the head's tokenizer"
  cal=$(bx a 120 "python3 scripts/phase6_load.py $(head_url) --key-file $A/api-key.txt --calibrate --prompt-tokens $prompt_tokens" | tail -n 1)
  cpt=$(python3 -I -c 'import json, sys; print(json.loads(sys.argv[1])["chars_per_token"])' "$cal")
  ptok=$(python3 -I -c 'import json, sys; print(json.loads(sys.argv[1])["prompt_tokens"])' "$cal")
  echo "chars per token $cpt, prompt $ptok tokens"
  sess load "{\"concurrency\": $conc, \"duration_s\": $trial_s, \"prompt_tokens\": $ptok, \"max_tokens\": $max_tokens, \"stream\": true}"
  sess max_repeats "$max_repeats"
  for mode in $modes; do
    valid=0
    n=0
    while [ "$valid" -lt "$trials" ]; do
      n=$((n + 1))
      if [ "$n" -gt $((trials + max_repeats)) ]; then
        say "$mode: $valid of $trials valid trials after $((n - 1)) attempts (at most $max_repeats repeats): the mode stays incomplete"
        break
      fi
      if [ -s "$out/trials/$mode-$n/meta.json" ]; then
        say "$mode attempt $n exists, skipped"
      else
        run_attempt "$mode" "$n"
      fi
      if trial_valid "$out/trials/$mode-$n"; then valid=$((valid + 1)); else say "$mode attempt $n found no request in flight on B at the fault: it is repeated, and reported"; fi
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
