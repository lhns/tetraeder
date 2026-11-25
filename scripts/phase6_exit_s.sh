#!/usr/bin/env bash
# Phase 6 exit, Case S (ADR 0091, answers 6 to 8, ADR 0092 rerun rules): Qwen3-235B-A22B Q4_K_M on two boxes (one RTX 3090 each, a
# 64 GB RAM budget by balloon), one segment per node, pp2048 and tg128 within 15% of a prediction frozen before the first judged run.
# Runs on the DRIVER machine and works on the boxes through `taxameter ssh --id`; the calibration is scripts/sc_box.sh on each box
# (its stages, with its gates), the verdict is scripts/phase6_exit_s_summary.py. No Rust, no cargo locally.
#
# Two invocations, with a commit between them (the frozen prediction goes to git before any judged run):
#   STAGES="gates setup budget plan linkgate fetch calib predict" bash scripts/phase6_exit_s.sh   ... commit predictions/, link/, plans/, calib/
#   FROZEN_SHA=<FROZEN.sha256> STAGES="judged aalast linkend summary" bash scripts/phase6_exit_s.sh
# Stages:
#   gates     both boxes, before any download: power, idle, one NUMA node, co-tenant by engine_sample.py. A miss ends the session
#   setup     build on both boxes, then certificates (they need the binary), cluster.toml after create, the node daemons. No
#             download yet (`create` went through scripts/sc_offers.sh: real --download-gb 150, disk_gb 230, --ports 7000)
#   budget    per node, sc_box.sh `env balloon`: the RAM budget under EMULATE_RAM_GB is measured before anything is profiled (ADR 0092
#             section 4). No model is needed: the disk curve and the model's layout come after the fetch, in calib
#   plan      the layout is the committed one (GATE_LAYOUT, default bench/results/segment-confirm-20261125b/layout.json), a link probe
#             (link/probe-plan.json) and `cluster profile --ram-budget` (each node's measured RAM budget on its CPU; discovery names the
#             models volume `nvme0`), then `tetraeder-plan candidates --cluster --covering`: the analytic pick among plans that compute on
#             both nodes and whose every segment is covering (VRAM plus RAM slots hold all experts of its layers, within the measured
#             budgets and 24 GiB). A plan on one node or a non-covering plan is never a candidate (ADR 0091 History 2026-11-25). No such
#             candidate, or a pick whose stages do not partition the layers, aborts here, before any download. Each node's V and R are the
#             pick's (plans/slots.txt): the anchors run them, the workers run them, the summary checks both
#   linkgate  the link gate BEFORE the download (advisor, delegated by the owner, 2026-11-25): two full probes (frames to 64 MiB, no
#             --max-bytes) at least 10 minutes apart (LINK_GATE_WAIT_S, never below 600 on a box); the judged pick itself
#             (plans/judged-plan.toml) priced on the plan's graph with each probe's link curves swapped in and nothing else changed.
#             Pp or tg moving by more than 0.08, or probes less than 600 s apart, aborts the pair: destroy both boxes, nothing was
#             downloaded. Files link/probe-gate{1,2}.json, link/gate-times.json, link/gate/
#   fetch     the 142 GB model on both boxes (REQUIRE_ONE_NUMA=1); refuses unless the link gate passed. Then `tetraeder cluster model-id
#             --install` on each box (the id of the five shards; detached, it hashes them), the two ids compared, and the model's own
#             layout compared with the committed one the plan was made on (a difference aborts: the pick was for another model)
#   calib     the disk curve (`sc_box.sh disk`: layout of the model, NVMe curve, graph seeded with the measured RAM budget), the start
#             link probe (link/probe-start.json, swapped into the plan's graph as link/hw.json), the whole-model trace on the leader
#             (routing trace, and the hidden rows entering the cut), the trace and the rows to node b, anchors of each node's own segment
#             truncated to a quarter and a half (real hidden rows; layer 0 takes tokens), two rounds, with the pick's slots, the fit
#             (band limit 0.10, refused above)
#   dryrun    NOT part of a session's flow: the whole of the above and the judged round on a tiny sharded MoE (tests/fixtures), with the
#             real binaries, before the download, on the boxes themselves (RESULTS_NAME of its own, its own P6 work dir); see below
#   predict   the fitted node graphs merged into the cluster graph (`tetraeder cluster merge`, on node a), `tetraeder-plan predict --cluster --judged` (routing trace only),
#             SHA256SUMS and FROZEN.sha256 of everything it used; refuses on any failed pre-freeze rule. With SECOND_PREDICT_CMD (runs
#             on node a, writes $PRED_OUT) a report-only prediction priced from a second prompt's routing trace, recorded before the
#             judged runs, is kept as predictions/predict-cluster-p2.json (not frozen, not the judged prompt)
#   judged    refuses unless FROZEN_SHA equals the frozen hash and every frozen file matches; rounds of the judged plan until three
#             are clean or eight have run (unclean rounds are rerun on the same prediction, all reported); the early A/A after round
#             2 aborts the session outside 0.08 on T
#   aalast    the A/A last: the judged plan once more, the last run of the session
#   linkend   the end probe; the frozen judged plan is priced again with its link curves and nothing else changed
#             (predictions/predict-cluster-end.json). The link rule (advisor, delegated by the owner, 2026-11-25): predicted pp and
#             tg each move by at most 0.08, else the session is not clean; the verdict must hold against the frozen and the end-priced
#             prediction. The link is also probed before every judged round (link/probe-<round>.json); raw points are reported only
#   summary   session.json and the verdict
# THE JUDGED ROUND is `tetraeder cluster judged-round` (ADR 0091, 0092 owner answer 1), run once on node a with the head's key: it assigns the
# plan's segments to both daemons, runs the warm request, then a timed pp2048 and tg128 request that computes every prompt token (the
# prompt is the frozen one, read from the hidden-row dump), with each segment's counters read from its worker, and writes on node a
# ROUND_DIR/cluster.json ({"runs": [{prefill_tok_s, decode_tok_s, prompt: {tokens, fnv1a64}, ...}]}) and ROUND_DIR/seg-<node>.json
# (exit_regime shape: warm, prefill_io, decode_layers, the segment's own rates and times). The slot counts are the plan's own
# (`expert_cache` of the segment plan the worker reads). This script adds what only a box knows: seg-<node>.gpu.txt (the `before:` and
# `after:` nvidia-smi lines of sc_box.sh run_one) and balloon_ok and cotenant_ok in each seg file. REFERENCE_CMD (optional, node a, env
# ROUND_FILE) writes the report-only single-node round.
#
# Env: LINK_GATE_WAIT_S (600), GATE_LAYOUT, RESULTS_NAME (required), ID_A, ID_B, T_CREATED_A, T_CREATED_B, STAGES, FROZEN_SHA, REFERENCE_CMD, RESULTS_ROOT,
# A1_LAYERS and A2_LAYERS (11 and 23 layers after a span's first), MAX_ROUNDS (8), EMULATE_RAM_GB (64), SPANS (dry runs only: "a 0..47,b 47..94"
# instead of the pick's). The slots of the anchors and of the workers are the pick's own, per node (plans/slots.txt): there is no global V and R.
# TINY=1: the tiny sharded MoE of tests/fixtures (8 layers cut 4 and 4, a pinned plan, 4 + 4 slots over 8 experts) instead of the 235B, with the real
# binaries; the `dryrun` stage is this on the boxes (DRYRUN_NAME, DRYRUN_STAGES), before the download and before `budget`: it runs one judged round of
# the real `tetraeder cluster judged-round` and checks it with `phase6_exit_s_summary.py round`; it prices and freezes nothing.
# LOOPBACK=1: the same dry run on this machine: both "boxes" are directories and processes here, the REAL binaries (TETRAEDER_BIN, default
# target/debug, with tetraeder-plan and tetraeder-bench beside it), no GPU (sc_box.sh with SC_CPU=1), no balloon (the budget is read by the real
# balloon_guard.py from a made-up cgroup directory) and a made-up throttle record; timing numbers prove the wiring, not the model. CI runs it.
# LOOPBACK=fake: the dry run with the stand-in tools of scripts/phase6_loopback.py (LB_SCENARIO breaks one rule on purpose); no GPU.
cd "$(dirname "$0")/.." || exit 1
repo=$(pwd -W 2> /dev/null || pwd) # a Windows driver needs C:/ paths for python3
export MSYS_NO_PATHCONV=1
LOOPBACK="${LOOPBACK:-0}"
lb_fake=0
[ "$LOOPBACK" = fake ] && { LOOPBACK=1; lb_fake=1; }
TINY="${TINY:-0}" # the tiny sharded MoE of tests/fixtures instead of the 235B (the dry run); the real loopback is always tiny
[ "$LOOPBACK" = 1 ] && [ "$lb_fake" = 0 ] && TINY=1
say() { echo "[$(date +%T)] $*"; }
die() { say "FATAL: $*"; exit 1; }
name="${RESULTS_NAME:?set RESULTS_NAME}"
out="${RESULTS_ROOT:-$repo/bench/results}/$name"
stages="${STAGES:-gates setup budget plan linkgate fetch calib predict}"
spans="${SPANS:-}" # the pick's, from plans/spans.txt, once the plan stage has run
a1_layers="${A1_LAYERS:-11}" a2_layers="${A2_LAYERS:-23}"
sc_rel="../tetraeder/runs/sc" # sc_box.sh's run dir, relative to a box's P6 dir
s_ctx="${S_CTX:-4096}"
if [ "$LOOPBACK" = 1 ]; then
  lb="${LB_DIR:-$repo/runs/phase6-loopback-s}"
  port_a="${LB_PORT_A:-17401}" port_b="${LB_PORT_B:-17402}" head_port="${LB_HEAD_PORT:-18480}" poll_s=0.1 wait_step=0.5
  gate_wait="${LINK_GATE_WAIT_S:-2}"
  export P6_MIN_GAP_S=1 # the summary's 600 s between the gate probes, shortened for the dry run only
  if [ "$lb_fake" = 0 ]; then
    EMULATE_RAM_GB="${EMULATE_RAM_GB:-0}"
    # the real binaries (CI builds them): tetraeder, tetraeder-plan and tetraeder-bench side by side; no GPU, no balloon (EMULATE_RAM_GB=0)
    LB_BIN="${TETRAEDER_BIN:-$repo/target/debug/tetraeder}"
    case "$LB_BIN" in /* | ?:*) ;; *) LB_BIN="$repo/$LB_BIN" ;; esac
    export LB_BIN
    bin_dir=$(dirname "$LB_BIN")
    LB_BOX_ENV="THREADS=${THREADS:-2} PAR=1 CTX=512 EMULATE_RAM_GB=0 SC_CPU=1 SC_BIN_DIR=$bin_dir SC_PROFILE=${SC_PROFILE:-}"
    export LB_BOX_ENV
    sc_rel="sc"
    s_ctx="${S_CTX:-512}"
  fi
else
  gate_wait="${LINK_GATE_WAIT_S:-600}"
  [ "$gate_wait" -ge 600 ] || { echo "LINK_GATE_WAIT_S below 600: the gate needs probes at least 10 minutes apart"; exit 2; }
  head_port=8080 poll_s=0.2 wait_step=5
fi
if [ "$TINY" = 1 ]; then # the tiny MoE: 8 layers cut 4 and 4, anchors of 1 and 2 layers, ubatch 16, 64 prompt tokens and 16 generated
  a1_layers="${A1_LAYERS:-1}" a2_layers="${A2_LAYERS:-2}"
fi
tiny_src="${TINY_SRC:-tests/fixtures/tiny-qwen3moe-l8-f16.gguf}"
tiny_plan="${TINY_PLAN:-tests/fixtures/phase6/tiny-case-s-plan.toml}"
mkdir -p "$out"/{gates,certs,cluster,link,plans,calib,env,hidden,trace,predictions,judged,reference}
# shellcheck source=scripts/phase6_exit_lib.sh
. "$repo/scripts/phase6_exit_lib.sh"

summ() { python3 -I scripts/phase6_exit_s_summary.py "$@"; }
need_spans() {
  [ -n "$spans" ] || spans=$(tr -d '\r' < "$out/plans/spans.txt" 2> /dev/null) || true
  [ -n "$spans" ] || die "no spans: run the plan stage first"
}
span_of() { echo "$spans" | tr ',' '\n' | awk -v n="$1" '$1 == n { print $2 }'; }
first_of() { span_of "$1" | cut -d. -f1; }
end_of() { span_of "$1" | awk -F'[.][.]' '{ print $2 }'; }
slots_of() { tr -d '\r' < "$out/plans/slots.txt" 2> /dev/null | awk -v n="$1" '$1 == n { print $2, $3 }'; } # "V R" of a node: the pick's own
n_expert_of() { python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["n_expert"])' "$out/layout.json"; }
tet() { if [ "$lb_fake" = 1 ]; then echo "$lb/tetraeder"; elif [ "$LOOPBACK" = 1 ]; then echo "$LB_BIN"; else echo target/release/tetraeder; fi; }
tplan() { if [ "$lb_fake" = 1 ]; then echo "$lb/tetraeder-plan"; elif [ "$LOOPBACK" = 1 ]; then echo "$bin_dir/tetraeder-plan"; else echo target/release/tetraeder-plan; fi; }
resrel() { if [ "$LOOPBACK" = 1 ]; then echo "results/$name-n$1"; else echo "../tetraeder/bench/results/$name-n$1"; fi; } # under P6
shim() { # shim NAME SUBCOMMAND-WORDS: a loopback executable that runs the stand-in
  mkdir -p "$lb"
  printf '#!/usr/bin/env bash\nexec python3 "%s/scripts/phase6_loopback.py" %s "$@"\n' "$(cd "$repo" && { pwd -W 2> /dev/null || pwd; })" "$2" > "$lb/$1"
  chmod +x "$lb/$1"
}
loopback_tools() { shim tetraeder "" && shim tetraeder-plan plan; }
tiny_first() { echo "$(p6dir "$1")/tiny/tiny-00001-of-00003.gguf"; } # the tiny model's first shard on a node

# one sc_box.sh run on a node (STAGES of sc_box.sh), waited for; the results dir of the box is <name>-n<node>. Its tag is part of the launch
# name and of the DONE line, so a stage of this run is never taken for the finished stage of another run or of a dry run
scbox() { # scbox NODE TAG "STAGES"
  local n=$1 tag=$2 st=$3 s="" e="" a1="" a2="" cut="" w v r P
  P=$(p6dir "$n")
  if [ "$lb_fake" = 1 ]; then
    [ -n "$spans" ] && { s=$(first_of "$n") e=$(end_of "$n"); }
    (cd "$repo" && python3 scripts/phase6_loopback.py sc-box --node "$n" --out "$lb/$n/$(resrel "$n")" --stages "$st" --span "${s:-0}..${e:-94}") | tail -n 1
    return 0
  fi
  w="RESULTS_NAME=$name-n$n EMULATE_RAM_GB=${EMULATE_RAM_GB:-64} SC_LOG=$P/logs/sc_box.log SC_RUN_TAG=$name-$n-$tag"
  if [ -n "$spans" ]; then
    s=$(first_of "$n") e=$(end_of "$n") cut=$(end_of a)
    a1=$((s + a1_layers)) a2=$((s + a2_layers))
    read -r v r <<< "$(slots_of "$n")"
    [ -n "$v" ] && [ -n "$r" ] || die "no slots of node $n in plans/slots.txt: the plan stage has not run"
    w="$w SPLIT=$s END=$e A1_END=$a1 A2_END=$a2 HIDDEN_AT=$cut V=$v R=$r"
  fi
  if [ "$TINY" = 1 ]; then
    w="$w MODEL=$(tiny_first "$n") PROMPT_FILE= UBATCH=16 PREFILL=64 DECODE=16 DUMP_V=4 DUMP_R=4 FIT_BAND_MAX=${FIT_BAND_MAX:-1000}"
  fi
  [ "$LOOPBACK" = 1 ] && w="$w RESULTS_ROOT=$P/results SC_LOCK=$P/sc_box.lock SC_RUN_DIR=$P/sc"
  bx "$n" 60 "bash scripts/phase6_box.sh launch sc-$n-$name-$tag env $w STAGES='$st' bash scripts/sc_box.sh" | head -n 2
}
scbox_wait() { # scbox_wait NODE TAG "STAGES"
  [ "$lb_fake" = 1 ] && return 0
  local line
  line=$(wait_line "$1" logs/sc_box.log "SC_BOX_DONE rc=[0-9]+.* tag=$name-$1-$2\$" 14400) || die "sc_box.sh on node $1 did not finish ($3)"
  case "$line" in *"rc=0"*) say "node $1 sc_box.sh [$3] done" ;; *) die "sc_box.sh on node $1 failed ($3): $line" ;; esac
}

st_gates() {
  local b rc=0
  if [ "$LOOPBACK" != 1 ]; then
    # `bash scripts/sc_offers.sh create --offer ID --ports 7000` wrote disk_gb 230 and priced the 150 GB download into the cap
    python3 -I -c 'import json, sys; sys.exit(0 if json.load(open(sys.argv[1])).get("disk_gb", 0) >= 230 else 1)' "${VAST_STATE_DIR:-.vast}/config.json" \
      || die "disk_gb in ${VAST_STATE_DIR:-.vast}/config.json is below 230: create the boxes with scripts/sc_offers.sh (real --download-gb 150, disk 230)"
  fi
  for b in a b; do
    say "gates on box $b"
    bx "$b" 400 "bash scripts/phase6_box.sh gates" | tail -n 8
    for f in lscpu.txt cotenant.json gates.json power-gate.txt idle-gate.txt; do get "$b" "gates/$f" "$out/gates/$b/$f" || rc=1; done
    python3 -I scripts/sc_gates.py numa "$out/gates/$b/lscpu.txt" || rc=1
    python3 -I scripts/sc_gates.py cotenant "$out/gates/$b/cotenant.json" || rc=1
    python3 -I -c 'import json, sys; g = json.load(open(sys.argv[1])); sys.exit(0 if g.get("power_ok") is True and g.get("idle_ok") is True else 1)' \
      "$out/gates/$b/gates.json" || { echo "GATE FAILED $b: power or idle"; rc=1; }
  done
  [ "$rc" = 0 ] || die "a gate failed on a box: destroy it before the 142 GB download and try the next offer"
  say "gates done: both boxes pass"
}

st_setup() {
  local b ea eb now
  if [ "$LOOPBACK" = 1 ]; then
    rm -rf "$lb"
    if [ "$lb_fake" = 1 ]; then loopback_tools; else [ -x "$LB_BIN" ] || die "no tetraeder binary at $LB_BIN (cargo build -p tetraeder, or set TETRAEDER_BIN)"; fi
  else
    # `cluster cert` is the tetraeder binary: build first, or the certificate stays empty (rehearsal 2). The tag differs from the budget stage's
    say "build on both boxes (the certificates need the binary)"
    for b in a b; do scbox "$b" bld "build" > /dev/null; done
    for b in a b; do scbox_wait "$b" bld build; done
  fi
  for b in a b; do
    bx "$b" 120 "bash scripts/phase6_box.sh cert $b" | tail -n 1
    get "$b" keys/cert.pem "$out/certs/$b.pem"
  done
  ea=$(bx a 120 "bash scripts/phase6_box.sh endpoint" | tail -n 1)
  eb=$(bx b 120 "bash scripts/phase6_box.sh endpoint" | tail -n 1)
  local extra=()
  [ "$LOOPBACK" = 1 ] && extra=(--node-listen "a=$port_a" --node-listen "b=$port_b")
  python3 -I scripts/phase6_cluster_toml.py --cluster phase6-exit-s --head a --node "a=$ea" --node "b=$eb" --out-dir "$out/cluster" "${extra[@]}"
  for b in a b; do
    put "$b" "$out/cluster/cluster-$b.toml" cluster.toml
    put "$b" "$out/certs/a.pem" a.pem
    put "$b" "$out/certs/b.pem" b.pem
  done
  now=$(python3 -I -c 'import time; print(time.time())')
  sess t_cluster_toml "$now"
  if [ "$LOOPBACK" = 1 ]; then
    sess t_created "{\"a\": $(awk -v n="$now" 'BEGIN { printf "%.3f", n - 30 }'), \"b\": $(awk -v n="$now" 'BEGIN { printf "%.3f", n - 20 }')}"
  else
    sess t_created "{\"a\": ${T_CREATED_A:?set T_CREATED_A}, \"b\": ${T_CREATED_B:?set T_CREATED_B}}"
  fi
  # the workers compute the experts outside their slots on the CPU (the exit harness's MissPolicy::Cpu); the slot counts are the plan's;
  # one sequence of ctx tokens (2048 prompt and 128 generated need 2176)
  for b in b a; do bx "$b" 60 "CTX=$s_ctx PAR=1 WORKER_EXTRA='--miss cpu' bash scripts/phase6_box.sh node"; done
  sess band_source '"bench/spec/phase0.toml"'
  sess boxes "{\"a\": {\"id\": \"${ID_A:-loopback}\"}, \"b\": {\"id\": \"${ID_B:-loopback}\"}}"
  say "setup done"
}

# the link gate before the download: two full probes 10 minutes apart, the judged pick priced with each
gate_time() { # gate_time KEY: the driver's clock at the end of a gate probe
  python3 -I -c 'import json, os, sys, time; p, k = sys.argv[1:3]; d = json.load(open(p)) if os.path.exists(p) else {}; d[k] = time.time(); json.dump(d, open(p, "w"))' \
    "$out/link/gate-times.json" "$1"
}
gate_price() { # gate_price HW_FILE OUT_FILE (under the work dir of node a): the judged pick on the graph, uniform routing, no replay
  local P
  P=$(p6dir a)
  bx a 600 "$(tplan) predict --cluster $P/judged-plan.toml --hw $P/$1 --layout $P/layout.json --prompt 2048 --gen 128 --out $P/$2" | tail -n 2
  get a "$2" "$out/link/gate/$2"
}

st_linkgate() {
  local P
  P=$(p6dir a)
  [ -s "$out/plans/judged-plan.toml" ] || die "no judged plan: the plan stage runs before the link gate (the gate prices the pick)"
  [ "$lb_fake" = 1 ] && [ ! -x "$lb/tetraeder-plan" ] && loopback_tools
  mkdir -p "$out/link/gate"
  rm -f "$out/link/gate-times.json"
  say "link gate: the first full probe"
  probe_link gate1 || die "the first full probe failed (a truncated probe is no fallback): the pair is aborted, destroy both boxes"
  gate_time gate1
  # the plan's graph with only the link curves swapped: what differs between the two prices is the link
  bx a 300 "$(tet) cluster merge $P/hw-plan.json --swap-link $P/probe-gate1.json -o $P/hw-gate1.json" | tail -n 1
  gate_price hw-gate1.json predict-gate1.json
  say "link gate: waiting $gate_wait s for the second probe"
  sleep "$gate_wait"
  probe_link gate2 || die "the second full probe failed (a truncated probe is no fallback): the pair is aborted, destroy both boxes"
  gate_time gate2
  bx a 300 "$(tet) cluster merge $P/hw-plan.json --swap-link $P/probe-gate2.json -o $P/hw-gate2.json" | tail -n 1
  gate_price hw-gate2.json predict-gate2.json
  summ gate "$out" || die "the link gate failed: destroy both boxes, nothing was downloaded (the 142 GB model comes only after this gate)"
  say "linkgate done: the pair passes, the download may start"
}

# the tiny dry-run model on a node: the fixture cut into three shards, installed under its id (what the 235B gets after its download)
tiny_model() { # tiny_model NODE: prints the id
  local n=$1 P id
  P=$(p6dir "$n")
  bx "$n" 120 "mkdir -p $P/tiny && $(tet) cluster model-split $tiny_src $P/tiny/tiny --shards 3" | tail -n 1 > /dev/null
  id=$(bx "$n" 300 "bash scripts/phase6_box.sh model-id $(tiny_first "$n")" | tail -n 1)
  case "$id" in *"P6_MODELID_DONE rc=0 id="*) echo "${id##*id=}" ;; *) die "model-id of the tiny model on node $n: $id" ;; esac
}

st_fetch() {
  local b ids=()
  if [ "$TINY" = 1 ]; then
    [ "$lb_fake" = 1 ] && { say "loopback: no download"; return 0; }
    # nothing to download: the plan stage split and installed the tiny model; the ids of the two nodes must agree
    for b in a b; do ids+=("$(bx "$b" 60 "cat $(p6dir "$b")/model.sha256" | tail -n 1)"); done
    [ "${#ids[0]}" = 64 ] && [ "${ids[0]}" = "${ids[1]}" ] || die "the nodes disagree on the tiny model's id: a ${ids[0]}, b ${ids[1]}"
    sess model_sha256 "\"${ids[0]}\""
    say "fetch done: tiny model id ${ids[0]}"
    return 0
  fi
  [ "$lb_fake" = 1 ] && { say "loopback: no download"; return 0; }
  summ gate "$out" > /dev/null || die "the link gate has not passed (link/gate.json): no download"
  say "the 142 GB model on both boxes (REQUIRE_ONE_NUMA=1)"
  for b in a b; do bx "$b" 120 "bash scripts/phase6_box.sh launch fetch-235b env REQUIRE_ONE_NUMA=1 bash scripts/fetch_exit_model.sh" | tail -n 1; done
  for b in a b; do
    wait_line "$b" ../tetraeder/runs/vast/fetch_exit.log 'FETCH_DONE rc=0' 14400 > /dev/null || die "the model download on $b did not finish ok"
  done
  # the id of the five shards (`tetraeder cluster model-id --install`): hashing 142 GB outlasts an ssh call, so detached, on both at once
  say "model-id on both boxes (hashes the shards)"
  for b in a b; do bx "$b" 120 "bash scripts/phase6_box.sh launch modelid-235b bash scripts/phase6_box.sh model-id" | tail -n 1; done
  for b in a b; do
    wait_line "$b" logs/model-id.log '^P6_MODELID_DONE rc=0 id=' 7200 > /dev/null || die "model-id on $b did not finish ok (logs/model-id.log)"
    ids+=("$(bx "$b" 60 "cat $(p6dir "$b")/model.sha256" | tail -n 1)")
  done
  [ "${#ids[0]}" = 64 ] && [ "${ids[0]}" = "${ids[1]}" ] || die "the nodes disagree on the model id: a ${ids[0]}, b ${ids[1]}"
  sess model_sha256 "\"${ids[0]}\""
  say "fetch done: model id ${ids[0]}"
}

# the RAM budget of each node is measured under the balloon before the graph is profiled (ADR 0092 section 4); no model is needed
st_budget() {
  local n cg
  say "env and balloon on both nodes"
  if [ "$TINY" = 1 ] && [ "$lb_fake" = 0 ]; then
    # the dry run holds no balloon (the tiny model needs none): sc_box.sh `env`, then the budget the real balloon_guard.py reads from the
    # cgroup (a box's own; on a CI runner a directory of made-up numbers)
    for n in a b; do scbox "$n" b1 "env" > /dev/null; done
    for n in a b; do scbox_wait "$n" b1 "env"; done
    for n in a b; do
      cg="${CGROUP_DIR:-/sys/fs/cgroup}"
      if [ "$LOOPBACK" = 1 ]; then
        cg="$lb/$n/cgroup"
        mkdir -p "$cg" "$lb/$n/$(resrel "$n")"
        printf '17179869184\n' > "$cg/memory.max"
        printf '1000000000\n' > "$cg/memory.current"
        printf 'inactive_file 500000000\n' > "$cg/memory.stat"
      fi
      bx "$n" 60 "python3 -I scripts/balloon_guard.py budget --cgroup $cg --env-json $(p6dir "$n")/$(resrel "$n")/env.json" | tail -n 1
    done
  else
    scbox a b1 "env balloon" > /dev/null
    scbox b b1 "env balloon" > /dev/null
    scbox_wait a b1 "env balloon"
    scbox_wait b b1 "env balloon"
  fi
  for n in a b; do
    mkdir -p "$out/env/$n"
    get "$n" "$(resrel "$n")/env.json" "$out/env/$n/env.json"
  done
  say "budget done"
}

probe_link() { # probe_link NAME: the link probe of the session, into link/probe-NAME.json
  local P
  P=$(p6dir a)
  bx a 900 "$(tet) cluster probe --cluster $P/cluster.toml --key $P/keys/key.pem --cert $P/keys/cert.pem --repeats ${PROBE_REPEATS:-10} -o $P/probe-$1.json" | tail -n 1
  get a "probe-$1.json" "$out/link/probe-$1.json"
}

# the plan: before the download. The layout is the committed one (the tiny dry run: the fixture's own), the graph carries each node's measured
# RAM budget and the pick is among covering plans only (ADR 0091 History 2026-11-25)
st_plan() {
  local P ra rb n_expert nl f cand
  P=$(p6dir a)
  [ "$lb_fake" = 1 ] && [ ! -x "$lb/tetraeder" ] && loopback_tools
  if [ "$TINY" = 1 ] && [ "$lb_fake" = 0 ]; then
    say "the tiny model on both nodes, its layout"
    for f in a b; do tiny_model "$f" > /dev/null; done
    bx a 120 "$(tet) plan layout $(tiny_first a) --out $P/layout.json" | tail -n 1
    get a layout.json "$out/layout.json"
  else
    put a "${GATE_LAYOUT:-$repo/bench/results/segment-confirm-20261125b/layout.json}" layout.json
    cp "${GATE_LAYOUT:-$repo/bench/results/segment-confirm-20261125b/layout.json}" "$out/layout.json"
  fi
  n_expert=$(n_expert_of)
  say "cluster probe and profile"
  local cl="--cluster $P/cluster.toml --key $P/keys/key.pem --cert $P/keys/cert.pem"
  bx a 900 "$(tet) cluster probe $cl --repeats ${PLAN_PROBE_REPEATS:-20} -o $P/probe-plan.json" | tail -n 2
  # the cluster graph carries each node's measured RAM budget on its CPU (ADR 0092 section 4); discovery names the models volume nvme0
  ra=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["ram_budget"]["ram_budget_bytes"])' "$out/env/a/env.json")
  rb=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["ram_budget"]["ram_budget_bytes"])' "$out/env/b/env.json")
  bx a 300 "$(tet) cluster profile $cl --probe $P/probe-plan.json --ram-budget a=$ra --ram-budget b=$rb -o $P/hw-plan.json" | tail -n 2
  get a probe-plan.json "$out/link/probe-plan.json"
  get a hw-plan.json "$out/plans/hw-plan.json"
  if [ "$TINY" = 1 ]; then
    # the planner prices a placement with a GPU only, and a CI runner has none: the plan is pinned (a head-led two-stage plan of the
    # tiny MoE, 4 + 4 slots covering its 8 experts), written by hand in tests/fixtures and parsed by a test
    say "the pinned plan of the tiny model"
    cp "$repo/$tiny_plan" "$out/plans/judged-plan.toml"
  else
    say "covering candidates across the nodes, then the pick"
    cand=$(bx a 300 "rm -rf $P/candidates; $(tplan) candidates --cluster --covering --hw $P/hw-plan.json --layout $P/layout.json --plans-dir $P/candidates --out $P/candidates.json --prompt 2048; echo CAND_RC=\$?" | tail -n 4)
    echo "$cand"
    case "$cand" in *CAND_RC=0*) ;; *) die "no covering plan that computes on both nodes (the planner refused, see above): nothing is downloaded" ;; esac
    get a candidates.json "$out/plans/candidates.json"
    mkdir -p "$out/plans/candidates"
    for f in $(python3 -I -c 'import json, sys; print(" ".join(c["plan"] for c in json.load(open(sys.argv[1]))["candidates"] if c.get("plan")))' "$out/plans/candidates.json"); do
      get a "candidates/$f" "$out/plans/candidates/$f"
    done
    summ pick "$out/plans/candidates.json" "$out/plans/candidates" "$out/plans/judged-plan.toml" --n-expert "$n_expert" \
      || die "no covering plan that computes on both nodes: Case S has no judged candidate, nothing is downloaded"
  fi
  summ span "$out/plans/judged-plan.toml" || die "the pick does not compute on both nodes"
  summ slots "$out/plans/judged-plan.toml" --n-expert "$n_expert" > "$out/plans/slots.txt" || die "the pick is not a covering plan: nothing is downloaded"
  nl=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["n_layer"])' "$out/layout.json")
  summ spans "$out/plans/judged-plan.toml" --n-layer "$nl" > "$out/plans/spans.raw" || die "the pick's stages do not cut the $nl layers one node each, leader first"
  tr -d '\r' < "$out/plans/spans.raw" | tr '\n' ',' | sed 's/,$//' > "$out/plans/spans.txt"
  rm -f "$out/plans/spans.raw"
  spans=$(cat "$out/plans/spans.txt")
  put a "$out/plans/judged-plan.toml" judged-plan.toml
  python3 -I - "$out/session.json" "$spans" "$out/plans/slots.txt" << 'EOF'
import json, os, sys
p, spans, slots = sys.argv[1:4]
d = json.load(open(p)) if os.path.exists(p) else {}
sl = {ln.split()[0]: [int(ln.split()[1]), int(ln.split()[2])] for ln in open(slots) if ln.strip()}
d["segments"] = [{"node": n.split()[0], "span": [int(x) for x in n.split()[1].split("..")], "slots": sl[n.split()[0]]} for n in spans.split(",")]
d["segments"][0]["embed"], d["segments"][-1]["head"] = True, True
json.dump(d, open(p, "w"), indent=2)
EOF
  say "plan done: $spans, slots $(tr '\n' ',' < "$out/plans/slots.txt") (the pick's own: the anchors and the workers run them)"
}

st_calib() {
  local n f
  need_spans
  say "disk curve and the model's layout on both nodes"
  scbox a c0 "disk" > /dev/null
  scbox b c0 "disk" > /dev/null
  scbox_wait a c0 "disk"
  scbox_wait b c0 "disk"
  # the plan was made on the committed layout (before the download): the model's own layout must be that one
  get a "$(resrel a)/layout.json" "$out/layout-model.json"
  summ layout-same "$out/layout.json" "$out/layout-model.json" || die "the model's layout is not the committed one the plan was made on: the pick is for another model"
  say "the start link probe, swapped into the plan's graph as link/hw.json"
  probe_link start
  bx a 300 "$(tet) cluster merge $(p6dir a)/hw-plan.json --swap-link $(p6dir a)/probe-start.json -o $(p6dir a)/hw.json" | tail -n 1
  get a hw.json "$out/link/hw.json"
  say "node a: the whole-model trace, hidden rows entering layer $(end_of a)"
  scbox a c1 "trace" > /dev/null
  scbox_wait a c1 "trace"
  if [ "$lb_fake" != 1 ]; then
    say "the trace and the hidden rows to node b"
    for f in hidden.bin routing.trace; do
      get a "$sc_rel/$f" "$out/.xfer-$f"
      gzip -f "$out/.xfer-$f"
      put b "$out/.xfer-$f.gz" "$sc_rel/$f.gz"
      bx b 600 "gunzip -f $(p6dir b)/$sc_rel/$f.gz"
      rm -f "$out/.xfer-$f.gz"
    done
    for f in routing hidden; do get a "$(resrel a)/trace/$f.sha256" "$out/trace/$f.sha256"; done
    put b "$out/trace/routing.sha256" "$(resrel b)/trace/routing.sha256"
    put b "$out/trace/hidden.sha256" "$(resrel b)/trace/hidden.sha256"
    for n in a b; do
      bx "$n" 600 "sha256sum $(p6dir "$n")/$sc_rel/hidden.bin | cut -d' ' -f1" | tail -n 1 > "$out/hidden/$n.sha256"
    done
  else
    for f in routing hidden; do get a "$(resrel a)/trace/$f.sha256" "$out/trace/$f.sha256"; done
    for n in a b; do cp "$out/trace/hidden.sha256" "$out/hidden/$n.sha256"; done
  fi
  say "anchors and the fit on both nodes (the pick's slots)"
  scbox a c2 "anchors predict" > /dev/null
  scbox b c2 "anchors predict" > /dev/null
  scbox_wait a c2 "anchors predict"
  scbox_wait b c2 "anchors predict"
  for n in a b; do
    mkdir -p "$out/calib/$n/anchors"
    for k in a1-r1 a1-r2 a2-r1 a2-r2; do get "$n" "$(resrel "$n")/anchors/$k.json" "$out/calib/$n/anchors/$k.json"; done
    for k in pred.json fitted.toml replay.json; do get "$n" "$(resrel "$n")/predictions/$k" "$out/calib/$n/$k"; done
  done
  python3 -I - "$out" << 'EOF'
import json, os, sys
out = sys.argv[1]
p = os.path.join(out, "session.json")
d = json.load(open(p))
a = json.load(open(os.path.join(out, "calib", "a", "anchors", "a1-r1.json")))
r = a["runs"][-1] if "runs" in a else a
d["prompt"] = {"tokens": 2048, "gen": 128, "fnv1a64": r["prompt"]["fnv1a64"]}
json.dump(d, open(p, "w"), indent=2)
EOF
  if [ "$TINY" = 1 ] && [ "$lb_fake" = 0 ]; then
    say "dry run: the pre-calibration rules of the 235B session (link gate, 64 GB budgets, fit band) are not checked"
  else
    summ precheck "$out" calib || die "calibration gates failed (fit band over the limit, anchors on fixed-seed input, gates, budget, covering slots or hidden rows): nothing frozen"
  fi
  say "calib done"
}

# price HW_FILE OUT_FILE (under the work dir of node a): the judged plan, the frozen replays, nothing else differs between callers
price() {
  local P tsha ph
  P=$(p6dir a)
  tsha=$(awk '{ print $1 }' "$out/trace/routing.sha256")
  ph=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["prompt"]["fnv1a64"])' "$out/session.json")
  bx a 600 "$(tplan) predict --cluster $P/judged-plan.toml --hw $P/$1 --layout $P/layout.json --tier-replay $(span_of a)=$P/replay-a.json --tier-replay $(span_of b)=$P/replay-b.json --trace-sha256 $tsha --prompt-hash $ph --judged --prompt 2048 --gen 128 --out $P/$2"
}

# the fitted node graphs go into the cluster graph on node a (`tetraeder cluster merge`, a Rust reader)
merge_fitted() {
  local P
  P=$(p6dir a)
  put a "$out/calib/a/fitted.toml" fitted-a.toml
  put a "$out/calib/b/fitted.toml" fitted-b.toml
  bx a 300 "$(tet) cluster merge $P/hw.json --node a=$P/fitted-a.toml --node b=$P/fitted-b.toml -o $P/hw-cluster.json" | tail -n 2
  get a hw-cluster.json "$out/predictions/hw-cluster.json"
}

st_merge() { # the dry run's stage of the same step
  need_spans
  merge_fitted
  say "merge done"
}

st_predict() {
  local P tsha ph
  need_spans
  P=$(p6dir a)
  [ "$lb_fake" = 1 ] && [ ! -x "$lb/tetraeder-plan" ] && loopback_tools
  tsha=$(awk '{ print $1 }' "$out/trace/routing.sha256")
  ph=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["prompt"]["fnv1a64"])' "$out/session.json")
  merge_fitted
  put a "$out/calib/a/replay.json" replay-a.json
  put a "$out/calib/b/replay.json" replay-b.json
  # the prediction is judged: routing from the trace only, the sha256 of the trace it was priced from, one replay per segment
  price hw-cluster.json predict-cluster.json > "$out/predictions/predict.log" 2>&1 || { cat "$out/predictions/predict.log"; die "predict --cluster --judged refused: nothing is frozen"; }
  tail -n 2 "$out/predictions/predict.log"
  get a predict-cluster.json "$out/predictions/predict-cluster.json"
  if [ -n "${SECOND_PREDICT_CMD:-}" ]; then # report only: the routing trace of a second prompt (recorded now, before any judged run)
    bx a 3600 "env PRED_OUT=$P/predict-cluster-p2.json bash -c '$SECOND_PREDICT_CMD'" > /dev/null \
      && get a predict-cluster-p2.json "$out/predictions/predict-cluster-p2.json" || say "the second-prompt prediction was not written (report only)"
  fi
  (cd "$out" && sha256sum layout.json link/probe-start.json link/hw.json predictions/hw-cluster.json predictions/predict-cluster.json \
    plans/judged-plan.toml plans/candidates.json plans/spans.txt plans/slots.txt trace/routing.sha256 trace/hidden.sha256 calib/a/pred.json calib/b/pred.json \
    calib/a/replay.json calib/b/replay.json calib/a/fitted.toml calib/b/fitted.toml > predictions/SHA256SUMS \
    && (cd predictions && sha256sum SHA256SUMS > FROZEN.sha256))
  python3 -I -c 'import json, sys, time; json.dump({"t_frozen": time.time(), "sha256": open(sys.argv[1]).read().split()[0]}, open(sys.argv[2], "w"))' \
    "$out/predictions/FROZEN.sha256" "$out/predictions/frozen.json"
  summ precheck "$out" freeze || die "a pre-freeze rule failed: the prediction is not frozen (delete predictions/FROZEN.sha256)"
  cat "$out/predictions/FROZEN.sha256"
  say "predict done: frozen. Commit predictions/, link/, plans/, calib/, trace/ and push; then run the judged stages with FROZEN_SHA=$(awk '{ print $1 }' "$out/predictions/FROZEN.sha256")"
}

# one judged round into $out/judged/<label>: on both boxes, then the files back
gpu_line() { # gpu_line NODE before|after: the nvidia-smi line of sc_box.sh run_one (clock-event reasons, power draw and limits)
  if [ "$LOOPBACK" = 1 ]; then echo "$2: 0x0000000000000000, 100 W, 350 W, 350 W"; return 0; fi # no GPU to read on a CI runner
  bx "$1" 60 "nvidia-smi --query-gpu=clocks_event_reasons.active,power.draw,power.limit,power.default_limit --format=csv,noheader 2>&1 | tail -n 1" | tail -n 1 | sed "s/^/$2: /"
}

# the exit code of sc_balloon.sh check on a node: 0 while the balloon holds the emulated RAM
balloon_rc() { bx "$1" 60 "EMULATE_RAM_GB=${EMULATE_RAM_GB:-64} bash scripts/sc_balloon.sh check > /dev/null 2>&1; echo \$?" | tail -n 1; }

# the flags only a box knows, into a segment file: balloon_ok (sc_balloon.sh check before and after the round) and cotenant_ok (the
# gates stage's reading; no single engine process exists to sample during a round)
box_flags() { # box_flags NODE JSON PRE_RC POST_RC
  python3 -I - "$out/gates/$1/cotenant.json" "$2" "$3" "$4" << 'EOF'
import json, sys
gate, path, pre, post = sys.argv[1:5]
d = json.load(open(path, encoding="utf-8"))
ok = pre == "0" and post == "0"
d["balloon_ok"] = ok
if not ok:
    d["balloon_reason"] = f"sc_balloon.sh check rc {pre} before, {post} after the round"
c = json.load(open(gate, encoding="utf-8"))
d["cotenant_ok"] = c.get("ok") is True
if c.get("ok") is not True:
    d["cotenant_reason"] = f"gates reading {c}"
json.dump(d, open(path, "w", encoding="utf-8"), indent=2)
EOF
}

# one judged round into $out/judged/<label>: `tetraeder cluster judged-round` on node a (the head), then the files back
run_round() { # run_round LABEL
  local label=$1 d="$out/judged/$1" n k t0 P
  mkdir -p "$d"
  P=$(p6dir a)
  if [ "$label" = last ]; then probe_link aalast; else probe_link "$label"; fi # raw points per round: reported, not gated
  if [ "$lb_fake" = 1 ]; then
    t0=$(python3 -I -c 'import time; print(time.time())')
    bx a 300 "$(tet) cluster judged-round --cluster $P/cluster.toml --key $P/keys/key.pem --cert $P/keys/cert.pem --plan $P/judged-plan.toml --out $P/judged/$label" | tail -n 1
    for k in cluster.json seg-a.json seg-a.gpu.txt seg-b.json seg-b.gpu.txt; do get a "judged/$label/$k" "$d/$k"; done
  else
    local sha pre_a pre_b done_line envs="PROMPT_TOKENS=${PROMPT_TOKENS:-2048} GEN_TOKENS=${GEN_TOKENS:-128}"
    [ "$TINY" = 1 ] && envs="PROMPT_TOKENS=64 GEN_TOKENS=16"
    [ "$LOOPBACK" = 1 ] && envs="$envs HIDDEN=$P/sc/hidden.bin"
    sha=$(bx a 60 "cat $P/model.sha256" | tail -n 1)
    [ -n "$sha" ] || die "no model.sha256 on node a"
    for n in a b; do gpu_line "$n" before > "$d/seg-$n.gpu.txt"; done
    pre_a=$(balloon_rc a) pre_b=$(balloon_rc b)
    t0=$(bx a 60 "bash scripts/phase6_box.sh launch judged-$name-$label env $envs bash scripts/phase6_box.sh judged-round $label $sha" | head -n 1)
    done_line=$(wait_line a "logs/launch-judged-$name-$label.log" 'ROUND_DONE' 14400) || die "judged round $label did not finish on node a"
    case "$done_line" in *"rc=0"*) ;; *) die "judged round $label failed on node a: $done_line (logs/launch-judged-$name-$label.log)" ;; esac
    for n in a b; do gpu_line "$n" after >> "$d/seg-$n.gpu.txt"; done
    for k in cluster.json seg-a.json seg-b.json; do get a "judged/$label/$k" "$d/$k"; done
    box_flags a "$d/seg-a.json" "$pre_a" "$(balloon_rc a)"
    box_flags b "$d/seg-b.json" "$pre_b" "$(balloon_rc b)"
  fi
  python3 -I -c 'import json, sys; json.dump({"t_start": float(sys.argv[1]), "frozen_sha256": sys.argv[2]}, open(sys.argv[3], "w"))' "$t0" "${FROZEN_SHA:-}" "$d/meta.json"
  if [ -n "${REFERENCE_CMD:-}" ] && [ "$label" != last ]; then
    bx a 14400 "env ROUND_FILE=$(p6dir a)/reference-$label.json bash -c '$REFERENCE_CMD'" > /dev/null && get a "reference-$label.json" "$out/reference/$label.json"
  fi
}

freeze_gate() {
  [ -n "${FROZEN_SHA:-}" ] || die "set FROZEN_SHA to the frozen hash: commit the prediction first"
  python3 -I scripts/sc_gates.py freeze --results "$out" --frozen-sha "$FROZEN_SHA" || die "the frozen prediction does not match FROZEN_SHA or a frozen file changed: nothing is judged"
  summ precheck "$out" freeze || die "a pre-freeze rule failed: nothing is judged"
}

st_judged() {
  local k ph
  if [ "$TINY" = 1 ] && [ "$lb_fake" = 0 ]; then
    # the dry run: one round of the real judged-round on the tiny model. Nothing was priced or frozen, so the round is judged by its own
    # rules only (the files complete, covering, no disk bytes in prefill, the prompt of the anchors, flags clean)
    ph=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["prompt"]["fnv1a64"])' "$out/session.json")
    say "judged round (dry run, one round)"
    run_round r1
    summ round "$out/judged/r1" --prompt-hash "$ph" || die "the dry run's judged round is not clean"
    say "judged done: the round is clean"
    return 0
  fi
  freeze_gate
  [ "$lb_fake" = 1 ] && [ ! -x "$lb/tetraeder" ] && loopback_tools
  while summ next-round "$out" > /dev/null; do
    k=$(($(ls -d "$out"/judged/r[0-9]* 2> /dev/null | wc -l) + 1))
    [ "$k" -le "${MAX_ROUNDS:-8}" ] || break
    say "judged round $k"
    run_round "r$k"
    summ round "$out/judged/r$k" --prompt-hash "$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["prompt"]["fnv1a64"])' "$out/session.json")" || say "round $k is unclean: it is rerun, not dropped"
    if [ "$k" = 2 ]; then
      # the early A/A (ADR 0091 answer 8): the first two rounds are one pair of the same plan right after warm-up
      python3 -I scripts/sc_gates.py aa "$out/judged/r1/cluster.json" "$out/judged/r2/cluster.json" | tee "$out/judged/aa-early.txt"
      [ "${PIPESTATUS[0]}" = 0 ] || die "early A/A outside the band: the session cannot be clean, abort, fetch and destroy"
    fi
  done
  summ next-round "$out" || true
  say "judged done"
}

st_aalast() {
  freeze_gate
  [ "$lb_fake" = 1 ] && [ ! -x "$lb/tetraeder" ] && loopback_tools
  run_round last
  rm -rf "$out/judged/aa-last"
  mv "$out/judged/last" "$out/judged/aa-last"
  python3 -I scripts/sc_gates.py aa "$out/judged/r1/cluster.json" "$out/judged/aa-last/cluster.json" | tee "$out/judged/aa-last.txt" || true
  say "aalast done: it must be the last run of the judged plan"
}

st_linkend() {
  need_spans
  probe_link end
  # the frozen plan priced again with the end probe's curves: only the link differs from the frozen graph
  bx a 300 "$(tet) cluster merge $(p6dir a)/hw-cluster.json --swap-link $(p6dir a)/probe-end.json -o $(p6dir a)/hw-cluster-end.json" | tail -n 1
  get a hw-cluster-end.json "$out/predictions/hw-cluster-end.json"
  price hw-cluster-end.json predict-cluster-end.json > "$out/predictions/predict-end.log" 2>&1 || { cat "$out/predictions/predict-end.log"; die "the end-priced prediction was refused"; }
  tail -n 2 "$out/predictions/predict-end.log"
  get a predict-cluster-end.json "$out/predictions/predict-cluster-end.json"
  summ link "$out/link/probe-start.json" "$out/link/probe-end.json" || true # raw points, reported
}

st_summary() {
  summ summary "$out"
  python3 -I -c 'import json, sys; v = json.load(open(sys.argv[1])); sys.exit(0 if v["verdict"] == "pass" else 1)' "$out/verdict-s.json"
}

# the dry run on the boxes themselves, before anything is downloaded: the same stages on the tiny sharded MoE with the real binaries (release,
# CUDA), its own results name, no balloon (the tiny model needs none; run it before the `budget` stage so the balloon is not held yet)
st_dryrun() {
  [ "$TINY" = 1 ] && die "dryrun inside a dry run"
  local rn="${DRYRUN_NAME:-$name-dryrun}"
  say "dry run on the tiny sharded MoE, real binaries on both boxes: results $rn"
  env TINY=1 EMULATE_RAM_GB=0 RESULTS_NAME="$rn" STAGES="${DRYRUN_STAGES:-budget plan fetch calib merge judged}" bash "$repo/scripts/phase6_exit_s.sh" \
    || die "the dry run failed: fix it before the 142 GB download"
  say "dryrun passed ($rn)"
}

check_order() {
  local s seen=0
  for s in $stages; do
    declare -F "st_$s" > /dev/null || die "unknown stage $s"
    [ "$seen" = 1 ] && [ "$s" = judged ] && die "judged after aalast: the A/A last must be the last run of the plan"
    [ "$s" = aalast ] && seen=1
  done
}
check_order
rc=0
for s in $stages; do
  say "STAGE $s"
  "st_$s" || rc=$?
done
exit "$rc"
