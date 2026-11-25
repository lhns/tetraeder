#!/usr/bin/env bash
# Phase 6 exit, Case S (ADR 0091, answers 6 to 8, ADR 0092 rerun rules): Qwen3-235B-A22B Q4_K_M on two boxes (one RTX 3090 each, a
# 64 GB RAM budget by balloon), one segment per node, pp2048 and tg128 within 15% of a prediction frozen before the first judged run.
# Runs on the DRIVER machine and works on the boxes through `taxameter ssh --id`; the calibration is scripts/sc_box.sh on each box
# (its stages, with its gates), the verdict is scripts/phase6_exit_s_summary.py. No Rust, no cargo locally.
#
# Two invocations, with a commit between them (the frozen prediction goes to git before any judged run):
#   STAGES="gates setup plan calib predict" bash scripts/phase6_exit_s.sh          ... commit predictions/, link/, plans/, calib/
#   FROZEN_SHA=<FROZEN.sha256> STAGES="judged aalast linkend summary" bash scripts/phase6_exit_s.sh
# Stages:
#   gates     both boxes, before any download: power, idle, one NUMA node, co-tenant by engine_sample.py. A miss ends the session
#   setup     certificates, cluster.toml after create, the 142 GB model on both boxes (`create` went through scripts/sc_offers.sh:
#             real --download-gb 150, disk_gb 230), build, the node daemons
#   plan      `cluster profile` and the start link probe (band 0.08 later), `tetraeder-plan candidates --cluster`, the analytic pick
#             among plans that compute on both nodes (a plan on one node is never a candidate), the spans checked against the regime
#             the segment confirmation validated
#   calib     per node, sc_box.sh: env, disk, balloon (the RAM budget measured), the whole-model trace on the leader (routing trace and
#             hidden rows), the hidden rows to node b, anchors of the node's own segment truncated to a quarter and a half (real hidden
#             rows; layer 0 takes tokens), two rounds, the fit (band limit 0.10, refused above)
#   predict   the fitted node graphs merged into the cluster graph, `tetraeder-plan predict --cluster --judged` (routing trace only),
#             SHA256SUMS and FROZEN.sha256 of everything it used; refuses on any failed pre-freeze rule
#   judged    refuses unless FROZEN_SHA equals the frozen hash and every frozen file matches; rounds of the judged plan until three
#             are clean or eight have run (unclean rounds are rerun on the same prediction, all reported); the early A/A after round
#             2 aborts the session outside 0.08 on T
#   aalast    the A/A last: the judged plan once more, the last run of the session
#   linkend   the probe again; 16 KiB and 32 MiB must agree with the start probe within 0.08, else the rounds are rerun after a new
#             probe and prediction (a new session)
#   summary   session.json and the verdict
# THE JUDGED ROUND: no Rust command runs pp2048 then tg128 across the two nodes with the warm request, no prefix reuse and each
# segment's counters in an exit_regime-shaped JSON (tree node cluster-exit-round). Until it exists the judged and aalast stages refuse
# unless JUDGED_ROUND_CMD names a command that runs on BOTH boxes (env NODE=a|b, ROUND_DIR, PLAN, FROZEN_SHA, HIDDEN) and writes, on
# node a, ROUND_DIR/cluster.json ({"runs": [{prefill_tok_s, decode_tok_s, prompt: {tokens, fnv1a64}}]}) and on each node
# ROUND_DIR/seg-<node>.json (exit_regime shape: warm, prefill_io, decode_layers, balloon_ok, cotenant_ok) with seg-<node>.gpu.txt
# (the `before:`/`after:` nvidia-smi lines of sc_box.sh run_one). REFERENCE_CMD (optional, node a, env ROUND_FILE) writes the report-only
# single-node round.
#
# Env: RESULTS_NAME (required), ID_A, ID_B, T_CREATED_A, T_CREATED_B, STAGES, FROZEN_SHA, JUDGED_ROUND_CMD, REFERENCE_CMD, RESULTS_ROOT,
# SPANS ("a 0..47,b 47..94": the regime the confirmation validated with 32 VRAM and 96 RAM slots, V and R), A1/A2 per node (11 and 23
# layers after the span's first), MAX_ROUNDS (8), CLEAN_ROUNDS (3), EMULATE_RAM_GB (64).
# LOOPBACK=1: the dry run with the stand-in tools of scripts/phase6_loopback.py (LB_SCENARIO breaks one rule on purpose); no GPU.
cd "$(dirname "$0")/.." || exit 1
repo=$PWD
export MSYS_NO_PATHCONV=1
LOOPBACK="${LOOPBACK:-0}"
say() { echo "[$(date +%T)] $*"; }
die() { say "FATAL: $*"; exit 1; }
name="${RESULTS_NAME:?set RESULTS_NAME}"
out="${RESULTS_ROOT:-$repo/bench/results}/$name"
stages="${STAGES:-gates setup plan calib predict}"
spans="${SPANS:-a 0..47,b 47..94}"
vs="${V:-32}" rs="${R:-96}"
if [ "$LOOPBACK" = 1 ]; then
  lb="${LB_DIR:-$repo/runs/phase6-loopback-s}"
  port_a="${LB_PORT_A:-17401}" port_b="${LB_PORT_B:-17402}" head_port="${LB_HEAD_PORT:-18480}" poll_s=0.1 wait_step=0.5
else
  head_port=8080 poll_s=0.2 wait_step=5
fi
mkdir -p "$out"/{gates,certs,cluster,link,plans,calib,env,hidden,trace,predictions,judged,reference}
# shellcheck source=scripts/phase6_exit_lib.sh
. "$repo/scripts/phase6_exit_lib.sh"

summ() { python3 -I scripts/phase6_exit_s_summary.py "$@"; }
span_of() { echo "$spans" | tr ',' '\n' | awk -v n="$1" '$1 == n { print $2 }'; }
first_of() { span_of "$1" | cut -d. -f1; }
end_of() { span_of "$1" | awk -F'[.][.]' '{ print $2 }'; }
tet() { if [ "$LOOPBACK" = 1 ]; then echo "$lb/tetraeder"; else echo target/release/tetraeder; fi; }
tplan() { if [ "$LOOPBACK" = 1 ]; then echo "$lb/tetraeder-plan"; else echo target/release/tetraeder-plan; fi; }
resrel() { if [ "$LOOPBACK" = 1 ]; then echo "results/$name-n$1"; else echo "../tetraeder/bench/results/$name-n$1"; fi; } # under P6
shim() { # shim NAME SUBCOMMAND-WORDS: a loopback executable that runs the stand-in
  mkdir -p "$lb"
  printf '#!/usr/bin/env bash\nexec python3 "%s/scripts/phase6_loopback.py" %s "$@"\n' "$(cd "$repo" && { pwd -W 2> /dev/null || pwd; })" "$2" > "$lb/$1"
  chmod +x "$lb/$1"
}
loopback_tools() { shim tetraeder "" && shim tetraeder-plan plan; }

# one sc_box.sh run on a node (STAGES of sc_box.sh), waited for; the results dir of the box is <name>-n<node>
scbox() { # scbox NODE TAG "STAGES"
  local n=$1 tag=$2 st=$3 s e a1 a2
  s=$(first_of "$n") e=$(end_of "$n")
  a1=$((s + ${A1_LAYERS:-11})) a2=$((s + ${A2_LAYERS:-23}))
  if [ "$LOOPBACK" = 1 ]; then
    (cd "$repo" && python3 scripts/phase6_loopback.py sc-box --node "$n" --out "$lb/$n/$(resrel "$n")" --stages "$st" --span "$s..$e") | tail -n 1
    return 0
  fi
  bx "$n" 60 "bash scripts/phase6_box.sh launch sc-$n-$tag env RESULTS_NAME=$name-n$n SPLIT=$s END=$e A1_END=$a1 A2_END=$a2 HIDDEN_AT=$(end_of a) V=$vs R=$rs EMULATE_RAM_GB=${EMULATE_RAM_GB:-64} STAGES='$st' bash scripts/sc_box.sh" | head -n 2
}
scbox_wait() { # scbox_wait NODE "STAGES"
  [ "$LOOPBACK" = 1 ] && return 0
  local line
  line=$(wait_line "$1" ../tetraeder/runs/sc/sc_box.log "SC_BOX_DONE rc=" 14400) || die "sc_box.sh on node $1 did not finish ($2)"
  case "$line" in *"rc=0"*) say "node $1 sc_box.sh [$2] done" ;; *) die "sc_box.sh on node $1 failed ($2): $line" ;; esac
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
  if [ "$LOOPBACK" = 1 ]; then rm -rf "$lb"; loopback_tools; fi
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
    say "the 142 GB model on both boxes (REQUIRE_ONE_NUMA=1) and the build"
    for b in a b; do
      bx "$b" 120 "bash scripts/phase6_box.sh launch fetch-235b env REQUIRE_ONE_NUMA=1 bash scripts/fetch_exit_model.sh" | tail -n 1
      scbox "$b" b1 "build" > /dev/null
    done
    for b in a b; do
      wait_line "$b" ../tetraeder/runs/vast/fetch_exit.log 'FETCH_DONE rc=0' 14400 > /dev/null || die "the model download on $b did not finish ok"
      scbox_wait "$b" build
    done
  fi
  bx b 60 "bash scripts/phase6_box.sh node"
  bx a 60 "bash scripts/phase6_box.sh node"
  python3 -I - "$out/session.json" "$spans" "$vs" "$rs" << 'EOF'
import json, os, sys
p, spans, v, r = sys.argv[1:5]
d = json.load(open(p)) if os.path.exists(p) else {}
d["segments"] = [{"node": n.split()[0], "span": [int(x) for x in n.split()[1].split("..")], "slots": [int(v), int(r)]} for n in spans.split(",")]
d["segments"][0]["embed"], d["segments"][-1]["head"] = True, True
d["band_source"] = "bench/spec/phase0.toml"
d["boxes"] = {"a": {"id": os.environ.get("ID_A", "loopback")}, "b": {"id": os.environ.get("ID_B", "loopback")}}
json.dump(d, open(p, "w"), indent=2)
EOF
  say "setup done"
}

st_plan() {
  local P
  P=$(p6dir a)
  [ "$LOOPBACK" = 1 ] && [ ! -x "$lb/tetraeder" ] && loopback_tools
  say "layout, cluster profile and the start probe"
  bx a 600 "$(tet) plan layout .hf_cache/Qwen__Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf --out $P/layout.json" | tail -n 1
  local cl="--cluster $P/cluster.toml --key $P/keys/key.pem --cert $P/keys/cert.pem"
  bx a 900 "$(tet) cluster probe $cl --repeats 20 -o $P/probe-start.json" | tail -n 2
  bx a 300 "$(tet) cluster profile $cl --probe $P/probe-start.json -o $P/hw.json" | tail -n 2
  get a probe-start.json "$out/link/probe-start.json"
  get a hw.json "$out/link/hw.json"
  get a layout.json "$out/layout.json"
  say "candidates across the nodes, then the pick"
  bx a 300 "rm -rf $P/candidates; $(tplan) candidates --cluster --hw $P/hw.json --layout $P/layout.json --plans-dir $P/candidates --out $P/candidates.json --prompt 2048" | tail -n 3
  get a candidates.json "$out/plans/candidates.json"
  mkdir -p "$out/plans/candidates"
  local f
  for f in $(python3 -I -c 'import json, sys; print(" ".join(c["plan"] for c in json.load(open(sys.argv[1]))["candidates"] if c.get("plan")))' "$out/plans/candidates.json"); do
    get a "candidates/$f" "$out/plans/candidates/$f"
  done
  summ pick "$out/plans/candidates.json" "$out/plans/candidates" "$out/plans/judged-plan.toml" || die "no plan that computes on both nodes: Case S has no judged candidate"
  summ span "$out/plans/judged-plan.toml" || die "the pick does not compute on both nodes"
  local got want
  got=$(summ spans "$out/plans/judged-plan.toml" | tr -d '\r' | tr '\n' ',' | sed 's/,$//')
  want="$spans"
  [ "$got" = "$want" ] || die "the pick's spans '$got' are not the regime the segment confirmation validated '$want' (ADR 0092 section 6): it goes back to the ADR"
  put a "$out/plans/judged-plan.toml" judged-plan.toml
  say "plan done: $got"
}

st_calib() {
  local n
  say "node a: env, disk, balloon, the whole-model trace; node b: env, disk, balloon"
  scbox a c1 "env disk balloon trace" > /dev/null
  scbox b c1 "env disk balloon" > /dev/null
  scbox_wait a "env disk balloon trace"
  scbox_wait b "env disk balloon"
  if [ "$LOOPBACK" != 1 ]; then
    say "the trace and the hidden rows to node b"
    for f in runs/sc/hidden.bin runs/sc/routing.trace; do
      get a "../tetraeder/$f" "$out/.xfer-$(basename "$f")"
      gzip -f "$out/.xfer-$(basename "$f")"
      put b "$out/.xfer-$(basename "$f").gz" "../tetraeder/$f.gz"
      bx b 600 "gunzip -f /root/tetraeder/$f.gz"
      rm -f "$out/.xfer-$(basename "$f").gz"
    done
    for f in routing hidden; do get a "$(resrel a)/trace/$f.sha256" "$out/trace/$f.sha256"; done
    put b "$out/trace/routing.sha256" "$(resrel b)/trace/routing.sha256"
    put b "$out/trace/hidden.sha256" "$(resrel b)/trace/hidden.sha256"
    for n in a b; do
      bx "$n" 600 "sha256sum /root/tetraeder/runs/sc/hidden.bin | cut -d' ' -f1" | tail -n 1 > "$out/hidden/$n.sha256"
    done
  else
    for f in routing hidden; do get a "$(resrel a)/trace/$f.sha256" "$out/trace/$f.sha256"; done
    for n in a b; do cp "$out/trace/hidden.sha256" "$out/hidden/$n.sha256"; done
  fi
  say "anchors and the fit on both nodes"
  scbox a c2 "anchors predict" > /dev/null
  scbox b c2 "anchors predict" > /dev/null
  scbox_wait a "anchors predict"
  scbox_wait b "anchors predict"
  for n in a b; do
    mkdir -p "$out/calib/$n/anchors" "$out/env/$n"
    for k in a1-r1 a1-r2 a2-r1 a2-r2; do get "$n" "$(resrel "$n")/anchors/$k.json" "$out/calib/$n/anchors/$k.json"; done
    for k in pred.json fitted.toml replay.json; do get "$n" "$(resrel "$n")/predictions/$k" "$out/calib/$n/$k"; done
    get "$n" "$(resrel "$n")/env.json" "$out/env/$n/env.json"
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
  summ precheck "$out" calib || die "calibration gates failed (fit band over the limit, anchors on fixed-seed input, gates, budget or hidden rows): nothing frozen"
  say "calib done"
}

st_predict() {
  local P tsha ph
  P=$(p6dir a)
  [ "$LOOPBACK" = 1 ] && [ ! -x "$lb/tetraeder-plan" ] && loopback_tools
  python3 -I scripts/phase6_exit_s_merge.py "$out/link/hw.json" --node "a=$out/calib/a/fitted.toml" --node "b=$out/calib/b/fitted.toml" \
    -o "$out/predictions/hw-cluster.json"
  tsha=$(awk '{ print $1 }' "$out/trace/routing.sha256")
  ph=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["prompt"]["fnv1a64"])' "$out/session.json")
  put a "$out/predictions/hw-cluster.json" hw-cluster.json
  put a "$out/calib/a/replay.json" replay-a.json
  put a "$out/calib/b/replay.json" replay-b.json
  # the prediction is judged: routing from the trace only, the sha256 of the trace it was priced from, one replay per segment
  if ! bx a 600 "$(tplan) predict --cluster $P/judged-plan.toml --hw $P/hw-cluster.json --layout $P/layout.json --tier-replay $(span_of a)=$P/replay-a.json --tier-replay $(span_of b)=$P/replay-b.json --trace-sha256 $tsha --prompt-hash $ph --judged --prompt 2048 --gen 128 --out $P/predict-cluster.json" > "$out/predictions/predict.log" 2>&1; then
    cat "$out/predictions/predict.log"
    die "predict --cluster --judged refused: nothing is frozen"
  fi
  tail -n 2 "$out/predictions/predict.log"
  get a predict-cluster.json "$out/predictions/predict-cluster.json"
  (cd "$out" && sha256sum layout.json link/probe-start.json link/hw.json predictions/hw-cluster.json predictions/predict-cluster.json \
    plans/judged-plan.toml plans/candidates.json trace/routing.sha256 trace/hidden.sha256 calib/a/pred.json calib/b/pred.json \
    calib/a/replay.json calib/b/replay.json calib/a/fitted.toml calib/b/fitted.toml > predictions/SHA256SUMS \
    && (cd predictions && sha256sum SHA256SUMS > FROZEN.sha256))
  python3 -I -c 'import json, sys, time; json.dump({"t_frozen": time.time(), "sha256": open(sys.argv[1]).read().split()[0]}, open(sys.argv[2], "w"))' \
    "$out/predictions/FROZEN.sha256" "$out/predictions/frozen.json"
  summ precheck "$out" freeze || die "a pre-freeze rule failed: the prediction is not frozen (delete predictions/FROZEN.sha256)"
  cat "$out/predictions/FROZEN.sha256"
  say "predict done: frozen. Commit predictions/, link/, plans/, calib/, trace/ and push; then run the judged stages with FROZEN_SHA=$(awk '{ print $1 }' "$out/predictions/FROZEN.sha256")"
}

# one judged round into $out/judged/<label>: on both boxes, then the files back
run_round() { # run_round LABEL
  local label=$1 d="$out/judged/$1" n t0 t0x
  mkdir -p "$d"
  if [ "$LOOPBACK" = 1 ]; then
    t0=$(python3 -I -c 'import time; print(time.time())')
    (cd "$repo" && python3 scripts/phase6_loopback.py judged-round --nodes a,b --round "${label#r}" --out "$d")
  else
    [ -n "${JUDGED_ROUND_CMD:-}" ] || die "no judged-round command exists: the two-node pp2048 and tg128 round with per-segment counters is not built (tree node cluster-exit-round). Set JUDGED_ROUND_CMD (see the header)"
    for n in a b; do
      t0x=$(bx "$n" 60 "bash scripts/phase6_box.sh launch judged-$label env NODE=$n ROUND_DIR=$(p6dir "$n")/judged/$label PLAN=$(p6dir "$n")/judged-plan.toml FROZEN_SHA=$FROZEN_SHA HIDDEN=/root/tetraeder/runs/sc/hidden.bin bash -c '$JUDGED_ROUND_CMD'" | head -n 1)
      [ "$n" = a ] && t0=$t0x
    done
    for n in a b; do wait_line "$n" "logs/launch-judged-$label.log" 'ROUND_DONE' 7200 > /dev/null || die "judged round $label did not finish on node $n"; done
    for k in cluster.json seg-a.json seg-a.gpu.txt; do get a "judged/$label/$k" "$d/$k"; done
    for k in seg-b.json seg-b.gpu.txt; do get b "judged/$label/$k" "$d/$k"; done
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
  local k
  freeze_gate
  [ "$LOOPBACK" = 1 ] && [ ! -x "$lb/tetraeder" ] && loopback_tools
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
  [ "$LOOPBACK" = 1 ] && [ ! -x "$lb/tetraeder" ] && loopback_tools
  run_round last
  rm -rf "$out/judged/aa-last"
  mv "$out/judged/last" "$out/judged/aa-last"
  python3 -I scripts/sc_gates.py aa "$out/judged/r1/cluster.json" "$out/judged/aa-last/cluster.json" | tee "$out/judged/aa-last.txt" || true
  say "aalast done: it must be the last run of the judged plan"
}

st_linkend() {
  local P
  P=$(p6dir a)
  bx a 900 "$(tet) cluster probe --cluster $P/cluster.toml --key $P/keys/key.pem --cert $P/keys/cert.pem --repeats 20 -o $P/probe-end.json" | tail -n 1
  get a probe-end.json "$out/link/probe-end.json"
  summ link "$out/link/probe-start.json" "$out/link/probe-end.json" \
    || say "the link drifted past the band: the rounds are rerun after a new probe and prediction (ADR 0091), so this session gives no verdict"
}

st_summary() {
  summ summary "$out"
  python3 -I -c 'import json, sys; v = json.load(open(sys.argv[1])); sys.exit(0 if v["verdict"] == "pass" else 1)' "$out/verdict-s.json"
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
