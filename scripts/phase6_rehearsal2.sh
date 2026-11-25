#!/usr/bin/env bash
# Phase 6 rehearsal 2 (ADR 0091, tree node bench-phase6-rehearsal-2): the steps that follow `scripts/phase6_exit_f.sh` on the same two
# boxes (same RESULTS_NAME, ID_A, ID_B; no loopback). Runs on the DRIVER machine through `taxameter ssh --id` (scripts/phase6_exit_lib.sh).
# Stages (STAGES, each safe to repeat):
#   prefix   head restarted with replicas-prefix.toml (the declared prefill rate lowered so a fetch is priced to win), then
#            scripts/phase6_prefix.py twice: holder on the head's own replica, and holder on the remote replica. Metrics and
#            the admit lines (source=, common=) are the evidence of a fetch
#   drain    load on the head, `tetraeder node --drain --id b` 30 s in (no cluster plan), load carried on 40 s; which requests B admitted
#   reset    head and both daemons stopped, daemons started again (the replica engines hold the VRAM the profile needs)
#   probe    NAME: `cluster probe` on the idle cluster into link/probe-NAME.json
#   plan30   layout of the 30B, `cluster profile`, `tetraeder-plan candidates --cluster`, the pick that computes on both nodes
#   freeze   `tetraeder-plan predict --cluster` of the pick on the profiled graph (judged only with TRACE_ARGS), SHA256SUMS, FROZEN.sha256
#   serve30  head restarted on the plan: `serve --cluster-plan PLAN --cluster-hw HW --no-prefix-reuse`
#   rounds   ROUNDS (3) timed requests of the frozen prompt through the head (scripts/phase6_split_round.py), a probe before each
#   linkend  the end probe, the frozen plan priced again with its link curves only (--swap-link), the drift per size
# Env: RESULTS_NAME, ID_A, ID_B, STAGES, ROUNDS, PROMPT_TOKENS (2048), GEN (128), PREFILL_TPS (100), TRACE_ARGS (extra args of `predict`).
cd "$(dirname "$0")/.." || exit 1
repo=$(pwd -W 2> /dev/null || pwd) # a Windows driver needs C:/ paths for python3
export MSYS_NO_PATHCONV=1
LOOPBACK=0
say() { echo "[$(date +%T)] $*"; }
die() { say "FATAL: $*"; exit 1; }
name="${RESULTS_NAME:?set RESULTS_NAME}"
out="${RESULTS_ROOT:-$repo/bench/results}/$name"
stages="${STAGES:?set STAGES}"
rounds="${ROUNDS:-3}"
ptok="${PROMPT_TOKENS:-2048}" gen="${GEN:-128}"
wait_step=5
mkdir -p "$out"/{prefix,drain,link,plans,predictions,split30}
# shellcheck source=scripts/phase6_exit_lib.sh
. "$repo/scripts/phase6_exit_lib.sh"
P=$(p6dir a)
Pb=$(p6dir b)
tet=target/release/tetraeder
tplan=target/release/tetraeder-plan
gguf=.hf_cache/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf
cl="--cluster $P/cluster.toml --key $P/keys/key.pem --cert $P/keys/cert.pem"
summ() { python3 -I scripts/phase6_exit_s_summary.py "$@"; }

head_up() { # waits for the head's /health (the engine loads first)
  local t0=$SECONDS
  until bx a 30 "python3 -I -c 'import http.client, sys; c = http.client.HTTPConnection(\"127.0.0.1\", 8080, timeout=5); c.request(\"GET\", \"/health\"); sys.exit(0 if c.getresponse().status == 200 else 1)'" > /dev/null 2>&1; do
    [ $((SECONDS - t0)) -lt 400 ] || die "the head does not answer /health after 400 s"
    sleep 3
  done
}
restart_head() { # restart_head SERVE_ARGS...: the head only; the daemons keep running
  bx a 60 "bash scripts/phase6_box.sh stop serve" | tail -n 1
  sleep 3
  bx a 60 "RUST_LOG=info,tetraeder::step=debug bash scripts/phase6_box.sh serve $*" | tail -n 1
  head_up
}
metrics() { bx a 60 "python3 -I -c 'import http.client; c = http.client.HTTPConnection(\"127.0.0.1\", 8080, timeout=10); c.request(\"GET\", \"/metrics\"); print(c.getresponse().read().decode())' | grep -E 'prefix|kv' | grep -v '^#'" || true; }

st_prefix() {
  local tps="${PREFILL_TPS:-100}"
  sed "s/^cap = \(.*\)$/cap = \1\nprefill_tps = $tps/" "$out/cluster/replicas.toml" > "$out/cluster/replicas-prefix.toml"
  put a "$out/cluster/replicas-prefix.toml" replicas-prefix.toml
  say "head on replicas-prefix.toml (declared prefill $tps tok/s)"
  restart_head "--replicas $P/replicas-prefix.toml ${PREFIX_HEAD_ARGS:-}"
  bx a 120 "python3 scripts/phase6_load.py http://127.0.0.1:8080 --key-file $P/api-key.txt --concurrency 4 --duration 15 --prompt-tokens 300 --max-tokens 32 --label warm-prefix --out $P/warm-prefix.jsonl" | tail -n 1
  local mode
  for mode in head-holder remote-holder; do
    local extra=""
    [ "$mode" = remote-holder ] && extra="--first-on-b"
    say "prefix fetch, $mode"
    bx a 60 "date +%s.%N > $P/prefix-$mode.t0"
    metrics > "$out/prefix/metrics-$mode-before.txt"
    bx a 900 "python3 scripts/phase6_prefix.py http://127.0.0.1:8080 --key-file $P/api-key.txt --prompt-tokens 3000 $extra --out $P/prefix-$mode.json" | tail -n 5
    metrics > "$out/prefix/metrics-$mode-after.txt"
    get a "prefix-$mode.json" "$out/prefix/prefix-$mode.json"
    get a "prefix-$mode.t0" "$out/prefix/prefix-$mode.t0"
  done
  # the admit lines of both replicas after the head restart: where each request ran, and `source=` or `common=` of the fetch
  bx a 60 "grep -a ' admit ' $P/logs/serve.log | tail -n 60" > "$out/prefix/admits-head.txt" || true
  bx b 60 "grep -a ' admit ' $Pb/logs/node.log | tail -n 60" > "$out/prefix/admits-b.txt" || true
  bx a 60 "grep -a -i 'prefix\|fetch\|grant' $P/logs/serve.log | tail -n 60" > "$out/prefix/serve-fetch-lines.txt" || true
  bx b 60 "grep -a -i 'prefix\|fetch\|grant' $Pb/logs/node.log | tail -n 60" > "$out/prefix/node-b-fetch-lines.txt" || true
  say "prefix done"
}

st_drain() {
  say "drain without a plan: load 70 s, drain of b at +30 s"
  local t0 td
  t0=$(bx a 60 "bash scripts/phase6_box.sh launch drain-load env python3 scripts/phase6_load.py http://127.0.0.1:8080 --key-file $P/api-key.txt --concurrency 8 --duration 70 --prompt-tokens 400 --max-tokens 128 --label drain --out $P/drain-load.jsonl" | head -n 1)
  sleep 30
  td=$(bx a 120 "date +%s.%N; $tet node $cl --drain --id b 2>&1 | tail -n 2" | tr '\n' ' ')
  echo "drain command at (a clock): $td; load started at $t0"
  printf '%s\n' "{\"load_t0\": \"$t0\", \"drain_output\": \"$td\"}" > "$out/drain/meta.json"
  wait_line a "logs/launch-drain-load.log" '^phase6_load ' 200 > /dev/null || say "the drain load did not finish"
  get a drain-load.jsonl "$out/drain/load.jsonl"
  bx b 120 "bash scripts/phase6_box.sh admits $Pb/admits-drain.txt > /dev/null"
  get b admits-drain.txt "$out/drain/b-admits.txt"
  bx a 60 "grep -a -i 'drain' $P/logs/serve.log | tail -n 20" > "$out/drain/serve-drain-lines.txt" || true
  say "drain done"
}

st_reset() {
  bx a 60 "bash scripts/phase6_box.sh stop serve" | tail -n 1
  bx a 60 "bash scripts/phase6_box.sh stop node" | tail -n 1
  bx b 60 "bash scripts/phase6_box.sh stop node" | tail -n 1
  sleep 5
  bx b 60 "bash scripts/phase6_box.sh node"
  bx a 60 "bash scripts/phase6_box.sh node"
  sleep 10
  for b in a b; do bx "$b" 60 "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader" | tail -n 1; done
  say "reset done: both daemons idle"
}

st_probe() { # STAGES=probe PROBE=NAME
  local n="${PROBE:?set PROBE=name}"
  bx a 900 "$tet cluster probe $cl --repeats ${PROBE_REPEATS:-10} --max-bytes ${PROBE_MAX_BYTES:-67108864} -o $P/probe-$n.json" | tail -n 1
  get a "probe-$n.json" "$out/link/probe-$n.json"
  say "probe $n done"
}

st_plan30() {
  bx a 300 "$tet plan layout $gguf --out $P/layout.json" | tail -n 1
  get a layout.json "$out/plans/layout.json"
  bx a 300 "$tet cluster profile $cl --probe $P/probe-start.json -o $P/hw.json" | tail -n 2
  get a hw.json "$out/link/hw.json"
  bx a 300 "rm -rf $P/candidates; $tplan candidates --cluster --hw $P/hw.json --layout $P/layout.json --plans-dir $P/candidates --out $P/candidates.json --prompt $ptok" | tail -n 3
  get a candidates.json "$out/plans/candidates.json"
  mkdir -p "$out/plans/candidates"
  local f
  for f in $(python3 -I -c 'import json, sys; print(" ".join(c["plan"] for c in json.load(open(sys.argv[1]))["candidates"] if c.get("plan")))' "$out/plans/candidates.json"); do
    get a "candidates/$f" "$out/plans/candidates/$f"
  done
  summ pick "$out/plans/candidates.json" "$out/plans/candidates" "$out/plans/judged-plan.toml" || die "no plan that computes on both nodes"
  summ span "$out/plans/judged-plan.toml" || die "the pick does not compute on both nodes"
  put a "$out/plans/judged-plan.toml" judged-plan.toml
  say "plan30 done"
}

price30() { # price30 HW OUT
  bx a 600 "$tplan predict --cluster $P/judged-plan.toml --hw $P/$1 --layout $P/layout.json ${TRACE_ARGS:-} --prompt $ptok --gen $gen --out $P/$2"
}

st_freeze() {
  put a "$out/link/hw.json" hw.json
  put a "$out/plans/judged-plan.toml" judged-plan.toml
  price30 hw.json predict-cluster.json | tail -n 3
  get a predict-cluster.json "$out/predictions/predict-cluster.json"
  (cd "$out" && sha256sum plans/layout.json link/probe-start.json link/hw.json plans/judged-plan.toml plans/candidates.json predictions/predict-cluster.json \
    > predictions/SHA256SUMS && (cd predictions && sha256sum SHA256SUMS > FROZEN.sha256))
  python3 -I -c 'import json, sys, time; json.dump({"t_frozen": time.time(), "sha256": open(sys.argv[1]).read().split()[0]}, open(sys.argv[2], "w"))' \
    "$out/predictions/FROZEN.sha256" "$out/predictions/frozen.json"
  cat "$out/predictions/FROZEN.sha256"
  say "frozen: commit and push predictions/, link/, plans/ before any measured request"
}

st_serve30() {
  put a "$out/link/hw.json" hw.json
  put a "$out/plans/judged-plan.toml" judged-plan.toml
  restart_head "--cluster-plan $P/judged-plan.toml --cluster-hw $P/hw.json --no-prefix-reuse"
  say "head on the plan"
}

st_rounds() {
  local k tag
  bx a 120 "python3 scripts/phase6_split_round.py http://127.0.0.1:8080 --key-file $P/api-key.txt --prompt-file bench/prompts/exit_prompt.txt --prompt-tokens $ptok --cut-only --prompt-out $P/frozen-prompt.txt --out $P/frozen-prompt.json" | tail -n 1
  get a frozen-prompt.json "$out/split30/frozen-prompt.json"
  # a different, shorter prompt warms the engines and the connections; the measured prompt is then still new to every prefix cache
  bx a 300 "python3 scripts/phase6_load.py http://127.0.0.1:8080 --key-file $P/api-key.txt --concurrency 1 --duration 10 --prompt-tokens 600 --max-tokens 32 --label warm30 --out $P/warm30.jsonl" | tail -n 1
  for k in $(seq 1 "$rounds"); do
    tag="mid-$k"
    PROBE=$tag st_probe
    bx a 900 "python3 scripts/phase6_split_round.py http://127.0.0.1:8080 --key-file $P/api-key.txt --prompt-file bench/prompts/exit_prompt.txt --prompt-tokens $ptok --gen $gen --nonce --label r$k --out $P/round-r$k.json" | tail -n 1
    get a "round-r$k.json" "$out/split30/round-r$k.json"
  done
  bx a 60 "grep -a ' admit ' $P/logs/serve.log | tail -n 20" > "$out/split30/admits-head.txt" || true
  bx b 60 "grep -a ' admit ' $Pb/logs/node.log | tail -n 20" > "$out/split30/admits-b.txt" || true
  say "rounds done"
}

st_linkend() {
  PROBE=end st_probe
  bx a 300 "$tet cluster merge $P/hw.json --swap-link $P/probe-end.json -o $P/hw-end.json" | tail -n 1
  get a hw-end.json "$out/predictions/hw-end.json"
  price30 hw-end.json predict-cluster-end.json | tail -n 3
  get a predict-cluster-end.json "$out/predictions/predict-cluster-end.json"
  summ link "$out/link/probe-start.json" "$out/link/probe-end.json" || true
  local args=("start=$out/link/probe-start.json")
  local f
  for f in "$out"/link/probe-mid-*.json; do [ -e "$f" ] && args+=("$(basename "$f" .json | sed 's/^probe-//')=$f"); done
  args+=("end=$out/link/probe-end.json")
  python3 -I scripts/phase6_link_drift.py "${args[@]}" --out "$out/link/drift.json" | tee "$out/link/drift.txt"
  say "linkend done"
}

for s in $stages; do declare -F "st_$s" > /dev/null || die "unknown stage $s"; done
rc=0
for s in $stages; do
  say "STAGE $s"
  "st_$s" || rc=$?
done
exit "$rc"
