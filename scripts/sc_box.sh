#!/usr/bin/env bash
# Session segment-confirm (tree node bench-235b-segment-confirm, ADR 0092 section 6, ADR 0091 Case S): one exit segment of the
# Qwen3-235B-A22B (layers SPLIT..END, no head, VRAM slots and RAM slots per layer, ubatch 2048) predicted from a routing trace
# and anchors, then measured. Stages (STAGES, in order; each ends with a marker in runs/sc/sc_box.log):
#   gates    first, before the model download (run it alone: STAGES=gates): `power_gate.sh` (every GPU at its default power
#            limit), `idle_gate.sh` (clock event reasons), lscpu shows exactly one NUMA node and the co-tenant reading of
#            `engine_sample.py cotenant` is under the limit (ADR 0091 answer 8; scripts/sc_gates.py), else abort
#   env      host facts of the box -> env/
#   build    cargo release build: tetraeder (CUDA), tetraeder-plan, tetraeder-bench and the exit_regime test binary
#   disk     stage 1: `tetraeder plan layout`, `tetraeder-bench disk-curve` (idle, with copy_bytes_per_s), the node graph
#            predictions/hw.toml (scripts/sp_make_hw.py: RTX 3090 prior, 24 GiB of VRAM, the box's threads and disk curve)
#   balloon  the RAM balloon under EMULATE_RAM_GB (scripts/sc_balloon.sh), then the RAM budget of ADR 0092 section 4 into
#            env.json and into hw.toml (cpu0 capacity_bytes)
#   trace    stage 2: whole-model streamed greedy run of the frozen exit prompt as two chained builders (DUMP_V, DUMP_R slots
#            each, no warm, report only): the routing trace and the hidden rows entering layer SPLIT; both hashed
#   anchors  stage 3: the segment truncated to A1_END and A2_END, SLOTS (VRAM, RAM), the hidden rows dumped from the frozen prompt
#            (TETRAEDER_EXIT_HIDDEN_IN, ADR 0091 answer 7), no head, warm 1, ANCHOR_ROUNDS (2) rounds each
#   predict  stage 4: `tetraeder-bench replay --layers SPLIT..END --warm 1` of the trace, `tetraeder-plan segment-predict`
#            (fit on the anchors, price the segment from the replay), then the freeze: SHA256SUMS and FROZEN.sha256. The freeze is
#            refused (nothing written) when an anchor ran on fixed-seed input or the fit band is above FIT_BAND_MAX (0.10)
#   judged   stage 5: refuses to run unless `sc_gates.py judged-start` passes (one NUMA node, the co-tenant reading under the
#            limit, FROZEN_SHA equal to predictions/FROZEN.sha256 and every frozen file matching, fit band within the limit,
#            anchors on real hidden rows); the segment with the trace run's hidden rows as input, warm 1, ROUNDS (3) rounds
#            (a rerun with more adds rounds). Right after round 2 the early A/A (r1 against r2, band 0.08 on T) aborts the
#            session when it is outside the band
#   aalast   the A/A last: the judged plan once more as judged/aa-last.json, behind the same gates. It must be the last stage
#            of the session; scripts/sc_summary.py counts the session clean only with it inside the band
#   report   stage 6, report only: the segment with fixed-seed input
# Case S of the Phase 6 exit (scripts/phase6_exit_s.sh) runs these stages per node with the node's own span: SPLIT=0 is a segment from
# layer 0 (it takes tokens: no hidden rows in, `--embed` for the planner, no anchors-real gate; the exit summary checks the prompt hash
# instead) and HIDDEN_AT is the layer whose rows the trace run dumps for the other node (default SPLIT).
# Env: RESULTS_NAME, STAGES, FROZEN_SHA, FIT_BAND_MAX, COTENANT_S, ROUNDS, ANCHOR_ROUNDS, THREADS, EMULATE_RAM_GB (64; 0 = no balloon),
# MODEL (the first shard), PROMPT_FILE (default bench/prompts/exit_prompt.txt; empty = pseudo-random ids), SPLIT, END, A1_END,
# A2_END, V, R, DUMP_V, DUMP_R, UBATCH, PREFILL, DECODE. Defaults are the 235B segment; the rehearsal on the synthetic model
# overrides them (docs in bench/results/<name>/README.md). Never overwrite this file while it runs.
# The session ends with `$out/DONE` (also on an abort): taxameter's fetch and destroy need it. Tests (tests/test_sc_box.py) set
# SC_LOCK, RESULTS_ROOT, SC_LOG and SC_BOX_OVERRIDES (a file sourced before main, to replace run_one and the gates).
if [ -n "${PIN_CPUS:-}" ] && [ -z "${SC_PINNED:-}" ]; then SC_PINNED=1 exec taskset -c "$PIN_CPUS" bash "$0" "$@"; fi
exec 8> "${SC_LOCK:-/root/tetraeder_sc_box.lock}"
flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
root=$PWD
name="${RESULTS_NAME:?set RESULTS_NAME}"
out="${RESULTS_ROOT:-$root/bench/results}/$name"
mkdir -p "$out"/{env,disk,trace,anchors,predictions,judged,report,build} runs/sc
log="${SC_LOG:-runs/sc/sc_box.log}"
stages="${STAGES:-gates env build disk balloon trace anchors predict}"
say() { echo "[$(date +%T)] $*"; }
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:/root/.local/bin:$PATH"
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh # native ggml as in the build stage: another setting rebuilds ggml
archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
[ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
quota=$(bash scripts/cpu_quota.sh 2> /dev/null || nproc)
threads=$(bash scripts/clamp_threads.sh "${THREADS:-$quota}")
model="${MODEL:-$root/.hf_cache/Qwen__Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf}"
shard="${DISK_FILE:-$root/.hf_cache/Qwen__Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00002-of-00005.gguf}"
[ -e "$shard" ] || shard="$model"
split="${SPLIT:-47}" hidden_at="${HIDDEN_AT:-${SPLIT:-47}}" end="${END:-94}" a1="${A1_END:-58}" a2="${A2_END:-70}"
vs="${V:-32}" rs="${R:-96}" dv="${DUMP_V:-16}" dr="${DUMP_R:-28}"
ub="${UBATCH:-2048}" pre="${PREFILL:-2048}" dec="${DECODE:-128}"
rounds="${ROUNDS:-3}" arounds="${ANCHOR_ROUNDS:-2}"
prompt_src="${PROMPT_FILE-$root/bench/prompts/exit_prompt.txt}"
prompt=""
if [ -n "$prompt_src" ]; then prompt=/root/exit_prompt.txt; cp "$prompt_src" "$prompt"; fi
tet="$root/target/release/tetraeder"
plan="$root/target/release/tetraeder-plan"
bench="$root/target/release/tetraeder-bench"
prior="$root/crates/planner/fixtures/rtx3090_prior.toml"
hidden="$root/runs/sc/hidden.bin"
trace="$root/runs/sc/routing.trace"
p="$out/predictions"
export EMULATE_RAM_GB="${EMULATE_RAM_GB:-64}"
die() {
  say "FATAL: $*"
  echo "SC_BOX_DONE rc=1"
  echo "SC_BOX_DONE rc=1 stages='$stages' $(date -Is)" > "$out/DONE" # fetch and destroy need it, also after an abort
  touch "$out/gpu-sampler.stop"
  exit 1
}

# ---- one engine run: the exit_regime test with the sampler, balloon check and host blocks (as scripts/sp_arms.sh) ----------
run_one() { # run_one TAG JSON [ENV=VALUE...]
  local tag=$1 json=$2
  shift 2
  [ -s "$json" ] && { say "$tag exists, skipped"; return 0; }
  local samples="$root/runs/sc/samples-$tag.jsonl" t0=$SECONDS bpre=ok
  [ -n "${HOST_DISK_DEV:-}" ] || HOST_DISK_DEV=$(python3 scripts/engine_sample.py probe-disk "$model")
  bash scripts/sc_balloon.sh check > "$out/report/balloon-pre-$tag.txt" 2>&1 || bpre="dead before the run"
  say "$tag start: $* threads $threads"
  nvidia-smi --query-gpu=clocks_event_reasons.active,power.draw,power.limit,power.default_limit --format=csv,noheader 2>&1 | sed 's/^/before: /' > "${json%.json}.gpu.txt"
  env "$@" TETRAEDER_EXIT_GGUF="$model" ${prompt:+TETRAEDER_EXIT_PROMPT_FILE="$prompt"} TETRAEDER_EXIT_GPU=1 TETRAEDER_EXIT_OP_OFFLOAD=1 \
    TETRAEDER_EXIT_UBATCH="$ub" TETRAEDER_EXIT_PREFILL="$pre" TETRAEDER_EXIT_DECODE="$dec" TETRAEDER_EXIT_THREADS="$threads" \
    TETRAEDER_EXIT_OUT="$json" TETRAEDER_EXIT_MACHINE="vast $(hostname) segment-confirm $tag" \
    bash scripts/oom_engine.sh timeout -k 30 60m cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda \
    --test exit_regime -- --ignored --nocapture > "${json%.json}.log" 2>&1 8>&- &
  local pid=$!
  TETRAEDER_STAGE="sc-$tag" python3 scripts/engine_sample.py sample --pid "$pid" --disk "$HOST_DISK_DEV" --out "$samples" 8>&- &
  local spid=$!
  wait "$pid"
  local rc=$?
  wait "$spid"
  nvidia-smi --query-gpu=clocks_event_reasons.active,power.draw,power.limit,power.default_limit --format=csv,noheader 2>&1 | sed 's/^/after: /' >> "${json%.json}.gpu.txt"
  if [ -s "$json" ]; then
    python3 scripts/engine_sample.py summarize --samples "$samples" --threads "$threads" --builder-json "$json" \
      --out "$samples.summary" --merge-into "$json" > /dev/null || say "host summary failed for $tag"
    python3 -I scripts/cotenant_guard.py merge --json "$json" || true
    local why ok=true
    bash scripts/sc_balloon.sh check > /dev/null 2>&1 || { ok=false; why="dead or shrunk after the run"; }
    [ "$bpre" = ok ] || { ok=false; why="${why:+$why; }$bpre"; }
    python3 -I scripts/balloon_guard.py merge --json "$json" --ok "$ok" --reason "${why:-}" || true
    gzip -c "$samples" > "${json%.json}.samples.jsonl.gz" 2> /dev/null
  else
    say "$tag: no result JSON (rc $rc); last log lines:"
    tail -n 12 "${json%.json}.log"
  fi
  say "$tag done rc=$rc in $((SECONDS - t0)) s"
}

# the segment's exit_regime environment (the planner's segment-predict writes the same keys; stage predict checks it)
seg_env() { # seg_env END
  echo "TETRAEDER_EXIT_LAYERS=$split..$1" "TETRAEDER_EXIT_SLOTS=$vs" "TETRAEDER_EXIT_RAM_SLOTS=$rs" "TETRAEDER_EXIT_PREFILL_WARM=1"
}

# ADR 0091 answer 8 and docs/box-driver.md: one NUMA node and a quiet host, read by the repo's own sampler. Run it before the
# model download (STAGES=gates); the judged stages check both again from the files it writes.
st_gates() {
  bash scripts/power_gate.sh > "$out/env/power-gate.txt" 2>&1 || { cat "$out/env/power-gate.txt"; die "a GPU is not at its default power limit: destroy the box"; }
  bash scripts/idle_gate.sh > "$out/env/idle-gate.txt" 2>&1 || { cat "$out/env/idle-gate.txt"; die "idle gate: clock event reasons set while idle: destroy the box"; }
  lscpu > "$out/env/lscpu.txt" 2>&1
  python3 -I scripts/sc_gates.py numa "$out/env/lscpu.txt" || die "not a one-NUMA-node host: destroy the box before the download"
  python3 scripts/engine_sample.py cotenant --seconds "${COTENANT_S:-20}" --out "$out/env/cotenant.json" \
    || die "co-tenant reading over the limit or without counters (env/cotenant.json): destroy the box"
  python3 -I scripts/sc_gates.py cotenant "$out/env/cotenant.json" || die "co-tenant gate"
  say "gates done"
}

st_env() {
  local f="$out/env"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap,power.limit,power.default_limit,clocks.max.sm,pci.bus_id --format=csv > "$f/gpu.csv" 2>&1
  nvidia-smi -q -d CLOCK,PERFORMANCE,POWER > "$f/nvidia-smi-clock-performance.txt" 2>&1
  lscpu > "$f/lscpu.txt" 2>&1
  free -b > "$f/free.txt" 2>&1
  df -h /root > "$f/df.txt" 2>&1
  {
    echo "host $(hostname) date $(date -Is)"
    echo "memory.max $(cat /sys/fs/cgroup/memory.max 2> /dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2> /dev/null)"
    echo "cpu.max $(cat /sys/fs/cgroup/cpu.max 2> /dev/null)"
    echo "cpu quota (cores) $quota, threads $threads, nproc $(nproc)"
    echo "TETRAEDER_CUDA_ARCHS ${TETRAEDER_CUDA_ARCHS:-}"
    echo "commit $(cat runs/vast/.pushed_commit 2> /dev/null || echo unknown)"
  } > "$f/box.txt"
  say "env done"
}

st_build() {
  bash scripts/ensure_patches.sh 8>&- || die "patch queue"
  local jobs=$((quota / 2))
  [ "$jobs" -lt 4 ] && jobs=4
  [ "$jobs" -gt 24 ] && jobs=24
  say "build start (cargo jobs $jobs)"
  CARGO_BUILD_JOBS=$jobs timeout -k 30 60m cargo build --release -p tetraeder -p tetraeder-planner -p tetraeder-bench \
    --features tetraeder/cuda > "$out/build/main.log" 2>&1 8>&- || { tail -n 30 "$out/build/main.log"; die "main build failed"; }
  say "main build done"
  CARGO_BUILD_JOBS=$jobs timeout -k 30 60m cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda \
    --test exit_regime --no-run > "$out/build/exit-regime.log" 2>&1 8>&- || { tail -n 30 "$out/build/exit-regime.log"; die "exit_regime build failed"; }
  say "build done"
}

st_disk() {
  for f in "$model" "$tet" "$plan" "$bench"; do [ -e "$f" ] || die "missing $f"; done
  local ramb=$((EMULATE_RAM_GB * 1000000000)) # the balloon stage replaces it with the measured budget; 64 GB without a balloon
  [ "$ramb" -gt 0 ] || ramb=64000000000
  "$tet" plan layout "$model" --out "$out/layout.json" || die "layout"
  # the box's NVMe queue curve and one reader thread's copy rate, beside nothing else (ADR 0092 section 2)
  "$bench" disk-curve --file "$shard" --depths 1,2,4,8,16,32 --seconds 4 --out "$out/disk/disk-curve.json" || die "disk-curve"
  python3 -I scripts/sp_make_hw.py --prior "$prior" --ram-bytes "$ramb" --gpu-bytes 25769803776 \
    --threads "$threads" --disk-curve "$out/disk/disk-curve.json" --out "$p/hw.toml" || die "hw graph"
  python3 -I -c "import json,sys; d=json.load(open(sys.argv[1])); print('copy_bytes_per_s', d.get('copy_bytes_per_s'))" "$out/disk/disk-curve.json"
  say "disk done"
}

st_balloon() {
  RESULTS="$out" HW="$p/hw.toml" bash scripts/sc_balloon.sh start 8>&- || die "balloon"
  grep -A8 ram_budget "$out/env.json" 2> /dev/null | head -n 12
  grep -A2 'id = "cpu0"' "$p/hw.toml" | head -n 3
  say "balloon done"
}

gpu_sampler_start() {
  rm -f "$out/gpu-sampler.stop"
  OUT="$out/gpu" SAMPLER_PARENT=$$ setsid nohup bash scripts/gpu_sampler.sh 8>&- > /dev/null 2>&1 &
}

st_trace() {
  [ -s "$out/trace/whole-model.json" ] && { say "trace exists"; return 0; }
  rm -f "$hidden" "$trace"
  run_one trace-whole "$out/trace/whole-model.json" "TETRAEDER_EXIT_SLOTS=$dv" "TETRAEDER_EXIT_RAM_SLOTS=$dr" \
    "TETRAEDER_EXIT_HIDDEN_OUT=$hidden_at,$hidden" "TETRAEDER_EXIT_ROUTING_OUT=$trace"
  [ -s "$hidden" ] && [ -s "$trace" ] || die "the trace run left no trace or hidden rows"
  python3 -I - "$trace" "$hidden" "$out/trace" << 'EOF'
import hashlib, os, sys
for path, name in ((sys.argv[1], "routing"), (sys.argv[2], "hidden")):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    open(os.path.join(sys.argv[3], f"{name}.sha256"), "w").write(f"{h.hexdigest()}  {name} {os.path.getsize(path)} bytes\n")
    print(name, h.hexdigest(), os.path.getsize(path))
EOF
  say "trace done"
}

st_anchors() {
  local r hin=()
  if [ "$split" != 0 ]; then
    [ -s "$hidden" ] || die "no hidden rows: run the trace stage first"
    hin=("TETRAEDER_EXIT_HIDDEN_IN=$hidden") # a span from layer 0 takes tokens
  fi
  for r in $(seq 1 "$arounds"); do
    # shellcheck disable=SC2046
    run_one "a1-r$r" "$out/anchors/a1-r$r.json" $(seg_env "$a1") "${hin[@]}"
    # shellcheck disable=SC2046
    run_one "a2-r$r" "$out/anchors/a2-r$r.json" $(seg_env "$a2") "${hin[@]}"
  done
  say "anchors done"
}

anchor_args() { # anchor_args DIR PREFIX: --anchor flags for the two spans
  local d=$1 pf=$2 r f1="" f2=""
  for r in $(seq 1 "$arounds"); do f1="$f1${f1:+,}$d/$pf${pf:+-}a1-r$r.json"; f2="$f2${f2:+,}$d/$pf${pf:+-}a2-r$r.json"; done
  echo "--anchor a1:$split..$a1:$f1 --anchor a2:$split..$a2:$f2"
}

st_predict() {
  [ -s "$trace" ] || die "no routing trace on the box"
  local tsha
  tsha=$(awk '{print $1}' "$out/trace/routing.sha256")
  rm -f "$p/SHA256SUMS" "$p/FROZEN.sha256" # a refused freeze leaves no frozen prediction behind
  # ADR 0091 answer 7: anchors on real hidden rows, and no freeze on a fit band over the limit
  local embed=""
  if [ "$split" = 0 ]; then
    embed="--embed"
  else
    python3 -I scripts/sc_gates.py anchors-real "$out"/anchors/a*-r*.json || die "anchors are not on real hidden rows: nothing frozen"
  fi
  "$bench" replay --trace "$trace" --vram "$vs" --ram "$rs" --policies lru --vram-policy static --layers "$split..$end" --warm 1 \
    --out "$p/replay.json" || die "replay"
  # shellcheck disable=SC2046,SC2086
  "$plan" segment-predict $embed --hw "$p/hw.toml" --layout "$out/layout.json" --layers "$split..$end" --pin "$vs,$rs" --ubatch "$ub" \
    --prompt "$pre" --gen "$dec" $(anchor_args "$out/anchors" "") --tier-replay "$p/replay.json" --trace-sha256 "$tsha" \
    --max-band "${FIT_BAND_MAX:-0.10}" --out "$p/pred.json" --hw-out "$p/fitted.toml" || die "segment-predict (a fit band over the limit refuses the freeze)"
  # the environment the planner wrote must be the one the anchors ran (apart from the span's end and the warm count)
  python3 -I - "$p/pred.json" "$vs" "$rs" "$ub" "$pre" "$dec" "$threads" << 'EOF' || die "pred.json environment differs from the anchors'"
import json, sys
env = json.load(open(sys.argv[1]))["exit_regime_env"]
want = {"TETRAEDER_EXIT_SLOTS": sys.argv[2], "TETRAEDER_EXIT_RAM_SLOTS": sys.argv[3], "TETRAEDER_EXIT_UBATCH": sys.argv[4],
        "TETRAEDER_EXIT_PREFILL": sys.argv[5], "TETRAEDER_EXIT_DECODE": sys.argv[6], "TETRAEDER_EXIT_THREADS": sys.argv[7]}
bad = {k: (env.get(k), v) for k, v in want.items() if env.get(k) != v}
if bad:
    print("environment mismatch", bad)
    sys.exit(1)
pr = json.load(open(sys.argv[1]))["prediction"]
print("footprint gpu %d host %d (budget gpu %d ram %d); replay prefill disk B/token %.3f, decode disk B/token %.1f, layer rows %d"
      % (pr["footprint_gpu"], pr["footprint_host"], pr["gpu_budget"], pr["ram_budget"],
         pr["replay_prefill_disk_bytes_per_token"], pr["replay_decode_disk_bytes_per_token"], pr["replay_layer_rows"]))
if pr["footprint_gpu"] > pr["gpu_budget"] or pr["footprint_host"] > pr["ram_budget"]:
    print("the pick does not fit its budgets")
    sys.exit(1)
EOF
  python3 -I scripts/sc_gates.py fitband "$p/pred.json" --limit "${FIT_BAND_MAX:-0.10}" || die "fit band over the limit: nothing frozen"
  (cd "$out" && sha256sum layout.json disk/disk-curve.json predictions/hw.toml predictions/fitted.toml predictions/replay.json \
    predictions/pred.json trace/routing.sha256 trace/hidden.sha256 anchors/*.json $(ls env.json 2> /dev/null) > predictions/SHA256SUMS \
    && (cd predictions && sha256sum SHA256SUMS > FROZEN.sha256))
  cat "$p/FROZEN.sha256"
  echo "box commit $(cat runs/vast/.pushed_commit 2> /dev/null || echo unknown)"
  say "predict done: frozen, fetch and commit predictions/ before the judged stage"
}

# Everything the judged stages need, as code (scripts/sc_gates.py judged-start): one NUMA node, the co-tenant reading, the frozen
# hash and files, the fit band, anchors on real hidden rows. The hidden rows file is not in the repo, so it is checked here.
freeze_gate() {
  python3 -I scripts/sc_gates.py judged-start --results "$out" --frozen-sha "${FROZEN_SHA:-}" > "$out/judged/gates.txt" 2>&1 \
    || { cat "$out/judged/gates.txt"; die "a judged gate failed (see $out/judged/gates.txt)"; }
  local h
  h=$(sha256sum "$hidden" | awk '{print $1}')
  [ "$h" = "$(awk '{print $1}' "$out/trace/hidden.sha256")" ] || die "the hidden rows file differs from the traced one"
  say "judged gates ok ($(awk '{print $1}' "$p/FROZEN.sha256"))"
}

judged_env() { # the frozen environment of pred.json plus the real hidden rows and the warm request
  python3 -I -c "
import json, sys
for k, v in sorted(json.load(open(sys.argv[1]))['exit_regime_env'].items()):
    print(f'{k}={v}')
" "$p/pred.json"
  echo "TETRAEDER_EXIT_PREFILL_WARM=1"
}

st_judged() {
  freeze_gate
  local r envs=() line
  while IFS= read -r line; do [ -n "$line" ] && envs+=("$line"); done < <(judged_env)
  for r in $(seq 1 "$rounds"); do
    run_one "judged-r$r" "$out/judged/r$r.json" "${envs[@]}" "TETRAEDER_EXIT_HIDDEN_IN=$hidden"
    # the early A/A (ADR 0091 answer 8): the first two rounds are one pair of the same plan right after warm-up
    if [ "$r" = 2 ]; then
      python3 -I scripts/sc_gates.py aa "$out/judged/r1.json" "$out/judged/r2.json" | tee "$out/judged/aa-early.txt"
      [ "${PIPESTATUS[0]}" = 0 ] || die "early A/A outside the band: the session cannot be clean, abort and destroy the box"
    fi
  done
  say "judged done"
}

# the A/A last: the judged plan once more, behind the same gates, as the last run of the session
st_aalast() {
  freeze_gate
  local line envs=()
  while IFS= read -r line; do [ -n "$line" ] && envs+=("$line"); done < <(judged_env)
  run_one "aa-last" "$out/judged/aa-last.json" "${envs[@]}" "TETRAEDER_EXIT_HIDDEN_IN=$hidden"
  [ -s "$out/judged/aa-last.json" ] || die "the A/A last left no result"
  python3 -I scripts/sc_gates.py aa "$out/judged/r1.json" "$out/judged/aa-last.json" | tee "$out/judged/aa-last.txt" || true
  say "aalast done"
}

st_report() {
  [ -s "$p/FROZEN.sha256" ] || die "no frozen prediction"
  local r line envs=()
  while IFS= read -r line; do [ -n "$line" ] && envs+=("$line"); done < <(judged_env)
  for r in 1 2; do # the fixed-seed segment: the routing gap between anchor-style input and the real rows
    run_one "seed-r$r" "$out/report/seed-r$r.json" "${envs[@]}"
  done
  say "report done"
}

# report only: more rounds of the judged configuration (the round-to-round spread that three rounds cannot show) and the
# anchors again, into report/ so that judged/ keeps exactly the rounds of the rule
st_extra() {
  [ -s "$p/FROZEN.sha256" ] || die "no frozen prediction"
  local r line envs=()
  while IFS= read -r line; do [ -n "$line" ] && envs+=("$line"); done < <(judged_env)
  for r in $(seq 1 "${EXTRA_ROUNDS:-5}"); do
    run_one "extra-r$r" "$out/report/extra-r$r.json" "${envs[@]}" "TETRAEDER_EXIT_HIDDEN_IN=$hidden"
  done
  for r in $(seq 1 "${EXTRA_ANCHOR_ROUNDS:-2}"); do
    # shellcheck disable=SC2046
    run_one "anchor-repeat-a1-r$r" "$out/report/anchor-repeat-a1-r$r.json" $(seg_env "$a1")
    # shellcheck disable=SC2046
    run_one "anchor-repeat-a2-r$r" "$out/report/anchor-repeat-a2-r$r.json" $(seg_env "$a2")
  done
  say "extra done"
}

# report only: the judged configuration with the whole script pinned (taskset by the caller) to one NUMA node, to test
# whether the round-to-round spread of the unpinned rounds is thread placement
st_pinned() {
  [ -s "$p/FROZEN.sha256" ] || die "no frozen prediction"
  local r line envs=()
  say "affinity: $(taskset -cp $$ 2> /dev/null)"
  while IFS= read -r line; do [ -n "$line" ] && envs+=("$line"); done < <(judged_env)
  for r in $(seq 1 "${PINNED_ROUNDS:-4}"); do
    run_one "pinned-r$r" "$out/report/pinned-r$r.json" "${envs[@]}" "TETRAEDER_EXIT_HIDDEN_IN=$hidden"
  done
  say "pinned done"
}

# aalast is the last judged-plan run of the session: nothing may follow it, or it is not the A/A last
check_order() {
  local s seen=0
  for s in $stages; do
    [ "$seen" = 1 ] && die "stage $s after aalast: the A/A last must be the last stage"
    [ "$s" = aalast ] && seen=1
    declare -F "st_$s" > /dev/null || die "unknown stage $s"
  done
}

main() {
  check_order
  say "sc_box start stages='$stages' threads=$threads out=$out model=$model span=$split..$end slots=$vs,$rs"
  gpu_sampler_start
  local s
  for s in $stages; do
    say "STAGE $s"
    "st_$s"
  done
  touch "$out/gpu-sampler.stop"
  say "SC_BOX_DONE rc=0 stages='$stages'"
  echo "SC_BOX_DONE rc=0 stages='$stages' $(date -Is)" > "$out/DONE"
}
# shellcheck disable=SC1090
[ -n "${SC_BOX_OVERRIDES:-}" ] && . "$SC_BOX_OVERRIDES"
main >> "$log" 2>&1
