#!/usr/bin/env bash
# Phase 4 verdict session on a GPU box (ADR 0065, "Verdict"): the gates first (neither own builder has run on CUDA:
# parity, step shapes, a short probe against llama-server), then record the Claude Code request log, pre-pass for the
# baseline's KV mode, interleaved pairs against llama-server per model with a libllama control after each pair, the
# TTFT pairs, env.json and verdict.json (scripts/phase4_summary.py).
# Started detached by `taxameter run scripts/phase4_box.sh`; ends with PHASE4_BOX_DONE in runs/vast/phase4_box.log.
# Results: bench/results/<session>/phase4/ (file names are fixed by the ADR). A rerun resumes: a run whose output
# exists is skipped; delete the files of an unclean pair to rerun it.
#
# Stages, in order (STAGE <name> lines in the log):
#   build       release build with CUDA and the python env
#   gpu-parity  per model, TETRAEDER_REQUIRE_GPU=1 cargo test: q8b gpu_dense (dense fixture, Qwen3-8B one sequence) and
#               gpu_real_batched (Qwen3-8B, 3 sequences with a prompt chunk mixed in, against libllama batched);
#               q30b gpu_multi_seq (MoE fixture) and gpu_real_batched (Qwen3-30B-A3B the same way); logs in
#               gpu-parity-<model>.log. STOP RULE, per model: a model whose parity failed is stopped (no pre-pass, probe
#               or pairs for it); gpu-shapes still runs
#   gpu-shapes  crates/scheduler/tests/step_shapes.rs on CUDA with Qwen3-8B for engines llama, llama_streams and dense
#               (TETRAEDER_SHAPES_GGUF, _NGL, _ENGINES); records gpu-shapes.txt, never gates
#   prepass     per model, llama-server c4 once with -kvu and once with -no-kvu; the higher throughput is the baseline
#               variant (phase4_summary.py --pick-variant), used by the probe and the pairs
#   probe       one short c1 and c4 (the verdict workload, c4 with --requests 8) per judged engine against llama-server
#               (the picked variant), probe-*.json; phase4_summary.py --probe writes probe.json. STOP RULE, per model: a
#               judged engine whose c4 throughput ratio is below 0.85 (the pass line is about 0.925, below 0.85 the pairs
#               cannot pass) is stopped like a failed parity: verdict.json records the model with pass null and the
#               reason, and its pairs are skipped; the other model's pairs and the TTFT pairs (dense 8B, even after a
#               stopped q8b: the TTFT pairs judge prefix reuse, not the c4 ratio) still run. The session ends (PHASE4_BOX_DONE rc=3, before record-log and pairs) only
#               when every judged model is stopped, so the driver fetches and destroys the box. 0.85 or above continues
#   record-log  scripts/claude_code_proxy.sh over 5 tasks, 1 rep; the tetraeder request log moves to
#               bench/private/ (clear-text prompts, gitignored); skipped when the log exists
#   env         env.json
#   pairs       per model, rep 1 (baseline, judged) then control, rep 2 (judged, baseline) then control. If one of the two
#               pairs is unclean (errors, throttle, co-tenant; phase4_summary.py --rerun-reps) and BUDGET_MIN allows, one
#               more pair (rep 3, the order of the unclean rep) runs; the verdict uses the first two clean pairs
#   ttft        Qwen3-8B replay of the log, reuse on / off: (on, off) then (off, on); then the informative llama-server run
#   summary     phase4_summary.py -> verdict.json
#
# Models (assumed downloaded, .hf_cache/<org>__<model>/): q8b Qwen3-8B-Q4_K_M (judged on --engine dense), q30b
# Qwen3-30B-A3B-Q4_K_M (judged on --engine moe; Q30B_ENGINE=llama only for a control run: libllama, counts false).
# Env knobs: COMMIT (env.json git_commit: the box checkout has no .git, so the driver passes COMMIT=$(git rev-parse HEAD)),
# SESSION (date), MODELS ("q8b q30b"), THREADS (CPU quota), BUDGET_MIN (150: runs that would start later
# are skipped, loudly, and the verdict is then null), Q30B_ENGINE, PHASE4_LOG (request log on the box, default
# bench/private/claude-code-requests.jsonl), RECORD_TASKS, TTFT_INFO (1), P4_OUT (results dir), DRY_RUN=1 (print the
# sequence and the commands, run nothing; needs no GPU, lock or build).
# Seeds: each test of a server start has its own seed (c1 rep*10+1, c4 rep*10+4; probe 91/94, pre-pass 84), the same on
# every engine for the same test and rep: repeated prompts on one server would measure llama-server's prompt cache.
# CPU rehearsal (.github/workflows/phase4-rehearsal.yml; the plumbing, not the verdict): SKIP_GPU=1 (no CUDA, no
# nvidia-smi: throttle records are written as idle, gpu-parity and gpu-shapes are skipped), BASELINE_BIN, TET_BIN, PY,
# MODEL_Q8B, MODEL_Q30B, P4_LOCK, CTX, NGL_LLAMA, PROMPT_TOKENS, MAX_TOKENS, N_C1, N_C4, N_PROBE, REPLAY_CTX,
# REPLAY_MAX_PROMPT, EXPECT_CTX, REPLAY_LIMIT, PROBE_MIN (stop line, 0.85), P4_LOG_FILE.
# DRY_RUN only: DRY_GPU_PARITY plays a failed gpu-parity: `fail` (every model) or model names ("q30b", "q8b q30b").
# A server that exits while loading (a rejected flag) ends the whole session at once: the box bills idle time.
# cargo runs with fd 8 closed: the sccache server it starts would otherwise keep the lock after the job
DRY="${DRY_RUN:-0}"
skip_gpu="${SKIP_GPU:-0}"
if [ "$DRY" != 1 ]; then
  exec 8> "${P4_LOCK:-/root/tetraeder_phase4_box.lock}"
  flock -n 8 || exit 0
fi
cd "$(dirname "$0")/.." || exit 1
if [ "$DRY" != 1 ]; then
  if [ "$skip_gpu" != 1 ]; then bash scripts/ensure_patches.sh 8>&- || exit 1; fi # ggml patch queue (ADR 0052)
  if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
  export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
  # shellcheck source=scripts/box_libs.sh
  . scripts/box_libs.sh
  command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
  if [ "$skip_gpu" != 1 ]; then
    # shellcheck disable=SC1091
    . "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
    # this box's GPU only (another arch list rebuilds ggml-cuda for every arch)
    if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
      archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
      [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
    fi
  fi
fi
log="${P4_LOG_FILE:-runs/vast/phase4_box.log}"
session="${SESSION:-$(date +%Y%m%d)}"
out="${P4_OUT:-$PWD/bench/results/$session/phase4}"
mkdir -p "$out" "$(dirname "$log")"
t_start=$(date +%s)
if [ "$DRY" = 1 ]; then threads="${THREADS:-8}"; else threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"; fi
budget_s=$((${BUDGET_MIN:-150} * 60))
models="${MODELS:-q8b q30b}"
q30b_engine="${Q30B_ENGINE:-moe}"
models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
tet="${TET_BIN:-target/release/tetraeder}"
baseline_bin="${BASELINE_BIN:-/opt/llama.cpp-baseline/bin/llama-server}"
venv=/root/gwvenv
py="${PY:-$venv/bin/python}"
[ "$DRY" = 1 ] && py=python3
ctx="${CTX:-32768}" ngl_ll="${NGL_LLAMA:-99}"
ptok="${PROMPT_TOKENS:-512}" mtok="${MAX_TOKENS:-128}"
n_c1="${N_C1:-8}" n_c4="${N_C4:-16}" n_probe="${N_PROBE:-8}"
rctx="${REPLAY_CTX:-40960}" rmax="${REPLAY_MAX_PROMPT:-36000}" rexpect="${EXPECT_CTX:-40960}"
probe_min="${PROBE_MIN:-0.85}"
req_log="${PHASE4_LOG:-bench/private/claude-code-requests.jsonl}"
record_tasks="${RECORD_TASKS:-fix-rounding,fix-eom-leap,feature-split-evenly,feature-late-fee,explain-parser}"
port_tet=18080
port_ll=18081
server_pid=""

declare -A MODEL_FILE=(
  [q30b]="${MODEL_Q30B:-$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf}"
  [q8b]="${MODEL_Q8B:-$models_dir/Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf}"
)
declare -A MODEL_ALIAS=([q30b]=qwen3-30b-a3b [q8b]=qwen3-8b)

# stopped[MODEL]=reason: the per-model stop rule (failed gpu-parity, probe below the line); parity_bad[MODEL]=1 for parity
declare -A stopped parity_bad
active() { local m; for m in $models; do [ -z "${stopped[$m]:-}" ] && echo "$m"; done; }

say() { echo "[$(date +%T)] $*"; }
plan() { echo "PLAN $*"; }

# Names in the result files: llama-server, tetraeder-dense, tetraeder-moe, tetraeder-llama.
judged_name() { # judged_name MODEL
  case $1 in
    q8b) echo tetraeder-dense ;;
    q30b) if [ "$q30b_engine" = moe ]; then echo tetraeder-moe; else echo tetraeder-llama; fi ;;
  esac
}
engine_flag() { echo "${1#tetraeder-}"; } # tetraeder-dense -> dense

die() {
  say "FATAL: $*"
  stop_server
  touch "$out/DONE"
  say "PHASE4_BOX_DONE rc=1"
  exit 1
}

# Starts a server detached, pid in $server_pid; $1 = server log, rest = command. Killed by pid only.
start_server() {
  local slog=$1
  shift
  if [ "$DRY" = 1 ]; then
    plan "server log=${slog##*/}: $*"
    return 0
  fi
  : > "$slog"
  setsid nohup "$@" > "$slog" 2>&1 8>&- &
  server_pid=$!
}

stop_server() {
  [ -n "$server_pid" ] || return 0
  kill "$server_pid" 2> /dev/null
  for _ in $(seq 1 30); do kill -0 "$server_pid" 2> /dev/null || break; sleep 1; done
  kill -9 "$server_pid" 2> /dev/null
  wait "$server_pid" 2> /dev/null
  server_pid=""
  sleep 3 # GPU memory is released after the process is gone
}
trap stop_server EXIT

# wait_ready URL SECONDS SERVER_LOG: success when URL answers 200. A server that exits ends the session (die).
wait_ready() {
  local url=$1 secs=$2 slog=$3
  [ "$DRY" = 1 ] && return 0
  for _ in $(seq 1 $((secs / 2))); do
    if ! kill -0 "$server_pid" 2> /dev/null; then
      tail -n 15 "$slog" | sed 's/^/  server: /'
      die "server exited while loading ($slog); a rejected flag? see the lines above"
    fi
    curl -sf -o /dev/null "$url" && return 0
    sleep 2
  done
  die "server not ready after ${secs}s ($slog)"
}

over_budget() { [ $(($(date +%s) - t_start)) -gt "$budget_s" ]; }

# gpu_note RUN before|after: the throttle reasons bitmask (hex) of every GPU, one `<when>: <hex>` line each, then one
# `power-<when>: <power.limit>, <power.draw>` line per GPU (informative; phase4_summary.py copies it to verdict.json)
gpu_note() {
  local f="$out/gpu-$1.txt" when=$2 v
  if [ "$DRY" = 1 ]; then
    plan "nvidia-smi throttle reasons $when -> gpu-$1.txt"
    return 0
  fi
  if [ "$skip_gpu" = 1 ]; then
    echo "$when: 0x0000000000000000" >> "$f" # no GPU: counts as idle
    return 0
  fi
  v=$(nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader 2> /dev/null) \
    || v=$(nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2> /dev/null)
  [ -n "$v" ] || say "WARNING: no throttle reasons from nvidia-smi for $1 $when (the run counts as unclean)"
  echo "$v" | sed "s/^ *\([^ ]\)/$when: \1/" >> "$f"
  nvidia-smi --query-gpu=power.limit,power.draw --format=csv,noheader 2> /dev/null \
    | sed "s/^ *\([^ ]\)/power-$when: \1/" >> "$f" || true
}

# ---- stage: record-log -------------------------------------------------------------------------------------
record_log() {
  if [ -s "$req_log" ]; then
    say "record-log: $req_log exists, sha256 $(sha256sum "$req_log" | cut -d' ' -f1)"
    return 0
  fi
  local rdir=bench/results/phase4-record/claude-code-proxy src
  if [ "$DRY" = 1 ]; then
    plan "RESULTS_DATE=phase4-record REPS=1 TASKS=$record_tasks bash scripts/claude_code_proxy.sh"
    plan "mv $rdir/tetraeder-r1.reqlog.jsonl $req_log"
    return 0
  fi
  mkdir -p "$(dirname "$req_log")"
  RESULTS_DATE=phase4-record REPS=1 TASKS="$record_tasks" bash scripts/claude_code_proxy.sh 8>&-
  src=$rdir/tetraeder-r1.reqlog.jsonl
  [ -s "$src" ] || die "record-log: no $src (see runs/vast/claude_code_proxy.log)"
  mv "$src" "$req_log"
  rm -f "$rdir"/*.reqlog.jsonl # clear-text prompts: only bench/private/ keeps one
  say "record-log: $(wc -l < "$req_log") requests, sha256 $(sha256sum "$req_log" | cut -d' ' -f1)"
}

# ---- stage: build ------------------------------------------------------------------------------------------
build() {
  if [ "$DRY" = 1 ]; then
    plan "cargo build --release -p tetraeder --features cuda; python env $venv"
    return 0
  fi
  say "build start"
  local features="--features cuda"
  [ "$skip_gpu" = 1 ] && features=""
  # shellcheck disable=SC2086
  timeout -k 30 50m cargo build --release -p tetraeder $features 8>&- || die "build failed"
  if [ -n "${PY:-}" ]; then
    say "build done (python: $PY)"
    return 0
  fi
  if [ ! -x "$py" ] || ! "$py" -c 'import httpx, tomllib' 2> /dev/null; then
    # the box image has python3 without ensurepip (no venv module): fetch uv then
    if ! command -v uv > /dev/null 2>&1 && ! python3 -c 'import ensurepip' 2> /dev/null; then
      curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.local/bin sh > /dev/null 2>&1
      export PATH="/root/.local/bin:$PATH"
    fi
    if command -v uv > /dev/null 2>&1; then
      uv venv --allow-existing "$venv" && uv pip install --python "$py" httpx
    else
      python3 -m venv "$venv" && "$py" -m pip install -q httpx
    fi || die "python env failed"
  fi
  say "build done"
}

# ---- stage: gpu-parity ------------------------------------------------------------------------------------
# No timed run on a box whose GPU builder disagrees with the CPU reference. gpu_dense covers one and three sequences;
# gpu_multi_seq is the MoE builder with several sequences, resident and streamed. Returns 1 on a failure (main stops).
gpu_parity() {
  local base="cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda" m cmd bad=0
  local tail="-- --nocapture --test-threads 1"
  mkdir -p "$out"
  for m in $models; do
    case $m in
      q8b) cmd="$base --test gpu_dense --test gpu_real_batched $tail --skip gpu_qwen3_30b" ;;
      q30b) cmd="$base --test gpu_multi_seq --test gpu_real_batched $tail --skip gpu_qwen3_8b" ;;
    esac
    if [ "$DRY" = 1 ]; then
      plan "TETRAEDER_REQUIRE_GPU=1 $cmd > gpu-parity-$m.log"
      case " ${DRY_GPU_PARITY:-ok} " in *" fail "* | *" $m "*) parity_bad[$m]=1 stopped[$m]="gpu-parity failed" bad=1 ;; esac
      continue
    fi
    [ "$skip_gpu" = 1 ] && { say "gpu-parity $m skipped (SKIP_GPU=1)"; continue; }
    say "gpu-parity $m start"
    # shellcheck disable=SC2086
    if TETRAEDER_REQUIRE_GPU=1 TETRAEDER_MODELS_DIR="$models_dir" timeout -k 30 40m $cmd > "$out/gpu-parity-$m.log" 2>&1 8>&-; then
      say "gpu-parity $m done"
    else
      say "gpu-parity $m FAILED (see $out/gpu-parity-$m.log): $m is stopped, the others go on"
      parity_bad[$m]=1 stopped[$m]="gpu-parity failed" bad=1
    fi
  done
  return $bad
}

# ---- stage: gpu-shapes ------------------------------------------------------------------------------------
# One decode per step shape on CUDA (the test of loop-bench, on Qwen3-8B): decode4x8, decode1x8_alone and
# decode1x8_resident per engine tell whether one KV stream per sequence (llama_streams) beats the unified pool
# (llama, dense) on the GPU. Records only; D2's driver reads gpu-shapes.txt.
gpu_shapes() {
  local cmd="cargo test --release -p tetraeder-scheduler --features cuda --test step_shapes -- --nocapture --test-threads 1"
  local engines="llama,llama_streams,dense"
  if [ "$DRY" = 1 ]; then
    plan "TETRAEDER_SHAPES_GGUF=<q8b> TETRAEDER_SHAPES_NGL=99 TETRAEDER_SHAPES_ENGINES=$engines $cmd > gpu-shapes.txt"
    return 0
  fi
  [ "$skip_gpu" = 1 ] && { say "gpu-shapes skipped (SKIP_GPU=1)"; return 0; }
  say "gpu-shapes start"
  # shellcheck disable=SC2086
  TETRAEDER_REQUIRE_GPU=1 TETRAEDER_SHAPES_GGUF="${MODEL_FILE[q8b]}" TETRAEDER_SHAPES_NGL=99 \
    TETRAEDER_SHAPES_ENGINES="$engines" TETRAEDER_SHAPES_THREADS="$threads" timeout -k 30 30m $cmd \
    > "$out/gpu-shapes.txt" 2>&1 8>&- || say "WARNING: gpu-shapes failed (see $out/gpu-shapes.txt); it does not gate"
  grep -E 'SHAPE engine=\S+ shape=(decode4x8|decode1x8_alone|decode1x8_resident) ' "$out/gpu-shapes.txt" | sed 's/^/  /'
  say "gpu-shapes done"
}

# ---- stage: probe -----------------------------------------------------------------------------------------
# probe_run MODEL NAME: one server start, c1 and c4 (8 requests) of the verdict workload; the seeds differ per test
# (a repeated prompt would measure the prompt cache), the same on both engines. The baseline runs the variant the
# pre-pass picked.
probe_run() {
  local m=$1 name=$2 url alias variant=""
  local c1="$out/probe-$m-$name-c1.json" c4="$out/probe-$m-$name-c4.json"
  [ -f "$c1" ] && [ -f "$c4" ] && { say "probe $m $name: done before"; return 0; }
  if [ "$name" = llama-server ]; then
    variant=$(pick_variant "$m") || variant=""
    [ -n "$variant" ] || { say "WARNING: no pre-pass result for $m: probe baseline runs -kvu"; variant=kvu; }
  fi
  say "probe $m $name start${variant:+ ($variant)}"
  server_for "$m" "$name" "$variant" "$out/server-probe-$m-$name.log"
  warm_up "$url" "$alias"
  bench_file "$c1" "probe-$m-$name-c1" "$url" "$alias" --prompt-tokens "$ptok" --max-tokens "$mtok" --concurrency 1 --requests "$n_probe" --seed 91
  bench_file "$c4" "probe-$m-$name-c4" "$url" "$alias" --prompt-tokens "$ptok" --max-tokens "$mtok" --concurrency 4 --requests "$n_probe" --seed 94
  stop_server
  say "probe $m $name done"
}

# probe_verdict [PARITY]: probe.json from the probe files (and a failed gpu-parity). Returns 3 on the stop rule.
probe_verdict() {
  local args=() m
  for m in $models; do
    args+=(--probe-run "$m:$(judged_name "$m")")
    [ -n "${parity_bad[$m]:-}" ] && args+=(--parity "$m:failed")
  done
  if [ "$DRY" = 1 ] && ! ls "$out"/probe-*-c4.json > /dev/null 2>&1 && [ "${#parity_bad[@]}" = 0 ]; then
    plan "phase4_summary.py $out --probe ${args[*]} -> probe.json; a model with gpu-parity failed or a c4 ratio < 0.85 is stopped (its pairs skipped), rc 3 when all are"
    return 0
  fi
  "$py" scripts/phase4_summary.py "$out" --probe "${args[@]}" --probe-min "$probe_min"
}

# probe: the models with a failed parity are not probed (stopped already); probe_verdict then names every stopped model
probe() {
  local m
  for m in $(active); do
    probe_run "$m" llama-server
    probe_run "$m" "$(judged_name "$m")"
  done
  probe_verdict
}

# stop_probed: probe.json's stopped models join `stopped` (DRY with no probe files: none)
stop_probed() {
  local m
  [ -s "$out/probe.json" ] || return 0
  for m in $("$py" scripts/phase4_summary.py "$out" --stopped); do
    stopped[$m]="${stopped[$m]:-probe c4 ratio below $probe_min}"
  done
}

# ---- stage: env --------------------------------------------------------------------------------------------
env_json() {
  if [ "$DRY" = 1 ]; then
    plan "env.json (gpu, driver, clocks, throttle reasons, cpu quota, threads, commit, baseline version, log hash)"
    return 0
  fi
  [ -f "$out/env.json" ] && return 0
  nvidia-smi -q -d CLOCK,PERFORMANCE > "$out/nvidia-smi-clock-performance.txt" 2>&1
  GW_OUT="$out" GW_THREADS="$threads" GW_QUOTA="$(bash scripts/cpu_quota.sh 2> /dev/null)" \
    GW_LLAMA="$("$baseline_bin" --version 2>&1 | head -3 | tr '\n' ' ')" \
    GW_COMMIT="${COMMIT:-$(git rev-parse HEAD 2> /dev/null)}" GW_HOST="$(hostname)" \
    GW_LOG_SHA="$(sha256sum "$req_log" 2> /dev/null | cut -d' ' -f1)" GW_Q30B="$q30b_engine" \
    GW_GPU="$(nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,clocks_throttle_reasons.active,memory.total --format=csv,noheader 2> /dev/null)" \
    GW_VAST="${VAST_CONTAINERLABEL:-} ${CONTAINER_ID:-}" "$py" - << 'EOF'
import json, os
e = os.environ
name, drv, sm, smmax, mem, throttle, vram = ([x.strip() for x in e["GW_GPU"].split(",")] + [""] * 7)[:7]
json.dump({
    "host": e["GW_HOST"], "vast_container": e["GW_VAST"].strip(), "gpu": name, "driver": drv, "vram": vram,
    "clock_sm_mhz": sm, "clock_sm_max_mhz": smmax, "clock_mem_mhz": mem, "throttle_reasons_active": throttle,
    "cpu_quota_cores": e["GW_QUOTA"], "threads": e["GW_THREADS"], "git_commit": e["GW_COMMIT"],
    "baseline_llama_server": e["GW_LLAMA"],
    # empty when the record-log stage did not run on this box (no request log to hash): expected, not an error
    "request_log_sha256": e["GW_LOG_SHA"], "q30b_engine": e["GW_Q30B"],
    "see": "nvidia-smi-clock-performance.txt",
}, open(os.path.join(e["GW_OUT"], "env.json"), "w"), indent=2)
EOF
}

# ---- benchmark runs ----------------------------------------------------------------------------------------
# bench_file FILE LABEL URL ALIAS ARGS...: one gateway_bench.py call; the file is kept even with errors (the verdict
# reads summary.errors and calls the run unclean)
bench_file() {
  local f=$1 label=$2 url=$3 alias=$4
  shift 4
  [ "$DRY" = 1 ] && { plan "gateway_bench.py $url --no-keepalive --label $label $* -> ${f##*/}"; return 0; }
  timeout -k 30 15m "$py" scripts/gateway_bench.py "$url" --model "$alias" --temperature 0 --no-keepalive --label "$label" \
    --out "$f.tmp" "$@" >> "$out/bench.log" 2>&1
  [ -s "$f.tmp" ] && mv "$f.tmp" "$f"
  return 0
}

warm_up() {
  [ "$DRY" = 1 ] && { plan "warm-up: 1 request, 128 prompt tokens"; return 0; }
  "$py" scripts/gateway_bench.py "$1" --model "$2" --prompt-tokens 128 --max-tokens 16 --requests 1 --no-keepalive > /dev/null 2>&1
}

# server_for MODEL NAME VARIANT SLOG: starts the server of a run, sets $url and $alias
server_for() {
  local m=$1 name=$2 variant=$3 slog=$4 f=${MODEL_FILE[$1]}
  alias=${MODEL_ALIAS[$m]}
  if [ "$name" = llama-server ]; then
    local kv=-kvu
    [ "$variant" = nokvu ] && kv=-no-kvu
    url="http://127.0.0.1:$port_ll"
    start_server "$slog" "$baseline_bin" -m "$f" -ngl "$ngl_ll" -np 4 -c "$ctx" -fa on -ctk f16 -ctv f16 -t "$threads" \
      --host 127.0.0.1 --port "$port_ll" $kv
    wait_ready "$url/health" 900 "$slog"
  else
    url="http://127.0.0.1:$port_tet"
    local extra=()
    [ "$name" = tetraeder-llama ] && extra=(--flash-attn on)
    start_server "$slog" "$tet" serve -m "$f" --engine "$(engine_flag "$name")" --ngl -1 --parallel 4 --ctx "$ctx" \
      -t "$threads" --port "$port_tet" --alias "$alias" --accept-any-model "${extra[@]}"
    wait_ready "$url/v1/models" 900 "$slog"
  fi
}

# Host sampler beside the server, best effort: its host block (outside cores) feeds cotenant_guard.py
sampler_start() { # sampler_start RUN MODEL
  sampler_pid=""
  [ "$DRY" = 1 ] && { plan "engine_sample.py sample --pid <server> (host block for cotenant_guard)"; return 0; }
  if [ -z "${disk_dev[$2]:-}" ]; then
    disk_dev[$2]=$(python3 scripts/engine_sample.py probe-disk "${MODEL_FILE[$2]}" 2> /dev/null)
  fi
  [ -n "${disk_dev[$2]}" ] || return 0
  samples="runs/vast/p4-$1.samples"
  python3 scripts/engine_sample.py sample --pid "$server_pid" --disk "${disk_dev[$2]}" --out "$samples" \
    > /dev/null 2>&1 8>&- &
  sampler_pid=$!
}
declare -A disk_dev

# sampler_finish RUN T_START T_END FILES...: needs the server stopped (the sampler follows its pid)
sampler_finish() {
  local run=$1 t0=$2 t1=$3 f
  shift 3
  [ "$DRY" = 1 ] && { plan "cotenant_guard.py merge for $*"; return 0; }
  [ -n "$sampler_pid" ] && wait "$sampler_pid" 2> /dev/null
  if [ -n "$sampler_pid" ] && [ -s "$samples" ]; then
    printf '{"t_start": %s, "t_first": %s, "t_end": %s, "prompt_tokens": 1, "decode_tokens": 2}\n' "$t0" "$t0" "$t1" \
      > "runs/vast/p4-$run.window.json"
    for f in "$@"; do
      [ -s "$f" ] && { python3 scripts/engine_sample.py summarize --samples "$samples" --threads "$threads" \
        --kt-json "runs/vast/p4-$run.window.json" --out "$samples.summary" --merge-into "$f" > /dev/null \
        || say "host summary failed for $run"; }
    done
  fi
  for f in "$@"; do [ -s "$f" ] && { python3 -I scripts/cotenant_guard.py merge --json "$f" || true; }; done
}

# pick_variant MODEL: the baseline's KV mode from the pre-pass (phase4_summary.py holds the rule)
pick_variant() {
  [ "$DRY" = 1 ] && { echo kvu; return 0; }
  "$py" scripts/phase4_summary.py "$out" --pick-variant "$1"
}

# one_run MODEL NAME REP: one server start: warm-up, c1, c4 (the gateway_bench workload of the ADR). Each test has
# its own seed, the same on every engine for that test and rep (repeated prompts would measure the prompt cache).
one_run() {
  local m=$1 name=$2 rep=$3 run="$1-$2-r$3" variant="" url alias
  local c1="$out/bench-$m-$name-c1-r$rep.json" c4="$out/bench-$m-$name-c4-r$rep.json"
  if [ -f "$c1" ] && [ -f "$c4" ]; then
    say "run $run: done before"
    return 0
  fi
  if over_budget; then
    say "BUDGET spent: skip run $run (the verdict will be null)"
    return 0
  fi
  if [ "$name" = llama-server ]; then
    variant=$(pick_variant "$m") || die "no pre-pass result for $m (prepass-$m-kvu.json, prepass-$m-nokvu.json)"
  fi
  say "run $run start${variant:+ ($variant)}"
  : > "$out/gpu-$run.txt"
  server_for "$m" "$name" "$variant" "$out/server-$m-$name-r$rep.log"
  gpu_note "$run" before
  sampler_start "$run" "$m"
  warm_up "$url" "$alias"
  local t0 t1
  t0=$(date +%s.%N)
  bench_file "$c1" "$run-c1" "$url" "$alias" --prompt-tokens "$ptok" --max-tokens "$mtok" --concurrency 1 --requests "$n_c1" --seed $((rep * 10 + 1))
  bench_file "$c4" "$run-c4" "$url" "$alias" --prompt-tokens "$ptok" --max-tokens "$mtok" --concurrency 4 --requests "$n_c4" --seed $((rep * 10 + 4))
  t1=$(date +%s.%N)
  gpu_note "$run" after
  stop_server
  sampler_finish "$run" "$t0" "$t1" "$c1" "$c4"
  say "run $run done"
}

# ---- stage: prepass ----------------------------------------------------------------------------------------
prepass() {
  local m v f url alias
  for m in $(active); do
    for v in kvu nokvu; do
      f="$out/prepass-$m-$v.json"
      [ -f "$f" ] && continue
      over_budget && { say "BUDGET spent: skip prepass $m $v"; continue; }
      say "prepass $m $v start"
      : > "$out/gpu-prepass-$m-$v.txt"
      server_for "$m" llama-server "$v" "$out/server-prepass-$m-$v.log"
      gpu_note "prepass-$m-$v" before
      warm_up "$url" "$alias"
      bench_file "$f" "prepass-$m-$v" "$url" "$alias" --prompt-tokens "$ptok" --max-tokens "$mtok" --concurrency 4 --requests "$n_c4" --seed 84
      gpu_note "prepass-$m-$v" after
      stop_server
    done
    if [ "$DRY" != 1 ]; then say "prepass $m: baseline variant $(pick_variant "$m")"; fi
  done
}

# ---- stage: pairs ------------------------------------------------------------------------------------------
# pair_runs MODEL JUDGED REP ORDER_REP: REP's pair, odd ORDER_REP = (baseline, judged), even = (judged, baseline), then
# the control (libllama behind the same scheduler: reported, never judged; on q30b without the own MoE builder the
# judged run already is libllama)
pair_runs() {
  local m=$1 j=$2 rep=$3 order=$4
  if [ $((order % 2)) = 1 ]; then
    one_run "$m" llama-server "$rep"
    one_run "$m" "$j" "$rep"
  else
    one_run "$m" "$j" "$rep"
    one_run "$m" llama-server "$rep"
  fi
  [ "$j" != tetraeder-llama ] && one_run "$m" tetraeder-llama "$rep"
  return 0
}

# unclean_reps MODEL: the unclean judged rep (1 or 2) one more pair can replace, else empty
unclean_reps() {
  if [ "$DRY" = 1 ] && ! ls "$out"/bench-"$1"-*-r1.json > /dev/null 2>&1; then return 0; fi
  "$py" scripts/phase4_summary.py "$out" --rerun-reps "$1"
}

pairs() {
  local m j rep bad
  for m in $(active); do
    j=$(judged_name "$m")
    for rep in 1 2; do pair_runs "$m" "$j" "$rep" "$rep"; done
    bad=$(unclean_reps "$m")
    if [ -z "$bad" ]; then continue; fi
    if over_budget; then
      say "BUDGET spent: no rerun pair for $m although rep $bad is unclean"
    else
      say "rerun pair $m: rep $bad is unclean, running rep 3"
      pair_runs "$m" "$j" 3 "$bad"
    fi
  done
}

# ---- stage: ttft -------------------------------------------------------------------------------------------
# replay_run MODE REP: MODE on|off|llama-server
replay_run() {
  local mode=$1 rep=$2 f=${MODEL_FILE[q8b]} run="replay-q8b-$1-r$2" slog url="http://127.0.0.1:$port_tet"
  local res="$out/$run.json" health=/v1/models limit=()
  [ -n "${REPLAY_LIMIT:-}" ] && limit=(--limit "$REPLAY_LIMIT")
  slog="$out/server-$run.log"
  [ -f "$res" ] && { say "run $run: done before"; return 0; }
  over_budget && { say "BUDGET spent: skip run $run"; return 0; }
  say "run $run start"
  : > "$out/gpu-$run.txt"
  case $mode in
    on) start_server "$slog" "$tet" serve -m "$f" --engine dense --ngl -1 --parallel 4 --ctx "$rctx" -t "$threads" \
      --port "$port_tet" --alias qwen3-8b --accept-any-model ;;
    off) start_server "$slog" "$tet" serve -m "$f" --engine dense --ngl -1 --parallel 4 --ctx "$rctx" -t "$threads" \
      --port "$port_tet" --alias qwen3-8b --accept-any-model --no-prefix-reuse --cache-ram 0 ;;
    llama-server)
      url="http://127.0.0.1:$port_ll"
      health=/health
      start_server "$slog" "$baseline_bin" -m "$f" -ngl 99 -np 4 -kvu -c "$rctx" -fa on -ctk f16 -ctv f16 -t "$threads" \
        --host 127.0.0.1 --port "$port_ll" --cache-ram 8192 ;;
  esac
  wait_ready "$url$health" 900 "$slog"
  gpu_note "$run" before
  if [ "$DRY" = 1 ]; then
    plan "gateway_replay.py $url $req_log --speed 0 --concurrency 1 --max-prompt-tokens $rmax --gguf <q8b> --expect-context-length $rexpect -> $run.json"
  else
    timeout -k 30 60m "$py" scripts/gateway_replay.py "$url" "$req_log" --speed 0 --concurrency 1 \
      --max-prompt-tokens "$rmax" --gguf "$f" --expect-context-length "$rexpect" "${limit[@]}" --out "$res.tmp" >> "$out/replay.log" 2>&1
    local rc=$?
    # rc 2: the context-length check failed (no result file); rc 1: failed requests (file kept, the verdict sees them)
    [ "$rc" = 2 ] && die "replay: Qwen3-8B context_length is not $rexpect (see $out/replay.log)"
    [ -s "$res.tmp" ] && mv "$res.tmp" "$res"
  fi
  gpu_note "$run" after
  stop_server
  say "run $run done"
}

ttft() {
  if [ "$DRY" != 1 ] && [ ! -s "$req_log" ]; then
    say "ttft: no request log at $req_log, skipped (the verdict will be null)"
    return 0
  fi
  case " $models " in *" q8b "*) ;; *) say "ttft: needs q8b, skipped"; return 0 ;; esac
  replay_run on 1
  replay_run off 1
  replay_run off 2
  replay_run on 2
  [ "${TTFT_INFO:-1}" = 1 ] && replay_run llama-server 1
}

# ---- stage: summary ----------------------------------------------------------------------------------------
summary() {
  if [ "$DRY" = 1 ]; then
    plan "phase4_summary.py $out -> verdict.json; kv_unified.txt from the baseline server logs"
    return 0
  fi
  grep -H "kv_unified" "$out"/server-prepass-*.log "$out"/server-*-llama-server-r*.log > "$out/kv_unified.txt" 2> /dev/null \
    || say "WARNING: no kv_unified line in the llama-server logs (baseline_kv_unified will be null)"
  "$py" scripts/phase4_summary.py "$out"
}

main() {
  say "phase4_box start (threads=$threads models=\"$models\" budget=${BUDGET_MIN:-150} min q30b=$q30b_engine) out=$out"
  say "STAGE build"
  build
  say "STAGE gpu-parity"
  local rc=0 m
  gpu_parity || say "gpu-parity failed for:$(for m in "${!parity_bad[@]}"; do echo -n " $m"; done)"
  # a failed model is skipped, not the session: step shapes record only; pre-pass and probe cover the other models
  say "STAGE gpu-shapes"
  gpu_shapes
  say "STAGE prepass"
  prepass
  say "STAGE probe"
  probe || rc=$?
  stop_probed
  for m in $models; do
    [ -n "${stopped[$m]:-}" ] && say "STOPPED $m: ${stopped[$m]} (its pairs are skipped, pass null in verdict.json)"
  done
  if [ -z "$(active)" ] || [ "$rc" = 3 ]; then
    say "every judged model is stopped (probe.json): the session ends before record-log and the pairs"
    stop_session
    return 3
  fi
  say "STAGE record-log"
  record_log
  say "STAGE env"
  env_json
  say "STAGE pairs"
  pairs
  say "STAGE ttft"
  ttft
  say "STAGE summary"
  summary
  rc=$?
  touch "$out/DONE"
  say "PHASE4_BOX_DONE rc=$rc"
  return $rc
}

# The stop rule ended the session: nothing timed ran beyond the probe; the driver fetches and destroys the box.
stop_session() {
  stop_server
  touch "$out/DONE"
  say "PHASE4_BOX_DONE rc=3"
}

if [ "$DRY" = 1 ]; then
  main
else
  main >> "$log" 2>&1
fi
