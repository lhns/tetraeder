#!/usr/bin/env bash
# Phase 3b gateway exit tests on a GPU box (ADR 0046): build, SDK conformance on real models, paired benchmark of
# `tetraeder serve` against llama-server on the same box, API overhead, env and summary.
# Started detached by `taxameter run scripts/gateway_box.sh`; ends with GATEWAY_BOX_DONE in runs/vast/gateway_box.log.
# Results: bench/results/<date>/gateway/ (rerun resumes: stages and runs whose output exists are skipped).
#
# Models (assumed downloaded, .hf_cache/<org>__<model>/): Qwen3-30B-A3B-Q4_K_M (MoE, builder + libllama) and
# Qwen3-8B-Q4_K_M (dense, libllama), on one RTX 3090 (24 GB). Env knobs: REPS (3), MODELS ("q8b q30b"),
# BUDGET_MIN (100: benchmark runs that would start after this many minutes since the script began are skipped, so
# a slow box still ends with complete repetitions), THREADS, C4_CTX (32768 total for -np 4, see below).
#
# Runtime estimate on a 3090 (sccache warm / cold): build 5 / 20 min; conformance 2 x 4 min (model load 1 min,
# about 120 requests); benchmark per repetition and model: 4 server starts (llama-server -np 1, -np 4,
# tetraeder auto, tetraeder --engine llama) x (load 1 min + warmup + 5 runs ~2 min) = 12 min, so 3 reps x 2
# models = 70 min (the 8B loads faster: about 55); overhead benches 5 min. Total 90 min warm, 105 min cold,
# bounded by BUDGET_MIN plus the stage in flight.
#
# Deviation from the spec's "-np 4 -c 131072": 32K per slot needs 13 GB (30B) or 19 GB (8B) of KV beside the
# weights and does not fit 24 GB. The -np 4 config uses -c 32768 (8K per slot), enough for every request here
# (at most 4.2K prompt + 128 generated).
# cargo runs with fd 8 closed: the sccache server it starts would otherwise keep the lock after the job
exec 8> /root/tetraeder_gateway_box.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
# this box's GPU only (another arch list rebuilds ggml-cuda for every arch)
if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
log=runs/vast/gateway_box.log
out="$PWD/bench/results/$(date +%Y%m%d)/gateway"
mkdir -p "$out" runs/vast
t_start=$(date +%s)
threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"
reps="${REPS:-3}"
budget_s=$((${BUDGET_MIN:-100} * 60))
c4_ctx="${C4_CTX:-32768}"
models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
tet=target/release/tetraeder
llama_bin=/opt/llama.cpp/bin
venv=/root/gwvenv
py="$venv/bin/python"
port_tet=18080
port_ll=18081
server_pid=""

declare -A MODEL_FILE=(
  [q30b]="$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
  [q8b]="$models_dir/Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
)
declare -A MODEL_ALIAS=([q30b]=qwen3-30b-a3b [q8b]=qwen3-8b)

say() { echo "[$(date +%T)] $*"; }

# Starts a server detached, pid in $server_pid; $1 = server log, rest = command. Killed by pid only.
start_server() {
  local slog=$1
  shift
  : > "$slog"
  setsid nohup "$@" > "$slog" 2>&1 &
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

# wait_ready URL SECONDS: the server answers 200 on URL (llama-server /health is 503 while loading)
wait_ready() {
  local url=$1 secs=$2
  for _ in $(seq 1 $((secs / 2))); do
    kill -0 "$server_pid" 2> /dev/null || { say "server exited while loading"; return 1; }
    curl -sf -o /dev/null "$url" && return 0
    sleep 2
  done
  say "server not ready after ${secs}s"
  return 1
}

over_budget() { [ $(($(date +%s) - t_start)) -gt "$budget_s" ]; }

# ---- a. build ----------------------------------------------------------------------------------------------
build() {
  say "build start"
  timeout -k 30 50m cargo build --release -p tetraeder --features cuda 8>&- || { say "build failed"; return 1; }
  if [ ! -x "$py" ] || ! "$py" -c 'import openai, anthropic, jsonschema, pytest, httpx, pydantic' 2> /dev/null; then
    # the box image has python3 without ensurepip (no venv module): fetch uv then
    if ! command -v uv > /dev/null 2>&1 && ! python3 -c 'import ensurepip' 2> /dev/null; then
      curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.local/bin sh > /dev/null 2>&1
      export PATH="/root/.local/bin:$PATH"
    fi
    if command -v uv > /dev/null 2>&1; then
      uv venv --allow-existing "$venv" && uv pip install --python "$py" openai anthropic jsonschema pytest httpx pydantic
    else
      python3 -m venv "$venv" && "$py" -m pip install -q openai anthropic jsonschema pytest httpx pydantic
    fi || { say "python env failed"; return 1; }
  fi
  say "build done"
}

# ---- b. conformance on real models -------------------------------------------------------------------------
conformance() {
  local m=$1 f=${MODEL_FILE[$1]} alias=${MODEL_ALIAS[$1]}
  [ -f "$out/conformance-$m.junit.xml" ] && { say "conformance $m: done before"; return 0; }
  [ -f "$f" ] || { say "conformance $m: missing $f"; return 1; }
  say "conformance $m start"
  start_server "$out/conformance-$m.server.log" "$tet" serve -m "$f" --ctx 32768 --port "$port_tet" --alias "$alias" \
    --accept-any-model --ngl -1 -t "$threads" --request-log "$out/conformance-$m.reqlog.jsonl"
  if wait_ready "http://127.0.0.1:$port_tet/v1/models" 600; then
    TETRAEDER_BASE_URL="http://127.0.0.1:$port_tet" TETRAEDER_MODEL="$alias" \
      TETRAEDER_REQUEST_LOG="$out/conformance-$m.reqlog.jsonl" \
      timeout -k 30 25m "$py" -m pytest tests/conformance -q -p no:cacheprovider \
      --junitxml="$out/conformance-$m.junit.xml" > "$out/conformance-$m.pytest.txt" 2>&1
    say "conformance $m rc=$? (junit: $out/conformance-$m.junit.xml)"
  fi
  stop_server
}

# ---- c. paired benchmark -----------------------------------------------------------------------------------
# bench_run MODEL ENGINE TEST REP URL -- gateway_bench.py arguments
bench_run() {
  local m=$1 engine=$2 test=$3 rep=$4 url=$5
  shift 5
  local f="$out/bench-$m-$engine-$test-r$rep.json"
  [ -f "$f" ] && return 0
  timeout -k 30 15m "$py" scripts/gateway_bench.py "$url" --model "${MODEL_ALIAS[$m]}" --temperature 0 \
    --label "$m-$engine-$test-r$rep" --out "$f.tmp" "$@" >> "$out/bench.log" 2>&1 && mv "$f.tmp" "$f"
}

# run_tests MODEL ENGINE REP URL TESTS...: the benchmark runs of one server
run_tests() {
  local m=$1 engine=$2 rep=$3 url=$4 t
  shift 4
  # warm up (CUDA graphs, caches, first-request costs)
  "$py" scripts/gateway_bench.py "$url" --model "${MODEL_ALIAS[$m]}" --prompt-tokens 128 --max-tokens 16 --requests 1 \
    > /dev/null 2>&1
  for t in "$@"; do
    case $t in
      c1) bench_run "$m" "$engine" c1 "$rep" "$url" --prompt-tokens 512 --max-tokens 128 --concurrency 1 --requests 8 ;;
      c4) bench_run "$m" "$engine" c4 "$rep" "$url" --prompt-tokens 512 --max-tokens 128 --concurrency 4 --requests 16 ;;
      p4096) bench_run "$m" "$engine" p4096 "$rep" "$url" --prompt-tokens 4096 --max-tokens 64 --concurrency 1 --requests 4 ;;
      mt) bench_run "$m" "$engine" mt "$rep" "$url" --prompt-tokens 512 --max-tokens 64 --multi-turn 8 ;;
    esac
  done
}

missing() { # missing MODEL ENGINE REP TESTS...: success if any output is absent
  local m=$1 e=$2 r=$3 t
  shift 3
  for t in "$@"; do [ -f "$out/bench-$m-$e-$t-r$r.json" ] || return 0; done
  return 1
}

# server_pass MODEL ENGINE REP TESTS...: start one server config, run its tests, stop it
server_pass() {
  local m=$1 engine=$2 rep=$3 f=${MODEL_FILE[$1]} alias=${MODEL_ALIAS[$1]}
  shift 3
  local fname=$engine # engine name in result files: both llama-server configs are one engine
  case $engine in llama-server-*) fname=llama-server ;; esac
  missing "$m" "$fname" "$rep" "$@" || return 0
  over_budget && { say "budget spent: skip $m $engine r$rep"; return 0; }
  say "pass $m $engine r$rep ($*) start"
  local slog="$out/server-$m-$engine-r$rep.log"
  case $engine in
    llama-server-np1)
      start_server "$slog" "$llama_bin/llama-server" -m "$f" -ngl 99 -c 32768 -t "$threads" --port "$port_ll" -np 1
      wait_ready "http://127.0.0.1:$port_ll/health" 600 && run_tests "$m" "$fname" "$rep" "http://127.0.0.1:$port_ll" "$@" ;;
    llama-server-np4)
      start_server "$slog" "$llama_bin/llama-server" -m "$f" -ngl 99 -c "$c4_ctx" -t "$threads" --port "$port_ll" -np 4
      wait_ready "http://127.0.0.1:$port_ll/health" 600 && run_tests "$m" "$fname" "$rep" "http://127.0.0.1:$port_ll" "$@" ;;
    tetraeder | tetraeder-llama)
      local eng=auto
      [ "$engine" = tetraeder-llama ] && eng=llama
      start_server "$slog" "$tet" serve -m "$f" --engine "$eng" --ngl -1 -t "$threads" --ctx 32768 --port "$port_tet" \
        --alias "$alias" --accept-any-model
      wait_ready "http://127.0.0.1:$port_tet/v1/models" 600 && run_tests "$m" "$fname" "$rep" "http://127.0.0.1:$port_tet" "$@" ;;
  esac
  stop_server
  say "pass $m $engine r$rep done"
}

overhead() {
  local m=$1 f=${MODEL_FILE[$1]}
  if [ ! -f "$out/overhead-$m-tetraeder-bench.json" ]; then
    say "overhead $m tetraeder bench"
    timeout -k 30 15m "$tet" bench -m "$f" -p 512 -n 128 -r 3 -t "$threads" --ngl -1 \
      > "$out/overhead-$m-tetraeder-bench.json.tmp" 2> "$out/overhead-$m-tetraeder-bench.err" \
      && mv "$out/overhead-$m-tetraeder-bench.json.tmp" "$out/overhead-$m-tetraeder-bench.json"
  fi
  if [ ! -f "$out/overhead-$m-llama-bench.json" ]; then
    say "overhead $m llama-bench"
    timeout -k 30 15m "$llama_bin/llama-bench" -m "$f" -p 512 -n 128 -r 3 -t "$threads" -ngl 99 -o json \
      > "$out/overhead-$m-llama-bench.json.tmp" 2> "$out/overhead-$m-llama-bench.err" \
      && mv "$out/overhead-$m-llama-bench.json.tmp" "$out/overhead-$m-llama-bench.json"
  fi
}

# ---- d. env.json -------------------------------------------------------------------------------------------
env_json() {
  [ -f "$out/env.json" ] && return 0
  nvidia-smi -q -d CLOCK,PERFORMANCE > "$out/nvidia-smi-clock-performance.txt" 2>&1
  GW_OUT="$out" GW_THREADS="$threads" GW_QUOTA="$(bash scripts/cpu_quota.sh 2> /dev/null)" \
    GW_LLAMA="$("$llama_bin/llama-server" --version 2>&1 | head -3 | tr '\n' ' ')" \
    GW_COMMIT="$(git rev-parse HEAD 2> /dev/null)" GW_HOST="$(hostname)" \
    GW_GPU="$(nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,clocks_throttle_reasons.active,memory.total --format=csv,noheader 2> /dev/null)" \
    GW_VAST="${VAST_CONTAINERLABEL:-} ${CONTAINER_ID:-}" python3 - << 'EOF'
import json, os
e = os.environ
name, drv, sm, smmax, mem, throttle, vram = ([x.strip() for x in e["GW_GPU"].split(",")] + [""] * 7)[:7]
json.dump({
    "host": e["GW_HOST"], "vast_container": e["GW_VAST"].strip(), "gpu": name, "driver": drv, "vram": vram,
    "clock_sm_mhz": sm, "clock_sm_max_mhz": smmax, "clock_mem_mhz": mem, "throttle_reasons_active": throttle,
    "cpu_quota_cores": e["GW_QUOTA"], "threads": e["GW_THREADS"], "git_commit": e["GW_COMMIT"],
    "llama_cpp": e["GW_LLAMA"], "see": "nvidia-smi-clock-performance.txt",
}, open(os.path.join(e["GW_OUT"], "env.json"), "w"), indent=2)
EOF
}

{
  say "gateway_box start (threads=$threads reps=$reps budget=${BUDGET_MIN:-100} min) out=$out"
  env_json
  rc=0
  build || rc=1
  if [ "$rc" = 0 ]; then
    for m in ${MODELS:-q8b q30b}; do conformance "$m"; done
    # repetitions outermost: a spent budget still leaves complete pairs for every engine
    for rep in $(seq 1 "$reps"); do
      for m in ${MODELS:-q8b q30b}; do
        # c=4: llama-server batches 4 slots (-np 4), tetraeder serves one sequence at a time until Phase 4
        server_pass "$m" llama-server-np1 "$rep" c1 p4096 mt
        server_pass "$m" llama-server-np4 "$rep" c4
        server_pass "$m" tetraeder "$rep" c1 p4096 mt c4
        # the 8B is libllama under engine auto already; the 30B also runs libllama on the same file
        [ "$m" = q30b ] && server_pass "$m" tetraeder-llama "$rep" c1 p4096 mt c4
      done
    done
    for m in ${MODELS:-q8b q30b}; do overhead "$m"; done
    env_json
    "$py" scripts/gateway_summary.py "$out" || rc=1
  fi
  touch "$out/DONE"
  say "GATEWAY_BOX_DONE rc=$rc"
} >> "$log" 2>&1
