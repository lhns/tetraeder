#!/usr/bin/env bash
# Same-box A/B of the own engines (`tetraeder serve --engine moe` on Qwen3-30B-A3B, `--engine dense` on Qwen3-8B): A = an
# earlier commit (ab30b: Phase 4 verdict 6b07795; regression gate: f68d2a0), B = the branch head pushed to the box.
# Diagnostic, no llama-server (bench/results/ab30b-20261122, bench/results/regression-gate-20261123).
# Started detached by `taxameter run scripts/ab_moe_box.sh`; ends with AB_BOX_DONE in runs/vast/ab_moe_box.log.
# Stages: build (A and B release builds in separate checkouts), env, runs (per model of MODELS: 3 rounds A B, B A, A B; one
# server start per side and round: warm-up, c4, then c1 of the Phase 4 workload, seeds per round, the same on both sides),
# ttft (TTFT=1: the Qwen3-8B request-log replay of ttft_diag.sh, once on A, once on B), summary
# (scripts/ab_moe_summary.py -> ab-summary-<model>.json, ab-ttft.json). A rerun resumes: a run whose c1 and c4 files exist
# is skipped. Result files are ab-<model>-<A|B>-<c4|c1>-r<n>.json.
# Source of A: a tarball of `git archive <A commit>` (the box has no history), put at A_TAR; unpacked to A_DIR.
# Env: SESSION, THREADS (CPU quota), ROUNDS (3), MODELS ("q30b q8b"), A_COMMIT, COMMIT (B), A_DIR, A_TAR, BUDGET_MIN (100),
# AB_OUT, MODEL_Q30B, MODEL_Q8B, TTFT (1), REQ_LOG (bench/private/claude-code-requests.jsonl), DRY_RUN=1 (print the plan, run nothing), SKIP_GPU=1 (tests).
DRY="${DRY_RUN:-0}"
skip_gpu="${SKIP_GPU:-0}"
if [ "$DRY" != 1 ]; then
  exec 8> "${AB_LOCK:-/root/tetraeder_ab_moe_box.lock}"
  flock -n 8 || exit 0
fi
cd "$(dirname "$0")/.." || exit 1
b_dir="$PWD"
a_dir="${A_DIR:-/root/tetraeder-A}"
a_tar="${A_TAR:-/root/ab_A.tar.gz}"
if [ "$DRY" != 1 ]; then
  if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
  export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
  if [ "$skip_gpu" != 1 ]; then
    # shellcheck disable=SC1091
    . "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
    archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
    [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
  fi
fi
log="${AB_LOG_FILE:-runs/vast/ab_moe_box.log}"
session="${SESSION:-ab30b-$(date +%Y%m%d)}"
out="${AB_OUT:-$PWD/bench/results/$session/ab}"
mkdir -p "$out" "$(dirname "$log")"
if [ "$DRY" = 1 ]; then
  threads="${THREADS:-8}"
else
  threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"
fi
rounds="${ROUNDS:-3}"
budget_s=$((${BUDGET_MIN:-100} * 60))
t_start=$(date +%s)
models="${MODELS:-q30b q8b}"
ttft_on="${TTFT:-1}"
req_log="${REQ_LOG:-bench/private/claude-code-requests.jsonl}"
rctx=40960 rmax=36000
declare -A MODEL_FILE=(
  [q30b]="${MODEL_Q30B:-$PWD/.hf_cache/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf}"
  [q8b]="${MODEL_Q8B:-$PWD/.hf_cache/Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf}")
declare -A MODEL_ENGINE=([q30b]=moe [q8b]=dense)
declare -A MODEL_NAME=([q30b]=qwen3-30b-a3b [q8b]=qwen3-8b)
model="${MODEL_FILE[${models%% *}]}" # the file the disk probe reads (one disk)
mname="" # the served model name of the current run
ctx=32768 ptok=512 mtok=128 n_c1=8 n_c4=16
port=18080
venv=/root/gwvenv
py="${PY:-$venv/bin/python}"
[ "$DRY" = 1 ] && py=python3
server_pid=""
disk_dev=""
sampler_pid=""

say() { echo "[$(date +%T)] $*"; }
plan() { echo "PLAN $*"; }

# 1 Hz GPU sampler for the whole session (scripts/gpu_sampler.sh -> gpu-samples.csv); ends on gpu-sampler.stop
if [ "$DRY" = 1 ]; then
  [ "$skip_gpu" = 1 ] || plan "gpu_sampler.sh -> gpu-samples.csv"
elif [ "$skip_gpu" != 1 ]; then
  OUT="$out" SAMPLER_PARENT=$$ setsid nohup bash scripts/gpu_sampler.sh 8>&- > /dev/null 2>&1 &
fi
stop_sampler() { [ "$DRY" = 1 ] || touch "$out/gpu-sampler.stop"; }

bin_of() { if [ "$1" = A ]; then echo "${TET_A:-$a_dir/target/release/tetraeder}"; else echo "${TET_B:-$b_dir/target/release/tetraeder}"; fi; }

die() {
  say "FATAL: $*"
  stop_server
  stop_sampler
  touch "$out/DONE"
  say "AB_BOX_DONE rc=1"
  exit 1
}

start_server() { # start_server LOG CMD...
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

wait_ready() { # wait_ready URL SECONDS LOG: a server that exits ends the session (the box bills idle time)
  [ "$DRY" = 1 ] && return 0
  for _ in $(seq 1 $(($2 / 2))); do
    if ! kill -0 "$server_pid" 2> /dev/null; then
      tail -n 15 "$3" | sed 's/^/  server: /'
      die "server exited while loading ($3); a rejected flag? see the lines above"
    fi
    curl -sf -o /dev/null "$1" && return 0
    sleep 2
  done
  die "server not ready after ${2}s ($3)"
}

over_budget() { [ $(($(date +%s) - t_start)) -gt "$budget_s" ]; }

gpu_note() { # gpu_note RUN before|after: throttle reasons bitmask (hex) per GPU
  local f="$out/gpu-$1.txt" when=$2 v
  if [ "$DRY" = 1 ]; then
    plan "nvidia-smi throttle reasons $when -> gpu-$1.txt"
    return 0
  fi
  if [ "$skip_gpu" = 1 ]; then
    echo "$when: 0x0000000000000000" >> "$f"
    return 0
  fi
  v=$(nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader 2> /dev/null) \
    || v=$(nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2> /dev/null)
  [ -n "$v" ] || say "WARNING: no throttle reasons from nvidia-smi for $1 $when"
  echo "$v" | sed "s/^ *\([^ ]\)/$when: \1/" >> "$f"
}

# ---- stage: build ----------------------------------------------------------------------------------------
# build_one DIR: patch queue, then the release build with CUDA in DIR (own target dir); runs in a subshell
build_one() {
  cd "$1" || exit 1
  bash scripts/ensure_patches.sh 8>&- || exit 1
  # shellcheck source=scripts/box_libs.sh
  . scripts/box_libs.sh
  command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
  features="--features cuda"
  [ "$skip_gpu" = 1 ] && features=""
  # shellcheck disable=SC2086
  CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$jobs}" timeout -k 30 50m cargo build --release -p tetraeder $features 8>&-
}

build() {
  if [ "$DRY" = 1 ]; then
    plan "unpack $a_tar to $a_dir; cargo build --release -p tetraeder --features cuda in A and in B (two target dirs); python env $venv"
    return 0
  fi
  if [ ! -d "$a_dir/crates" ]; then
    [ -s "$a_tar" ] || die "no $a_tar"
    mkdir -p "$a_dir" && tar -xzf "$a_tar" -C "$a_dir" || die "unpack A failed"
  fi
  jobs=$((threads / 2))
  [ "$jobs" -lt 4 ] && jobs=4
  say "build start (A $a_dir, B $b_dir, cargo jobs $jobs each, in parallel)"
  (build_one "$a_dir" > "$out/build-A.log" 2>&1) &
  pa=$!
  (build_one "$b_dir" > "$out/build-B.log" 2>&1) &
  pb=$!
  wait "$pa" || die "build A failed (see $out/build-A.log)"
  wait "$pb" || die "build B failed (see $out/build-B.log)"
  if [ "$skip_gpu" != 1 ]; then
    for s in A B; do [ -x "$(bin_of $s)" ] || die "no binary for $s"; done
  fi
  if [ -z "${PY:-}" ] && { [ ! -x "$py" ] || ! "$py" -c 'import httpx, tomllib' 2> /dev/null; }; then
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

# ---- stage: env ------------------------------------------------------------------------------------------
env_json() {
  if [ "$DRY" = 1 ]; then
    plan "env.json (gpu, driver, clocks, cpu model, host cpus, quota, threads, A and B commits)"
    return 0
  fi
  [ -f "$out/env.json" ] && return 0
  nvidia-smi -q -d CLOCK,PERFORMANCE,POWER > "$out/nvidia-smi-clock-performance.txt" 2>&1
  GW_OUT="$out" GW_THREADS="$threads" GW_QUOTA="$(bash scripts/cpu_quota.sh 2> /dev/null)" \
    GW_CPU="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)" GW_NPROC="$(nproc)" \
    GW_A="${A_COMMIT:-}" GW_B="${COMMIT:-}" GW_HOST="$(hostname)" \
    GW_GPU="$(nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,clocks_throttle_reasons.active,memory.total,power.limit,power.default_limit --format=csv,noheader 2> /dev/null)" \
    GW_VAST="${VAST_CONTAINERLABEL:-} ${CONTAINER_ID:-}" "$py" - << 'PYEOF'
import json, os
e = os.environ
k = ["gpu", "driver", "clock_sm_mhz", "clock_sm_max_mhz", "clock_mem_mhz", "throttle_reasons_active", "vram", "power_limit", "power_default_limit"]
g = dict(zip(k, ([x.strip() for x in e["GW_GPU"].split(",")] + [""] * 9)[:9]))
json.dump({"host": e["GW_HOST"], "vast_container": e["GW_VAST"].strip(), **g, "cpu_model": e["GW_CPU"],
           "host_cpus": e["GW_NPROC"], "cpu_quota_cores": e["GW_QUOTA"], "threads": e["GW_THREADS"],
           "commit_A": e["GW_A"], "commit_B": e["GW_B"]},
          open(os.path.join(e["GW_OUT"], "env.json"), "w"), indent=2)
PYEOF
}

# ---- runs ------------------------------------------------------------------------------------------------
bench_file() { # bench_file FILE LABEL URL ARGS...: the file is kept even with errors (the summary reads them)
  local f=$1 label=$2 url=$3
  shift 3
  [ "$DRY" = 1 ] && { plan "gateway_bench.py $url --no-keepalive --label $label $* -> ${f##*/}"; return 0; }
  timeout -k 30 15m "$py" scripts/gateway_bench.py "$url" --model "$mname" --temperature 0 --no-keepalive --label "$label" \
    --out "$f.tmp" "$@" >> "$out/bench.log" 2>&1
  [ -s "$f.tmp" ] && mv "$f.tmp" "$f"
  return 0
}

warm_up() {
  [ "$DRY" = 1 ] && { plan "warm-up: 1 request, 128 prompt tokens"; return 0; }
  "$py" scripts/gateway_bench.py "$1" --model "$mname" --prompt-tokens 128 --max-tokens 16 --requests 1 --no-keepalive > /dev/null 2>&1
}

sampler_start() { # host sampler beside the server, best effort: its host block feeds cotenant_guard.py
  sampler_pid=""
  [ "$DRY" = 1 ] && { plan "engine_sample.py sample --pid <server> (host block for cotenant_guard)"; return 0; }
  [ -n "$disk_dev" ] || disk_dev=$(python3 scripts/engine_sample.py probe-disk "$model" 2> /dev/null)
  [ -n "$disk_dev" ] || return 0
  samples="runs/vast/ab-$1.samples"
  python3 scripts/engine_sample.py sample --pid "$server_pid" --disk "$disk_dev" --out "$samples" > /dev/null 2>&1 8>&- &
  sampler_pid=$!
}

sampler_finish() { # sampler_finish RUN T0 T1 FILES...: needs the server stopped (the sampler follows its pid)
  local run=$1 t0=$2 t1=$3 f
  shift 3
  [ "$DRY" = 1 ] && { plan "cotenant_guard.py merge for $*"; return 0; }
  [ -n "$sampler_pid" ] && wait "$sampler_pid" 2> /dev/null
  if [ -n "$sampler_pid" ] && [ -s "$samples" ]; then
    printf '{"t_start": %s, "t_first": %s, "t_end": %s, "prompt_tokens": 1, "decode_tokens": 2}\n' "$t0" "$t0" "$t1" > "runs/vast/ab-$run.window.json"
    for f in "$@"; do
      [ -s "$f" ] && { python3 scripts/engine_sample.py summarize --samples "$samples" --threads "$threads" \
        --kt-json "runs/vast/ab-$run.window.json" --out "$samples.summary" --merge-into "$f" > /dev/null || say "host summary failed for $run"; }
    done
  fi
  for f in "$@"; do [ -s "$f" ] && { python3 -I scripts/cotenant_guard.py merge --json "$f" || true; }; done
}

one_run() { # one_run MODEL SIDE ROUND: fresh server, warm-up, c4, c1; seeds per round, the same on both sides
  local m=$1 side=$2 rnd=$3 run="$1-$2-r$3" url="http://127.0.0.1:$port"
  local c4="$out/ab-$run-c4.json" c1="$out/ab-$run-c1.json"
  mname=${MODEL_NAME[$m]}
  model=${MODEL_FILE[$m]}
  if [ -f "$c1" ] && [ -f "$c4" ]; then
    say "run $run: done before"
    return 0
  fi
  if over_budget; then
    say "BUDGET spent: skip run $run"
    return 0
  fi
  say "run $run start"
  : > "$out/gpu-$run.txt"
  start_server "$out/server-$run.log" "$(bin_of "$side")" serve -m "$model" --engine "${MODEL_ENGINE[$m]}" --ngl -1 --parallel 4     --ctx "$ctx" -t "$threads" --port "$port" --alias "$mname" --accept-any-model
  wait_ready "$url/v1/models" 900 "$out/server-$run.log"
  gpu_note "$run" before
  sampler_start "$run"
  warm_up "$url"
  local t0 t1
  t0=$(date +%s.%N)
  bench_file "$c4" "$run-c4" "$url" --prompt-tokens "$ptok" --max-tokens "$mtok" --concurrency 4 --requests "$n_c4" --seed $((rnd * 10 + 4))
  bench_file "$c1" "$run-c1" "$url" --prompt-tokens "$ptok" --max-tokens "$mtok" --concurrency 1 --requests "$n_c1" --seed $((rnd * 10 + 1))
  t1=$(date +%s.%N)
  gpu_note "$run" after
  stop_server
  sampler_finish "$run" "$t0" "$t1" "$c4" "$c1"
  say "run $run done"
}

# ttft_run SIDE: the request-log replay of ttft_diag.sh on Qwen3-8B (dense, --cache-ram 8192, ctx $rctx), one server per side
ttft_run() {
  local side=$1 run="ttft-$1" url="http://127.0.0.1:$port"
  local res="$out/ab-$run.json" f=${MODEL_FILE[q8b]}
  [ -f "$res" ] && { say "run $run: done before"; return 0; }
  say "run $run start"
  : > "$out/gpu-$run.txt"
  start_server "$out/server-$run.log" "$(bin_of "$side")" serve -m "$f" --engine dense --ngl -1 --parallel 4 --ctx "$rctx"     -t "$threads" --port "$port" --alias qwen3-8b --accept-any-model --cache-ram 8192
  wait_ready "$url/v1/models" 900 "$out/server-$run.log"
  gpu_note "$run" before
  if [ "$DRY" = 1 ]; then
    plan "gateway_replay.py $url $req_log --speed 0 --concurrency 1 --max-prompt-tokens $rmax --gguf <q8b> --expect-context-length $rctx -> ab-$run.json"
  else
    timeout -k 30 60m "$py" scripts/gateway_replay.py "$url" "$req_log" --speed 0 --concurrency 1       --max-prompt-tokens "$rmax" --gguf "$f" --expect-context-length "$rctx" --out "$res.tmp" >> "$out/replay.log" 2>&1
    local rc=$?
    [ "$rc" = 2 ] && die "replay: Qwen3-8B context_length is not $rctx (see $out/replay.log)"
    [ -s "$res.tmp" ] && mv "$res.tmp" "$res"
  fi
  gpu_note "$run" after
  stop_server
  say "run $run done"
}

ttft() {
  [ "$ttft_on" = 1 ] || return 0
  if [ "$DRY" != 1 ]; then
    [ -s "$req_log" ] || { say "ttft: no request log at $req_log, skipped"; return 0; }
    [ "$(sha256sum "$req_log" | cut -d' ' -f1)" = f79b4a6d810f71632fb0633553b32d3c6c18132a137a4005e2129688f95f7b35 ]       || die "request log sha256 mismatch"
  fi
  ttft_run A
  ttft_run B
}

summary() {
  if [ "$DRY" = 1 ]; then
    plan "ab_moe_summary.py $out --models ${models// /,} -> ab-summary-<model>.json, ab-ttft.json"
    return 0
  fi
  "$py" scripts/ab_moe_summary.py "$out" --rounds "$rounds" --models "${models// /,}"
}

main() {
  say "ab_moe_box start (threads=$threads rounds=$rounds A=${A_COMMIT:-?} B=${COMMIT:-?}) out=$out"
  say "STAGE build"
  build
  say "STAGE env"
  env_json
  say "STAGE runs"
  local rnd rc m
  for m in $models; do
    for rnd in $(seq 1 "$rounds"); do
      if [ $((rnd % 2)) = 1 ]; then
        one_run "$m" A "$rnd"
        one_run "$m" B "$rnd"
      else
        one_run "$m" B "$rnd"
        one_run "$m" A "$rnd"
      fi
      summary > /dev/null 2>&1 || true # incremental
    done
  done
  say "STAGE ttft"
  ttft
  say "STAGE summary"
  summary
  rc=$?
  stop_sampler
  touch "$out/DONE"
  say "AB_BOX_DONE rc=$rc"
  return $rc
}

if [ "$DRY" = 1 ]; then
  main
else
  main >> "$log" 2>&1
fi
