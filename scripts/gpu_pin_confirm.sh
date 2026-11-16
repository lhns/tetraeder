#!/usr/bin/env bash
# Pin confirmation on a GPU box (ADR 0073, "Consequences"): the engine on llama.cpp b12426 on CUDA, and a logged c4
# probe. No verdict pairs. Started detached by `taxameter run --id <id> scripts/gpu_pin_confirm.sh --env SESSION=<s>`;
# log runs/vast/pin_confirm.log ends with PIN_CONFIRM_DONE rc=<n>. Results: bench/results/<session>/pin-confirm/.
# Stages (STAGE lines in the log; a finished stage leaves stage-<name>.ok in the results dir, a rerun resumes):
#   build     tetraeder release with CUDA (as D2a) and the python env
#   smoke6    scripts/gpu_smoke6.sh: mul_mat_id_skip (patch 0004), gpu_multi_seq, gpu_dense, executor gpu_ tests
#   batched   scripts/gpu_real_batched.sh: q8b and q30b against libllama batched, alone control, anchors
#   golden    skipped: the 0.6B goldens (crates/engine/tests/golden.rs, crates/model/tests/dense_golden.rs) are CPU only
#   shapes    step_shapes on CUDA, Qwen3-8B, engines llama,dense, ROUNDS rounds (default 3)
#   probe     per model, 2 interleaved rounds of llama-server b12426 (8B nokvu, 30B kvu) and tetraeder (8B dense, 30B
#             moe, RUST_LOG=info,tetraeder::step=debug): warm-up, c1, c4, as the D2a probe; throttle before/after
#   summary   scripts/pin_confirm_summary.py -> summary.md, summary.json
# Env: SESSION, MODELS ("q8b q30b"), ROUNDS (3, shapes), PROBE_ROUNDS (2), THREADS, CTX, BASELINE_BIN.
# Models are assumed in .hf_cache/<org>__<model>/ (provision --models), run offline.
exec 8> /root/tetraeder_pin_confirm.lock
flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
root=$PWD
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
export HF_HUB_OFFLINE=1 TETRAEDER_MODELS_DIR="$root/.hf_cache"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
session="${SESSION:-$(date +%Y%m%d)}"
export RESULTS_DATE="$session/pin-confirm" # gpu_smoke6.sh and gpu_real_batched.sh write under bench/results/$RESULTS_DATE
out="$root/bench/results/$RESULTS_DATE"
log="$root/runs/vast/pin_confirm.log"
mkdir -p "$out" "$(dirname "$log")"
exec >> "$log" 2>&1
t_start=$(date +%s)
threads="${THREADS:-$(bash scripts/cpu_quota.sh 8>&-)}"
models="${MODELS:-q8b q30b}"
rounds="${ROUNDS:-3}"
probe_rounds="${PROBE_ROUNDS:-2}"
models_dir="$root/.hf_cache"
tet="$root/target/release/tetraeder"
baseline_bin="${BASELINE_BIN:-/opt/llama.cpp-baseline/bin/llama-server}"
venv=/root/gwvenv
py="${PY:-$venv/bin/python}"
ctx="${CTX:-32768}"
ptok=512 mtok=128 n_probe=8
port_tet=18080
port_ll=18081
server_pid=""

declare -A MODEL_FILE=(
  [q30b]="$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
  [q8b]="$models_dir/Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
)
declare -A MODEL_ALIAS=([q30b]=qwen3-30b-a3b [q8b]=qwen3-8b)
# the baseline's KV mode that D2a's pre-pass picked (20261116d2a2)
declare -A VARIANT=([q30b]=kvu [q8b]=nokvu)
declare -A JUDGED=([q30b]=moe [q8b]=dense)

say() { echo "[$(date +%T)] $*"; }
die() {
  say "FATAL: $*"
  stop_server
  echo "rc=1" > "$out/DONE"
  say "PIN_CONFIRM_DONE rc=1"
  exit 1
}
start_server() { # LOG CMD...; detached, pid in $server_pid, killed by pid only
  local slog=$1
  shift
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
wait_ready() { # URL SECONDS SERVER_LOG
  local url=$1 secs=$2 slog=$3
  for _ in $(seq 1 $((secs / 2))); do
    if ! kill -0 "$server_pid" 2> /dev/null; then
      tail -n 15 "$slog" | sed 's/^/  server: /'
      die "server exited while loading ($slog)"
    fi
    curl -sf -o /dev/null "$url" && return 0
    sleep 2
  done
  die "server not ready after ${secs}s ($slog)"
}
gpu_note() { # RUN before|after: throttle hex and clocks
  local f="$out/gpu-$1.txt" when=$2 v
  v=$(nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader 2> /dev/null) \
    || v=$(nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2> /dev/null)
  [ -n "$v" ] || say "WARNING: no throttle reasons from nvidia-smi for $1 $when"
  echo "$v" | sed "s/^ *\([^ ]\)/$when: \1/" >> "$f"
  echo "$when clocks: $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader 2> /dev/null)" >> "$f"
}
# stage NAME: true when the stage still has to run
stage() {
  say "STAGE $1 (elapsed $(((($(date +%s) - t_start)) / 60)) min)"
  if [ -f "$out/stage-$1.ok" ]; then
    say "stage $1: done before"
    return 1
  fi
  return 0
}
ok() { touch "$out/stage-$1.ok"; }

# ---- build ---------------------------------------------------------------------------------------------------
build() {
  stage build || return 0
  case "$CUDA_ENV_STATUS" in ok | ok_preload) say "cuda: $CUDA_ENV_STATUS" ;; *) die "no cuda ($CUDA_ENV_STATUS)" ;; esac
  bash scripts/ensure_patches.sh 8>&- || die "patch queue failed"
  say "vendor/PIN: $(tr '\n' ' ' < vendor/PIN)"
  say "baseline: $("$baseline_bin" --version 2>&1 | head -3 | tr '\n' ' ') | $(cat /opt/llama.cpp-baseline/PIN 2> /dev/null | tr '\n' ' ')"
  timeout -k 30 50m cargo build --release -p tetraeder --features cuda 8>&- || die "build failed"
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
  box_libs_path
  nvidia-smi -q -d CLOCK,PERFORMANCE > "$out/nvidia-smi-clock-performance.txt" 2>&1
  ok build
}

# ---- smoke6, batched -----------------------------------------------------------------------------------------
# The two scripts take their own locks, write their logs under runs/vast and results under bench/results/$RESULTS_DATE.
smoke6() {
  stage smoke6 || return 0
  bash scripts/gpu_smoke6.sh 8>&-
  say "smoke6 rc=$? ($(cat "$out/DONE6" 2> /dev/null))"
  cp runs/vast/gpu-smoke6.log "$out/gpu-smoke6-run.log" 2> /dev/null
  cp runs/vast/gpu-smoke5.log "$out/gpu-smoke5-run.log" 2> /dev/null
  [ -f "$out/DONE6" ] || die "smoke6 left no DONE6"
  ok smoke6
}
batched() {
  stage batched || return 0
  bash scripts/gpu_real_batched.sh 8>&-
  say "batched rc=$? ($(cat "$out/real-batched/DONE" 2> /dev/null))"
  cp runs/vast/gpu-real-batched.log "$out/gpu-real-batched-run.log" 2> /dev/null
  [ -f "$out/real-batched/DONE" ] || die "batched left no DONE"
  ok batched
}

# ---- shapes --------------------------------------------------------------------------------------------------
shapes() {
  stage shapes || return 0
  local r raw
  for r in $(seq 1 "$rounds"); do
    raw="$out/shapes-r$r.txt"
    if [ -s "$raw" ] && grep -q '^SHAPE' "$raw"; then continue; fi
    say "shapes round $r"
    TETRAEDER_REQUIRE_GPU=1 TETRAEDER_SHAPES_GGUF="${MODEL_FILE[q8b]}" TETRAEDER_SHAPES_NGL=99 \
      TETRAEDER_SHAPES_THREADS="$threads" TETRAEDER_SHAPES_ENGINES=llama,dense \
      timeout -k 30 30m cargo test --release -p tetraeder-scheduler --features cuda --test step_shapes \
      -- --nocapture --test-threads 1 > "$raw.tmp" 2>&1 8>&- || say "WARNING: shapes round $r failed (see $raw)"
    [ -s "$raw.tmp" ] && mv "$raw.tmp" "$raw"
    gpu_note "shapes-r$r" after
  done
  grep -h '^SHAPE' "$out"/shapes-r*.txt | sed 's/^/  /'
  ok shapes
}

# ---- probe ---------------------------------------------------------------------------------------------------
bench_file() { # FILE LABEL URL ALIAS ARGS...: as D2a
  local f=$1 label=$2 url=$3 alias=$4
  shift 4
  timeout -k 30 15m "$py" scripts/gateway_bench.py "$url" --model "$alias" --temperature 0 --no-keepalive --label "$label" \
    --out "$f.tmp" "$@" >> "$out/bench.log" 2>&1
  [ -s "$f.tmp" ] && mv "$f.tmp" "$f"
  return 0
}
warm_up() { "$py" scripts/gateway_bench.py "$1" --model "$2" --prompt-tokens 128 --max-tokens 16 --requests 1 --no-keepalive > /dev/null 2>&1; }

# probe_run MODEL NAME ROUND: NAME = llama-server | tetraeder. The seeds are the same on both engines for a round.
probe_run() {
  local m=$1 name=$2 rd=$3 run="$1-$2-r$3" url alias=${MODEL_ALIAS[$1]} f=${MODEL_FILE[$1]} kv=-kvu
  local c1="$out/probe-$run-c1.json" c4="$out/probe-$run-c4.json" slog="$out/server-$run.log"
  [ -f "$c1" ] && [ -f "$c4" ] && { say "probe $run: done before"; return 0; }
  say "probe $run start"
  : > "$out/gpu-$run.txt"
  if [ "$name" = llama-server ]; then
    [ "${VARIANT[$m]}" = nokvu ] && kv=-no-kvu
    url="http://127.0.0.1:$port_ll"
    start_server "$slog" "$baseline_bin" -m "$f" -ngl 99 -np 4 -c "$ctx" -fa on -ctk f16 -ctv f16 -t "$threads" \
      --host 127.0.0.1 --port "$port_ll" $kv
    wait_ready "$url/health" 900 "$slog"
  else
    url="http://127.0.0.1:$port_tet"
    RUST_LOG=info,tetraeder::step=debug start_server "$slog" "$tet" serve -m "$f" --engine "${JUDGED[$m]}" --ngl -1 \
      --parallel 4 --ctx "$ctx" -t "$threads" --port "$port_tet" --alias "$alias" --accept-any-model
    wait_ready "$url/v1/models" 900 "$slog"
  fi
  gpu_note "$run" before
  warm_up "$url" "$alias"
  bench_file "$c1" "probe-$run-c1" "$url" "$alias" --prompt-tokens "$ptok" --max-tokens "$mtok" --concurrency 1 --requests "$n_probe" --seed $((90 + rd))
  bench_file "$c4" "probe-$run-c4" "$url" "$alias" --prompt-tokens "$ptok" --max-tokens "$mtok" --concurrency 4 --requests "$n_probe" --seed $((93 + rd))
  gpu_note "$run" after
  stop_server
  say "probe $run done: $("$py" -c "import json,sys;s=json.load(open(sys.argv[1]))['summary'];print('c4 tps %.1f errors %s' % (s['throughput_tps'], s['errors']))" "$c4" 2> /dev/null)"
}
probe() {
  stage probe || return 0
  local m rd
  for m in $models; do
    [ -s "${MODEL_FILE[$m]}" ] || { say "WARNING: no model ${MODEL_FILE[$m]}: $m skipped"; continue; }
    for rd in $(seq 1 "$probe_rounds"); do
      if [ $((rd % 2)) = 1 ]; then
        probe_run "$m" llama-server "$rd"
        probe_run "$m" tetraeder "$rd"
      else
        probe_run "$m" tetraeder "$rd"
        probe_run "$m" llama-server "$rd"
      fi
    done
  done
  ok probe
}

# ---- env, summary --------------------------------------------------------------------------------------------
summary() {
  stage summary || return 0
  GW_HOST="$(hostname)" GW_COMMIT="${COMMIT:-$(git rev-parse HEAD 2> /dev/null)}" GW_THREADS="$threads" \
    GW_GPU="$(nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,memory.total --format=csv,noheader 2> /dev/null)" \
    GW_BASELINE="$("$baseline_bin" --version 2>&1 | head -3 | tr '\n' ' ')" GW_PIN="$(tr '\n' ' ' < vendor/PIN)" \
    GW_BPIN="$(cat /opt/llama.cpp-baseline/PIN 2> /dev/null | tr '\n' ' ')" GW_QUOTA="$(bash scripts/cpu_quota.sh 2> /dev/null)" \
    GW_VAST="${VAST_CONTAINERLABEL:-} ${CONTAINER_ID:-}" GW_OUT="$out" python3 - << 'EOF'
import json, os
e = os.environ
json.dump({k.lower()[3:]: v.strip() for k, v in e.items() if k.startswith("GW_") and k != "GW_OUT"},
          open(os.path.join(e["GW_OUT"], "env.json"), "w"), indent=2)
EOF
  python3 -I scripts/pin_confirm_summary.py "$out" || say "WARNING: summary failed"
  ok summary
}

main() {
  say "pin_confirm start (threads=$threads models=\"$models\" rounds=$rounds probe_rounds=$probe_rounds) out=$out"
  rm -f "$out/DONE"
  build
  smoke6
  batched
  say "STAGE golden: skipped, the 0.6B golden tests are CPU only"
  shapes
  probe
  summary
  # taxameter fetch skips *.txt and *.md: log copies come back too
  local f
  for f in "$out"/*.txt "$out"/*.md; do [ -s "$f" ] && cp "$f" "$f.log"; done
  echo "rc=0" > "$out/DONE"
  say "PIN_CONFIRM_DONE rc=0 (elapsed $(((($(date +%s) - t_start)) / 60)) min)"
}
main
