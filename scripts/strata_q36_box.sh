#!/usr/bin/env bash
# bench-strata-q36 (ADR 0085): Qwen3.6-35B-A3B (qwen35moe) UD-Q4_K_XL on one GPU box, three engines on the same GGUF:
#   a = llama-server at the baseline pin (/opt/llama.cpp-baseline, -fa on -np 4, kvu/no-kvu picked by a pre-pass),
#   b = `tetraeder serve --engine auto` (libllama path), c = Strata v0.2.20 (its Terminal installer and engine package).
# Started detached by `taxameter run scripts/strata_q36_box.sh`; ends with STRATA_BOX_DONE in runs/vast/strata_q36_box.log.
# Stages (STAGES, default all; a finished stage leaves stage-<name>.ok in the results dir):
#   build    cargo build tetraeder (CUDA), the kv_gather test binary, python env, mapped_read.cu, Strata install, env.json
#   probes   ADR 0084 step 1: kv_gather CPU bench, mapped_read (CUDA host read vs cudaMemcpy)
#   prepass  llama-server c4 with -kvu and with -no-kvu (the higher throughput is the baseline, ADR 0065)
#   rounds   3 interleaved rounds (a b c, b c a, c a b): fresh server, warm-up, c4, c1 per run
#   aa       A/A of the own Qwen3-30B-A3B engine (--engine moe), two runs (bench-30b-c4-gap), if the budget allows
#   smoke    (not in the default list) one start of each engine, for a first look
# Env: SESSION, COMMIT, THREADS (CPU quota), ROUNDS (3), BUDGET_MIN (130, runs that would start later are skipped), STAGES,
# STRATA_ARGS (the `strata launch` flags after the common ones), MODEL_Q36, MODEL_Q30B.
exec 8> /root/tetraeder_strata_q36_box.lock
flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
root=$PWD
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$HOME/.local/bin:/root/.local/bin:$PATH"
export HF_HUB_OFFLINE=1
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
[ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
session="${SESSION:-strata-q36-$(date +%Y%m%d)}"
out="$root/bench/results/$session"
log=runs/vast/strata_q36_box.log
mkdir -p "$out" runs/vast
threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"
rounds="${ROUNDS:-3}"
budget_s=$((${BUDGET_MIN:-130} * 60))
t_start=$(date +%s)
stages="${STAGES:-build probes prepass rounds aa}"
model="${MODEL_Q36:-$root/.hf_cache/unsloth__Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
model30="${MODEL_Q30B:-$root/.hf_cache/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf}"
alias36=qwen3.6-35b-a3b
alias30=qwen3-30b-a3b
ctx=32768 ptok=512 mtok=128 n_c1=8 n_c4=16
port_ll=18081 port_tet=18080 port_st=18090
baseline_bin=/opt/llama.cpp-baseline/bin/llama-server
tet="$root/target/release/tetraeder"
venv=/root/gwvenv
py="$venv/bin/python"
strata_feed="https://github.com/Niko1221/Strata/releases/download/v0.2.20"
server_pid=""
strata_up=0
url="" alias=""

say() { echo "[$(date +%T)] $*"; }
stage_done() { [ -f "$out/stage-$1.ok" ]; }

# 1 Hz GPU sampler for the whole session (gpu-samples.csv); ends on gpu-sampler.stop
OUT="$out" SAMPLER_PARENT=$$ setsid nohup bash scripts/gpu_sampler.sh 8>&- > /dev/null 2>&1 &
stop_sampler() { touch "$out/gpu-sampler.stop"; }

start_server() { # start_server LOG CMD...
  local slog=$1
  shift
  : > "$slog"
  setsid nohup "$@" > "$slog" 2>&1 8>&- &
  server_pid=$!
}

stop_server() {
  if [ "$strata_up" = 1 ]; then
    strata launch --stop > /dev/null 2>&1 8>&-
    strata_up=0
  fi
  [ -n "$server_pid" ] || return 0
  kill "$server_pid" 2> /dev/null
  for _ in $(seq 1 30); do kill -0 "$server_pid" 2> /dev/null || break; sleep 1; done
  kill -9 "$server_pid" 2> /dev/null
  wait "$server_pid" 2> /dev/null
  server_pid=""
  wait_gpu_free
}
trap stop_server EXIT

# GPU memory is released after the process is gone (Strata may take up to 120 s to unload): wait for under 1 GB used
wait_gpu_free() {
  local used
  for _ in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | head -1 | tr -d ' ')
    [ -n "$used" ] && [ "$used" -lt 1024 ] && return 0
    sleep 2
  done
  say "WARNING: GPU memory still ${used:-?} MiB used after 180 s"
}

die() {
  say "FATAL: $*"
  stop_server
  stop_sampler
  touch "$out/DONE"
  say "STRATA_BOX_DONE rc=1"
  exit 1
}

wait_ready() { # wait_ready URL SECONDS LOG: a server that exits ends the run's start (returns 1), the caller decides
  local u=$1 secs=$2 slog=$3
  for _ in $(seq 1 $((secs / 2))); do
    if [ -n "$server_pid" ] && ! kill -0 "$server_pid" 2> /dev/null; then
      tail -n 15 "$slog" | sed 's/^/  server: /'
      say "server exited while loading ($slog)"
      return 1
    fi
    curl -sf -o /dev/null "$u" && return 0
    sleep 2
  done
  say "server not ready after ${secs}s ($slog)"
  return 1
}

over_budget() { [ $(($(date +%s) - t_start)) -gt "$budget_s" ]; }

gpu_note() { # gpu_note RUN before|after: throttle reasons bitmask (hex)
  local v
  v=$(nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader 2> /dev/null) \
    || v=$(nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2> /dev/null)
  echo "$v" | sed "s/^ *\([^ ]\)/$2: \1/" >> "$out/gpu-$1.txt"
}

# ---- servers -----------------------------------------------------------------------------------------------
# server_up NAME VARIANT LOG: starts engine NAME (llama-server | tetraeder | strata | moe30), sets url and alias
server_up() {
  local name=$1 variant=$2 slog=$3
  case $name in
    llama-server)
      # variant = kvu | nokvu, optionally with -ub2048 (physical batch 2048) and -nocp (no context checkpoints, no host
      # prompt cache: the unique prompts of this workload never reuse them)
      local kv=-kvu
      case $variant in nokvu*) kv=-no-kvu ;; esac
      case $variant in *-ub2048*) kv="$kv -ub 2048" ;; esac
      case $variant in *-nocp*) kv="$kv --ctx-checkpoints 0 --cache-ram 0" ;; esac
      url="http://127.0.0.1:$port_ll"
      alias=$alias36
      start_server "$slog" "$baseline_bin" -m "$model" -ngl 99 -np 4 -c "$ctx" -fa on -ctk f16 -ctv f16 -t "$threads" \
        --host 127.0.0.1 --port "$port_ll" $kv
      wait_ready "$url/health" 900 "$slog" ;;
    tetraeder)
      url="http://127.0.0.1:$port_tet"
      alias=$alias36
      start_server "$slog" "$tet" serve -m "$model" --engine auto --ngl -1 --parallel 4 --ctx "$ctx" -t "$threads" \
        --flash-attn on --no-prefix-reuse --cache-ram 0 --port "$port_tet" --alias "$alias" --accept-any-model
      wait_ready "$url/v1/models" 900 "$slog" ;;
    tetraeder-default)
      # informational: no workaround flags; on qwen35moe a reused slot fails with HTTP 500 (the recurrent state cannot drop positions)
      url="http://127.0.0.1:$port_tet"
      alias=$alias36
      start_server "$slog" "$tet" serve -m "$model" --engine auto --ngl -1 --parallel 4 --ctx "$ctx" -t "$threads"         --flash-attn on --port "$port_tet" --alias "$alias" --accept-any-model
      wait_ready "$url/v1/models" 900 "$slog" ;;
    moe30)
      url="http://127.0.0.1:$port_tet"
      alias=$alias30
      start_server "$slog" "$tet" serve -m "$model30" --engine moe --ngl -1 --parallel 4 --ctx "$ctx" -t "$threads" \
        --port "$port_tet" --alias "$alias" --accept-any-model
      wait_ready "$url/v1/models" 900 "$slog" ;;
    strata)
      url="http://127.0.0.1:$port_st"
      # context per slot: the engine's slots each keep their own cache; 8192 covers 512 + 128 tokens and matches
      # llama-server's 32768 / 4 per slot. No tuning (the command line skips it unless --tune), no pictures, no key.
      # The feed pins the engine package to the v0.2.20 release folder.
      if [ -f "$out/strata-failed" ] || ! command -v strata > /dev/null 2>&1; then
        say "strata: not available (install or an earlier launch failed)"
        return 1
      fi
      # shellcheck disable=SC2086
      if ! strata launch --feed "$strata_feed" --model-path "$model" --context 8192 --parallel 4 --vision off --access local \
        --port "$port_st" --no-api-key --yes --no-tui ${STRATA_ARGS:-} > "$slog" 2>&1 8>&-; then
        tail -n 15 "$slog" | sed 's/^/  strata: /'
        echo "strata launch failed, see $slog" > "$out/strata-failed"
        return 1
      fi
      strata_up=1
      server_pid=""
      if ! wait_ready "$url/v1/models" 420 "$slog"; then
        echo "strata did not serve, see $slog" > "$out/strata-failed"
        return 1
      fi
      alias=$(curl -sf "$url/v1/models" | "$py" -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')
      strata_record ;;
  esac
}

# strata_record: what Strata ran with (flags, effective slots, config), once per session
strata_record() {
  [ -f "$out/strata-effective.txt" ] && return 0
  {
    echo "== strata launch --status"
    strata launch --status --output json 2>&1 8>&-
    echo "== strata status --json"
    strata status --url "$url" --json 2>&1 8>&-
    echo "== process command lines"
    ps -eo pid,args | awk '/[s]trata/ {print}'
    echo "== launch config files"
    find "$HOME/.local/share/strata" -path '*launch*' -name '*.json' -newer /root/strata_install.sh 2> /dev/null | while read -r f; do
      echo "-- $f"
      cat "$f"
    done
    echo "== GPU memory"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
  } > "$out/strata-effective.txt" 2>&1
}

bench_file() { # bench_file FILE LABEL URL ARGS...: the file is kept even with errors (the summary reads them)
  local f=$1 label=$2 u=$3
  shift 3
  timeout -k 30 15m "$py" scripts/gateway_bench.py "$u" --model "$alias" --temperature 0 --no-keepalive --label "$label" \
    --out "$f.tmp" "$@" >> "$out/bench.log" 2>&1
  [ -s "$f.tmp" ] && mv "$f.tmp" "$f"
  return 0
}

warm_up() {
  "$py" scripts/gateway_bench.py "$1" --model "$alias" --prompt-tokens 128 --max-tokens 16 --requests 1 --no-keepalive > /dev/null 2>&1
}

# one_run RUN ENGINE VARIANT SEEDN: fresh server, warm-up, c4, then c1; files bench-RUN-c4.json / -c1.json
one_run() {
  local run=$1 name=$2 variant=$3 seed=$4
  local c4="$out/bench-$run-c4.json" c1="$out/bench-$run-c1.json"
  if [ -f "$c1" ] && [ -f "$c4" ]; then say "run $run: done before"; return 0; fi
  if over_budget; then say "BUDGET spent: skip run $run"; return 0; fi
  say "run $run start ($name${variant:+ $variant})"
  : > "$out/gpu-$run.txt"
  if ! server_up "$name" "$variant" "$out/server-$run.log"; then
    say "run $run: server did not start"
    echo "server did not start" > "$out/failed-$run.txt"
    stop_server
    return 0
  fi
  gpu_note "$run" before
  warm_up "$url"
  bench_file "$c4" "$run-c4" "$url" --prompt-tokens "$ptok" --max-tokens "$mtok" --concurrency 4 --requests "$n_c4" --seed $((seed * 10 + 4))
  bench_file "$c1" "$run-c1" "$url" --prompt-tokens "$ptok" --max-tokens "$mtok" --concurrency 1 --requests "$n_c1" --seed $((seed * 10 + 1))
  gpu_note "$run" after
  nvidia-smi --query-gpu=memory.used --format=csv,noheader >> "$out/gpu-$run.txt" 2> /dev/null
  stop_server
  say "run $run done"
}

# ---- stage: build ------------------------------------------------------------------------------------------
st_build() {
  say "build start (cargo jobs $((threads / 2 < 4 ? 4 : threads / 2)))"
  (
    export CARGO_BUILD_JOBS=$((threads / 2 < 4 ? 4 : threads / 2))
    bash scripts/ensure_patches.sh 8>&- || exit 1
    timeout -k 30 50m cargo build --release -p tetraeder --features cuda 8>&- || exit 1
    # the test binary of the CPU probe (no GPU needed); built here so the probe stage only runs it
    timeout -k 30 30m cargo test --release -p tetraeder-model --test kv_gather --no-run 8>&- || exit 1
  ) > "$out/build.log" 2>&1 || die "build failed (see $out/build.log)"
  if [ ! -x "$py" ] || ! "$py" -c 'import httpx' 2> /dev/null; then
    if ! command -v uv > /dev/null 2>&1 && ! python3 -c 'import ensurepip' 2> /dev/null; then
      curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.local/bin sh > /dev/null 2>&1
    fi
    if command -v uv > /dev/null 2>&1; then
      uv venv --allow-existing "$venv" && uv pip install --python "$py" httpx
    else
      python3 -m venv "$venv" && "$py" -m pip install -q httpx
    fi || die "python env failed"
  fi
  mkdir -p "$out/probes"
  nvcc -O2 -arch="sm_${archs%%;*}" -o /root/mapped_read scripts/probes/mapped_read.cu > "$out/probes/mapped_read.build.log" 2>&1 \
    || say "WARNING: mapped_read build failed (see probes/mapped_read.build.log)"
  strata_install
  env_json
  touch "$out/stage-build.ok"
  say "build done"
}

strata_install() {
  command -v strata > /dev/null 2>&1 && { say "strata already installed"; return 0; }
  # the release folder of v0.2.20, not "latest": the installer checks the archive's SHA-256 against the folder's manifest
  curl -fsSL --connect-timeout 15 --max-time 120 "$strata_feed/install.sh" -o /root/strata_install.sh \
    && STRATA_FEED="$strata_feed" sh /root/strata_install.sh > "$out/strata-install.log" 2>&1 \
    || { say "WARNING: strata install failed (see strata-install.log)"; return 1; }
  strata --version >> "$out/strata-install.log" 2>&1
  # what a launch would download (must be the engine only, never the model) and which engine package it picks
  strata launch --plan --feed "$strata_feed" --model-path "$model" --context 8192 --parallel 4 --vision off --access local \
    --port "$port_st" --no-api-key ${STRATA_ARGS:-} > "$out/strata-plan.txt" 2>&1 8>&-
  cp "$HOME/.local/share/strata/terminal"/*/install-manifest.txt "$out/" 2> /dev/null
  say "strata installed"
}

env_json() {
  [ -f "$out/env.json" ] && return 0
  nvidia-smi -q -d CLOCK,PERFORMANCE,POWER > "$out/nvidia-smi-clock-performance.txt" 2>&1
  GW_OUT="$out" GW_THREADS="$threads" GW_QUOTA="$(bash scripts/cpu_quota.sh 2> /dev/null)" \
    GW_CPU="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)" GW_NPROC="$(nproc)" \
    GW_COMMIT="${COMMIT:-}" GW_HOST="$(hostname)" \
    GW_LLAMA="$($baseline_bin --version 2>&1 | head -3 | tr '\n' ' ')" GW_PIN="$(cat /opt/llama.cpp-baseline/PIN 2> /dev/null | tr '\n' ' ')" \
    GW_STRATA="$(strata --version 2>&1 | head -1)" GW_MEM="$(free -g | awk '/Mem:/ {print $2}')" \
    GW_GPU="$(nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,clocks_throttle_reasons.active,memory.total,power.limit,power.default_limit --format=csv,noheader 2> /dev/null)" \
    GW_VAST="${VAST_CONTAINERLABEL:-} ${CONTAINER_ID:-}" "$py" - << 'PYEOF'
import json, os
e = os.environ
k = ["gpu", "driver", "clock_sm_mhz", "clock_sm_max_mhz", "clock_mem_mhz", "throttle_reasons_active", "vram", "power_limit", "power_default_limit"]
g = dict(zip(k, ([x.strip() for x in e["GW_GPU"].split(",")] + [""] * 9)[:9]))
json.dump({"host": e["GW_HOST"], "vast_container": e["GW_VAST"].strip(), **g, "cpu_model": e["GW_CPU"],
           "host_cpus": e["GW_NPROC"], "cpu_quota_cores": e["GW_QUOTA"], "threads": e["GW_THREADS"], "ram_gb": e["GW_MEM"],
           "llama_server": e["GW_LLAMA"], "llama_pin": e["GW_PIN"], "strata": e["GW_STRATA"], "model": os.path.basename(os.environ.get("MODEL", ""))},
          open(os.path.join(e["GW_OUT"], "env.json"), "w"), indent=2)
PYEOF
}

# ---- stage: probes -----------------------------------------------------------------------------------------
st_probes() {
  mkdir -p "$out/probes"
  say "probe kv_gather (CPU, threads $threads)"
  TETRAEDER_KV_GATHER_OUT="$out/probes/kv-gather.json" TETRAEDER_KV_GATHER_LENS=2048,8192,32768 \
    timeout -k 30 25m cargo test --release -p tetraeder-model --test kv_gather -- --ignored --nocapture cpu_kv_gather_bench \
    > "$out/probes/kv-gather.log" 2>&1 8>&- || say "WARNING: kv_gather probe failed (see probes/kv-gather.log)"
  grep -E "ggml CPU tier" "$out/probes/kv-gather.log"
  if [ -x /root/mapped_read ]; then
    say "probe mapped_read (2 GB, 7 repeats)"
    /root/mapped_read 2 7 > "$out/probes/mapped-read-2gb.txt" 2>&1 8>&- || say "WARNING: mapped_read failed"
    /root/mapped_read 4 5 > "$out/probes/mapped-read-4gb.txt" 2>&1 8>&- || true
    tail -n 2 "$out/probes/mapped-read-2gb.txt"
  fi
  touch "$out/stage-probes.ok"
}

# ---- stage: prepass ----------------------------------------------------------------------------------------
st_prepass() {
  local v f
  for v in ${PREPASS_VARIANTS:-kvu nokvu kvu-ub2048 kvu-nocp kvu-ub2048-nocp}; do
    f="$out/prepass-$v.json"
    [ -f "$f" ] && continue
    say "prepass $v start"
    : > "$out/gpu-prepass-$v.txt"
    server_up llama-server "$v" "$out/server-prepass-$v.log" || { stop_server; die "llama-server did not start (prepass $v)"; }
    gpu_note "prepass-$v" before
    warm_up "$url"
    bench_file "$f" "prepass-$v" "$url" --prompt-tokens "$ptok" --max-tokens "$mtok" --concurrency 4 --requests "$n_c4" --seed 84
    gpu_note "prepass-$v" after
    stop_server
  done
  touch "$out/stage-prepass.ok"
}

pick_variant() { # the pre-pass variant with the highest c4 throughput
  "$py" - "$out" << 'PYEOF'
import json, sys
d = sys.argv[1]
t = {}
import glob, os
for f in glob.glob(f"{d}/prepass-*.json"):
    try:
        t[os.path.basename(f)[8:-5]] = json.load(open(f))["summary"]["throughput_tps"]
    except Exception:
        pass
print(max(t, key=t.get) if t else "kvu")
PYEOF
}

# ---- stage: rounds -----------------------------------------------------------------------------------------
st_rounds() {
  local variant rnd e order
  variant=$(pick_variant)
  echo "$variant" > "$out/baseline-variant.txt"
  say "baseline variant: $variant"
  # engines: a = llama-server, b = tetraeder, c = strata; the order rotates so every engine runs first, second and last
  for rnd in $(seq 1 "$rounds"); do
    case $((rnd % 3)) in
      1) order="a b c" ;;
      2) order="b c a" ;;
      0) order="c a b" ;;
    esac
    for e in $order; do
      case $e in
        a) one_run "a-r$rnd" llama-server "$variant" "$rnd" ;;
        b) one_run "b-r$rnd" tetraeder "" "$rnd" ;;
        c) one_run "c-r$rnd" strata "" "$rnd" ;;
      esac
    done
  done
  one_run "b0-default" tetraeder-default "" 1
  touch "$out/stage-rounds.ok"
}

# ---- stage: aa ---------------------------------------------------------------------------------------------
st_aa() {
  [ -f "$model30" ] || { say "aa: no $model30, skipped"; return 0; }
  local r
  for r in 1 2; do one_run "aa-r$r" moe30 "" 1; done # same seeds on both runs
  touch "$out/stage-aa.ok"
}

st_smoke() {
  local e
  for e in llama-server tetraeder strata; do
    say "smoke $e"
    if server_up "$e" kvu "$out/server-smoke-$e.log"; then
      warm_up "$url" && say "smoke $e: answered ($alias)"
      bench_file "$out/smoke-$e-c4.json" "smoke-$e-c4" "$url" --prompt-tokens "$ptok" --max-tokens "$mtok" --concurrency 4 \
        --requests 4 --seed 1
      "$py" - "$out/smoke-$e-c4.json" << 'PYEOF'
import json, sys
j = json.load(open(sys.argv[1]))
r = j.get("results", [])
print("  smoke summary:", j["summary"])
print("  smoke completion_tokens:", [x.get("completion_tokens") for x in r], "usage_from:", sorted({str(x.get("usage_from")) for x in r}),
      "head:", [x.get("text_head") for x in r][:2])
PYEOF
    fi
    stop_server
  done
  grep -h -iE "general.architecture" "$out"/server-smoke-*.log | head -3 | tee "$out/architecture.txt"
}

main() {
  say "strata_q36_box start (threads=$threads rounds=$rounds budget=${BUDGET_MIN:-130} min stages=$stages) out=$out"
  export MODEL="$model"
  local s
  for s in $stages; do
    if stage_done "$s"; then say "STAGE $s: done before"; continue; fi
    say "STAGE $s"
    "st_$s"
  done
  stop_sampler
  touch "$out/DONE"
  say "STRATA_BOX_DONE rc=0"
}

main >> "$log" 2>&1
