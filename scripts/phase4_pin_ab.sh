#!/usr/bin/env bash
# Pin A/B on a GPU box: is the c4 gap against llama-server caused by the newer llama.cpp in the baseline (b12426,
# /opt/llama.cpp-baseline) versus the engine's pin (b11976, vendor/llama.cpp with the patch queue)?
# Reuses the D2a workload and flags (scripts/phase4_box_d2a.sh: gateway_bench.py arguments, seeds, server flags,
# throttle notes); no pre-pass, the baseline variant is fixed from the D2a rerun (8B nokvu, 30B kvu).
# Started detached by `taxameter run scripts/phase4_pin_ab.sh`; log runs/vast/pin_ab.log ends with PIN_AB_DONE rc=<n>.
# Results: bench/results/<session>/pin-ab/ (bench-*.json, summary.json, summary.md, env.json, dense-regress/).
# Stages (STAGE lines in the log):
#   build        tetraeder release with CUDA (as D2a) and the python env
#   build-b11976 llama-server from the patched vendor/llama.cpp, the cmake flags of docker/box/Dockerfile's baseline
#   bench        per model, ROUNDS rounds, each (a) llama-server b12426, (b) llama-server b11976, (c) tetraeder; odd
#                rounds a,b,c, even rounds c,b,a. Each server start: warm-up, c1 (8 requests), c4 (16 requests)
#   shapes       scripts/gpu_dense_regress.sh: step_shapes (Qwen3-8B, all layers on GPU) at HEAD (b11976) against OLD
#                (track-pin-b12426), engines llama,dense, 3 rounds
#   summary      summary.json, summary.md
# Models run in order q8b, then q30b (only when at least 40 minutes of BUDGET_MIN remain after the shapes stage).
# A rerun resumes: a run whose output exists is skipped. Env: SESSION, MODELS ("q8b q30b"), ROUNDS (2), BUDGET_MIN (90),
# PIN_OLD (full sha of track-pin-b12426), SHAPE_ROUNDS (3), THREADS, CTX, BASELINE_BIN.
exec 8> /root/tetraeder_pin_ab.lock
flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
root=$PWD
bash scripts/ensure_patches.sh 8>&- || exit 1 # ggml patch queue (ADR 0052): vendor/llama.cpp is the patched b11976
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
log=runs/vast/pin_ab.log
session="${SESSION:-$(date +%Y%m%d)}"
out="$root/bench/results/$session/pin-ab"
mkdir -p "$out" "$(dirname "$log")"
t_start=$(date +%s)
threads="${THREADS:-$(bash scripts/cpu_quota.sh 8>&-)}"
budget_s=$((${BUDGET_MIN:-90} * 60))
models="${MODELS:-q8b q30b}"
rounds="${ROUNDS:-2}"
pin_old="${PIN_OLD:-d2e8696d5bfd40d84614bba42e81ae9d76e494bd}"
models_dir="$root/${TETRAEDER_MODELS_DIR:-.hf_cache}"
tet="$root/target/release/tetraeder"
bin_a="${BASELINE_BIN:-/opt/llama.cpp-baseline/bin/llama-server}"
bin_b=/root/pinab/bin/llama-server-b11976
venv=/root/gwvenv
py="${PY:-$venv/bin/python}"
ctx="${CTX:-32768}" ngl_ll=99
ptok=512 mtok=128 n_c1=8 n_c4=16
port_tet=18080
port_ll=18081
server_pid=""

declare -A MODEL_FILE=(
  [q30b]="$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
  [q8b]="$models_dir/Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
)
declare -A MODEL_ALIAS=([q30b]=qwen3-30b-a3b [q8b]=qwen3-8b)
# the baseline's KV mode that D2a's pre-pass picked (20261116d2a2); both llama-servers use it
declare -A VARIANT=([q30b]=kvu [q8b]=nokvu)
declare -A JUDGED=([q30b]=moe [q8b]=dense)

say() { echo "[$(date +%T)] $*"; }
die() {
  say "FATAL: $*"
  stop_server
  touch "$out/DONE"
  say "PIN_AB_DONE rc=1"
  exit 1
}

# Starts a server detached, pid in $server_pid; $1 = server log, rest = command. Killed by pid only.
start_server() {
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
      die "server exited while loading ($slog); a rejected flag? see the lines above"
    fi
    curl -sf -o /dev/null "$url" && return 0
    sleep 2
  done
  die "server not ready after ${secs}s ($slog)"
}

elapsed_min() { echo $((($(date +%s) - t_start) / 60)); }
remaining_min() { echo $((${BUDGET_MIN:-90} - $(elapsed_min))); }
over_hard() { [ $(($(date +%s) - t_start)) -gt 9000 ]; } # 150 min: the box lives 3 h

# gpu_note RUN before|after: throttle reasons bitmask (hex) and clocks, as D2a plus the clocks
gpu_note() {
  local f="$out/gpu-$1.txt" when=$2 v
  v=$(nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader 2> /dev/null) \
    || v=$(nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2> /dev/null)
  [ -n "$v" ] || say "WARNING: no throttle reasons from nvidia-smi for $1 $when"
  echo "$v" | sed "s/^ *\([^ ]\)/$when: \1/" >> "$f"
  echo "$when clocks: $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader 2> /dev/null)" >> "$f"
}

# bench_file FILE LABEL URL ALIAS ARGS...: one gateway_bench.py call, exactly as D2a
bench_file() {
  local f=$1 label=$2 url=$3 alias=$4
  shift 4
  timeout -k 30 15m "$py" scripts/gateway_bench.py "$url" --model "$alias" --temperature 0 --no-keepalive --label "$label" \
    --out "$f.tmp" "$@" >> "$out/bench.log" 2>&1
  [ -s "$f.tmp" ] && mv "$f.tmp" "$f"
  return 0
}
warm_up() { "$py" scripts/gateway_bench.py "$1" --model "$2" --prompt-tokens 128 --max-tokens 16 --requests 1 --no-keepalive > /dev/null 2>&1; }

# ---- builds ------------------------------------------------------------------------------------------------
build() {
  say "build start"
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
  say "build done"
}

# llama-server from the patched vendor tree; flags as the baseline in docker/box/Dockerfile (arch from the GPU)
build_b11976() {
  if [ -x "$bin_b" ]; then
    say "build-b11976: done before"
    return 0
  fi
  say "build-b11976 start (vendor/llama.cpp $(git -C vendor/llama.cpp rev-parse --short HEAD 2> /dev/null), archs ${TETRAEDER_CUDA_ARCHS:-86})"
  local b=/root/pinab/build-b11976
  mkdir -p /root/pinab/bin
  {
    cmake -S vendor/llama.cpp -B "$b" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="${TETRAEDER_CUDA_ARCHS:-86}" -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
      -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF &&
      timeout -k 30 50m cmake --build "$b" -j"${CARGO_BUILD_JOBS:-8}" --target llama-server
  } > "$out/build-b11976.log" 2>&1 8>&- || { tail -n 20 "$out/build-b11976.log"; die "llama-server b11976 build failed"; }
  install -D "$b/bin/llama-server" "$bin_b.tmp" && mv "$bin_b.tmp" "$bin_b"
  say "build-b11976 done: $("$bin_b" --version 2>&1 | head -2 | tr '\n' ' ')"
}

# ---- bench -------------------------------------------------------------------------------------------------
# server_for MODEL NAME: NAME = b12426 | b11976 | tetraeder; sets $url and $alias
server_for() {
  local m=$1 name=$2 slog=$3 f=${MODEL_FILE[$1]} bin kv=-kvu
  alias=${MODEL_ALIAS[$m]}
  case $name in
    b12426 | b11976)
      [ "$name" = b12426 ] && bin=$bin_a || bin=$bin_b
      [ "${VARIANT[$m]}" = nokvu ] && kv=-no-kvu
      url="http://127.0.0.1:$port_ll"
      start_server "$slog" "$bin" -m "$f" -ngl "$ngl_ll" -np 4 -c "$ctx" -fa on -ctk f16 -ctv f16 -t "$threads" \
        --host 127.0.0.1 --port "$port_ll" $kv
      wait_ready "$url/health" 900 "$slog"
      ;;
    tetraeder)
      url="http://127.0.0.1:$port_tet"
      start_server "$slog" "$tet" serve -m "$f" --engine "${JUDGED[$m]}" --ngl -1 --parallel 4 --ctx "$ctx" \
        -t "$threads" --port "$port_tet" --alias "$alias" --accept-any-model
      wait_ready "$url/v1/models" 900 "$slog"
      ;;
  esac
}

# one_run MODEL NAME REP: one server start: warm-up, c1, c4 (D2a's workload and seeds, the same on every engine)
one_run() {
  local m=$1 name=$2 rep=$3 run="$1-$2-r$3" url alias
  local c1="$out/bench-$m-$name-c1-r$rep.json" c4="$out/bench-$m-$name-c4-r$rep.json"
  [ -f "$c1" ] && [ -f "$c4" ] && { say "run $run: done before"; return 0; }
  over_hard && { say "HARD limit: skip run $run"; return 0; }
  say "run $run start (elapsed $(elapsed_min) min)"
  : > "$out/gpu-$run.txt"
  server_for "$m" "$name" "$out/server-$m-$name-r$rep.log"
  gpu_note "$run" before
  warm_up "$url" "$alias"
  bench_file "$c1" "$run-c1" "$url" "$alias" --prompt-tokens "$ptok" --max-tokens "$mtok" --concurrency 1 --requests "$n_c1" --seed $((rep * 10 + 1))
  bench_file "$c4" "$run-c4" "$url" "$alias" --prompt-tokens "$ptok" --max-tokens "$mtok" --concurrency 4 --requests "$n_c4" --seed $((rep * 10 + 4))
  gpu_note "$run" after
  stop_server
  say "run $run done: $("$py" -c "import json,sys;s=json.load(open(sys.argv[1]))['summary'];print('c4 tps %.1f errors %s' % (s['throughput_tps'], s['errors']))" "$c4" 2> /dev/null)"
}

bench_model() {
  local m=$1 rep n
  [ -s "${MODEL_FILE[$m]}" ] || { say "WARNING: no model ${MODEL_FILE[$m]}: $m skipped"; return 0; }
  for rep in $(seq 1 "$rounds"); do
    if [ $((rep % 2)) = 1 ]; then n="b12426 b11976 tetraeder"; else n="tetraeder b11976 b12426"; fi
    for name in $n; do one_run "$m" "$name" "$rep"; done
  done
}

# ---- env, shapes, summary ----------------------------------------------------------------------------------
env_json() {
  [ -f "$out/env.json" ] && return 0
  nvidia-smi -q -d CLOCK,PERFORMANCE > "$out/nvidia-smi-clock-performance.txt" 2>&1
  GW_OUT="$out" GW_THREADS="$threads" GW_QUOTA="$(bash scripts/cpu_quota.sh 2> /dev/null)" \
    GW_A="$("$bin_a" --version 2>&1 | head -3 | tr '\n' ' ')" GW_B="$("$bin_b" --version 2>&1 | head -3 | tr '\n' ' ')" \
    GW_PIN_A="$(cat /opt/llama.cpp-baseline/PIN 2> /dev/null)" GW_VENDOR="$(git -C vendor/llama.cpp rev-parse HEAD 2> /dev/null)" \
    GW_COMMIT="$(git rev-parse HEAD 2> /dev/null)" GW_HOST="$(hostname)" GW_VARIANT="q8b=${VARIANT[q8b]} q30b=${VARIANT[q30b]}" \
    GW_GPU="$(nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,clocks_throttle_reasons.active,memory.total --format=csv,noheader 2> /dev/null)" \
    GW_VAST="${VAST_CONTAINERLABEL:-} ${CONTAINER_ID:-}" "$py" - << 'EOF'
import json, os
e = os.environ
name, drv, sm, smmax, mem, throttle, vram = ([x.strip() for x in e["GW_GPU"].split(",")] + [""] * 7)[:7]
json.dump({
    "host": e["GW_HOST"], "vast_container": e["GW_VAST"].strip(), "gpu": name, "driver": drv, "vram": vram,
    "clock_sm_mhz": sm, "clock_sm_max_mhz": smmax, "clock_mem_mhz": mem, "throttle_reasons_active": throttle,
    "cpu_quota_cores": e["GW_QUOTA"], "threads": e["GW_THREADS"], "git_commit": e["GW_COMMIT"],
    "llama_server_b12426": e["GW_A"], "baseline_pin_file": e["GW_PIN_A"], "llama_server_b11976": e["GW_B"],
    "vendor_llama_cpp_commit": e["GW_VENDOR"], "baseline_variants": e["GW_VARIANT"],
}, open(os.path.join(e["GW_OUT"], "env.json"), "w"), indent=2)
EOF
}

shapes() {
  local d="$out/dense-regress" f
  if [ -s "$d/table.md" ]; then
    say "shapes: done before"
    return 0
  fi
  say "STAGE shapes start (elapsed $(elapsed_min) min)"
  rm -f "$d/DONE"
  SESSION="$session" OUT_DIR="$d" OLD="$pin_old" ROUNDS="${SHAPE_ROUNDS:-3}" ENG_OLD=llama,dense ENG_HEAD=llama,dense \
    ALLOW_PARTIAL=1 bash scripts/gpu_dense_regress.sh 8>&- || true
  tail -n 5 runs/vast/dense_regress.log
  [ -s "$d/table.md" ] || say "WARNING: shapes produced no table (runs/vast/dense_regress.log)"
  # fetch skips *.txt and *.md: log copies come back too
  for f in "$d"/table.md "$d"/shapes.txt "$d"/raw-*.txt; do [ -s "$f" ] && cp "$f" "$f.log"; done
  cp runs/vast/dense_regress.log "$d/dense_regress_run.log" 2> /dev/null
  say "STAGE shapes done"
}

summary() {
  "$py" - "$out" << 'EOF'
import json, os, re, glob, sys, statistics
out = sys.argv[1]
rows = {}
for f in sorted(glob.glob(out + '/bench-*-c[14]-r*.json')):
    m = re.match(r'bench-(\w+?)-(b12426|b11976|tetraeder)-c([14])-r(\d+)\.json', os.path.basename(f))
    if not m: continue
    model, eng, c, rep = m[1], m[2], int(m[3]), int(m[4])
    s = json.load(open(f))['summary']
    gpu = os.path.join(out, f'gpu-{model}-{eng}-r{rep}.txt')
    rows[(model, rep, eng, c)] = {
        'throughput_tps': s['throughput_tps'], 'ttft_median_s': s['ttft_median_s'], 'ttft_p90_s': s['ttft_p90_s'],
        'decode_tps_median': s['decode_tps_median'], 'errors': s['errors'],
        'throttle': open(gpu).read().split('\n') if os.path.exists(gpu) else None,
    }
json.dump({f'{k[0]}|r{k[1]}|{k[2]}|c{k[3]}': v for k, v in rows.items()}, open(out + '/summary.json', 'w'), indent=1)
w = []
def g(model, rep, eng, c, key):
    r = rows.get((model, rep, eng, c))
    return r[key] if r else None
def fmt(x, p=1): return '-' if x is None else f'{x:.{p}f}'
def ratio(a, b): return '-' if a is None or not b else f'{a / b:.3f}'
for model in sorted({k[0] for k in rows}):
    for c, key in ((4, 'throughput_tps'), (1, 'throughput_tps')):
        w += ['', f'{model} c{c}: throughput_tps (decode_tps median; ttft median/p90 s; errors)', '',
              'round | (a) b12426 | (b) b11976 | (c) tetraeder | (b)/(a) | (c)/(b) | (c)/(a)', '--- | --- | --- | --- | --- | --- | ---']
        for rep in sorted({k[1] for k in rows if k[0] == model}):
            cells, v = [], {}
            for eng in ('b12426', 'b11976', 'tetraeder'):
                r = rows.get((model, rep, eng, c))
                v[eng] = r['throughput_tps'] if r else None
                cells.append('-' if not r else f"{r['throughput_tps']:.1f} ({r['decode_tps_median']:.1f}; {r['ttft_median_s']:.3f}/{r['ttft_p90_s']:.3f}; e{r['errors']})")
            w.append(f'r{rep} | ' + ' | '.join(cells) + f" | {ratio(v['b11976'], v['b12426'])} | {ratio(v['tetraeder'], v['b11976'])} | {ratio(v['tetraeder'], v['b12426'])}")
        means = {}
        for eng in ('b12426', 'b11976', 'tetraeder'):
            xs = [r['throughput_tps'] for k, r in rows.items() if k[0] == model and k[2] == eng and k[3] == c]
            means[eng] = statistics.mean(xs) if xs else None
        w.append(f"mean | {fmt(means['b12426'])} | {fmt(means['b11976'])} | {fmt(means['tetraeder'])} | {ratio(means['b11976'], means['b12426'])} | {ratio(means['tetraeder'], means['b11976'])} | {ratio(means['tetraeder'], means['b12426'])}")
open(out + '/summary.md', 'w').write('\n'.join(w) + '\n')
print('\n'.join(w))
EOF
}

main() {
  say "phase4_pin_ab start (threads=$threads models=\"$models\" rounds=$rounds budget=${BUDGET_MIN:-90} min) out=$out"
  say "baseline: $("$bin_a" --version 2>&1 | head -2 | tr '\n' ' ') $(cat /opt/llama.cpp-baseline/PIN 2> /dev/null)"
  say "STAGE build"
  build
  say "STAGE build-b11976 (elapsed $(elapsed_min) min)"
  build_b11976
  env_json
  local m first=1
  for m in $models; do
    if [ "$first" = 0 ] && [ "$(remaining_min)" -lt 40 ]; then
      say "SKIP $m: $(remaining_min) min of the budget remain (needs 40)"
      continue
    fi
    say "STAGE bench $m (elapsed $(elapsed_min) min)"
    bench_model "$m"
    if [ "$first" = 1 ]; then
      first=0
      shapes # right after the first model: the key number; the next model needs the remaining budget
    fi
  done
  [ "$first" = 1 ] && shapes
  say "STAGE summary"
  summary
  touch "$out/DONE"
  say "PIN_AB_DONE rc=0 (elapsed $(elapsed_min) min)"
}

main >> "$log" 2>&1
