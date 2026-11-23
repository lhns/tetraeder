#!/usr/bin/env bash
# Extra stages of the regression-gate box session (node bench-regression-gate), after or beside scripts/ab_moe_box.sh:
#   llama  build llama-server with CUDA from the patched vendor tree (target/llama-server/bin; the bench's baseline
#          /opt/llama.cpp-baseline is a different commit and unpatched). Can run beside the A/B builds, never beside runs.
#   conf   the parity cases (tests/conformance/parity/test_parity.py; test_parity_lib.py reads docs/, which is not pushed) on CUDA: llama-server (above) against target/release/tetraeder
#          (CUDA build of the checkout), tiny fixtures, TETRAEDER_REQUIRE_PARITY=1; parity.log (-rA), server logs and the
#          device lines (parity-devices.txt) in $out.
#   q36    Qwen3.6-35B-A3B UD-Q4_K_XL, `serve --engine auto` with DEFAULT flags (prefix reuse and RAM cache on), c4 and c1
#          of the strata-q36 workload, ROUNDS_Q36 fresh servers; expects zero errors (the client counts error frames and
#          empty streams) and 128 tokens per request.
# Env: SESSION, STAGES ("llama conf q36"), THREADS, ROUNDS_Q36 (2), MODEL_Q36, DRY_RUN=1.
# Started detached by `taxameter run scripts/gate_extra_box.sh --env STAGES=...`; ends with GATE_EXTRA_DONE in
# runs/vast/gate_extra_box.log. Reruns resume (a stage with its .ok file is skipped).
DRY="${DRY_RUN:-0}"
if [ "$DRY" != 1 ]; then
  exec 8> /root/tetraeder_gate_extra_box.lock
  flock -n 8 || exit 0
fi
cd "$(dirname "$0")/.." || exit 1
root="$PWD"
if [ "$DRY" != 1 ]; then
  if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
  export PATH="/opt/cargo/bin:$HOME/.cargo/bin:/root/.local/bin:$PATH"
  # shellcheck disable=SC1091
  . "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
log="${GATE_LOG:-runs/vast/gate_extra_box.log}"
session="${SESSION:-regression-gate-$(date +%Y%m%d)}"
out="${GATE_OUT:-$root/bench/results/$session/extra}"
mkdir -p "$out" "$(dirname "$log")"
stages="${STAGES:-llama conf q36}"
rounds="${ROUNDS_Q36:-2}"
if [ "$DRY" = 1 ]; then threads="${THREADS:-8}"; else threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"; fi
model36="${MODEL_Q36:-$root/.hf_cache/unsloth__Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
tet="${TET_BIN:-$root/target/release/tetraeder}"
llama_bin="$root/target/llama-server/bin/llama-server"
py=/root/gwvenv/bin/python
[ "$DRY" = 1 ] && py=python3
port=18080
server_pid=""
say() { echo "[$(date +%T)] $*"; }

stop_server() {
  [ -n "$server_pid" ] || return 0
  kill "$server_pid" 2> /dev/null
  for _ in $(seq 1 30); do kill -0 "$server_pid" 2> /dev/null || break; sleep 1; done
  kill -9 "$server_pid" 2> /dev/null
  wait "$server_pid" 2> /dev/null
  server_pid=""
  sleep 3
}
trap stop_server EXIT
die() {
  say "FATAL: $*"
  stop_server
  touch "$out/DONE"
  say "GATE_EXTRA_DONE rc=1"
  exit 1
}

st_llama() {
  [ -f "$out/stage-llama.ok" ] && return 0
  say "llama-server CUDA build (vendor/llama.cpp $(git -C vendor/llama.cpp rev-parse --short HEAD 2> /dev/null), archs ${TETRAEDER_CUDA_ARCHS:-86})"
  if [ "$DRY" = 1 ]; then say "PLAN cmake -S vendor/llama.cpp -B target/llama-server -DGGML_CUDA=ON; build llama-server"; return 0; fi
  bash scripts/ensure_patches.sh 8>&- || die "patch queue"
  {
    cmake -S vendor/llama.cpp -B target/llama-server -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="${TETRAEDER_CUDA_ARCHS:-86}" -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
      -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF &&
      timeout -k 30 50m cmake --build target/llama-server -j"${CARGO_BUILD_JOBS:-8}" --target llama-server
  } > "$out/build-llama-server.log" 2>&1 8>&- || { tail -n 20 "$out/build-llama-server.log"; die "llama-server build failed"; }
  "$llama_bin" --version > "$out/llama-server-version.txt" 2>&1
  touch "$out/stage-llama.ok"
  say "llama-server build done"
}

st_conf() {
  [ -f "$out/stage-conf.ok" ] && return 0
  say "parity suite on CUDA"
  [ "$DRY" = 1 ] && { say "PLAN TETRAEDER_BIN=$tet LLAMA_SERVER_BIN=$llama_bin python -m pytest -rA tests/conformance/parity/test_parity.py"; return 0; }
  [ -x "$llama_bin" ] || die "no $llama_bin (stage llama first)"
  [ -x "$tet" ] || die "no $tet (the A/B stage builds it)"
  # pyproject.toml is not pushed (taxameter.toml [push]), so no `uv run`: the bench venv plus the suite's three imports
  if ! "$py" -c 'import httpx, pytest, yaml' 2> /dev/null; then
    command -v uv > /dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.local/bin sh > /dev/null 2>&1; }
    [ -x "$py" ] || uv venv --allow-existing /root/gwvenv
    uv pip install --python "$py" httpx pytest pyyaml > "$out/pip-parity.log" 2>&1 || die "parity python env failed"
  fi
  nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>&1 | sed 's/^/before: /' > "$out/gpu-conf.txt"
  rm -rf "$out/parity-tmp"
  TETRAEDER_BIN="$tet" LLAMA_SERVER_BIN="$llama_bin" TETRAEDER_REQUIRE_PARITY=1     timeout -k 30 60m "$py" -m pytest -q -rA -p no:cacheprovider --basetemp="$out/parity-tmp" tests/conformance/parity/test_parity.py > "$out/parity.log" 2>&1 8>&-
  echo "rc=$?" >> "$out/parity.log"
  nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>&1 | sed 's/^/after: /' >> "$out/gpu-conf.txt"
  # proof the servers ran on CUDA: the device and offload lines of both servers' logs
  grep -h -i -E 'CUDA[0-9]|offloaded|using device|ggml_cuda_init|found [0-9]+ CUDA' "$out"/parity-tmp/*/*.log 2> /dev/null     | sed 's/^[0-9: .TZ-]*//' | sort | uniq -c | sort -rn | head -40 > "$out/parity-devices.txt"
  tail -n 5 "$out/parity.log"
  touch "$out/stage-conf.ok"
}

# q36_run N: fresh server with default flags, warm-up, c4 then c1
q36_run() {
  local n=$1 run="q36-default-r$1" url="http://127.0.0.1:$port"
  [ -f "$out/bench-$run-c1.json" ] && return 0
  say "run $run start"
  if [ "$DRY" = 1 ]; then say "PLAN serve -m $model36 --engine auto --ngl -1 --parallel 4 --ctx 32768 -t $threads --flash-attn on (default prefix reuse and cache)"; return 0; fi
  : > "$out/server-$run.log"
  setsid nohup "$tet" serve -m "$model36" --engine auto --ngl -1 --parallel 4 --ctx 32768 -t "$threads" --flash-attn on \
    --port "$port" --alias qwen36 --accept-any-model > "$out/server-$run.log" 2>&1 8>&- &
  server_pid=$!
  for _ in $(seq 1 450); do
    kill -0 "$server_pid" 2> /dev/null || { tail -n 15 "$out/server-$run.log"; die "server exited while loading ($run)"; }
    curl -sf -o /dev/null "$url/v1/models" && break
    sleep 2
  done
  curl -sf -o /dev/null "$url/v1/models" || die "server not ready ($run)"
  nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>&1 | sed 's/^/before: /' > "$out/gpu-$run.txt"
  "$py" scripts/gateway_bench.py "$url" --model qwen36 --prompt-tokens 128 --max-tokens 16 --requests 1 --no-keepalive > /dev/null 2>&1
  local t conc reqs seed
  for t in c4 c1; do
    conc=4 reqs=16 seed=$((n * 10 + 4))
    [ "$t" = c1 ] && { conc=1 reqs=8 seed=$((n * 10 + 1)); }
    timeout -k 30 15m "$py" scripts/gateway_bench.py "$url" --model qwen36 --temperature 0 --no-keepalive --label "$run-$t" \
      --prompt-tokens 512 --max-tokens 128 --concurrency "$conc" --requests "$reqs" --seed "$seed" \
      --out "$out/bench-$run-$t.json.tmp" >> "$out/bench.log" 2>&1
    [ -s "$out/bench-$run-$t.json.tmp" ] && mv "$out/bench-$run-$t.json.tmp" "$out/bench-$run-$t.json"
  done
  nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>&1 | sed 's/^/after: /' >> "$out/gpu-$run.txt"
  stop_server
  say "run $run done"
}

st_q36() {
  [ -f "$out/stage-q36.ok" ] && return 0
  [ "$DRY" = 1 ] || [ -f "$model36" ] || { say "q36: no $model36, skipped"; return 0; }
  local n
  for n in $(seq 1 "$rounds"); do q36_run "$n"; done
  [ "$DRY" = 1 ] && return 0
  "$py" scripts/gate_extra_summary.py "$out" "$rounds" > "$out/q36-summary.json"
  cat "$out/q36-summary.json"
  touch "$out/stage-q36.ok"
}

main() {
  say "gate_extra_box start (stages=$stages threads=$threads) out=$out"
  local s
  for s in $stages; do
    say "STAGE $s"
    "st_$s"
  done
  touch "$out/DONE"
  say "GATE_EXTRA_DONE rc=0"
}

if [ "$DRY" = 1 ]; then main; else main >> "$log" 2>&1; fi
