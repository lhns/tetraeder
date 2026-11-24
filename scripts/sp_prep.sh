#!/usr/bin/env bash
# Session stream-predict (tree node bench-235b-stream-predict), preparation while the 235B downloads. Correctness only,
# nothing here is timed. Stages (STAGES, default "env build parity"):
#   env     host facts of the box (gpu, cpu, cgroup memory and cpu limits, disk) -> env/
#   build   cargo release build of tetraeder (CUDA), tetraeder-plan, tetraeder-bench and the exit_regime test binary, and
#           llama-server with CUDA (scripts/gate_extra_box.sh llama) in parallel
#   parity  node serving-cuda-parity: device proof for --engine dense and moe, the parity suite on CUDA (a) as the suite
#           runs, twice, (b) with prefix reuse off on tetraeder and cache_prompt off on llama-server, (c) each of the four
#           failing cases alone on fresh servers, and llama-server against itself (scripts/parity_llama_stability.py)
# Env: RESULTS_NAME (bench/results/<name>/), STAGES, THREADS. Ends with SP_PREP_DONE in runs/vast/sp_prep.log.
exec 8> /root/tetraeder_sp_prep.lock
flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
root=$PWD
name="${RESULTS_NAME:?set RESULTS_NAME}"
out="$root/bench/results/$name"
mkdir -p "$out/env" "$out/parity" runs/vast
log=runs/vast/sp_prep.log
stages="${STAGES:-env build parity}"
say() { echo "[$(date +%T)] $*"; }
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:/root/.local/bin:$PATH"
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
[ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
quota=$(bash scripts/cpu_quota.sh 2> /dev/null || nproc)
threads="${THREADS:-$quota}"
tet="$root/target/release/tetraeder"
llama_bin="$root/target/llama-server/bin/llama-server"
py=/root/gwvenv/bin/python
server_pid=""

die() { say "FATAL: $*"; echo "SP_PREP_DONE rc=1"; exit 1; }

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
    echo "cpu quota (cores) $quota, nproc $(nproc)"
    echo "TETRAEDER_CUDA_ARCHS ${TETRAEDER_CUDA_ARCHS:-}"
    echo "commit $(cat runs/vast/.pushed_commit 2> /dev/null || echo unknown) tetraeder_git ${TETRAEDER_GIT_COMMIT:-}"
  } > "$f/box.txt"
  say "env done"
}

st_build() {
  bash scripts/ensure_patches.sh 8>&- || die "patch queue"
  # shellcheck source=scripts/box_libs.sh
  . scripts/box_libs.sh
  local jobs=$((quota / 2))
  [ "$jobs" -lt 4 ] && jobs=4
  [ "$jobs" -gt 24 ] && jobs=24
  say "build start (cargo jobs $jobs)"
  (STAGES=llama SESSION="$name" GATE_OUT="$out/llama-build" CARGO_BUILD_JOBS=$jobs bash scripts/gate_extra_box.sh 8>&-) &
  local pl=$!
  CARGO_BUILD_JOBS=$jobs timeout -k 30 50m cargo build --release -p tetraeder -p tetraeder-planner -p tetraeder-bench \
    --features tetraeder/cuda > "$out/build-main.log" 2>&1 8>&- || { tail -n 20 "$out/build-main.log"; die "main build failed"; }
  say "main build done"
  CARGO_BUILD_JOBS=$jobs timeout -k 30 50m cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda \
    --test exit_regime --no-run > "$out/build-exit-regime.log" 2>&1 8>&- || { tail -n 20 "$out/build-exit-regime.log"; die "exit_regime build failed"; }
  say "exit_regime build done"
  wait "$pl"
  [ -x "$llama_bin" ] || die "no llama-server ($out/llama-build)"
  say "build done"
}

stop_server() {
  [ -n "$server_pid" ] || return 0
  kill "$server_pid" 2> /dev/null
  for _ in $(seq 1 20); do kill -0 "$server_pid" 2> /dev/null || break; sleep 1; done
  kill -9 "$server_pid" 2> /dev/null
  wait "$server_pid" 2> /dev/null
  server_pid=""
  sleep 2
}

# device_check ENGINE MODEL: the server on the tiny fixture with --ngl -1; its log's device lines and the GPU memory
# before, with the server idle and after a request
device_check() {
  local e=$1 model=$2 port=18181 d="$out/parity" m0 m1 m2 i
  m0=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  setsid nohup env RUST_LOG=debug "$tet" serve -m "$model" --engine "$e" --ngl -1 --host 127.0.0.1 --port "$port" --ctx 4096 \
    --parallel 2 -t 2 --accept-any-model --chat-template-file tests/fixtures/chatml.jinja > "$d/device-$e.log" 2>&1 8>&- &
  server_pid=$!
  for i in $(seq 1 120); do curl -sf -o /dev/null "http://127.0.0.1:$port/health" && break; sleep 1; done
  m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  curl -s "http://127.0.0.1:$port/v1/completions" -H 'content-type: application/json' \
    -d '{"model":"tiny","prompt":"Once upon a time","max_tokens":16,"temperature":0}' > "$d/device-$e.response.json"
  m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$d/device-$e.compute-apps.csv" 2>&1
  echo "engine $e: gpu memory.used MiB before start $m0, server idle $m1, after a request $m2" | tee "$d/device-$e.txt"
  grep -a -i -E 'cuda|device|gpu|offload|vram|backend' "$d/device-$e.log" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-240 | head -n 30 >> "$d/device-$e.txt"
  stop_server
}

run_suite() { # run_suite TAG [ENV=VALUE...] [-- pytest args]: the parity suite on CUDA, log parity/<tag>.log
  local tag=$1
  shift
  local envs=() args=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done
  [ "${1:-}" = -- ] && shift
  args=("$@")
  [ ${#args[@]} -eq 0 ] && args=(tests/conformance/parity/test_parity.py)
  say "parity run $tag start"
  rm -rf "$out/parity/tmp-$tag"
  env "${envs[@]}" TETRAEDER_BIN="$tet" LLAMA_SERVER_BIN="$llama_bin" TETRAEDER_REQUIRE_PARITY=1 \
    timeout -k 30 40m "$py" -m pytest -q -rA -p no:cacheprovider --basetemp="$out/parity/tmp-$tag" "${args[@]}" \
    > "$out/parity/$tag.log" 2>&1 8>&-
  echo "rc=$?" >> "$out/parity/$tag.log"
  tail -n 3 "$out/parity/$tag.log"
}

st_parity() {
  [ -x "$tet" ] && [ -x "$llama_bin" ] || die "build first"
  if ! "$py" -c 'import httpx, pytest, yaml' 2> /dev/null; then
    command -v uv > /dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.local/bin sh > /dev/null 2>&1; }
    [ -x "$py" ] || uv venv --allow-existing /root/gwvenv
    uv pip install --python "$py" httpx pytest pyyaml > "$out/parity/pip.log" 2>&1 || die "python env"
  fi
  "$llama_bin" --version > "$out/parity/llama-server-version.txt" 2>&1
  "$llama_bin" --help 2>&1 | grep -i -E 'cache-prompt|cache-ram|cache-reuse' > "$out/parity/llama-server-cache-flags.txt"
  device_check dense "$root/tests/fixtures/tiny-qwen3-f16.gguf"
  device_check moe "$root/tests/fixtures/tiny-qwen3moe-f16.gguf"
  nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>&1 | sed 's/^/before: /' > "$out/parity/gpu-throttle.txt"
  # (a) as the suite runs them, twice
  run_suite a1
  run_suite a2
  # (b) prefix reuse and the prompt caches off on both sides (the whole suite, as (a))
  run_suite b PARITY_TETRAEDER_EXTRA="--no-prefix-reuse --cache-ram 0" PARITY_LLAMA_EXTRA="--no-cache-prompt --cache-ram 0"
  # (c) each failing case alone, fresh servers
  local id
  for id in repeat-penalty seeded-sampler-names logprobs logprobs-shape; do
    run_suite "c-$id" -- "tests/conformance/parity/test_parity.py::test_case[$id[auto]]"
  done
  # llama-server against itself: two fresh servers per case, and the same request twice on one server
  "$py" scripts/parity_llama_stability.py --llama "$llama_bin" --out "$out/parity/llama-stability.json" \
    repeat-penalty seeded-sampler-names logprobs logprobs-shape > "$out/parity/llama-stability.log" 2>&1 8>&-
  tail -n 12 "$out/parity/llama-stability.log"
  nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>&1 | sed 's/^/after: /' >> "$out/parity/gpu-throttle.txt"
  {
    echo "case results per run (PASSED or FAILED of the four cases; [auto] is libllama)"
    for f in "$out"/parity/a1.log "$out"/parity/a2.log "$out"/parity/b.log "$out"/parity/c-*.log; do
      echo "== $(basename "$f" .log): $(grep -E '^(=+ )?[0-9]+ (passed|failed)|passed|failed' "$f" | tail -n 1)"
      grep -a -E '^(PASSED|FAILED|XFAIL|XPASS) .*(repeat-penalty|seeded-sampler-names|logprobs|logprobs-shape)\[auto\]' "$f"
    done
  } > "$out/parity/summary.txt"
  cat "$out/parity/summary.txt"
}

main() {
  say "sp_prep start stages=$stages threads=$threads out=$out"
  local s
  for s in $stages; do
    say "STAGE $s"
    "st_$s"
  done
  say "SP_PREP_DONE rc=0"
}
trap stop_server EXIT
main >> "$log" 2>&1
