#!/usr/bin/env bash
# Scheduler loop overhead without a GPU (CPU only): llama-server against `tetraeder serve --engine llama|dense`
# on one model, gateway_bench.py c1 and c4 (ADR 0065), interleaved ABC then CBA. Then the same for
# `--engine moe` (and llama) on a tiny synthetic MoE (MOE_MODEL, the committed fixture: its steps are so short that
# the loop is what is measured), results in <out-dir>/moe/. Used by loop-bench.yml.
#   scripts/loop_bench.sh <out-dir>
# Env: MODEL (.hf_cache/Qwen__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf), MOE_MODEL (tests/fixtures/tiny-qwen3moe-f16.gguf;
# empty or missing: no MoE pass), LLAMA_SERVER, TETRAEDER, PY (python3), THREADS (4), CTX (8192), PARALLEL (4),
# MAX_TOKENS (64). llama-server runs flash attention (`-fa on`) like the verdict baseline; tetraeder always does.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
out=${1:?usage: loop_bench.sh <out-dir>}
mkdir -p "$out"
base_out=$out
model="${MODEL:-.hf_cache/Qwen__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf}"
llama_server="${LLAMA_SERVER:-target/llama-server-build/bin/llama-server}"
moe_model="${MOE_MODEL-tests/fixtures/tiny-qwen3moe-f16.gguf}"
tet="${TETRAEDER:-target/release/tetraeder}"
py="${PY:-python3}"
threads="${THREADS:-4}" ctx="${CTX:-8192}" par="${PARALLEL:-4}" max_tokens="${MAX_TOKENS:-64}"
port=18080
server_pid=""

say() { echo "[$(date +%T)] $*"; }

start_server() { # $1 = log, rest = command; killed by pid only
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
}
trap stop_server EXIT

wait_ready() { # URL SECONDS
  local url=$1 secs=$2
  for _ in $(seq 1 $((secs / 2))); do
    kill -0 "$server_pid" 2> /dev/null || { say "server exited while loading"; return 1; }
    curl -sf -o /dev/null "$url" && return 0
    sleep 2
  done
  say "server not ready after ${secs}s"
  return 1
}

# Each test of a pass gets its own prompts (seed): repeated prompts would measure the servers' prompt caches,
# not the loop. The same test and rep get the same prompts on every engine.
bench() { # ENGINE TEST REP SEED, gateway_bench.py arguments after
  local engine=$1 test=$2 rep=$3 seed=$4
  shift 4
  "$py" scripts/gateway_bench.py "http://127.0.0.1:$port" --model qwen3 --temperature 0 --seed "$seed" \
    --prompt-tokens 512 --max-tokens "$max_tokens" --no-keepalive --label "$engine-$test-r$rep" \
    --out "$out/$engine-$test-r$rep.json" "$@" || say "$engine $test r$rep: client reported errors"
}

pass() { # ENGINE REP
  local engine=$1 rep=$2 ready
  say "pass $engine r$rep start"
  case $engine in
    llama-server)
      start_server "$out/server-$engine-r$rep.log" "$llama_server" -m "$model" -np "$par" -c "$ctx" -t "$threads" -ngl 0 \
        -fa on --cont-batching --port "$port"
      ready=/health ;;
    tetraeder-llama | tetraeder-dense | tetraeder-moe)
      # dense and moe run with the default --ngl (-1: the GPU if there is one, else the CPU)
      local ngl=()
      [ "$engine" = tetraeder-llama ] && ngl=(--ngl 0)
      RUST_LOG=info,tetraeder::step=debug start_server "$out/server-$engine-r$rep.log" "$tet" serve -m "$model" \
        --engine "${engine#tetraeder-}" --parallel "$par" --ctx "$ctx" -t "$threads" "${ngl[@]}" --port "$port" \
        --alias qwen3 --accept-any-model
      ready=/v1/models ;;
  esac
  if wait_ready "http://127.0.0.1:$port$ready" 300; then
    # the ggml modules each server mapped: both must run the same CPU variant
    grep -oE '[^ /]*(ggml|llama)[^ /]*\.so[^ ]*' "/proc/$server_pid/maps" | sort -u > "$out/modules-$engine-r$rep.txt"
    "$py" scripts/gateway_bench.py "http://127.0.0.1:$port" --model qwen3 --prompt-tokens 128 --max-tokens 8 \
      --requests 1 > /dev/null 2>&1 # warm up
    bench "$engine" c1 "$rep" $((rep * 10 + 1)) --concurrency 1 --requests 8
    bench "$engine" c4 "$rep" $((rep * 10 + 4)) --concurrency 4 --requests 16
  else
    tail -n 20 "$out/server-$engine-r$rep.log"
  fi
  stop_server
  say "pass $engine r$rep done"
}

# The engines of one model in both orders, ABC then CBA (the first is the baseline)
run_model() { # ENGINE...
  local rep=1 rev=() e order
  for e in "$@"; do rev=("$e" "${rev[@]}"); done
  for order in "$*" "${rev[*]}"; do
    for e in $order; do pass "$e" "$rep"; done
    rep=$((rep + 1))
  done
}

grep -m1 'model name' /proc/cpuinfo > "$out/cpu.txt"
run_model llama-server tetraeder-llama tetraeder-dense
if [ -n "$moe_model" ] && [ -f "$moe_model" ]; then
  model=$moe_model out=$base_out/moe
  mkdir -p "$out"
  run_model llama-server tetraeder-llama tetraeder-moe
fi
exit 0
