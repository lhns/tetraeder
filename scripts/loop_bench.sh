#!/usr/bin/env bash
# Scheduler loop overhead without a GPU (CPU only): llama-server against `tetraeder serve --engine llama|dense`
# on one model, gateway_bench.py c1 and c4 (ADR 0065), interleaved ABC then CBA. Used by loop-bench.yml.
#   scripts/loop_bench.sh <out-dir>
# Env: MODEL (.hf_cache/Qwen__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf), LLAMA_SERVER, TETRAEDER, PY (python3),
# THREADS (4), CTX (8192), PARALLEL (4), MAX_TOKENS (64).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
out=${1:?usage: loop_bench.sh <out-dir>}
mkdir -p "$out"
model="${MODEL:-.hf_cache/Qwen__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf}"
llama_server="${LLAMA_SERVER:-target/llama-server-build/bin/llama-server}"
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

bench() { # ENGINE TEST REP, gateway_bench.py arguments after
  local engine=$1 test=$2 rep=$3
  shift 3
  "$py" scripts/gateway_bench.py "http://127.0.0.1:$port" --model qwen3 --temperature 0 --seed 1 \
    --prompt-tokens 512 --max-tokens "$max_tokens" --label "$engine-$test-r$rep" \
    --out "$out/$engine-$test-r$rep.json" "$@" || say "$engine $test r$rep: client reported errors"
}

pass() { # ENGINE REP
  local engine=$1 rep=$2 ready
  say "pass $engine r$rep start"
  case $engine in
    llama-server)
      start_server "$out/server-$engine-r$rep.log" "$llama_server" -m "$model" -np "$par" -c "$ctx" -t "$threads" -ngl 0 \
        --cont-batching --port "$port"
      ready=/health ;;
    tetraeder-llama | tetraeder-dense)
      start_server "$out/server-$engine-r$rep.log" "$tet" serve -m "$model" --engine "${engine#tetraeder-}" \
        --parallel "$par" --ctx "$ctx" -t "$threads" --ngl 0 --port "$port" --alias qwen3 --accept-any-model
      ready=/v1/models ;;
  esac
  if wait_ready "http://127.0.0.1:$port$ready" 300; then
    "$py" scripts/gateway_bench.py "http://127.0.0.1:$port" --model qwen3 --prompt-tokens 128 --max-tokens 8 \
      --requests 1 > /dev/null 2>&1 # warm up
    bench "$engine" c1 "$rep" --concurrency 1 --requests 8
    bench "$engine" c4 "$rep" --concurrency 4 --requests 16
  else
    tail -n 20 "$out/server-$engine-r$rep.log"
  fi
  stop_server
  say "pass $engine r$rep done"
}

rep=1
for order in "llama-server tetraeder-llama tetraeder-dense" "tetraeder-dense tetraeder-llama llama-server"; do
  for e in $order; do pass "$e" "$rep"; done
  rep=$((rep + 1))
done
exit 0
