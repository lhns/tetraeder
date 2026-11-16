#!/usr/bin/env bash
# TEMPORARY (track-c4-loop): c4 loop diagnosis on a CPU runner. The verdict workload (gateway_bench c4: 16 requests,
# 4 concurrent, 512-token prompts, MAX_TOKENS 128, greedy; c1 8 requests) against llama-server and tetraeder
# variants on one model, interleaved forward then reverse. Step and admission logs on (tetraeder::step=debug).
#   scripts/c4_diag.sh <out-dir> <variant>...
# A variant is NAME=ENGINE[:extra serve args, comma separated]; NAME llama-server runs the vendored llama-server.
# Env: MODEL, LLAMA_SERVER, TETRAEDER (a NAME may set TETRAEDER_<NAME with - as _> for another binary), PY, THREADS
# (4), CTX (8192), MAX_TOKENS (128), REPS (2), TEMPLATE (a chat template file for both servers, optional).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
out=${1:?usage: c4_diag.sh <out-dir> <variant>...}
shift
mkdir -p "$out"
model="${MODEL:?}"
llama_server="${LLAMA_SERVER:-target/llama-server-build/bin/llama-server}"
py="${PY:-python3}"
threads="${THREADS:-4}" ctx="${CTX:-8192}" max_tokens="${MAX_TOKENS:-128}" reps="${REPS:-2}"
port=18080
server_pid=""
say() { echo "[$(date +%T)] $*"; }
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
}
trap stop_server EXIT
wait_ready() {
  for _ in $(seq 1 150); do
    kill -0 "$server_pid" 2> /dev/null || { say "server exited while loading"; return 1; }
    curl -sf -o /dev/null "$1" && return 0
    sleep 2
  done
  return 1
}
bench() { # NAME TEST REP SEED, args
  local name=$1 test=$2 rep=$3 seed=$4
  shift 4
  "$py" scripts/gateway_bench.py "http://127.0.0.1:$port" --model qwen3 --temperature 0 --seed "$seed" \
    --prompt-tokens 512 --max-tokens "$max_tokens" --no-keepalive --label "$name-$test-r$rep" \
    --out "$out/$name-$test-r$rep.json" "$@" || say "$name $test r$rep: client reported errors"
}
pass() { # VARIANT REP
  local v=$1 rep=$2 name engine extra ready bin
  name=${v%%=*}
  engine=${v#*=}
  extra=""
  case $engine in *:*) extra=${engine#*:}; engine=${engine%%:*} ;; esac
  say "pass $name r$rep start"
  if [ "$name" = llama-server ]; then
    local tpl=()
    [ -n "${TEMPLATE:-}" ] && tpl=(--jinja --chat-template-file "$TEMPLATE")
    start_server "$out/server-$name-r$rep.log" "$llama_server" -m "$model" -np 4 -c "$ctx" -t "$threads" -ngl 0 \
      -fa on --cont-batching --port "$port" "${tpl[@]}"
    ready=/health
  else
    local var="TETRAEDER_${name//-/_}" args=() tpl=()
    bin="${!var:-${TETRAEDER:-target/release/tetraeder}}"
    [ -n "$extra" ] && IFS=, read -r -a args <<< "$extra"
    [ -n "${TEMPLATE:-}" ] && tpl=(--chat-template-file "$TEMPLATE")
    RUST_LOG=info,tetraeder::step=debug start_server "$out/server-$name-r$rep.log" "$bin" serve -m "$model" \
      --engine "$engine" --parallel 4 --ctx "$ctx" -t "$threads" --ngl 0 --port "$port" --alias qwen3 \
      --accept-any-model "${tpl[@]}" "${args[@]}"
    ready=/v1/models
  fi
  if wait_ready "http://127.0.0.1:$port$ready"; then
    "$py" scripts/gateway_bench.py "http://127.0.0.1:$port" --model qwen3 --prompt-tokens 128 --max-tokens 8 \
      --requests 1 > /dev/null 2>&1
    bench "$name" c1 "$rep" $((rep * 10 + 1)) --concurrency 1 --requests 8
    bench "$name" c4 "$rep" $((rep * 10 + 4)) --concurrency 4 --requests 16
  else
    tail -n 20 "$out/server-$name-r$rep.log"
  fi
  stop_server
  say "pass $name r$rep done"
}
grep -m1 'model name' /proc/cpuinfo > "$out/cpu.txt"
rev=()
for v in "$@"; do rev=("$v" "${rev[@]}"); done
for rep in $(seq 1 "$reps"); do
  if [ $((rep % 2)) = 1 ]; then order=("$@"); else order=("${rev[@]}"); fi
  for v in "${order[@]}"; do pass "$v" "$rep"; done
done
exit 0
