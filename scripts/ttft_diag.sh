#!/usr/bin/env bash
# TTFT diagnosis on Qwen3-8B (ADR 0065, TTFT addenda; no verdict): where does the time of a prefix hit go?
# Same servers and replay as the ttft stage of phase4_box.sh. Runs, one GPU, sequential:
#   tet r1 (RUST_LOG=info,tetraeder::step=debug: the admit lines), llama-server r1, llama-server r2, tet r2.
# Started detached: `taxameter run --id <id> scripts/ttft_diag.sh --env SESSION=ttft-20261118`; ends with
# TTFT_DIAG_DONE in runs/vast/ttft_diag.log. Results: bench/results/$SESSION/ (replay JSONs, server logs, gpu files).
# A rerun skips runs whose JSON exists. TETRAEDER_STEP_TRACE=1 in the env adds the per-step trace lines
# (target tetraeder::trace) to the tet server logs; TET_R2_ENV gives tet r2 extra env words.
exec 8> /root/tetraeder_ttft_diag.lock
flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
bash scripts/ensure_patches.sh 8>&- || exit 1
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
[ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
log=runs/vast/ttft_diag.log
mkdir -p runs/vast
session="${SESSION:-ttft-20261118}"
out="$PWD/bench/results/$session"
mkdir -p "$out"
threads=$(bash scripts/cpu_quota.sh)
f="$PWD/.hf_cache/Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
tet=target/release/tetraeder
baseline_bin=/opt/llama.cpp-baseline/bin/llama-server
py=/root/gwvenv/bin/python
req_log=bench/private/claude-code-requests.jsonl
rctx=40960 rmax=36000
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
  touch "$out/gpu-sampler.stop"
  say "TTFT_DIAG_DONE rc=1"
  exit 1
}

run() { # run MODE REP: MODE tet|llama
  local mode=$1 rep=$2 name="replay-q8b-$1-r$2" url health=/v1/models
  local slog="$out/server-$name.log" res="$out/$name.json"
  [ -f "$res" ] && { say "$name done before"; return 0; }
  say "$name start"
  : > "$slog"
  case $mode in
    tet)
      url=http://127.0.0.1:18080
      # TET_R2_ENV: extra VAR=value words for tet r2 only (a paired variant, e.g. TETRAEDER_SNAPSHOT_SPARES=0).
      local extra=""
      [ "$rep" = 2 ] && extra="${TET_R2_ENV:-}"
      # shellcheck disable=SC2086
      env $extra RUST_LOG=info,tetraeder::step=debug setsid nohup "$tet" serve -m "$f" --engine dense --ngl -1 --parallel 4 \
        --ctx "$rctx" -t "$threads" --port 18080 --alias qwen3-8b --accept-any-model --cache-ram 8192 > "$slog" 2>&1 8>&- &
      server_pid=$! ;;
    llama)
      url=http://127.0.0.1:18081
      health=/health
      setsid nohup "$baseline_bin" -m "$f" -ngl 99 -np 4 -kvu -c "$rctx" -fa on -ctk f16 -ctv f16 -t "$threads" \
        --host 127.0.0.1 --port 18081 --cache-ram 8192 > "$slog" 2>&1 8>&- &
      server_pid=$! ;;
  esac
  for _ in $(seq 1 450); do
    kill -0 "$server_pid" 2> /dev/null || { tail -n 15 "$slog"; die "server exited while loading ($name)"; }
    curl -sf -o /dev/null "$url$health" && break
    sleep 2
  done
  curl -sf -o /dev/null "$url$health" || die "server not ready ($name)"
  nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>&1 | sed 's/^/before: /' > "$out/gpu-$name.txt"
  timeout -k 30 60m "$py" scripts/gateway_replay.py "$url" "$req_log" --speed 0 --concurrency 1 \
    --max-prompt-tokens "$rmax" --gguf "$f" --expect-context-length "$rctx" --out "$res.tmp" >> "$out/replay.log" 2>&1
  local rc=$?
  [ "$rc" = 2 ] && die "replay: context_length is not $rctx"
  [ -s "$res.tmp" ] && mv "$res.tmp" "$res"
  nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>&1 | sed 's/^/after: /' >> "$out/gpu-$name.txt"
  stop_server
  say "$name done rc=$rc"
}

main() {
  say "ttft_diag start threads=$threads out=$out"
  [ "$(sha256sum "$req_log" | cut -d' ' -f1)" = f79b4a6d810f71632fb0633553b32d3c6c18132a137a4005e2129688f95f7b35 ] \
    || die "request log sha256 mismatch"
  OUT="$out" SAMPLER_PARENT=$$ setsid nohup bash scripts/gpu_sampler.sh 8>&- > /dev/null 2>&1 &
  say "build start"
  timeout -k 30 50m cargo build --release -p tetraeder --features cuda 8>&- || die "build failed"
  if [ ! -x "$py" ] || ! "$py" -c 'import httpx' 2> /dev/null; then
    if ! command -v uv > /dev/null 2>&1 && ! python3 -c 'import ensurepip' 2> /dev/null; then
      curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.local/bin sh > /dev/null 2>&1
      export PATH="/root/.local/bin:$PATH"
    fi
    if command -v uv > /dev/null 2>&1; then
      uv venv --allow-existing /root/gwvenv && uv pip install --python "$py" httpx
    else
      python3 -m venv /root/gwvenv && "$py" -m pip install -q httpx
    fi || die "python env failed"
  fi
  say "build done"
  run tet 1
  run llama 1
  run llama 2
  run tet 2
  touch "$out/gpu-sampler.stop"
  touch "$out/DONE"
  say "TTFT_DIAG_DONE rc=0"
}
main >> "$log" 2>&1
