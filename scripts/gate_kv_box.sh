#!/usr/bin/env bash
# Report-only extras of the regression-gate session (node bench-regression-gate, job 4), on the box after gate_box.sh:
#   kv    Qwen3-30B-A3B (--engine moe): probe the context where `--kv auto` stops picking f16 (it logs its pick; `--kv f16`
#         is tried at the same context), run c4 and c1 with `--kv auto` there, then a paired f16 against q8_0 at ctx 32768
#         (3 interleaved rounds, the gate workload): the throughput cost of the quantized KV.
#   disk  Qwen3-8B request-log replay (the first REPLAY_LIMIT requests, ctx 40960, --cache-ram 2048 so entries are
#         evicted): RAM only (recompute) against RAM plus the disk tier (--cache-disk, ADR 0087); TTFT and total times.
# Env: SESSION, STAGES ("kv disk"), THREADS, KV_PROBE_CTXS, REPLAY_LIMIT (50), DRY_RUN=1. Ends with GATE_KV_DONE in
# runs/vast/gate_kv_box.log. Results: bench/results/$SESSION/kv/.
DRY="${DRY_RUN:-0}"
if [ "$DRY" != 1 ]; then
  exec 8> /root/tetraeder_gate_kv_box.lock
  flock -n 8 || exit 0
fi
cd "$(dirname "$0")/.." || exit 1
root="$PWD"
if [ "$DRY" != 1 ]; then
  export PATH="/root/.local/bin:$PATH"
fi
log="${GATE_LOG:-runs/vast/gate_kv_box.log}"
session="${SESSION:-regression-gate-$(date +%Y%m%d)}"
out="${GATE_OUT:-$root/bench/results/$session/kv}"
mkdir -p "$out" "$(dirname "$log")"
stages="${STAGES:-kv disk}"
if [ "$DRY" = 1 ]; then threads="${THREADS:-8}"; else threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"; fi
tet="${TET_BIN:-$root/target/release/tetraeder}"
m30="${MODEL_Q30B:-$root/.hf_cache/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf}"
m8="${MODEL_Q8B:-$root/.hf_cache/Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf}"
req_log=bench/private/claude-code-requests.jsonl
probe_ctxs="${KV_PROBE_CTXS:-65536 98304 131072 163840 196608 229376 262144}"
limit="${REPLAY_LIMIT:-50}"
# CACHE_RAM_MIB: the RAM budget of the disk pass; DISK_TAG names a rerun's files (the first pass, 2048 MiB, never evicted)
cache_ram="${CACHE_RAM_MIB:-2048}" tag="${DISK_TAG:-}"
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
  say "GATE_KV_DONE rc=1"
  exit 1
}

# start LOG CMD...: server in the background; sets server_pid
start() {
  local slog=$1
  shift
  if [ "$DRY" = 1 ]; then say "PLAN $*"; return 0; fi
  : > "$slog"
  setsid nohup "$@" > "$slog" 2>&1 8>&- &
  server_pid=$!
}

# ready LOG SECONDS: 0 when /v1/models answers, 1 when the server exited or the time ran out
ready() {
  [ "$DRY" = 1 ] && return 0
  for _ in $(seq 1 $(($2 / 2))); do
    kill -0 "$server_pid" 2> /dev/null || return 1
    curl -sf -o /dev/null "http://127.0.0.1:$port/v1/models" && return 0
    sleep 2
  done
  return 1
}

bench() { # bench FILE LABEL ARGS...
  local f=$1 label=$2
  shift 2
  [ "$DRY" = 1 ] && { say "PLAN gateway_bench.py $label $*"; return 0; }
  timeout -k 30 15m "$py" scripts/gateway_bench.py "http://127.0.0.1:$port" --model qwen3-30b-a3b --temperature 0 --no-keepalive \
    --label "$label" --out "$f.tmp" "$@" >> "$out/bench.log" 2>&1
  [ -s "$f.tmp" ] && mv "$f.tmp" "$f"
  return 0
}

# kv_run LABEL CTX KV: server with `--kv KV`, warm-up, c4 then c1 (seeds 14 and 11); 1 when the server did not start
kv_run() {
  local label=$1 ctx=$2 kv=$3 n=${4:-1}
  [ -f "$out/kv-$label-c1.json" ] && return 0
  say "kv run $label (ctx $ctx, --kv $kv) start"
  start "$out/server-kv-$label.log" "$tet" serve -m "$m30" --engine moe --ngl -1 --parallel 4 --ctx "$ctx" -t "$threads" \
    --port "$port" --alias qwen3-30b-a3b --accept-any-model --kv "$kv"
  if ! ready "$out/server-kv-$label.log" 600; then
    say "kv run $label: server did not start; $(grep -i -E 'kv auto|error|fit|free' "$out/server-kv-$label.log" | tail -n 3 | cut -c1-300)"
    stop_server
    return 1
  fi
  [ "$DRY" = 1 ] || "$py" scripts/gateway_bench.py "http://127.0.0.1:$port" --model qwen3-30b-a3b --prompt-tokens 128 --max-tokens 16 \
    --requests 1 --no-keepalive > /dev/null 2>&1
  nvidia-smi --query-gpu=clocks_event_reasons.active,memory.used --format=csv,noheader 2> /dev/null | sed 's/^/before: /' > "$out/gpu-kv-$label.txt"
  bench "$out/kv-$label-c4.json" "$label-c4" --prompt-tokens 512 --max-tokens 128 --concurrency 4 --requests 16 --seed $((n * 10 + 4))
  bench "$out/kv-$label-c1.json" "$label-c1" --prompt-tokens 512 --max-tokens 128 --concurrency 1 --requests 8 --seed $((n * 10 + 1))
  nvidia-smi --query-gpu=clocks_event_reasons.active,memory.used --format=csv,noheader 2> /dev/null | sed 's/^/after: /' >> "$out/gpu-kv-$label.txt"
  stop_server
  say "kv run $label done"
}

st_kv() {
  [ -f "$out/stage-kv.ok" ] && return 0
  [ "$DRY" = 1 ] || [ -f "$m30" ] || { say "kv: no $m30"; return 0; }
  # 1. the context where auto leaves f16: its pick line is in the server's stderr ("f16/f16 needs ...; q8_0/q8_0 fits")
  local ctx hit=""
  for ctx in $probe_ctxs; do
    say "probe --kv auto --ctx $ctx"
    start "$out/server-probe-$ctx.log" "$tet" serve -m "$m30" --engine moe --ngl -1 --parallel 4 --ctx "$ctx" -t "$threads" \
      --port "$port" --alias qwen3-30b-a3b --accept-any-model --kv auto
    ready "$out/server-probe-$ctx.log" 600
    local ok=$?
    stop_server
    grep -h -i -E 'kv auto|needs .* free' "$out/server-probe-$ctx.log" 2> /dev/null | head -n 3 | cut -c1-400 | sed "s/^/ctx $ctx: /" >> "$out/kv-probe.txt"
    [ "$DRY" = 1 ] && break
    if grep -q -E 'free; (q8_0|q4)' "$out/server-probe-$ctx.log" 2> /dev/null; then
      hit=$ctx
      say "auto leaves f16 at ctx $ctx (server start rc $ok)"
      break
    fi
  done
  if [ -n "$hit" ]; then
    kv_run "auto-ctx$hit" "$hit" auto || true
    kv_run "f16-ctx$hit" "$hit" f16 || true # expected to be refused: the f16 KV does not fit
  else
    say "kv: no probe context made auto leave f16 (see kv-probe.txt)"
  fi
  # 2. paired f16 against q8_0 at the gate context, three rounds interleaved
  local r
  for r in 1 2 3; do
    if [ $((r % 2)) = 1 ]; then
      kv_run "f16-r$r" 32768 f16 "$r" || die "f16 at 32768 did not start"
      kv_run "q8_0-r$r" 32768 q8_0 "$r" || die "q8_0 at 32768 did not start"
    else
      kv_run "q8_0-r$r" 32768 q8_0 "$r" || die "q8_0 at 32768 did not start"
      kv_run "f16-r$r" 32768 f16 "$r" || die "f16 at 32768 did not start"
    fi
  done
  touch "$out/stage-kv.ok"
}

# disk_run MODE: the replay of the first $limit requests, RAM cache of 2048 MiB, with or without the disk tier
disk_run() {
  local mode=$1 res="$out/disk$tag-$1.json" url="http://127.0.0.1:$port" extra=()
  [ -f "$res" ] && return 0
  [ "$mode" = disk ] && { rm -rf /root/kvdisk; extra=(--cache-disk /root/kvdisk --cache-disk-mib 30720); }
  say "disk run $mode start"
  start "$out/server-disk$tag-$mode.log" env RUST_LOG=info,tetraeder::step=debug "$tet" serve -m "$m8" --engine dense --ngl -1 --parallel 4 \
    --ctx 40960 -t "$threads" --port "$port" --alias qwen3-8b --accept-any-model --cache-ram "$cache_ram" "${extra[@]}"
  ready "$out/server-disk$tag-$mode.log" 600 || { say "disk run $mode: server did not start"; stop_server; return 1; }
  nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2> /dev/null | sed 's/^/before: /' > "$out/gpu-disk$tag-$mode.txt"
  if [ "$DRY" = 1 ]; then
    say "PLAN gateway_replay.py $url $req_log --speed 0 --concurrency 1 --max-prompt-tokens 36000 --limit $limit -> disk-$mode.json"
  else
    timeout -k 30 40m "$py" scripts/gateway_replay.py "$url" "$req_log" --speed 0 --concurrency 1 --max-prompt-tokens 36000 \
      --gguf "$m8" --expect-context-length 40960 --limit "$limit" --out "$res.tmp" >> "$out/replay.log" 2>&1
    [ -s "$res.tmp" ] && mv "$res.tmp" "$res"
  fi
  nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2> /dev/null | sed 's/^/after: /' >> "$out/gpu-disk$tag-$mode.txt"
  stop_server
  # the cache's own lines (save, restore, spill) without the step trace
  grep -a -i -E 'disk|restore|spill|evict' "$out/server-disk$tag-$mode.log" 2> /dev/null | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-300 | head -n 400 \
    > "$out/server-disk$tag-$mode.cache.log"
  rm -f "$out/server-disk$tag-$mode.log" # the step trace is large; the cache lines are kept
  say "disk run $mode done"
}

st_disk() {
  [ -f "$out/stage-disk$tag.ok" ] && return 0
  [ "$DRY" = 1 ] || [ -s "$req_log" ] || { say "disk: no request log"; return 0; }
  disk_run ram
  disk_run disk
  touch "$out/stage-disk$tag.ok"
}

main() {
  say "gate_kv_box start (stages=$stages threads=$threads) out=$out"
  local s
  for s in $stages; do
    say "STAGE $s"
    "st_$s"
  done
  touch "$out/DONE"
  say "GATE_KV_DONE rc=0"
}

if [ "$DRY" = 1 ]; then main; else main >> "$log" 2>&1; fi
