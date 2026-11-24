#!/usr/bin/env bash
# Session stream-predict, the stages after the arms (never beside a timed run). STAGES (default "gate disk"):
#   gate  the Qwen3-8B (--engine dense) c4 and c1 gate pair of node bench-regression-gate through scripts/ab_moe_box.sh:
#         A = f68d2a0, B = 41dcedd2 (binaries of scripts/sp_abbuild.sh), ROUNDS (3) interleaved A B, B A, A B rounds, the rule
#         of bench/results/regression-gate-20261123/README.md; a rerun with a larger ROUNDS adds rounds (finished runs are kept)
#   disk  the prefix disk tier: a synthetic multi-session replay (scripts/disk_tier_log.py, more sessions than slots) on Qwen3-8B
#         with a small --cache-ram, once without and once with --cache-disk; TTFT of the later turns, restore count and restore
#         p50 and p90 from the server's admit lines (scripts/disk_tier_summary.py). Report only.
# Env: RESULTS_NAME, STAGES, ROUNDS, THREADS, SESSIONS (12), TURNS (4), WORDS (1500), CACHE_RAM_MIB (512), CACHE_DISK_MIB (8192).
# Ends with SP_POST_DONE in runs/vast/sp_post.log.
exec 8> /root/tetraeder_sp_post.lock
flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
root=$PWD
name="${RESULTS_NAME:?set RESULTS_NAME}"
out="$root/bench/results/$name"
mkdir -p "$out/ab" "$out/disk" runs/vast
log=runs/vast/sp_post.log
stages="${STAGES:-gate disk}"
say() { echo "[$(date +%T)] $*"; }
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:/root/.local/bin:$PATH"
quota=$(bash scripts/cpu_quota.sh 2> /dev/null || nproc)
threads="${THREADS:-$quota}"
py=/root/gwvenv/bin/python
tet="$root/target/release/tetraeder"
m8="$root/.hf_cache/Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
port=18080
server_pid=""

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

st_gate() {
  [ -x /root/tetraeder-A/target/release/tetraeder ] && [ -x /root/tetraeder-B/target/release/tetraeder ] || { say "gate: run sp_abbuild.sh first"; return 1; }
  [ -f "$m8" ] || { say "gate: no $m8"; return 1; }
  SESSION="$name" AB_OUT="$out/ab" MODELS=q8b TTFT=0 SKIP_BUILD=1 ROUNDS="${ROUNDS:-3}" THREADS="$threads" BUDGET_MIN=70 \
    TET_A=/root/tetraeder-A/target/release/tetraeder TET_B=/root/tetraeder-B/target/release/tetraeder \
    A_COMMIT=f68d2a0c71035a42eabcba1799e689b2f6e779c9 COMMIT=41dcedd2af65d1981ca7593b27bedcd7f1a18421 \
    AB_LOCK=/root/tetraeder_sp_post_ab.lock bash scripts/ab_moe_box.sh 8>&-
  say "gate: ab_moe_box.sh rc=$?"
}

disk_run() { # disk_run MODE: ram (recompute) or disk
  local mode=$1 url="http://127.0.0.1:$port" extra=() res="$out/disk/disk-$1.json"
  [ -s "$res" ] && { say "disk run $mode done before"; return 0; }
  [ "$mode" = disk ] && { rm -rf /root/spdisk; extra=(--cache-disk /root/spdisk --cache-disk-mib "${CACHE_DISK_MIB:-8192}"); }
  say "disk run $mode start"
  : > "$out/disk/server-disk-$mode.log"
  setsid nohup env RUST_LOG=info,tetraeder::step=debug "$tet" serve -m "$m8" --engine dense --ngl -1 --parallel 4 --ctx 40960 \
    -t "$threads" --port "$port" --alias qwen3-8b --accept-any-model --cache-ram "${CACHE_RAM_MIB:-512}" "${extra[@]}" \
    > "$out/disk/server-disk-$mode.log" 2>&1 8>&- &
  server_pid=$!
  local i
  for i in $(seq 1 450); do
    kill -0 "$server_pid" 2> /dev/null || { tail -n 15 "$out/disk/server-disk-$mode.log"; say "disk run $mode: server exited"; return 1; }
    curl -sf -o /dev/null "$url/v1/models" && break
    sleep 2
  done
  nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>&1 | sed 's/^/before: /' > "$out/disk/gpu-disk-$mode.txt"
  "$py" scripts/gateway_bench.py "$url" --model qwen3-8b --prompt-tokens 128 --max-tokens 16 --requests 1 --no-keepalive > /dev/null 2>&1
  timeout -k 30 40m "$py" scripts/gateway_replay.py "$url" "$out/disk/requests.jsonl" --speed 0 --concurrency 1 --timeout 600 \
    --out "$res.tmp" >> "$out/disk/replay.log" 2>&1
  [ -s "$res.tmp" ] && mv "$res.tmp" "$res"
  nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2>&1 | sed 's/^/after: /' >> "$out/disk/gpu-disk-$mode.txt"
  stop_server
  # the admit lines and the cache's own lines; the full step trace is large
  grep -a -E 'admit seq=' "$out/disk/server-disk-$mode.log" | sed 's/\x1b\[[0-9;]*m//g' > "$out/disk/server-disk-$mode.admits.log"
  grep -a -i -E 'disk|spill|evict' "$out/disk/server-disk-$mode.log" | grep -a -v 'admit seq=' | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-300 | head -n 300 > "$out/disk/server-disk-$mode.cache.log"
  say "disk run $mode done"
}

st_disk() {
  [ -f "$m8" ] || { say "disk: no $m8"; return 1; }
  python3 -I scripts/disk_tier_log.py --sessions "${SESSIONS:-12}" --turns "${TURNS:-4}" --words "${WORDS:-1500}" --out "$out/disk/requests.jsonl"
  disk_run ram || return 1
  disk_run disk || return 1
  # the summary reads the admit lines from the logs; keep the large raw logs out of the results
  cp "$out/disk/server-disk-ram.admits.log" "$out/disk/server-disk-ram.log" 2> /dev/null
  cp "$out/disk/server-disk-disk.admits.log" "$out/disk/server-disk-disk.log" 2> /dev/null
  python3 -I scripts/disk_tier_summary.py "$out/disk" "${SESSIONS:-12}"
}

main() {
  say "sp_post start stages=$stages threads=$threads"
  local s
  for s in $stages; do
    say "STAGE $s"
    "st_$s"
  done
  say "SP_POST_DONE rc=0"
}
main >> "$log" 2>&1
