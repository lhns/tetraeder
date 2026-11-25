#!/usr/bin/env bash
# Box side of the Phase 6 rehearsal (ADR 0091, tree node bench-phase6-rehearsal): one stage per call, run on the box
# by the driver over taxameter ssh. Work dir P6 (default /root/p6): keys/ (key.pem never leaves), cluster.toml and the
# peers' certificates, models/<sha256>.gguf, node-state/, logs/, pids/. Long stages run detached and end with a
# `P6_<STAGE>_DONE rc=N` line in logs/<stage>.log.
#   build                 cargo release build of tetraeder (cuda) and tetraeder-plan
#   model                 download Qwen3-30B-A3B Q4_K_M, link it as models/<sha256>.gguf
#   cert ID               tetraeder cluster cert --node ID (skipped when keys exist)
#   node [BUFFERS]        start `tetraeder node` (autotuned buffers by default, or SEND,RECV)
#   serve ARGS...         start the head: `tetraeder serve --cluster ...` with the engine flags and ARGS
#   solo                  start one `tetraeder serve --engine moe` on loopback port 18080 (the one-node reference)
#   stop NAME             kill -9 the process in pids/NAME (node, serve, solo), children first
#   kill NAME SIGNAL      send SIGNAL to the process of pids/NAME and its children (worker, replica engine)
#   worker-kill           kill -9 the children of the node daemon only
#   gates                 the exit sessions' gates before any download (power, idle, one NUMA node, co-tenant by
#                         engine_sample.py), written to gates/ ; exit 1 and `P6_GATES_DONE rc=1` when one fails
#   admits OUT            epochs of the `admit` lines of logs/node.log (the replica engine's requests) into OUT
#   clock                 `date +%s.%N`, for the driver's clock offset between the boxes
# Env: THREADS (CPU quota, default cpu_quota.sh), PAR (parallel, 4), CTX (4096), P6, HEAD_HOST and HEAD_PORT (the head, 0.0.0.0:8080), TETRAEDER_BIN (the binary to run; the
# loopback dry run sets it), P6_LOOPBACK (1: no hardware, no CUDA; `gates` writes fixture readings).
cd "$(dirname "$0")/.." || exit 1
root=$PWD
P6="${P6:-/root/p6}"
stage="${1:-}"
[ -n "$stage" ] || { echo "usage: phase6_box.sh STAGE [ARGS]" >&2; exit 2; }
shift
mkdir -p "$P6"/{keys,models,node-state,logs,pids}
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
if [ -z "${P6_LOOPBACK:-}" ]; then
  # shellcheck disable=SC1091
  . "$root/scripts/box_libs.sh"
  box_libs_path
  # shellcheck disable=SC1091
  [ -f "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh" ] && . "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
fi
bin="${TETRAEDER_BIN:-$root/target/release/tetraeder}"
threads="${THREADS:-$(bash "$root/scripts/cpu_quota.sh" 2> /dev/null || nproc)}"
par="${PAR:-4}"
ctx="${CTX:-4096}"
gguf_rel=".hf_cache/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
sha() { cat "$P6/model.sha256" 2> /dev/null; }
# the engine flags of a replica or a whole-model server on this box
engine_flags=(--engine moe --ngl -1 -t "$threads" --ctx "$ctx" --parallel "$par" --accept-any-model --alias qwen3-30b-a3b)

detached() { # detached NAME CMD...: refuses a second start while pids/NAME is alive; log in logs/NAME.log
  local name=$1
  shift
  if [ -s "$P6/pids/$name" ] && kill -0 "$(cat "$P6/pids/$name")" 2> /dev/null; then
    echo "$name already running (pid $(cat "$P6/pids/$name"))"
    return 0
  fi
  echo "=== RUN $(date +%FT%T) $name ===" >> "$P6/logs/$name.log"
  setsid nohup "$@" >> "$P6/logs/$name.log" 2>&1 < /dev/null &
  echo $! > "$P6/pids/$name"
  echo "$name started pid $!"
}

case "$stage" in
  build)
    exec 8> "$P6/pids/build.lock"
    flock -n 8 || { echo "build already running"; exit 0; }
    archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
    [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
    echo "=== RUN $(date +%FT%T) build ===" >> "$P6/logs/build.log"
    (
      cd "$root" || exit 1
      export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-$threads}"
      cargo build --release -p tetraeder --features cuda 8>&- && cargo build --release -p tetraeder-planner --bin tetraeder-plan 8>&-
      echo "P6_BUILD_DONE rc=$?"
    ) >> "$P6/logs/build.log" 2>&1 8>&- &
    echo "build started $(date +%T)"
    ;;
  model)
    (
      cd "$root" || exit 1
      if [ ! -s "$gguf_rel" ]; then
        TAXAMETER_DIR="$PWD" bash "${TAXAMETER_HOME:-/root/.taxameter}/box.sh" models "Qwen/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf" 8>&-
      fi
      [ -s "$gguf_rel" ] || { echo "P6_MODEL_DONE rc=1 (no file)"; exit 1; }
      h=$(sha256sum "$gguf_rel" | cut -d' ' -f1)
      echo "$h" > "$P6/model.sha256"
      ln -sf "$root/$gguf_rel" "$P6/models/$h.gguf"
      echo "P6_MODEL_DONE rc=0 sha256=$h bytes=$(stat -c %s "$gguf_rel")"
    ) >> "$P6/logs/model.log" 2>&1 8>&- &
    echo "model started $(date +%T)"
    ;;
  cert)
    id="${1:?node id}"
    if [ -s "$P6/keys/key.pem" ]; then echo "keys exist"; else "$bin" cluster cert --node "$id" --out "$P6/keys"; fi
    echo "public certificate: $P6/keys/cert.pem"
    ;;
  node)
    buf="${1:-auto}"
    args=(node --cluster "$P6/cluster.toml" --key "$P6/keys/key.pem" --cert "$P6/keys/cert.pem" --models-dir "$P6/models"
      --state-dir "$P6/node-state" --buffers "$buf")
    for a in "${engine_flags[@]}"; do args+=("--worker-arg=$a" "--replica-arg=$a"); done
    # the replica engine's own prefix cache and request log
    args+=("--replica-arg=--request-log" "--replica-arg=$P6/logs/replica-requests.jsonl")
    # `tetraeder::step` at debug: its `admit` lines are the evidence of which requests B served (phase6_exit_f_summary.py)
    RUST_LOG="${RUST_LOG:-info,tetraeder::step=debug}" detached node "$bin" "${args[@]}"
    ;;
  serve)
    args=(serve --cluster "$P6/cluster.toml" --key "$P6/keys/key.pem" --cert "$P6/keys/cert.pem"
      -m "$P6/models/$(sha).gguf" --model-sha256 "$(sha)" --host "${HEAD_HOST:-0.0.0.0}" --port "${HEAD_PORT:-8080}" --api-key-file "$P6/api-key.txt"
      --request-log "$P6/logs/head-requests.jsonl" --metrics "${engine_flags[@]}" "$@")
    RUST_LOG="${RUST_LOG:-info}" detached serve "$bin" "${args[@]}"
    ;;
  solo)
    RUST_LOG="${RUST_LOG:-info}" detached solo "$bin" serve -m "$root/$gguf_rel" --host 127.0.0.1 --port 18080 "${engine_flags[@]}"
    ;;
  stop)
    name="${1:?name}"
    pid=$(cat "$P6/pids/$name" 2> /dev/null) || { echo "no pid for $name"; exit 0; }
    for c in $(pgrep -P "$pid"); do kill -9 "$c" 2> /dev/null; done
    kill -9 "$pid" 2> /dev/null
    echo "stopped $name ($pid)"
    ;;
  kill)
    name="${1:?name}"
    sig="${2:?signal}"
    pid=$(cat "$P6/pids/$name" 2> /dev/null) || { echo "no pid for $name"; exit 1; }
    for c in $(pgrep -P "$pid"); do kill "-$sig" "$c" 2> /dev/null; done
    kill "-$sig" "$pid" 2> /dev/null
    echo "$(date +%s.%N) sent $sig to $name ($pid) and its children"
    ;;
  gates)
    g="$P6/gates"
    mkdir -p "$g"
    rc=0
    if [ -n "${P6_LOOPBACK:-}" ]; then
      printf 'Architecture: x86_64\nNUMA node(s):                      1\n' > "$g/lscpu.txt"
      python3 -I -c "import json; json.dump({'source': 'engine_sample.py cotenant', 'outside_cores': 0.5, 'host_cpus': 16, 'limit': 4.0, 'ok': True}, open('$g/cotenant.json', 'w'))"
      echo "loopback: no GPU, nothing to gate" > "$g/power-gate.txt"
      cp "$g/power-gate.txt" "$g/idle-gate.txt"
      power=true idle=true
    else
      power=false idle=false
      bash scripts/power_gate.sh > "$g/power-gate.txt" 2>&1 && power=true
      bash scripts/idle_gate.sh > "$g/idle-gate.txt" 2>&1 && idle=true
      lscpu > "$g/lscpu.txt" 2>&1
      python3 scripts/engine_sample.py cotenant --seconds "${COTENANT_S:-20}" --out "$g/cotenant.json" > "$g/cotenant.txt" 2>&1
    fi
    python3 -I scripts/sc_gates.py numa "$g/lscpu.txt" || rc=1
    python3 -I scripts/sc_gates.py cotenant "$g/cotenant.json" || rc=1
    [ "$power" = true ] || { echo "GATE FAILED power: a GPU is not at its default power limit"; rc=1; }
    [ "$idle" = true ] || { echo "GATE FAILED idle: clock event reasons set while idle"; rc=1; }
    echo "{\"power_ok\": $power, \"idle_ok\": $idle}" > "$g/gates.json"
    echo "P6_GATES_DONE rc=$rc"
    exit "$rc"
    ;;
  admits)
    python3 -I scripts/phase6_exit_f_summary.py admits "$P6/logs/node.log" --out "${1:?out file}"
    wc -l < "${1}"
    ;;
  clock)
    date +%s.%N
    ;;
  endpoint)
    # this box's public address and the mapped port of 7000 (`create --ports 7000`), as the container sees them
    if [ -n "${P6_LOOPBACK:-}" ]; then
      echo "127.0.0.1:${P6_LB_PORT:?}"
      exit 0
    fi
    eval "$(xargs -0 -n1 < /proc/1/environ | grep -E '^(VAST_TCP_PORT_7000|PUBLIC_IPADDR)=' | sed 's/^/export /')"
    echo "${PUBLIC_IPADDR:?no PUBLIC_IPADDR}:${VAST_TCP_PORT_7000:?no VAST_TCP_PORT_7000}"
    ;;
  launch)
    # launch TAG CMD...: starts CMD detached once. taxameter repeats a command after a dropped connection (AGENTS.md), so a
    # second call with the same TAG starts nothing. The first line is the epoch of the start on this box's clock.
    tag="${1:?tag}"
    shift
    exec 8> "$P6/pids/launch-$tag.lock"
    flock -n 8 || { echo "launch $tag: another call is starting it"; exit 0; }
    [ -e "$P6/pids/launch-$tag.done" ] && { echo "launch $tag: already started"; exit 0; }
    : > "$P6/pids/launch-$tag.done"
    date +%s.%N
    setsid nohup "$@" >> "$P6/logs/launch-$tag.log" 2>&1 8>&- < /dev/null &
    echo "launched $tag pid $!"
    ;;
  worker-kill)
    pid=$(cat "$P6/pids/node" 2> /dev/null) || { echo "no node pid"; exit 1; }
    echo "$(date +%s.%N) kill -9 children of node $pid: $(pgrep -P "$pid" | tr '\n' ' ')"
    for c in $(pgrep -P "$pid"); do kill -9 "$c" 2> /dev/null; done
    ;;
  *)
    echo "unknown stage $stage" >&2
    exit 2
    ;;
esac
