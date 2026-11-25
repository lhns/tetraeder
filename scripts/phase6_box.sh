#!/usr/bin/env bash
# Box side of the Phase 6 rehearsal (ADR 0091, tree node bench-phase6-rehearsal): one stage per call, run on the box
# by the driver over taxameter ssh. Work dir P6 (default /root/p6): keys/ (key.pem never leaves), cluster.toml and the
# peers' certificates, models/<sha256>.gguf, node-state/, logs/, pids/. Long stages run detached and end with a
# `P6_<STAGE>_DONE rc=N` line in logs/<stage>.log.
#   build                 cargo release build of tetraeder (cuda) and tetraeder-plan
#   model                 download Qwen3-30B-A3B Q4_K_M, install it under its id (`tetraeder cluster model-id --install`)
#   model-id [FIRST]      the id of a model, single file or split (the first shard; default the 235B Q4_K_M under .hf_cache), installed
#                         in models/ and written to model.sha256; it hashes every shard, so run it through `launch`. Ends with a
#                         `P6_MODELID_DONE rc=N id=...` line in logs/model-id.log
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
#   clock                 `date +%s.%N`
#   trigger               start the Case F fault trigger (scripts/phase6_trigger.py, own pid file) on TRIGGER_PORT (7001); token in
#                         $P6/trigger-token.txt; node b only
#   endpoint [PORT]       this box's public address and the mapped port (default 7000, the trigger's 7001)
#   judged-round LABEL SHA  one judged Case S round (`tetraeder cluster judged-round`, node a, the head) into judged/LABEL; ends with
#                         a `ROUND_DONE rc=N` line in logs/launch-judged-LABEL.log (scripts/phase6_exit_s.sh)
# Env: THREADS (CPU quota, default cpu_quota.sh), PAR (parallel, 4), CTX (4096), WORKER_EXTRA (more worker flags of `node`, e.g. --miss cpu), P6, HEAD_HOST and HEAD_PORT (the head, 0.0.0.0:8080), TETRAEDER_BIN (the binary to run; the
# loopback dry run sets it), P6_LOOPBACK (1: no hardware, no CUDA; `gates` writes fixture readings; `build` only checks that TETRAEDER_BIN
# exists; `model` links the tiny MoE fixture, P6_LB_MODEL to change it), TRIGGER_PORT and TRIGGER_HOST (the trigger).
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
# the id of a model (a single file: its sha256; a split one: the sha256 of its shards' manifest) linked into the node's models folder under
# that id, and written to model.sha256; prints it, or fails
install_model() { # install_model FIRST_FILE
  local id
  id=$("$bin" cluster model-id "$1" --install "$P6/models" 2>> "$P6/logs/model-id.err" | tail -n 1) || return 1
  [[ "$id" =~ ^[0-9a-f]{64}$ ]] || return 1
  echo "$id" > "$P6/model.sha256"
  echo "$id"
}
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
    if [ -n "${P6_LOOPBACK:-}" ]; then # CI builds the binary; this only proves it is there, in the same log line the real build ends with
      echo "=== RUN $(date +%FT%T) build (loopback: $bin) ===" >> "$P6/logs/build.log"
      if [ -x "$bin" ]; then echo "P6_BUILD_DONE rc=0" >> "$P6/logs/build.log"; else echo "P6_BUILD_DONE rc=1 (no $bin)" >> "$P6/logs/build.log"; fi
      tail -n 1 "$P6/logs/build.log"
      exit 0
    fi
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
    if [ -n "${P6_LOOPBACK:-}" ]; then # the tiny MoE fixture under its id, as a box holds the 30B
      src="${P6_LB_MODEL:-$root/tests/fixtures/tiny-qwen3moe-f16.gguf}"
      if h=$(install_model "$src"); then
        echo "P6_MODEL_DONE rc=0 sha256=$h bytes=$(stat -c %s "$src")" >> "$P6/logs/model.log"
      else
        echo "P6_MODEL_DONE rc=1 (model-id of $src failed: logs/model-id.err)" >> "$P6/logs/model.log"
      fi
      tail -n 1 "$P6/logs/model.log"
      exit 0
    fi
    (
      cd "$root" || exit 1
      if [ ! -s "$gguf_rel" ]; then
        TAXAMETER_DIR="$PWD" bash "${TAXAMETER_HOME:-/root/.taxameter}/box.sh" models "Qwen/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf" 8>&-
      fi
      [ -s "$gguf_rel" ] || { echo "P6_MODEL_DONE rc=1 (no file)"; exit 1; }
      h=$(install_model "$root/$gguf_rel") || { echo "P6_MODEL_DONE rc=1 (model-id failed: logs/model-id.err)"; exit 1; }
      echo "P6_MODEL_DONE rc=0 sha256=$h bytes=$(stat -c %s "$gguf_rel")"
    ) >> "$P6/logs/model.log" 2>&1 8>&- &
    echo "model started $(date +%T)"
    ;;
  model-id)
    first="${1:-$root/.hf_cache/Qwen__Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf}"
    echo "=== RUN $(date +%FT%T) model-id $first ===" >> "$P6/logs/model-id.log"
    if h=$(install_model "$first"); then
      echo "P6_MODELID_DONE rc=0 id=$h" >> "$P6/logs/model-id.log"
    else
      echo "P6_MODELID_DONE rc=1 (logs/model-id.err)" >> "$P6/logs/model-id.log"
    fi
    tail -n 1 "$P6/logs/model-id.log"
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
    # shellcheck disable=SC2086 # a word list on purpose
    for a in ${WORKER_EXTRA:-}; do args+=("--worker-arg=$a"); done
    # the replica engine's own prefix cache and request log
    args+=("--replica-arg=--request-log" "--replica-arg=$P6/logs/replica-requests.jsonl")
    # `tetraeder::step` at debug: its `admit` lines are the evidence of which requests B served (phase6_exit_f_summary.py)
    RUST_LOG="${RUST_LOG:-info,tetraeder::step=debug}" detached node "$bin" "${args[@]}"
    ;;
  serve)
    args=(serve --cluster "$P6/cluster.toml" --key "$P6/keys/key.pem" --cert "$P6/keys/cert.pem"
      -m "$P6/models/$(sha).gguf" --model-sha256 "$(sha)" --host "${HEAD_HOST:-0.0.0.0}" --port "${HEAD_PORT:-8080}" --api-key-file "$P6/api-key.txt"
      --node-state "$P6/node-state" --request-log "$P6/logs/head-requests.jsonl" --metrics "${engine_flags[@]}" "$@")
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
    # this box's public address and the mapped port of PORT (`create --ports 7000,7001`), as the container sees them
    port="${1:-7000}"
    if [ -n "${P6_LOOPBACK:-}" ]; then
      if [ "$port" = 7000 ]; then echo "127.0.0.1:${P6_LB_PORT:?}"; else echo "127.0.0.1:${TRIGGER_PORT:?}"; fi
      exit 0
    fi
    [[ "$port" =~ ^[0-9]+$ ]] || { echo "bad port $port" >&2; exit 2; }
    eval "$(xargs -0 -n1 < /proc/1/environ | grep -E "^(VAST_TCP_PORT_$port|PUBLIC_IPADDR)=" | sed 's/^/export /')"
    p="VAST_TCP_PORT_$port"
    echo "${PUBLIC_IPADDR:?no PUBLIC_IPADDR}:${!p:?no $p (create --ports 7000,7001)}"
    ;;
  trigger)
    # not a child of the daemon: the faults kill the daemon's children and the daemon itself, never this
    detached trigger python3 -I "$root/scripts/phase6_trigger.py" serve --host "${TRIGGER_HOST:-0.0.0.0}" --port "${TRIGGER_PORT:-7001}" \
      --token-file "$P6/trigger-token.txt" --p6 "$P6" --repo "$root"
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
  judged-round)
    label="${1:?label}"
    sha_v="${2:?model sha256}"
    mkdir -p "$P6/judged/$label"
    "$bin" cluster judged-round --cluster "$P6/cluster.toml" --key "$P6/keys/key.pem" --cert "$P6/keys/cert.pem"       --plan "$P6/judged-plan.toml" --model-sha256 "$sha_v" --hidden "${HIDDEN:-$root/runs/sc/hidden.bin}"       --prompt-tokens "${PROMPT_TOKENS:-2048}" --gen "${GEN_TOKENS:-128}" --warm "${WARM:-1}" --out "$P6/judged/$label"
    echo "ROUND_DONE rc=$?"
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
