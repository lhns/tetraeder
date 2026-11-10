#!/usr/bin/env bash
# Claude Code against `tetraeder serve` and llama-server on a GPU box (ADR 0046): real headless Claude Code sessions
# on 10 coding tasks, both engines behind the same logging proxy, scored by task checks.
# Started detached by `taxameter run scripts/claude_code_proxy.sh`; ends with CLAUDE_CODE_PROXY_DONE in
# runs/vast/claude_code_proxy.log. Results: bench/results/<date>/claude-code-proxy/ (summary.json; a rerun on the
# same day, or with the same RESULTS_DATE, resumes: sessions whose output exists are skipped).
#
# Model (assumed downloaded): Qwen3-Coder-30B-A3B-Instruct-Q4_K_M (provision with
#   --models unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf).
# Env knobs: REPS (2), BUDGET_MIN (240: no block or session starts after this many minutes since the script began;
# a block that started in time may run GRACE_MIN (30) longer), CTX (131072), THREADS, MODEL, ALIAS, TASKS (comma
# separated ids), TASK_TIMEOUT (900 s per session), TET_ARGS / LLAMA_ARGS (extra server arguments), NODE_VERSION,
# CLAUDE_CODE_VERSION (npm tag, default latest), PERMISSIONS (skip = --dangerously-skip-permissions with IS_SANDBOX=1,
# accept = acceptEdits plus an allowed-tools list), STREAMING=1 (final conformance stage on the streaming MoE),
# DECODE_PROFILE (1: tetraeder serve reports per-request decode stages into its request log and summary.json
# `server_decode`; near-zero cost), MICROBENCH=1 (then scripts/decode_gap.sh: both engines and llama-bench at fixed
# depths without Claude Code, docs/m1-open.md item 4).
# Sessions run one at a time per server (-np 1); the engines alternate ABBA over the reps (one server start per block).
#
# VRAM: the 30B Q4_K_M weights are about 18.6 GB and the f16 KV cache about 98 KB per token (12.9 GB at 131072):
# CTX 131072 needs a 48 GB card (the target: RTX A6000, L40S, A40). On 24 GB lower CTX for both engines (49152 is a
# floor: Claude Code's own prompt is 15K to 30K tokens).
# cargo runs with fd 8 closed: the sccache server it starts would otherwise keep the lock after the job
exec 8> /root/tetraeder_claude_code_proxy.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
bash scripts/ensure_patches.sh 8>&- || exit 1  # ggml patch queue (ADR 0052)
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:/opt/node/bin:/root/.local/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
# this box's GPU only (another arch list rebuilds ggml-cuda for every arch)
if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
log=runs/vast/claude_code_proxy.log
export TETRAEDER_DECODE_PROFILE="${DECODE_PROFILE:-1}"
out="$PWD/bench/results/${RESULTS_DATE:-$(date +%Y%m%d)}/claude-code-proxy"
mkdir -p "$out" runs/vast
t_start=$(date +%s)
threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"
budget_min="${BUDGET_MIN:-240}"
models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
model="${MODEL:-$models_dir/unsloth__Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}"
alias="${ALIAS:-qwen3-coder-30b}"
ctx="${CTX:-131072}"
node_version="${NODE_VERSION:-v22.11.0}"
venv=/root/ccvenv
py="$venv/bin/python"
tet=target/release/tetraeder
server_pid=""

say() { echo "[$(date +%T)] $*"; }

stop_server() {
  [ -n "$server_pid" ] || return 0
  kill "$server_pid" 2> /dev/null
  for _ in $(seq 1 30); do kill -0 "$server_pid" 2> /dev/null || break; sleep 1; done
  kill -9 "$server_pid" 2> /dev/null
  wait "$server_pid" 2> /dev/null
  server_pid=""
}
trap stop_server EXIT

# ---- a. build and toolchain --------------------------------------------------------------------------------
setup() {
  [ -f "$model" ] || { say "missing model $model"; return 1; }
  say "build start"
  timeout -k 30 50m cargo build --release -p tetraeder --features cuda 8>&- || { say "build failed"; return 1; }
  # the box image has python3 without ensurepip (no venv module): uv makes the venv
  if ! command -v uv > /dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.local/bin sh > /dev/null 2>&1
  fi
  command -v uv > /dev/null 2>&1 || { say "uv install failed"; return 1; }
  local pkgs="pytest httpx"
  [ "${STREAMING:-0}" = 1 ] && pkgs="$pkgs openai anthropic jsonschema pydantic"
  # shellcheck disable=SC2086
  uv venv --allow-existing "$venv" && uv pip install --python "$py" $pkgs || { say "python env failed"; return 1; }
  # Node.js >= 18 from the official tarball, then Claude Code
  local major
  major=$(node --version 2> /dev/null | sed -E 's/^v([0-9]+).*/\1/')
  if [ -z "$major" ] || [ "$major" -lt 18 ]; then
    say "install node $node_version"
    mkdir -p /opt/node
    curl -fsSL "https://nodejs.org/dist/$node_version/node-$node_version-linux-x64.tar.gz" | tar -xz -C /opt/node --strip-components=1 \
      || { say "node install failed"; return 1; }
  fi
  say "node $(node --version) npm $(npm --version)"
  if ! command -v claude > /dev/null 2>&1; then
    npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION:-latest}" > "$out/npm-install.log" 2>&1 \
      || { say "claude code install failed (see npm-install.log)"; return 1; }
  fi
  say "claude $(claude --version 2>&1 | head -1)"
  say "setup done"
}

# ---- b. optional: conformance on the streaming MoE ---------------------------------------------------------
streaming() {
  grep -qs " passed" "$out/conformance-streaming.pytest.log" && { say "streaming conformance: done before"; return 0; }
  say "streaming conformance start"
  local slog="$out/server-streaming.log"
  : > "$slog"
  setsid nohup "$tet" serve -m "$model" --engine moe --ctx 32768 --port 18082 --alias "$alias" --accept-any-model --ngl -1 \
    -t "$threads" --expert-slots 64 --ram-slots 128 --miss cpu --prefetch > "$slog" 2>&1 8>&- &
  server_pid=$!
  for _ in $(seq 1 300); do
    kill -0 "$server_pid" 2> /dev/null || { say "streaming server exited while loading"; break; }
    curl -sf -o /dev/null http://127.0.0.1:18082/v1/models && break
    sleep 2
  done
  if curl -sf -o /dev/null http://127.0.0.1:18082/v1/models; then
    TETRAEDER_BASE_URL=http://127.0.0.1:18082 TETRAEDER_MODEL="$alias" \
      timeout -k 30 25m "$py" -m pytest tests/conformance -q -p no:cacheprovider \
      > "$out/conformance-streaming.pytest.log" 2>&1
    say "streaming conformance rc=$?"
  fi
  stop_server
}

{
  say "claude_code_proxy start (threads=$threads ctx=$ctx reps=${REPS:-2} budget=${budget_min} min) out=$out"
  rc=0
  setup || rc=1
  if [ "$rc" = 0 ]; then
    # the runner starts and stops each server itself (by pid) and skips sessions whose output exists
    "$py" scripts/claude_code_proxy.py --out "$out" --model "$model" --alias "$alias" --ctx "$ctx" --threads "$threads" \
      --reps "${REPS:-2}" --budget-min "$((budget_min - ($(date +%s) - t_start) / 60))" --grace-min "${GRACE_MIN:-30}" \
      --task-timeout "${TASK_TIMEOUT:-900}" --permissions "${PERMISSIONS:-skip}" --tet-bin "$tet" --tet-args "${TET_ARGS:-}" --llama-args "${LLAMA_ARGS:-}" \
      ${TASKS:+--tasks "$TASKS"} 8>&- || rc=1
    "$py" scripts/claude_code_proxy_summary.py "$out" || rc=1
    "$py" scripts/claude_code_cache_classes.py "$out" > "$out/cache_classes.txt" || rc=1
    [ "${STREAMING:-0}" = 1 ] && { streaming || rc=1; }
    if [ "${MICROBENCH:-0}" = 1 ]; then
      say "decode_gap stage (log runs/vast/decode_gap.log)"
      bash scripts/decode_gap.sh 8>&- || rc=1
    fi
  fi
  GW_OUT="$out" GW_THREADS="$threads" GW_CTX="$ctx" GW_MODEL="$model" GW_COMMIT="$(git rev-parse HEAD 2> /dev/null)" \
    GW_HOST="$(hostname)" GW_CLAUDE="$(claude --version 2> /dev/null | head -1)" \
    GW_LLAMA="$(/opt/llama.cpp/bin/llama-server --version 2>&1 | head -3 | tr '\n' ' ')" \
    GW_GPU="$(nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active,memory.total --format=csv,noheader 2> /dev/null)" \
    "$py" - << 'EOF'
import json, os
e = os.environ
json.dump({k[3:].lower(): v.strip() for k, v in e.items() if k.startswith("GW_") and k != "GW_OUT"},
          open(os.path.join(e["GW_OUT"], "env.json"), "w"), indent=2)
EOF
  touch "$out/DONE"
  say "CLAUDE_CODE_PROXY_DONE rc=$rc"
} >> "$log" 2>&1
