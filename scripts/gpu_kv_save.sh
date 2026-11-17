#!/usr/bin/env bash
# Prefix-cache save through pinned staging on a GPU box (ADR 0074): the paired save timing test (gpu_kv_save), the GPU
# snapshot round trips (gpu_multi_seq, ggml-sys gpu_host_reads), then per model one tetraeder c4 probe as in
# gpu_pin_confirm.sh with the admission log, whose save_ms lines are summarized. No verdict pairs.
# Started detached: `taxameter run --id <id> scripts/gpu_kv_save.sh --env SESSION=<s>`; the log runs/vast/kv_save.log
# ends with KV_SAVE_DONE rc=<n>. Results: bench/results/<session>/kv-save/. Env: SESSION, MODELS ("q8b q30b"), CTX.
exec 8> /root/tetraeder_kv_save.lock
flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
root=$PWD
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
export HF_HUB_OFFLINE=1 TETRAEDER_MODELS_DIR="$root/.hf_cache"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
[ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
out="$root/bench/results/${SESSION:?}/kv-save"
mkdir -p "$out" "$root/runs/vast"
exec >> "$root/runs/vast/kv_save.log" 2>&1
say() { echo "[$(date +%T)] $*"; }
done_rc() {
  echo "rc=$1" > "$out/DONE"
  say "KV_SAVE_DONE rc=$1"
  exit "$1"
}
case "$CUDA_ENV_STATUS" in ok | ok_preload) say "cuda: $CUDA_ENV_STATUS" ;; *) say "FATAL: no cuda ($CUDA_ENV_STATUS)"; done_rc 1 ;; esac
say "start $(git rev-parse HEAD) $(nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.mem --format=csv,noheader 2> /dev/null)"
bash scripts/ensure_patches.sh 8>&- || { say "FATAL: patch queue failed"; done_rc 1; }
box_libs_path
rc=0
tests() { # NAME CARGO-ARGS...: one test binary, its log in the results
  local name=$1
  shift
  say "test $name"
  TETRAEDER_REQUIRE_GPU=1 timeout -k 30 40m cargo test --release "$@" -- --nocapture --test-threads 1 > "$out/$name.log" 2>&1 8>&-
  local r=$?
  say "test $name rc=$r $(grep -hE 'MB, staged|host copy' "$out/$name.log" | tr '\n' ' ')"
  [ $r = 0 ] || rc=1
}
tests gpu_kv_save -p tetraeder-model --features tetraeder-ggml-sys/cuda --test gpu_kv_save
tests gpu_multi_seq -p tetraeder-model --features tetraeder-ggml-sys/cuda --test gpu_multi_seq
tests ggml_host_reads -p tetraeder-ggml-sys --features cuda --test ggml host_reads

# ---- probe: save_ms in the admission log of a c4 run ------------------------------------------------------------
timeout -k 30 50m cargo build --release -p tetraeder --features cuda 8>&- || { say "FATAL: build failed"; done_rc 1; }
py=/root/gwvenv/bin/python
if [ ! -x "$py" ] || ! "$py" -c 'import httpx' 2> /dev/null; then
  if ! command -v uv > /dev/null 2>&1 && ! python3 -c 'import ensurepip' 2> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.local/bin sh > /dev/null 2>&1
    export PATH="/root/.local/bin:$PATH"
  fi
  if command -v uv > /dev/null 2>&1; then
    uv venv --allow-existing /root/gwvenv && uv pip install --python "$py" httpx
  else
    python3 -m venv /root/gwvenv && "$py" -m pip install -q httpx
  fi || { say "FATAL: python env failed"; done_rc 1; }
fi
declare -A FILE=(
  [q30b]="$root/.hf_cache/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
  [q8b]="$root/.hf_cache/Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
)
declare -A ALIAS=([q30b]=qwen3-30b-a3b [q8b]=qwen3-8b)
declare -A ENGINE=([q30b]=moe [q8b]=dense)
threads=$(bash scripts/cpu_quota.sh 8>&-)
u=http://127.0.0.1:18080
for m in ${MODELS:-q8b q30b}; do
  [ -s "${FILE[$m]}" ] || { say "WARNING: no model ${FILE[$m]}"; continue; }
  slog="$out/server-$m.log"
  say "probe $m"
  RUST_LOG=info,tetraeder::step=debug setsid nohup "$root/target/release/tetraeder" serve -m "${FILE[$m]}" \
    --engine "${ENGINE[$m]}" --ngl -1 --parallel 4 --ctx "${CTX:-32768}" -t "$threads" --port 18080 --alias "${ALIAS[$m]}" \
    --accept-any-model > "$slog" 2>&1 8>&- &
  pid=$!
  for _ in $(seq 1 450); do
    curl -sf -o /dev/null $u/v1/models && break
    kill -0 $pid 2> /dev/null || break
    sleep 2
  done
  "$py" scripts/gateway_bench.py $u --model "${ALIAS[$m]}" --prompt-tokens 128 --max-tokens 16 --requests 1 --no-keepalive > /dev/null 2>&1
  for c in 1 4; do
    timeout -k 30 15m "$py" scripts/gateway_bench.py $u --model "${ALIAS[$m]}" --temperature 0 --no-keepalive \
      --label "kv-save-$m-c$c" --out "$out/probe-$m-c$c.json" --prompt-tokens 512 --max-tokens 128 --concurrency $c \
      --requests 8 --seed $((90 + c)) >> "$out/bench.log" 2>&1
  done
  kill $pid 2> /dev/null
  sleep 3
  kill -9 $pid 2> /dev/null
  wait $pid 2> /dev/null
  saves=$(grep -o 'save_ms=[0-9.]*' "$slog" | cut -d= -f2 | awk '$1 > 0' | sort -n | tr '\n' ' ')
  say "probe $m: save_ms (nonzero, sorted) $saves; $(grep -c 'CUDA error' "$slog") CUDA error lines; $(grep -h 'kv readback staging' "$slog" | head -1)"
done
done_rc $rc
