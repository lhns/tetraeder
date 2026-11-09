#!/usr/bin/env bash
# Resident decode gap on one GPU box (docs/m1-open.md item 4): the Claude Code proxy run decoded at 0.88x llama-server
# on an RTX A6000 (Qwen3-Coder-30B-A3B Q4_K_M, 128K context). This reproduces it without Claude Code, paired on one box,
# repeats interleaved:
#   llama-bench  raw llama.cpp at the engine pin (/opt/llama.cpp), -fa 1, -d <depths>, tg <DECODE>
#   decode_gap   crates/scheduler/tests/decode_gap.rs: `tetraeder serve`'s two engines (libllama engine, MoE builder)
#                through the scheduler backend at the same depths, eval and serve sampler timed apart, graph reuse
#                counters (libllama n_reused, builder step_graphs_built)
# Started detached by `taxameter run scripts/decode_gap.sh`; ends with DECODE_GAP_DONE rc=N in runs/vast/decode_gap.log.
# Results: bench/results/<date>/decode-gap/ (llama-bench-r<rep>.json, decode-gap-r<rep>.json, env.json).
# Env: MODEL, DEPTHS (0,20000,100000), DECODE (128), CTX (131072), THREADS (cpu quota), REPS (2).
# Runtime estimate (A6000, warm sccache): build 10 min, per rep llama-bench 3 min + decode_gap 2 x (load 1 min + 120K
# prefill tokens at about 3K tok/s = 40 s) = 6 min: about 25 min for 2 reps. VRAM: 18.6 GB weights + 12.9 GB KV at
# 131072 per engine, one engine at a time.
exec 8> /root/tetraeder_decode_gap.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
log=runs/vast/decode_gap.log
out="$PWD/bench/results/$(date +%Y%m%d)/decode-gap"
mkdir -p "$out" runs/vast
models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
model="${MODEL:-$models_dir/unsloth__Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf}"
depths="${DEPTHS:-0,20000,100000}"
decode="${DECODE:-128}"
ctx="${CTX:-131072}"
threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"
bench="${LLAMA_BENCH:-/opt/llama.cpp/bin/llama-bench}"
say() { echo "[$(date +%T)] $*"; }
test_cmd=(cargo test --release -p tetraeder-scheduler --features cuda --test decode_gap)

{
  say "decode_gap start (threads=$threads ctx=$ctx depths=$depths decode=$decode) out=$out"
  rc=0
  [ -f "$model" ] || { say "missing model $model"; rc=1; }
  if [ "$rc" = 0 ]; then
    timeout -k 30 50m "${test_cmd[@]}" --no-run 8>&- || { say "build failed"; rc=1; }
  fi
  for rep in $(seq 1 "${REPS:-2}"); do
    [ "$rc" = 0 ] || break
    # Interleaved: llama-bench, then both serve engines, per rep.
    f="$out/llama-bench-r$rep.json"
    if [ ! -s "$f" ]; then
      say "llama-bench r$rep"
      if timeout -k 30 20m "$bench" -m "$model" -ngl 99 -fa 1 -t "$threads" -b 2048 -ub 512 -p 0 -n "$decode" \
        -d "$depths" -r 3 -o json > "$f.tmp" 2> "$out/llama-bench-r$rep.log"; then
        mv "$f.tmp" "$f"
      else
        say "llama-bench failed"
        rc=1
      fi
    fi
    f="$out/decode-gap-r$rep.json"
    if [ "$rc" = 0 ] && [ ! -s "$f" ]; then
      say "decode_gap r$rep"
      TETRAEDER_GAP_GGUF="$model" TETRAEDER_GAP_DEPTHS="$depths" TETRAEDER_GAP_DECODE="$decode" \
        TETRAEDER_GAP_CTX="$ctx" TETRAEDER_GAP_THREADS="$threads" TETRAEDER_GAP_OUT="$f" \
        timeout -k 30 30m "${test_cmd[@]}" -- --nocapture 8>&- > "$out/decode-gap-r$rep.log" 2>&1 || { say "decode_gap failed"; rc=1; }
    fi
  done
  host="$(hostname)" gpu="$(nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active,memory.total --format=csv,noheader 2> /dev/null)" \
    commit="$(git rev-parse HEAD 2> /dev/null)" llama="$(cat /opt/llama.cpp/PIN 2> /dev/null)" \
    python3 -c 'import json, os, sys; json.dump({k: os.environ.get(k, "") for k in ("host", "gpu", "commit", "llama")} | {"threads": sys.argv[2], "ctx": sys.argv[3], "depths": sys.argv[4], "decode": sys.argv[5], "model": sys.argv[6]}, open(sys.argv[1] + "/env.json", "w"), indent=2)' \
    "$out" "$threads" "$ctx" "$depths" "$decode" "$model"
  say "DECODE_GAP_DONE rc=$rc"
} >> "$log" 2>&1
# sccache inherits fd 8 and would keep the flock after this script exits (as in nightly.sh)
command -v sccache > /dev/null 2>&1 && timeout 30 sccache --stop-server > /dev/null 2>&1
