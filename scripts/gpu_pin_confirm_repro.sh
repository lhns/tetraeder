#!/usr/bin/env bash
# Repro follow-up of gpu_pin_confirm.sh on the same box: gpu_multi_seq twice, then N repeats of the q8b tetraeder dense
# probe c1+c4 (the run that hit a CUDA illegal memory access in session 20261116pc). Results: <session>/pin-confirm/repro/.
# Started detached: `taxameter run --id <id> scripts/gpu_pin_confirm_repro.sh --env SESSION=<s>`; ends with REPRO_DONE.
exec 8> /root/tetraeder_pin_confirm_repro.lock
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
out="$root/bench/results/${SESSION:?}/pin-confirm/repro"
mkdir -p "$out" "$root/runs/vast"
exec >> "$root/runs/vast/pin_confirm_repro.log" 2>&1
say() { echo "[$(date +%T)] $*"; }
for i in 1 2; do
  say "gpu_multi_seq $i"
  TETRAEDER_REQUIRE_GPU=1 timeout -k 30 15m cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda \
    --test gpu_multi_seq -- --nocapture --test-threads 1 > "$out/gpu_multi_seq-$i.log" 2>&1 8>&-
  say "gpu_multi_seq $i rc=$?"
done
py=/root/gwvenv/bin/python
f="$root/.hf_cache/Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
threads=$(bash scripts/cpu_quota.sh 8>&-)
for i in $(seq 1 "${N:-4}"); do
  slog="$out/server-q8b-tetraeder-$i.log"
  say "q8b tetraeder run $i"
  RUST_LOG=info,tetraeder::step=debug setsid nohup "$root/target/release/tetraeder" serve -m "$f" --engine dense --ngl -1 \
    --parallel 4 --ctx 32768 -t "$threads" --port 18080 --alias qwen3-8b --accept-any-model > "$slog" 2>&1 8>&- &
  pid=$!
  for _ in $(seq 1 300); do
    curl -sf -o /dev/null http://127.0.0.1:18080/v1/models && break
    kill -0 $pid 2> /dev/null || break
    sleep 2
  done
  u=http://127.0.0.1:18080
  "$py" scripts/gateway_bench.py $u --model qwen3-8b --prompt-tokens 128 --max-tokens 16 --requests 1 --no-keepalive > /dev/null 2>&1
  for c in 1 4; do
    seed=91
    [ $c = 4 ] && seed=94
    timeout -k 30 5m "$py" scripts/gateway_bench.py $u --model qwen3-8b --temperature 0 --no-keepalive --label "repro-$i-c$c" \
      --out "$out/repro-$i-c$c.json" --prompt-tokens 512 --max-tokens 128 --concurrency $c --requests 8 --seed $seed >> "$out/bench.log" 2>&1
  done
  kill $pid 2> /dev/null
  sleep 3
  kill -9 $pid 2> /dev/null
  wait $pid 2> /dev/null
  say "run $i: $(grep -c 'CUDA error' "$slog") CUDA error lines"
  sleep 3
done
say REPRO_DONE
