#!/usr/bin/env bash
# Phase 3a planner exit-test measurements (ADR 0040), runs ON the box, started detached (taxameter run).
# Per fixture: VRAM balloon to the fixture's free VRAM, GGUF layout, the planner's enumerated candidates and bench spec
# (anchors, candidates, llama.cpp fit and hand-tuned -ot baselines), then one paired harness run.
# Results: bench/results/<date>/planner-<fixture>/ (layout, candidates, spec, harness JSON, vram and env notes).
# Log runs/vast/planner_campaign.log, ends with PLANNER_DONE rc=N. FIXTURES (default "moe dense") selects and orders
# fixtures, so a relaunch can skip finished ones. Time limit per fixture: FIXTURE_MIN (default 70).
exec 8> /root/tetraeder_planner_campaign.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
bash scripts/ensure_patches.sh 8>&- || exit 1  # ggml patch queue (ADR 0052)
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
mkdir -p runs/vast
log=runs/vast/planner_campaign.log
date_dir="$PWD/bench/results/$(date +%Y%m%d)"
mkdir -p "$date_dir"
models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"
rc=0
hold_pid=
finish() {
  [ -n "$hold_pid" ] && kill "$hold_pid" 2> /dev/null
  echo "rc=$rc" > "$date_dir/DONE"
  echo "[$(date +%T)] PLANNER_DONE rc=$rc"
  command -v sccache > /dev/null 2>&1 && timeout 30 sccache --stop-server > /dev/null 2>&1
}
trap finish EXIT
exec >> "$log" 2>&1

if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
echo "[$(date +%T)] start: cuda $CUDA_ENV_STATUS, threads $threads, archs ${TETRAEDER_CUDA_ARCHS:-default}"
[ "$CUDA_ENV_STATUS" = fail ] && { rc=13; exit; }
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active --format=csv

# fixture table: key, hf spec, file under models_dir, free VRAM the engines get (MiB)
declare -A HF FILE FREE
HF[moe]="Qwen/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf"
FILE[moe]="Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
FREE[moe]="${FREE_MOE_MIB:-14336}"
HF[dense]="Qwen/Qwen3-32B-GGUF:Qwen3-32B-Q4_K_M.gguf"
FILE[dense]="Qwen__Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf"
FREE[dense]="${FREE_DENSE_MIB:-18432}"
fixtures="${FIXTURES:-moe dense}"

# downloads run while the build runs; one fetch per model, in sequence
(
  for f in $fixtures; do
    [ -s "$models_dir/${FILE[$f]}" ] && { echo "[$(date +%T)] have $f"; continue; }
    echo "[$(date +%T)] fetch $f"
    TAXAMETER_DIR="$PWD" bash "${TAXAMETER_HOME:-/root/.taxameter}/box.sh" models "${HF[$f]}" || echo "fetch $f failed"
  done
  echo "[$(date +%T)] FETCH_DONE"
) 8>&- &
fetch_pid=$!

echo "[$(date +%T)] start build"
timeout -k 30 40m cargo build --release -p tetraeder -p tetraeder-bench -p tetraeder-planner \
  --features tetraeder/cuda 8>&- || { rc=10; exit; }
echo "[$(date +%T)] done build"
wait "$fetch_pid"
mem=$(cat /sys/fs/cgroup/memory.max 2> /dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2> /dev/null)
total=$(awk '/MemTotal/ { print $2 * 1024 }' /proc/meminfo)
case "$mem" in '' | max) mem=$total ;; esac
[ "$mem" -gt "$total" ] && mem=$total
ram_bytes=$((mem - 8000000000))
echo "[$(date +%T)] container memory $mem B, ram budget $ram_bytes B"

for f in $fixtures; do
  gguf="$models_dir/${FILE[$f]}"
  out="$date_dir/planner-$f"
  mkdir -p "$out"
  [ -s "$gguf" ] || { echo "no model for $f"; rc=20; continue; }
  echo "[$(date +%T)] fixture $f: balloon to ${FREE[$f]} MiB free"
  python3 scripts/vram_hold.py --leave-free-mib "${FREE[$f]}" > "$out/vram_hold.log" 2>&1 8>&- &
  hold_pid=$!
  for _ in $(seq 1 60); do grep -q 'holding' "$out/vram_hold.log" && break; sleep 2; done
  cat "$out/vram_hold.log"
  free_mib=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  echo "free_mib=$free_mib threads=$threads ram_bytes=$ram_bytes" > "$out/fixture.txt"
  target/release/tetraeder plan layout "$gguf" --out "$out/layout.json" || { rc=21; kill "$hold_pid"; hold_pid=; continue; }
  target/release/tetraeder-plan candidates --layout "$out/layout.json" --gpu-bytes $((free_mib * 1048576)) \
    --ram-bytes "$ram_bytes" --threads "$threads" --prompt 2048 --gen 128 --model-key "$f" --hf "${HF[$f]}" \
    --out-candidates "$out/candidates.json" --out-spec "$out/spec.toml" || { rc=22; kill "$hold_pid"; hold_pid=; continue; }
  cat "$gguf" > /dev/null  # page cache warm, so the first sample does not include the disk read
  echo "[$(date +%T)] start bench $f"
  timeout -k 30 "${FIXTURE_MIN:-70}m" target/release/tetraeder-bench run "$out/spec.toml" --out-dir "$out" \
    --llama-bench "${LLAMA_BENCH:-/opt/llama.cpp/bin/llama-bench}" --models-dir "$models_dir" 8>&-
  r=$?
  echo "[$(date +%T)] done bench $f rc=$r"
  # rc 2 means some pair errored (e.g. a candidate out of memory): recorded in the JSON, not fatal here
  [ "$r" -ne 0 ] && [ "$r" -ne 2 ] && rc=$r
  kill "$hold_pid" 2> /dev/null; wait "$hold_pid" 2> /dev/null; hold_pid=
done
