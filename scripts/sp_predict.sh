#!/usr/bin/env bash
# Session stream-predict: the frozen predictions of both arms, written BEFORE any measured run (tree node
# bench-235b-stream-predict, ADR 0091 "named risk"). Needs the model (fetch done), the builds of sp_prep.sh and the routing
# trace (bench/traces on the box, put from the main checkout's runs/traces).
#   disk-curve   the box's NVMe queue curve (idle), the one measured input
#   layout       tetraeder plan layout of the real GGUF
#   per arm (D: RAM budget 64 GB as EMULATE_RAM_GB=64; R: the container's whole RAM): the hardware graph (scripts/sp_make_hw.py),
#                the planner's pick and prediction under the uniform routing prior, a routing replay of the picked slot pair
#                on the recorded trace (tetraeder-bench replay), and the prediction with decode priced from it
#   freeze       sha256 of every input and output (SHA256SUMS), and the sha256 of that file (FROZEN.sha256)
# Writes only bench/results/<name>/predictions/. sp_arms.sh refuses to run unless FROZEN_SHA equals FROZEN.sha256.
# Env: RESULTS_NAME, THREADS, TRACE, RAM_D_BYTES (64e9). Ends with SP_PREDICT_DONE in runs/vast/sp_predict.log.
exec 8> /root/tetraeder_sp_predict.lock
flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
root=$PWD
name="${RESULTS_NAME:?set RESULTS_NAME}"
p="$root/bench/results/$name/predictions"
mkdir -p "$p" runs/vast
log=runs/vast/sp_predict.log
say() { echo "[$(date +%T)] $*"; }
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:/root/.local/bin:$PATH"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
quota=$(bash scripts/cpu_quota.sh 2> /dev/null || nproc)
threads="${THREADS:-$quota}"
model="$root/.hf_cache/Qwen__Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf"
big_shard="$root/.hf_cache/Qwen__Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00002-of-00005.gguf"
trace="${TRACE:-$root/bench/traces/routing-235b-20261114-041805.trace}"
tet="$root/target/release/tetraeder"
plan="$root/target/release/tetraeder-plan"
bench="$root/target/release/tetraeder-bench"
prior="$root/crates/planner/fixtures/rtx3090_prior.toml"
gpu_real=$(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) * 1048576 ))
# arm D is a 3090-class card (the prior's 24 GiB) whatever the box has; arm R may use the whole card
gpu_d="${GPU_D_BYTES:-25769803776}"
mem=$(cat /sys/fs/cgroup/memory.max 2> /dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2> /dev/null)
total=$(awk '/MemTotal/ { print $2 * 1024 }' /proc/meminfo)
case "$mem" in '' | max) mem=$total ;; esac
[ "$mem" -gt "$total" ] && mem=$total
ram_d="${RAM_D_BYTES:-64000000000}"
die() { say "FATAL: $*"; echo "SP_PREDICT_DONE rc=1"; exit 1; }

main() {
  say "sp_predict start threads=$threads gpu_real=$gpu_real gpu_d=$gpu_d mem=$mem ram_d=$ram_d"
  for f in "$model" "$tet" "$plan" "$bench" "$trace"; do [ -e "$f" ] || die "missing $f"; done
  "$tet" plan layout "$model" --out "$p/layout.json" || die "layout"
  # the disk curve beside nothing else: idle disk and CPU
  "$bench" disk-curve --file "$big_shard" --depths 1,2,4,8,16,32 --seconds 4 --out "$p/disk-curve.json" || die "disk-curve"
  local arm ram
  for arm in D R; do
    ram=$ram_d
    gpu_bytes=$gpu_d
    pin=()
    if [ "$arm" = R ]; then
      ram=$mem
      gpu_bytes=$gpu_real
      [ -n "${PIN_R:-}" ] && pin=(--pin "$PIN_R")
    fi
    python3 -I scripts/sp_make_hw.py --prior "$prior" --ram-bytes "$ram" --gpu-bytes "$gpu_bytes" --threads "$threads" \
      --disk-curve "$p/disk-curve.json" --out "$p/hw-$arm.toml" || die "hw $arm"
    python3 -I scripts/sp_make_hw.py --prior "$prior" --ram-bytes "$ram" --gpu-bytes "$gpu_bytes" --threads "$threads" \
      --keep-disk-prior --out "$p/hw-$arm-diskprior.toml" || die "hw $arm prior disk"
    # the pick under the uniform routing prior, on the measured disk curve (the arm's frozen config) and on the prior disk
    "$plan" node-predict --hw "$p/hw-$arm.toml" --layout "$p/layout.json" "${pin[@]}" --out "$p/pred-$arm-uniform.json" || die "predict $arm uniform"
    "$plan" node-predict --hw "$p/hw-$arm-diskprior.toml" --layout "$p/layout.json" "${pin[@]}" --out "$p/pred-$arm-uniform-diskprior.json" || die "predict $arm uniform diskprior"
    # the picked pair on the recorded routing trace
    local pair
    pair=$(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1]))['pick']; print(f\"{d['vram_slots']},{d['ram_slots']}\")" "$p/pred-$arm-uniform.json")
    say "arm $arm pick $pair"
    "$bench" replay --trace "$trace" --vram "${pair%,*}" --ram "${pair#*,}" --policies lru --vram-policy static \
      --out "$p/replay-$arm.json" || say "replay $arm failed (no trace-based prediction for this arm)"
    if [ -s "$p/replay-$arm.json" ]; then
      "$plan" node-predict --hw "$p/hw-$arm.toml" --layout "$p/layout.json" --pin "$pair" --tier-replay "$p/replay-$arm.json" \
        --out "$p/pred-$arm-trace.json" || say "predict $arm trace failed"
    fi
  done
  python3 -I -c "
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest(), 'routing trace')
" "$trace" > "$p/trace.sha256"
  (cd "$p" && sha256sum layout.json disk-curve.json hw-*.toml pred-*.json replay-*.json trace.sha256 > SHA256SUMS && sha256sum SHA256SUMS > FROZEN.sha256)
  cat "$p/FROZEN.sha256"
  python3 -I scripts/sp_table.py predictions "$p" || true
  say "SP_PREDICT_DONE rc=0"
}
main >> "$log" 2>&1
