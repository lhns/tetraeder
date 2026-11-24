#!/usr/bin/env bash
# Session stream-predict, second freeze for arm D (tree node bench-235b-stream-predict). The first freeze's pick for arm D
# (16 VRAM slots, 56 RAM slots, a 60.07 GB RAM tier under the 64 GB budget) was OOM-killed on every round before the first
# token: the process took the 64 GB (anon 60.9 GB at the kill, still rising). The planner's host footprint leaves about 3.2 GB
# for everything else; the run needed more. This script prices the pick again with the RAM budget 5 GB lower (59 GB, RAM_D2_BYTES),
# on the same layout, disk curve and routing trace, and freezes it in predictions/d2/ before arm D2 runs. Nothing else changes.
# The planner's RAM slot options are a quarter, half and all of the experts outside VRAM (28, 56, 112), so a budget that drops the
# pick to the next option halves the tier. PIN_D2 ("16,52") instead keeps the failed pick's VRAM slots and lowers its RAM slots by
# 4 (4.3 GB more headroom), the largest tier that was expected to fit; the budget stays 64 GB (RAM_D2_BYTES 64000000000).
# Needs predictions/ of sp_predict.sh. Env: RESULTS_NAME, TRACE, RAM_D2_BYTES, THREADS. Ends with SP_PREDICT_D2_DONE.
exec 8> /root/tetraeder_sp_predict_d2.lock
flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
root=$PWD
name="${RESULTS_NAME:?set RESULTS_NAME}"
p="$root/bench/results/$name/predictions"
d="$p/d2"
mkdir -p "$d" runs/vast
log=runs/vast/sp_predict_d2.log
say() { echo "[$(date +%T)] $*"; }
quota=$(bash scripts/cpu_quota.sh 2> /dev/null || nproc)
threads="${THREADS:-$quota}"
trace="${TRACE:?set TRACE}"
plan="$root/target/release/tetraeder-plan"
bench="$root/target/release/tetraeder-bench"
prior="$root/crates/planner/fixtures/rtx3090_prior.toml"
ram="${RAM_D2_BYTES:-59000000000}"
pin=()
[ -n "${PIN_D2:-}" ] && pin=(--pin "$PIN_D2")
gpu_d="${GPU_D_BYTES:-25769803776}"
die() { say "FATAL: $*"; echo "SP_PREDICT_D2_DONE rc=1"; exit 1; }
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

main() {
  say "sp_predict_d2 start ram=$ram pin=${PIN_D2:-none} threads=$threads"
  (cd "$p" && sha256sum -c FROZEN.sha256 && sha256sum -c SHA256SUMS) > /dev/null 2>&1 || die "the first freeze changed"
  python3 -I scripts/sp_make_hw.py --prior "$prior" --ram-bytes "$ram" --gpu-bytes "$gpu_d" --threads "$threads" \
    --disk-curve "$p/disk-curve.json" --out "$d/hw-D2.toml" || die "hw"
  "$plan" node-predict --hw "$d/hw-D2.toml" --layout "$p/layout.json" "${pin[@]}" --out "$d/pred-D2-uniform.json" || die "predict uniform"
  local pair
  pair=$(python3 -I -c "import json,sys; d=json.load(open(sys.argv[1]))['pick']; print(f\"{d['vram_slots']},{d['ram_slots']}\")" "$d/pred-D2-uniform.json")
  say "arm D2 pick $pair"
  "$bench" replay --trace "$trace" --vram "${pair%,*}" --ram "${pair#*,}" --policies lru --vram-policy static --out "$d/replay-D2.json" \
    || say "replay failed (no trace-based prediction)"
  if [ -s "$d/replay-D2.json" ]; then
    "$plan" node-predict --hw "$d/hw-D2.toml" --layout "$p/layout.json" --pin "$pair" --tier-replay "$d/replay-D2.json" \
      --out "$d/pred-D2-trace.json" || say "predict trace failed"
  fi
  (cd "$d" && sha256sum ../layout.json ../disk-curve.json ../trace.sha256 hw-D2.toml pred-D2-*.json replay-D2.json > SHA256SUMS && sha256sum SHA256SUMS > FROZEN.sha256)
  cat "$d/FROZEN.sha256"
  python3 -I scripts/sp_table.py predictions "$d" || true
  say "SP_PREDICT_D2_DONE rc=0"
}
main >> "$log" 2>&1
