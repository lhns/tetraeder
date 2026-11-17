#!/usr/bin/env bash
# GPU session sampler (ADR 0065, addendum "BoardLimit and Reliability bits judged under load").
# Writes $OUT/gpu-samples.csv at 1 Hz, no header, one line per GPU per second:
#   epoch,timestamp,reasons,power_w,sm_mhz,mem_mhz,temp_c
# epoch is `date +%s.%N` when the line was read; timestamp is nvidia-smi's own "YYYY/MM/DD HH:MM:SS.mmm" (box local
# time, the same clock and zone as `started` in the bench JSONs, so phase4_summary.py matches windows on it);
# reasons is the clock event reasons bitmask (hex).
# Env: OUT (required, results dir), SAMPLER_PARENT (optional pid; the sampler ends when it is gone).
# Start detached: `OUT=... setsid nohup bash scripts/gpu_sampler.sh 8>&- > /dev/null 2>&1 &`.
# Stops when $OUT/gpu-sampler.stop exists (checked per sample, so within about 1 s).
exec 8> "${SAMPLER_LOCK:-/root/tetraeder_gpu_sampler.lock}" # SAMPLER_LOCK: tests only
flock -n 8 || exit 0
: "${OUT:?OUT (results dir) is required}"
mkdir -p "$OUT"
stop="$OUT/gpu-sampler.stop"
csv="$OUT/gpu-samples.csv"
rm -f "$stop"
q=timestamp,clocks_event_reasons.active,power.draw,clocks.sm,clocks.mem,temperature.gpu
# Older drivers only know clocks_throttle_reasons.
if ! nvidia-smi --query-gpu="$q" --format=csv,noheader,nounits > /dev/null 2>&1; then
  q=${q/clocks_event_reasons/clocks_throttle_reasons}
fi
while IFS= read -r line; do
  [ -e "$stop" ] && break
  if [ -n "${SAMPLER_PARENT:-}" ] && ! kill -0 "$SAMPLER_PARENT" 2> /dev/null; then break; fi
  # "ts, 0x..., 28.1, 210, 405, 31" -> "epoch,ts,0x...,28.1,210,405,31"
  printf '%s,%s\n' "$(date +%s.%N)" "${line//, /,}" >> "$csv"
done < <(nvidia-smi --query-gpu="$q" --format=csv,noheader,nounits -lms 1000 2> /dev/null)
# nvidia-smi ends on SIGPIPE at its next write
exit 0
