#!/usr/bin/env bash
# Idle watchdog for a Vast.ai instance: stop the instance (halts GPU billing; disk is kept and still billed) after
# IDLE_MIN consecutive idle minutes. Runs ON the instance, authenticated by the Vast-injected CONTAINER_API_KEY.
#   nohup setsid bash scripts/idle_autostop.sh > /root/autostop.log 2>&1 &
# Busy = a cargo/rustc/nvcc/cmake/ninja process, an engine or bench binary under target/, bench.sh, nightly.sh or a
# llama-* tool, a vast_box.sh provision/models run (downloads and installs look idle), or GPU utilisation above BUSY_UTIL percent. The patterns are anchored on purpose: a bare "tetraeder"
# would match this script's own path (/root/tetraeder/scripts/...) and the box would never go idle.
# The vastai CLI only takes the key as --api-key (no header file or stdin), so it stays on argv.
# Restarting a stopped instance needs the Vast console (or an account API key); this script never destroys anything.
# Kept alive across container restarts by scripts/vast_box.sh ensure-watchdog (supervisor / onstart), see vastctl.py.
IDLE_MIN="${IDLE_MIN:-20}"
BUSY_UTIL="${BUSY_UTIL:-10}"
VASTAI="${VASTAI:-/opt/instance-tools/bin/vastai}"
idle=0
if [ -z "${CONTAINER_API_KEY:-}" ]; then
  echo "$(date +%T) ERROR: CONTAINER_API_KEY is not set: the stop call will fail and this box will never auto-stop"
fi
echo "$(date +%T) autostop armed: stop after ${IDLE_MIN} idle minutes (instance ${CONTAINER_ID})"
while true; do
  busy=0
  pgrep -f '(^|/)(cargo|rustc|nvcc|cmake|ninja)( |$)|target/(release|debug)/tetraede[r]|bench[.]s[h]|nightl[y][.]sh|vast_bo[x][.]sh (provision|models)|(^|/)llama-' > /dev/null && busy=1
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -nr | head -1)
  [ -n "$util" ] && [ "$util" -gt "$BUSY_UTIL" ] && busy=1
  if [ "$busy" = 1 ]; then idle=0; else idle=$((idle + 1)); fi
  echo "$(date +%T) busy=$busy util=${util:-?} idle_min=$idle"
  if [ "$idle" -ge "$IDLE_MIN" ]; then
    # Never exit after the stop call: when the host's network is down the call fails (seen on host 483833), so
    # keep retrying every minute while still idle. A successful stop ends this process with the container.
    echo "$(date +%T) idle for ${idle} min -> stopping instance ${CONTAINER_ID}"
    if "$VASTAI" stop instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY" > /tmp/autostop_call.log 2>&1; then
      echo "$(date +%T) stop call returned ok; retrying in 60 s if still running"
    else
      echo "$(date +%T) stop call FAILED (network?): $(tail -1 /tmp/autostop_call.log); retrying in 60 s"
    fi
  fi
  sleep 60
done
