#!/usr/bin/env bash
# Session segment-confirm (tree node bench-235b-segment-confirm): the RAM balloon of ADR 0092 section 4 as its own detached
# process, so it survives the pause between the frozen prediction and the judged rounds.
#   start   hold (memory.max - EMULATE_RAM_GB) as touched anonymous memory, wait until it holds the FULL target
#           (balloon_guard.py wait), then read the planner's RAM budget from cgroup v2 (balloon_guard.py budget) into
#           $RESULTS/env.json and, when HW names a graph, into its cpu0 capacity_bytes. Prints BALLOON_READY.
#   stop    end the balloon.
#   check   print ok or the reason (exit 1): the balloon is alive and holds its target.
# Env: RESULTS (bench/results/<name>), EMULATE_RAM_GB (64; 0 = no balloon, the budget is not read), HW, CGROUP_DIR,
# BALLOON_FULL_WAIT_S (1800). State in runs/sc/balloon.{pid,target}. Log runs/sc/balloon.log.
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/sc
log=runs/sc/balloon.log
say() { echo "[$(date +%T)] $*"; }
emulated=$((${EMULATE_RAM_GB:-64} * 1000000000))
cmd="${1:?start|stop|check}"

case "$cmd" in
  start)
    exec 8> /root/tetraeder_sc_balloon.lock
    flock -n 8 || { echo "balloon: another start is running"; exit 0; }
    if [ "${EMULATE_RAM_GB:-64}" = 0 ]; then say "no balloon (EMULATE_RAM_GB=0)" | tee -a "$log"; echo BALLOON_READY; exit 0; fi
    if [ -s runs/sc/balloon.pid ] && python3 -I scripts/balloon_guard.py check --pid "$(cat runs/sc/balloon.pid)" \
      --target "$(cat runs/sc/balloon.target)" > /dev/null 2>&1; then
      say "balloon already running" | tee -a "$log"
      echo BALLOON_READY
      exit 0
    fi
    total=$(awk '/MemTotal/ { print $2 * 1024 }' /proc/meminfo)
    mem=$(cat /sys/fs/cgroup/memory.max 2> /dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2> /dev/null)
    case "$mem" in '' | max) mem=$total ;; esac
    [ "$mem" -gt "$total" ] && mem=$total
    [ "$mem" -gt "$emulated" ] || { say "FATAL: memory $mem is not above the emulated $emulated" | tee -a "$log"; exit 1; }
    n=$((mem - emulated))
    say "balloon $n B (memory $mem, emulated $emulated)" | tee -a "$log"
    setsid nohup python3 -c "
import sys, time
n = int(sys.argv[1])
b = bytearray(n)
b[::4096] = b'\x01' * len(range(0, n, 4096))
print('balloon holds', n, 'B', flush=True)
time.sleep(10**9)
" "$n" >> "$log" 2>&1 8>&- &
    pid=$!
    echo "$pid" > runs/sc/balloon.pid
    echo "$n" > runs/sc/balloon.target
    if echo -1000 > /proc/$pid/oom_score_adj 2> /dev/null; then say "balloon pid $pid oom_score_adj -1000" | tee -a "$log"; else say "balloon pid $pid: oom_score_adj -1000 not allowed" | tee -a "$log"; fi
    python3 -I scripts/balloon_guard.py wait --pid "$pid" --target "$n" --timeout "${BALLOON_FULL_WAIT_S:-1800}" | tee -a "$log"
    [ "${PIPESTATUS[0]}" = 0 ] || { say "FATAL: the balloon did not reach its full target" | tee -a "$log"; exit 1; }
    mkdir -p "${RESULTS:-runs/sc}"
    python3 -I scripts/balloon_guard.py budget --cgroup "${CGROUP_DIR:-/sys/fs/cgroup}" --env-json "${RESULTS:-runs/sc}/env.json" \
      ${HW:+--hw "$HW"} | tee -a "$log"
    [ "${PIPESTATUS[0]}" = 0 ] || { say "FATAL: RAM budget not measured" | tee -a "$log"; exit 1; }
    echo BALLOON_READY
    ;;
  stop)
    [ -s runs/sc/balloon.pid ] && kill "$(cat runs/sc/balloon.pid)" 2> /dev/null
    rm -f runs/sc/balloon.pid runs/sc/balloon.target
    say "balloon stopped" | tee -a "$log"
    ;;
  check)
    [ "${EMULATE_RAM_GB:-64}" = 0 ] && { echo ok; exit 0; }
    [ -s runs/sc/balloon.pid ] || { echo "no balloon"; exit 1; }
    python3 -I scripts/balloon_guard.py check --pid "$(cat runs/sc/balloon.pid)" --target "$(cat runs/sc/balloon.target)"
    ;;
  *) echo "usage: sc_balloon.sh start|stop|check"; exit 2 ;;
esac
