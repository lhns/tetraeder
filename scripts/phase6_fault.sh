#!/usr/bin/env bash
# Case F fault injector on node B (ADR 0091), run detached on the box: waits DELAY seconds, then injects MODE and logs the
# epoch (seconds, from `date +%s.%N`) of every step to logs/fault-MODE.log. Modes:
#   worker   kill -9 of the node's children (the replica engine); the daemon restarts it. The restore is the moment a new
#            child of the daemon exists (`restore child-up`, polled every POLL_S seconds for up to 60 s)
#   daemon   kill -9 of the daemon and its children; RESTORE_AFTER seconds later the daemon is started again (`restore start daemon`)
#   stop     SIGSTOP of the daemon and its children (a silent node); SIGCONT after STOP_FOR seconds (`restore SIGCONT`)
# usage: phase6_fault.sh MODE DELAY   Env: FAULT_LOG (the log of this trial), STOP_FOR (60), RESTORE_AFTER (40), P6, POLL_S (0.2), TARGET_EPOCH (fire at this
# epoch of this box's clock instead of after DELAY seconds: the driver converts a time on the load's clock, so ssh latency
# does not time the fault).
cd "$(dirname "$0")/.." || exit 1
mode="${1:?mode}"
delay="${2:?delay}"
P6="${P6:-/root/p6}"
log="${FAULT_LOG:-$P6/logs/fault-$mode.log}"
now() { date +%s.%N; }
echo "$(now) armed mode=$mode delay=$delay target=${TARGET_EPOCH:-none}" >> "$log"
if [ -n "${TARGET_EPOCH:-}" ]; then
  wait_s=$(awk -v t="$TARGET_EPOCH" -v n="$(now)" 'BEGIN { d = t - n; if (d < 0) { d = 0; print "late" > "/dev/stderr" } printf "%.3f", d }' 2>> "$log")
  sleep "$wait_s"
else
  sleep "$delay"
fi
pid=$(cat "$P6/pids/node")
case "$mode" in
  worker)
    echo "$(now) fault kill -9 children of node $pid: $(pgrep -P "$pid" | tr '\n' ' ')" >> "$log"
    old=$(pgrep -P "$pid" | tr '\n' ' ')
    for c in $old; do kill -9 "$c" 2> /dev/null; done
    echo "$(now) fault done" >> "$log"
    # the restore: a child of the daemon that is not one of the killed
    for _ in $(seq 1 300); do
      new=""
      for c in $(pgrep -P "$pid"); do case " $old " in *" $c "*) ;; *) new="$new $c" ;; esac; done
      if [ -n "$new" ]; then
        echo "$(now) restore child-up$new" >> "$log"
        break
      fi
      sleep "${POLL_S:-0.2}"
    done
    ;;
  daemon)
    echo "$(now) fault kill -9 daemon $pid and children: $(pgrep -P "$pid" | tr '\n' ' ')" >> "$log"
    for c in $(pgrep -P "$pid"); do kill -9 "$c" 2> /dev/null; done
    kill -9 "$pid" 2> /dev/null
    echo "$(now) fault done" >> "$log"
    sleep "${RESTORE_AFTER:-40}"
    echo "$(now) restore start daemon" >> "$log"
    rm -f "$P6"/node-state/*
    bash scripts/phase6_box.sh node >> "$log" 2>&1
    echo "$(now) restore done" >> "$log"
    ;;
  stop)
    kids=$(pgrep -P "$pid" | tr '\n' ' ')
    echo "$(now) fault SIGSTOP daemon $pid and children: $kids" >> "$log"
    for c in $kids; do kill -STOP "$c" 2> /dev/null; done
    kill -STOP "$pid" 2> /dev/null
    echo "$(now) fault done" >> "$log"
    sleep "${STOP_FOR:-60}"
    echo "$(now) restore SIGCONT" >> "$log"
    kill -CONT "$pid" 2> /dev/null
    for c in $kids; do kill -CONT "$c" 2> /dev/null; done
    echo "$(now) restore done" >> "$log"
    ;;
  *)
    echo "unknown mode $mode" >> "$log"
    exit 2
    ;;
esac
