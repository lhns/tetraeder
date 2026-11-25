#!/usr/bin/env bash
# Case F fault injector on node B (ADR 0091), run detached on the box: waits DELAY seconds, then injects MODE and logs the
# epoch (seconds, from `date +%s.%N`) of every step to logs/fault-MODE.log. Modes:
#   worker   kill -9 of the node's children (the replica engine); the daemon restarts it
#   daemon   kill -9 of the daemon and its children; RESTORE_AFTER seconds later the daemon is started again
#   stop     SIGSTOP of the daemon and its children (a silent node); SIGCONT after STOP_FOR seconds
# usage: phase6_fault.sh MODE DELAY   Env: STOP_FOR (60), RESTORE_AFTER (40), P6.
cd "$(dirname "$0")/.." || exit 1
mode="${1:?mode}"
delay="${2:?delay}"
P6="${P6:-/root/p6}"
log="$P6/logs/fault-$mode.log"
now() { date +%s.%N; }
echo "$(now) armed mode=$mode delay=$delay" >> "$log"
sleep "$delay"
pid=$(cat "$P6/pids/node")
case "$mode" in
  worker)
    echo "$(now) fault kill -9 children of node $pid: $(pgrep -P "$pid" | tr '\n' ' ')" >> "$log"
    for c in $(pgrep -P "$pid"); do kill -9 "$c" 2> /dev/null; done
    echo "$(now) fault done" >> "$log"
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
