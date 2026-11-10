#!/usr/bin/env bash
# log_done.sh <log> <marker>: exit 0 when <marker> appears after the last "=== RUN" header of <log>. Logs append across
# relaunches; a marker from an earlier launch must not end a wait. In a wait loop on the box:
#   for i in $(seq 100); do bash scripts/log_done.sh runs/vast/session-b1-exit.log SESSION_DONE && break; sleep 5; done
awk -v m="$2" '/^=== RUN /{f=0} index($0,m){f=1} END{exit !f}' "$1" 2> /dev/null
