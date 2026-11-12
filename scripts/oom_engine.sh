#!/usr/bin/env bash
# oom_engine.sh CMD...: run CMD as an OOM-kill candidate before the RAM balloon (exit_run.sh). Raising oom_score_adj
# is always allowed; the engine inherits it through exec.
echo 1000 > /proc/$$/oom_score_adj 2> /dev/null || echo "oom_engine: could not set oom_score_adj" >&2
exec "$@"
