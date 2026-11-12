#!/usr/bin/env bash
# Usage: clamp_threads.sh N [QUOTA]. Prints N, or the whole-core CPU quota when N exceeds it: threads above the quota
# get throttled (20261112a ran 24 on 23.04; ADR 0058 addendum 2026-11-12).
# QUOTA defaults to scripts/cpu_quota.sh when that finds a quota (no nproc fallback). THREADS_ALLOW_OVER_QUOTA=1
# keeps N. The clamp line goes to stderr.
n="$1"
quota="${2:-}"
if [ -z "$quota" ]; then
  out=$(bash "$(dirname "$0")/cpu_quota.sh" 2> /dev/null) || out=""
  # cpu_quota.sh warns on stderr and falls back to nproc when no quota is set: only a silent run is a quota
  [ -z "$(bash "$(dirname "$0")/cpu_quota.sh" 2>&1 > /dev/null)" ] && quota="$out"
fi
if [ -n "$quota" ] && [ "$n" -gt "$quota" ] 2> /dev/null && [ "${THREADS_ALLOW_OVER_QUOTA:-0}" != 1 ]; then
  echo "clamp_threads.sh: THREADS=$n exceeds the CPU quota of $quota cores, using $quota (THREADS_ALLOW_OVER_QUOTA=1 keeps $n)" >&2
  n="$quota"
fi
echo "$n"
