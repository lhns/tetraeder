#!/usr/bin/env bash
# Usage: kt_threads.sh "<stage list>" [QUOTA] [AVX512=0|1]. Prints the default thread count of a session: the CPU quota
# (scripts/cpu_quota.sh), or on an AVX-512 CPU whose stages include KTransformers, KT_HEADROOM (default 4) below it.
# KT busy-spins its worker pool (50 ms before it sleeps) and used 29.7 cores at --kt-cpuinfer 28, 30.35 with the other
# processes of the container, on a quota of 30.7 (20261113d): throttled in 32% of decode periods. At quota-4 it stays
# about 2 cores under the quota. Both engines get the same count (AGENTS.md); an explicit THREADS bypasses this script.
stages="$1"
quota="${2:-$(bash "$(dirname "$0")/cpu_quota.sh" 2> /dev/null)}"
avx512="${3:-$(grep -q -w avx512f /proc/cpuinfo 2> /dev/null && echo 1 || echo 0)}"
if [[ " $stages " == *" ktransformers"* ]] && [ "$avx512" = 1 ] && [ "$quota" -gt "${KT_HEADROOM:-4}" ] 2> /dev/null; then
  echo $((quota - ${KT_HEADROOM:-4}))
else
  echo "$quota"
fi
