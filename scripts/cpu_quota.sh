#!/usr/bin/env bash
# Prints the container's CPU quota in whole cores (cgroup v2 cpu.max, else v1 cfs quota/period), the thread count
# both engines get in paired runs (AGENTS.md). nproc shows every host core (80 on one box), so it is used only when
# neither cgroup version sets a quota.
awk '$1 != "max" { n = int($1 / $2); if (n > 0) { print n; found = 1 } } END { exit !found }' \
  /sys/fs/cgroup/cpu.max 2> /dev/null && exit 0
for d in /sys/fs/cgroup/cpu,cpuacct /sys/fs/cgroup/cpu; do
  q=$(cat "$d/cpu.cfs_quota_us" 2> /dev/null) && p=$(cat "$d/cpu.cfs_period_us" 2> /dev/null) || continue
  [ "${q:--1}" -gt 0 ] && [ "${p:-0}" -gt 0 ] && [ $((q / p)) -gt 0 ] && { echo $((q / p)); exit 0; }
done
# no quota in either cgroup version: the container may use every core nproc shows
echo "cpu_quota.sh: no CPU quota found, using nproc $(nproc)" >&2
nproc
