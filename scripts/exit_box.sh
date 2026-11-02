#!/usr/bin/env bash
# The exit box's whole job, in order: Qwen3-30B-A3B GPU checks while the 235B downloads (moe_check, graphs_check,
# ot_pair2), then the exit run (exit_run.sh waits for fetch_exit_model.sh). Started detached by
# `taxameter run scripts/exit_box.sh` after `scripts/fetch_exit_model.sh` was started; ends with EXIT_BOX_DONE.
# BOX_STEPS picks the scripts (default all four); exit_run.sh reads STAGES.
exec 8> /root/tetraeder_exit_box.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
for s in ${BOX_STEPS:-moe_check graphs_check ot_pair2 exit_run}; do
  echo "[$(date +%T)] start $s" >> runs/vast/exit_box.log
  # each script takes its own lock on fd 8
  bash "scripts/$s.sh" 8>&-
  echo "[$(date +%T)] done $s" >> runs/vast/exit_box.log
done
echo "EXIT_BOX_DONE" >> runs/vast/exit_box.log
