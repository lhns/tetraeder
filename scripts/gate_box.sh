#!/usr/bin/env bash
# One detached launch of the regression-gate session (node bench-regression-gate): the A/B gate and TTFT
# (scripts/ab_moe_box.sh), then the extra stages (scripts/gate_extra_box.sh: llama-server CUDA build, parity on CUDA,
# Qwen3.6 default flags). Nothing overlaps a judged run. Env: SESSION (set it: the run can cross midnight), A_COMMIT, COMMIT,
# A_TAR, STAGES (extra stages), the knobs of both scripts. Ends with GATE_BOX_DONE in runs/vast/gate_box.log.
exec 9> /root/tetraeder_gate_box.lock
flock -n 9 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
: "${SESSION:?set SESSION}"
export SESSION
{
  echo "[$(date +%T)] gate_box start session=$SESSION"
  bash scripts/ab_moe_box.sh 8>&-
  echo "[$(date +%T)] ab_moe_box done rc=$?"
  bash scripts/gate_extra_box.sh 8>&-
  echo "[$(date +%T)] gate_extra_box done rc=$?"
  echo "GATE_BOX_DONE"
} >> runs/vast/gate_box.log 2>&1
