#!/usr/bin/env bash
# Track C box session: the planner campaign, then the detector campaign on the same box (one fetch of each model).
# Started detached; each campaign has its own lock, log and DONE line. Ends with C_SESSION_DONE in runs/vast/c_session.log.
exec 9> /root/tetraeder_c_session.lock; flock -n 9 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
{
  # TUNE=1: llama.cpp's expert-cache tuner on the 30B first (needs the baseline build in the box image); off by default
  if [ "${TUNE:-0}" = 1 ]; then
    echo "[$(date +%T)] moe cache tune"
    bash scripts/moe_cache_tune.sh 9>&-
  fi
  echo "[$(date +%T)] planner campaign"
  bash scripts/planner_campaign.sh 9>&-
  echo "[$(date +%T)] detector campaign"
  bash scripts/detector_campaign.sh 9>&-
  echo "[$(date +%T)] C_SESSION_DONE"
} >> runs/vast/c_session.log 2>&1
