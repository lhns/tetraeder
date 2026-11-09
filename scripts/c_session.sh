#!/usr/bin/env bash
# Track C box session: the planner campaign, then the detector campaign on the same box (one fetch of each model).
# Started detached; each campaign has its own lock, log and DONE line. Ends with C_SESSION_DONE in runs/vast/c_session.log.
exec 9> /root/tetraeder_c_session.lock; flock -n 9 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
rc=0
{
  # TUNE=1: llama.cpp's expert-cache tuner on the 30B first (needs the baseline build in the box image); off by default.
  # The tuner needs the 30B, which the planner campaign would only fetch later; it has its own time limit (TUNE_MIN).
  if [ "${TUNE:-0}" = 1 ]; then
    f30="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
    [ -s "$f30" ] || TAXAMETER_DIR="$PWD" bash "${TAXAMETER_HOME:-/root/.taxameter}/box.sh" models \
      "Qwen/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf" 9>&-
    echo "[$(date +%T)] moe cache tune"
    bash scripts/moe_cache_tune.sh 9>&-
    r=$?
    echo "[$(date +%T)] moe cache tune rc=$r"
    [ "$r" = 0 ] || rc=$r
  fi
  echo "[$(date +%T)] planner campaign"
  bash scripts/planner_campaign.sh 9>&- || rc=$?
  echo "[$(date +%T)] detector campaign"
  bash scripts/detector_campaign.sh 9>&- || rc=$?
  echo "[$(date +%T)] C_SESSION_DONE rc=$rc"
} >> runs/vast/c_session.log 2>&1
exit "$rc"
