#!/usr/bin/env bash
# Downloads the Phase 2 exit model (Qwen3-235B-A22B Q4_K_M, five shards, about 142 GB) with taxameter's box-side
# downloader (parallel ranged requests, ModelScope or Hugging Face, whichever probes faster). Started detached so
# GPU work can run meanwhile; ends with FETCH_DONE rc=N in runs/vast/fetch_exit.log.
exec 8> /root/tetraeder_fetch_exit.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
log=runs/vast/fetch_exit.log
q=Q4_K_M/Qwen3-235B-A22B-Q4_K_M
spec="Qwen/Qwen3-235B-A22B-GGUF:$q-00001-of-00005.gguf+$q-00002-of-00005.gguf+$q-00003-of-00005.gguf"
spec="$spec+$q-00004-of-00005.gguf+$q-00005-of-00005.gguf"
{
  echo "[$(date +%T)] start fetch, disk free $(df -h --output=avail . | tail -1)"
  TAXAMETER_DIR="$PWD" bash "${TAXAMETER_HOME:-/root/.taxameter}/box.sh" models "$spec"
  rc=$?
  echo "[$(date +%T)] FETCH_DONE rc=$rc"
} >> "$log" 2>&1
