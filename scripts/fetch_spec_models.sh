#!/usr/bin/env bash
# Downloads the ADR 0043 draft model (Qwen3-0.6B Q8_0, same tokenizer as the Qwen3 MoE targets) and, with
# SPEC_FETCH_30B=1, Qwen3-30B-A3B Q4_K_M for quick iterations. Ends with SPEC_FETCH_DONE rc=N in
# runs/vast/fetch_spec.log. Started detached next to scripts/fetch_exit_model.sh.
exec 8> /root/tetraeder_fetch_spec.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
log=runs/vast/fetch_spec.log
echo "=== RUN $(date +%FT%T) fetch_spec_models.sh ===" >> "$log"
spec="Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf"
[ "${SPEC_FETCH_30B:-0}" = 1 ] && spec="$spec,Qwen/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf"
{
  echo "[$(date +%T)] start fetch $spec"
  TAXAMETER_DIR="$PWD" bash "${TAXAMETER_HOME:-/root/.taxameter}/box.sh" models "$spec"
  rc=$?
  echo "[$(date +%T)] SPEC_FETCH_DONE rc=$rc"
} >> "$log" 2>&1
