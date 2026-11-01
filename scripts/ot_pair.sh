#!/usr/bin/env bash
# Paired baseline for the builder's CPU expert policy (ADR 0035): llama-bench with expert offload (`-ot`) and full GPU
# at real_moe's sizes (64-token prompt, 16 decode tokens) on this box, then scripts/moe_check.sh. Started detached by
# `taxameter run scripts/ot_pair.sh`; ends with OT_PAIR_DONE rc=N in its log.
exec 8> /root/tetraeder_ot_pair.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
date_dir="$PWD/bench/results/$(date +%Y%m%d)"
mkdir -p "$date_dir" runs/vast
log=runs/vast/ot_pair.log
model="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
bench="${LLAMA_BENCH:-/opt/llama.cpp/bin/llama-bench}"
upper='blk\.(2[4-9]|3[0-9]|4[0-7])\.ffn_(up|down|gate)_exps=CPU'
{
  echo "[$(date +%T)] start llama-bench, threads $(nproc)"
  rc=0
  for v in full ot-all ot-upper; do
    case $v in
      full) ot=() ;;
      ot-all) ot=(-ot exps=CPU) ;;
      ot-upper) ot=(-ot "$upper") ;;
    esac
    "$bench" -m "$model" -ngl 99 -fa 1 -p 64 -n 16 -r 3 -t "$(nproc)" "${ot[@]}" -o json \
      > "$date_dir/ot-pair-$v.json" || rc=$?
  done
  echo "[$(date +%T)] llama-bench rc=$rc"
} >> "$log" 2>&1
# fd 8 is this script's lock; moe_check takes its own
bash scripts/moe_check.sh 8>&-
echo "OT_PAIR_DONE rc=$rc" >> "$log"
