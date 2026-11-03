#!/usr/bin/env bash
# Track B round 2 on one box (about 200 GB disk): both downloads and the ik_llama.cpp build in parallel, then
# conversion speed against llama-quantize on the 8B and the 30B (BF16 sources), the 30B profile with a BF16
# reference and its preset comparison (Q4_K_M, Q3_K_M, Q5_K_M), and ik_llama quant types on the 8B.
# Ends with ROUND2_DONE in runs/vast/quality-round2.log.
exec 9> /root/tetraeder_round2.lock; flock -n 9 || exit 0
cd "$(dirname "$0")/../.." || exit 1
mkdir -p runs/vast
log=runs/vast/quality-round2.log
say() { echo "[$(date +%T)] $*" >> "$log"; }
say start
(IK_JOBS=4 bash scripts/quality/ik_build.sh > runs/vast/ik_build.log 2>&1 9>&- &)
(MODEL=qwen3-30b-bf16 FETCH_ONLY=1 bash scripts/quality/box_quality.sh 9>&- &)
MODEL=qwen3-8b FETCH_ONLY=1 bash scripts/quality/box_quality.sh 9>&-
until grep -q "FETCH_DONE\|fetch failed" runs/vast/quality-qwen3-30b-bf16-fetch.log 2> /dev/null; do sleep 10; done
say "downloads done: $(tail -1 runs/vast/quality-qwen3-30b-bf16-fetch.log)"
# conversion speed first (nothing else runs then), 8B and 30B
MODEL=qwen3-8b STEPS=convert bash scripts/quality/box_quality.sh 9>&-
say "8B convert done"
MODEL=qwen3-30b-bf16 STEPS=convert bash scripts/quality/box_quality.sh 9>&-
say "30B convert done"
MODEL=qwen3-30b-bf16 STEPS="calibrate compare" bash scripts/quality/box_quality.sh 9>&-
say "30B quality done"
until grep -q IK_BUILD_DONE runs/vast/ik_build.log 2> /dev/null; do sleep 10; done
if grep -q "IK_BUILD_DONE rc=0" runs/vast/ik_build.log; then
  MODEL=qwen3-8b STEPS=compare TAG=ik PRESETS=Q4_K_M IK_TYPES="IQ4_KS IQ4_K" bash scripts/quality/box_quality.sh 9>&-
  say "ik compare done"
fi
say ROUND2_DONE
