#!/usr/bin/env bash
# Phase 3a against ik_llama.cpp quant types (m1-open item 7): the committed Qwen3-8B profile mix against Q4_K_M,
# IQ4_KS and IQ4_K at matched bytes. ik_llama's quantizer builds while the BF16 source downloads.
# Ends with IK_QUALITY_DONE in runs/vast/quality-ik.log.
exec 9> /root/tetraeder_ik_quality.lock; flock -n 9 || exit 0
cd "$(dirname "$0")/../.." || exit 1
mkdir -p runs/vast
log=runs/vast/quality-ik.log
say() { echo "[$(date +%T)] $*" >> "$log"; }
say start
(IK_JOBS=${IK_JOBS:-4} bash scripts/quality/ik_build.sh >> runs/vast/ik_build.log 2>&1 9>&- &)
MODEL=qwen3-8b FETCH_ONLY=1 bash scripts/quality/box_quality.sh 9>&-
say "fetch done: $(tail -1 runs/vast/quality-qwen3-8b-fetch.log)"
until grep -q IK_BUILD_DONE runs/vast/ik_build.log 2> /dev/null; do sleep 10; done
say "$(grep IK_BUILD_DONE runs/vast/ik_build.log | tail -1)"
if grep -q "IK_BUILD_DONE rc=0" runs/vast/ik_build.log; then
  MODEL=qwen3-8b STEPS=compare TAG=ik PRESETS=Q4_K_M IK_TYPES="IQ4_KS IQ4_K" bash scripts/quality/box_quality.sh 9>&-
  say "compare done: $(tail -2 runs/vast/quality-qwen3-8b.log | head -1)"
fi
say IK_QUALITY_DONE
