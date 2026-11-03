#!/usr/bin/env bash
# Track B's whole box: the 8B campaign, then the 30B (fetched only after the 8B shards are gone: 80 GB disks).
exec 9> /root/tetraeder_quality_all.lock; flock -n 9 || exit 0
cd "$(dirname "$0")/../.." || exit 1
mkdir -p runs/vast
MODEL=qwen3-8b bash scripts/quality/box_quality.sh 9>&-
rm -rf /root/q/qwen3-8b .hf_cache/unsloth__Qwen3-8B-GGUF
MODEL=qwen3-30b bash scripts/quality/box_quality.sh 9>&-
echo "ALL_DONE $(date +%T)" >> runs/vast/quality-all.log
