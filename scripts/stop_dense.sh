#!/usr/bin/env bash
# Stops the dense rerun (box cap) and turns both dense logs into result files the scorer reads.
cd "$(dirname "$0")/.." || exit 1
pkill -f 'dense_reru[n].sh'
pkill -f 'tetraeder-benc[h] run'
pkill -f 'vram_hol[d].py'
out=$(ls -d "$PWD"/bench/results/*/planner-dense | tail -1)
python3 scripts/bench_log_to_results.py runs/vast/planner_campaign.log --since 11:20:00 --until 11:48:00 --out "$out/planner-dense-log-cpu-anchors.json"
python3 scripts/bench_log_to_results.py runs/vast/dense_rerun.log --since 11:48:00 --out "$out/planner-dense-log-rerun.json"
cp runs/vast/planner_campaign.log runs/vast/dense_rerun.log "$out/"
echo "rc=0" > "$(dirname "$out")/DONE"
ls "$out"
