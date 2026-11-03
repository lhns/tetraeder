#!/usr/bin/env bash
# One-off: the dense planner fixture without its CPU-only anchor pairs (a 32B on 2 CPU threads ran 1.7 tok/s), one
# repeat, so it fits the box cap. Stops the running dense harness first. Log runs/vast/dense_rerun.log, DENSE_RERUN_DONE.
exec 8> /root/tetraeder_dense_rerun.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
exec >> runs/vast/dense_rerun.log 2>&1
pkill -f 'planner_campaig[n].sh'
pkill -f 'tetraeder-benc[h] run'
pkill -f 'vram_hol[d].py'
sleep 5
out=$(ls -d "$PWD"/bench/results/*/planner-dense | tail -1)
python3 scripts/vram_hold.py --leave-free-mib 18432 > "$out/vram_hold_rerun.log" 2>&1 8>&- &
hold=$!
for _ in $(seq 1 60); do grep -q holding "$out/vram_hold_rerun.log" && break; sleep 2; done
cat "$out/vram_hold_rerun.log"
only=$(grep -o "^id = '[^']*'" "$out/spec.toml" | sed "s/id = '//; s/'$//" | grep '^info-' | grep -v 'anchor-cpu' | paste -sd,)
echo "[$(date +%T)] pairs: $only"
args=()
IFS=, read -ra ids <<< "$only"
for i in "${ids[@]}"; do args+=(--only "$i"); done
timeout -k 30 30m target/release/tetraeder-bench run "$out/spec.toml" --out-dir "$out" --repeats 1 "${args[@]}" \
  --llama-bench /opt/llama.cpp/bin/llama-bench --models-dir "$PWD/.hf_cache" 8>&-
echo "[$(date +%T)] DENSE_RERUN_DONE rc=$?"
kill "$hold"
echo "rc=0" > "$(dirname "$out")/DONE"
