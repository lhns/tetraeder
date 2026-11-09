#!/usr/bin/env bash
# Runs llama.cpp's cache tuner (scripts/moe-cache-tune.py, #30966) with the baseline llama-server on Qwen3-30B-A3B
# Q4_K_M and writes its run directory and report to bench/results/<date>/moe-cache-tune/. Its recommendation is a
# starting point for the 235B sweep (exit_run.sh stage llamacpp-cache), not a replacement for it.
# Env: THREADS (default: CPU quota), CTX (32768), MAX_MIB (tuner --max-mib), LOAD_MODE (default auto).
exec 8> /root/tetraeder_moe_cache_tune.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
log=runs/vast/moe_cache_tune.log
base="${LLAMA_BASELINE_DIR:-/opt/llama.cpp-baseline}"
out="$PWD/bench/results/$(date +%Y%m%d)/moe-cache-tune"
mkdir -p "$out"
models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
gguf="$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"
[ -x "$base/bin/llama-server" ] || { echo "missing $base/bin/llama-server" >> "$log"; exit 3; }
[ -f "$gguf" ] || { echo "missing $gguf" >> "$log"; exit 4; }
max=()
[ -n "${MAX_MIB:-}" ] && max=(--max-mib "$MAX_MIB")
echo "[$(date +%T)] moe-cache-tune start: threads $threads" >> "$log"
# the tuner reuses the runs in --out while the command stays the same, so a relaunch resumes
python3 "$base/scripts/moe-cache-tune.py" "${max[@]}" --out "$out/runs" -- \
  "$base/bin/llama-server" -m "$gguf" -cmoe -t "$threads" -c "${CTX:-32768}" -lm "${LOAD_MODE:-auto}" \
  > "$out/report.txt" 2>> "$log"
rc=$?
echo "[$(date +%T)] moe-cache-tune done rc=$rc" >> "$log"
python3 -I - "$out" "$rc" "$threads" << 'EOF'
import json, re, sys
out, rc, threads = sys.argv[1:4]
text = open(f"{out}/report.txt", errors="replace").read()
m = re.search(r"recommended: --moe-cache-mib (\d+) --moe-cache-cpu-share (\S+)", text)
json.dump({"rc": int(rc), "threads": int(threads), "model": "Qwen3-30B-A3B-Q4_K_M",
           "cache_mib": int(m.group(1)) if m else None, "cpu_share": float(m.group(2)) if m else None,
           "report": text}, open(f"{out}/tune.json", "w"), indent=1)
EOF
exit "$rc"
