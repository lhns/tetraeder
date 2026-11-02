#!/usr/bin/env bash
# Colibri baseline for the Phase 2 exit run: Qwen3-235B-A22B Q4_K_M (split GGUF, read in place) on its qwen36 engine,
# CUDA tier plus RAM expert cache plus streaming from disk, greedy decode.
#   scripts/colibri_exit.sh <model-first-shard> <threads> <ram-limit-bytes> <out-json> [deep|short]
# deep (default): prefill about 32768 tokens, then 64 decoded. short: 2048 and 32, the llama-bench `-p 2048 -n 32` size
# (the decode then starts at depth 2048, not 0; Colibri has no depth-0 decode flag). COLIBRI_PREFILL / COLIBRI_DECODE
# override the sizes. Needs scripts/colibri_build.sh first. Writes the JSON even on failure (ok=false) and exits
# non-zero then. Time limit 60 min. Same prompt text as scripts/exit_run.sh (/root/exit_prompt.txt), repeated and cut
# to the token count with Colibri's own tokenizer.
exec 8> /root/tetraeder_colibri_exit.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 4 ] || { echo "usage: $0 <model-first-shard> <threads> <ram-limit-bytes> <out-json> [deep|short]" >&2; exit 2; }
model=$1 threads=$2 ram_bytes=$3 out_json=$4 profile=${5:-deep}
case "$profile" in
  deep) prefill="${COLIBRI_PREFILL:-32768}" decode="${COLIBRI_DECODE:-64}" ;;
  short) prefill="${COLIBRI_PREFILL:-2048}" decode="${COLIBRI_DECODE:-32}" ;;
  *) echo "profile must be deep or short" >&2; exit 2 ;;
esac
src="${COLIBRI_SRC:-/opt/colibri-src}"
bin="$src/c/qwen36"
commit=$(git -C "$src" rev-parse HEAD 2> /dev/null)
[ -x "$bin" ] || { echo "missing $bin: run scripts/colibri_build.sh" >&2; exit 3; }
[ -f "$model" ] || { echo "missing model $model" >&2; exit 3; }
mkdir -p runs/vast "$(dirname "$out_json")"
log=runs/vast/colibri_exit.log
work="${COLIBRI_WORK:-/root/colibri_work}"
mkdir -p "$work"
# shellcheck disable=SC1091
[ -f "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh" ] && . "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
export PATH="/usr/local/cuda/bin:$PATH"

err="$work/run-$profile.err"
rc=0
{
  echo "[$(date +%T)] colibri $profile: prefill $prefill, decode $decode, threads $threads, ram $ram_bytes B, commit $commit"
  # model directory: config, tokenizer and gguf.json pointing at the shards (nothing is copied)
  dir="$work/$(basename "$model" .gguf).colibri"
  if [ ! -f "$dir/gguf.json" ]; then
    timeout -k 10 900 python3 -I "$src/c/tools/gguf_import.py" "$model" "$dir" --mode link || rc=40
  fi
  # prompt text: the exit-run file, repeated, cut by bytes until Colibri's tokenizer counts about $prefill tokens
  base=/root/exit_prompt.txt
  [ -f "$base" ] || cat AGENTS.md docs/adr/*.md > "$base"
  prompt="$work/prompt-$prefill.txt"
  if [ $rc = 0 ]; then
    bytes=$((prefill * 4)) # first guess for English text
    for _ in 1 2 3 4; do
      reps=$((bytes / $(stat -c %s "$base") + 1))
      for _ in $(seq "$reps"); do cat "$base"; done | head -c "$bytes" > "$prompt"
      n=$(SNAP="$dir" TOK_ONLY=1 COLI_NO_OMP_TUNE=1 timeout -k 10 600 "$bin" 0 4 "$prompt" 2> /dev/null | tail -1 | wc -w)
      echo "[$(date +%T)] prompt bytes $bytes -> $n tokens"
      [ "$n" -gt 0 ] || { rc=41; break; }
      # within 0.5 % below the target: done (never above, the KV is sized for it)
      [ "$n" -le "$prefill" ] && [ $((n * 1000)) -ge $((prefill * 995)) ] && break
      bytes=$((bytes * prefill / n - 8))
    done
    [ "$n" -le "$prefill" ] || rc=42
  fi
  if [ $rc = 0 ]; then
    # RAM_GB is the engine's whole budget (resident trunk plus expert cache), not only the cache.
    # COLI_LOOKUP=0: prompt-lookup drafts would match the repeated prompt and inflate decode.
    # COLI_NO_OMP_TUNE=1 with an explicit OMP_NUM_THREADS: no spin-wait team that burns the container's CPU quota.
    # NOSTREAM=1: the text comes at the end, the speed lines are not interleaved with it.
    # CUDA_EXPERT_GB=auto: the card's free memory after the KV and trunk; COLIBRI_VRAM_GB sets a number instead.
    echo "[$(date +%T)] run"
    env SNAP="$dir" N_NEW="$decode" Q36_MAXT=$((prefill + decode + 64)) RAM_GB="$(awk -v b="$ram_bytes" 'BEGIN { printf "%.1f", b / 1e9 }')" \
      COLI_CUDA=1 CUDA_EXPERT_GB="${COLIBRI_VRAM_GB:-auto}" OMP_NUM_THREADS="$threads" COLI_NO_OMP_TUNE="${COLI_NO_OMP_TUNE:-1}" \
      COLI_LOOKUP=0 NOSTREAM=1 COLI_TIMERS=1 \
      timeout -k 30 60m "$bin" 0 4 "$prompt" > "$work/run-$profile.out" 2> "$err"
    rc=$?
    echo "[$(date +%T)] engine rc=$rc"
  fi
} >> "$log" 2>&1

# JSON from the engine's stderr: "[enc] prompt tokens: N | generating M new tokens", "TTFT: s", "Speed: .. (T s for M tokens)",
# "PEAK RSS: GB", "Expert cache hit rate: P%". Prefill speed is prompt tokens / TTFT (TTFT also holds the first sampled token);
# decode speed is (M-1) tokens over the time after TTFT.
python3 -I - "$out_json" "$err" "$rc" "$profile" "$commit" "$threads" "$ram_bytes" "$model" "$prefill" "$decode" << 'EOF'
import json, os, re, subprocess, sys
out, err, rc, profile, commit, threads, ram, model, want_pre, want_dec = sys.argv[1:11]
rc = int(rc)
try:
    text = open(err, errors="replace").read() if rc < 40 else ""  # 40+: failed before the engine ran
except OSError:
    text = ""
def last(pat, cast=float):
    m = re.findall(pat, text)
    return cast(m[-1]) if m else None
np_ = last(r"prompt tokens: (\d+)", int)
nnew = last(r"generating (\d+) new tokens", int)
ttft = last(r"TTFT: ([\d.]+) s")
total = last(r"Speed: [\d.]+ tok/s \(([\d.]+)s for")
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
except OSError:
    gpu = None
d = {
    "engine": "colibri",
    "engine_arch": "qwen36 (qwen3moe GGUF read in place)",
    "commit": commit,
    "profile": profile,
    "model": os.path.basename(model),
    "threads": int(threads),
    "ram_budget_bytes": int(ram),
    "prompt_tokens_wanted": int(want_pre),
    "prompt_tokens": np_,
    "decode_tokens": nnew,
    "decode_depth": np_,
    "ttft_s": ttft,
    "total_s": total,
    "prefill_tok_s": np_ / ttft if np_ and ttft else None,
    "decode_tok_s": (nnew - 1) / (total - ttft) if nnew and nnew > 1 and total and ttft and total > ttft else None,
    "peak_rss_gb": last(r"PEAK RSS: ([\d.]+) GB"),
    "expert_cache_hit_pct": last(r"Expert cache hit rate: ([\d.]+)%"),
    "cuda_tier": "[gpu] " in text,
    "engine_rc": rc,
    "gpu": gpu,
    "raw_tail": text.splitlines()[-40:],
}
# a CPU-only fallback or a missing speed line is a failed baseline, not a slow one
d["ok"] = rc == 0 and d["prefill_tok_s"] is not None and d["decode_tok_s"] is not None and d["cuda_tier"]
json.dump(d, open(out, "w"), indent=1)
print(json.dumps({k: d[k] for k in ("ok", "prompt_tokens", "decode_tokens", "prefill_tok_s", "decode_tok_s", "engine_rc")}))
sys.exit(0 if d["ok"] else 1)
EOF
r=$?
echo "COLIBRI_EXIT_DONE profile=$profile rc=$r" >> "$log"
exit $r
