#!/usr/bin/env bash
# Track B quality run on a box (ADR 0044, 0045): build the CLI, fetch the BF16 model and wikitext-2, run the
# calibration job, the preset comparison and the conversion benchmark for MODEL (qwen3-8b or qwen3-30b).
# STEPS picks steps (default: calibrate compare convert). Results: bench/quality/, bench/results/<date>/.
# One box runs both models (downloads are billed per GB, tracks-common.md): qwen3-8b first while the 30B BF16
# downloads (FETCH_ONLY=1 MODEL=qwen3-30b), then qwen3-30b. Ends with QUALITY_DONE in runs/vast/quality-$MODEL.log. Resumable: every step skips finished work.
exec 8> "/root/tetraeder_quality${FETCH_ONLY:+_fetch_$MODEL}.lock"; flock -n 8 || exit 0
cd "$(dirname "$0")/../.." || exit 1
mkdir -p runs/vast bench/quality
MODEL="${MODEL:-qwen3-8b}"
log="runs/vast/quality-$MODEL${FETCH_ONLY:+-fetch}.log"
exec >> "$log" 2>&1
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"
date_dir="bench/results/$(date +%Y%m%d)"
mkdir -p "$date_dir"
W=/root/q/$MODEL
mkdir -p "$W" /root/q
gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2> /dev/null | head -1)
echo "[$(date +%T)] start $MODEL: threads $threads, gpu '$gpu', host $(hostname), mem $(free -g | awk '/Mem/{print $2}') GB, disk $(df -h --output=avail /root | tail -1)"

case "$MODEL" in
  qwen3-8b)
    spec="unsloth/Qwen3-8B-GGUF:Qwen3-8B-BF16.gguf"
    src="$PWD/.hf_cache/unsloth__Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf"
    base=BF16; span=${SPAN:-6}; formats="${FORMATS:-Q8_0 Q6_K Q5_K Q4_K IQ4_XS Q3_K Q2_K}"
    ppl_args="-ngl 99 -fa on"; ref_args="$ppl_args"; eval_chunks=${EVAL_CHUNKS:-100}
    ;;
  qwen3-30b)
    b=BF16/Qwen3-30B-A3B-BF16
    spec="unsloth/Qwen3-30B-A3B-GGUF:$b-00001-of-00002.gguf+$b-00002-of-00002.gguf"
    src="$PWD/.hf_cache/unsloth__Qwen3-30B-A3B-GGUF/$b-00001-of-00002.gguf"
    base=Q8_0; span=${SPAN:-12}; formats="${FORMATS:-Q6_K Q4_K Q3_K Q2_K}"
    # Q8_0 (32 GB) on a 24 GB card: experts of the upper layers stay in RAM; big batches amortize their copies
    ot='blk\.(2[6-9]|3[0-9]|4[0-7])\.ffn_.*_exps\.=CPU'
    ppl_args="-ngl 99 -fa on -b 2048 -ub 2048 -ot ${OT:-$ot}"
    ref_args="-ngl 99 -fa on -b 2048 -ub 2048 -ot exps=CPU"; eval_chunks=${EVAL_CHUNKS:-50}
    ;;
  *) echo "unknown MODEL $MODEL"; exit 1 ;;
esac

if [ ! -f /root/q/wikitext-2-raw/wiki.test.raw ]; then
  (cd /root/q && curl -fsSL -o wt.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip \
    && python3 -c "import zipfile; zipfile.ZipFile('wt.zip').extractall('.')") || { echo "wikitext fetch failed"; exit 1; }
fi
if [ ! -f "$src" ]; then
  echo "[$(date +%T)] fetch $spec"
  TAXAMETER_DIR="$PWD" bash "${TAXAMETER_HOME:-/root/.taxameter}/box.sh" models "$spec" || { echo "fetch failed"; exit 1; }
fi
[ -n "${FETCH_ONLY:-}" ] && { echo "[$(date +%T)] FETCH_DONE"; exit 0; }
echo "[$(date +%T)] build"
cargo build --release -p tetraeder -j 3 || { echo "build failed"; exit 1; }

for s in ${STEPS:-calibrate compare convert}; do
  echo "[$(date +%T)] step $s"
  rc=0
  case "$s" in
    calibrate)
      python3 scripts/quality/calibrate.py --source "$src" --work "$W" --text /root/q/wikitext-2-raw/wiki.train.raw \
        --base "$base" --formats $formats --span "$span" --threads "$threads" --ppl-args "$ppl_args" \
        --ref-args "$ref_args" --gpu "$gpu" --out "bench/quality/$MODEL.quality.json" || rc=$?
      # the per-format shards are not needed after the profile; the compare step needs the disk
      [ "$rc" = 0 ] && [ -z "${KEEP_SHARDS:-}" ] && rm -rf "$W/shards" "$W/mix"
      ;;
    compare)
      python3 scripts/quality/compare_mix.py --source "$src" --profile "bench/quality/$MODEL.quality.json" \
        --work "$W/compare" --text /root/q/wikitext-2-raw/wiki.test.raw --chunks "$eval_chunks" --presets ${PRESETS:-Q4_K_M Q3_K_M Q5_K_M} \
        --threads "$threads" --ppl-args "$ppl_args" --ref-args "$ref_args" --gpu "$gpu" \
        ${IK_TYPES:+--ik-bin /opt/ik_llama/bin --ik-types $IK_TYPES} \
        --out "$date_dir/quality-$MODEL.json" || rc=$?
      ;;
    convert)
      python3 scripts/quality/convert_bench.py --source "$src" --work "$W/convert" --threads "$threads" \
        --gpu "$gpu" --out "$date_dir/convert-$MODEL.json" || rc=$?
      rm -rf "$W/convert"
      ;;
  esac
  echo "[$(date +%T)] step $s rc=$rc"
done
command -v sccache > /dev/null 2>&1 && timeout 30 sccache --stop-server > /dev/null 2>&1
echo "QUALITY_DONE"
