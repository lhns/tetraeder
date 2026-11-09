#!/usr/bin/env bash
# Box session profiles (docs/sessions.md has box requirements, estimates and the fetch steps):
#   SESSION=b1-exit      Phase 2 exit verdict: builder with the ADR 0038 levers, KTransformers, llama.cpp's expert
#                        cache at the tuned CPU share (mmap and dio), a builder/KT repeat pair; ktransformers-avx2 and
#                        the cache repeat only if time is left
#   SESSION=b1b-levers   the lever stages on a second 235B box: builder, low-bit KV (kv-*), routing trace and replay,
#                        the native-ISA builder
#   SESSION=b2-proxy     Claude Code proxy rerun (48 GB card), then KV quality on the 30B and the scripts in B2_STAGES
#   SESSION=b3-planner   llama.cpp's cache tuner on the 30B, then the planner dense fixture
#   SESSION=b3-detector  detector campaign, then kernels-cpu
# Each profile fits its box cap with 20% margin: DEADLINE is SESSION_MIN (default per profile) after the first launch,
# and every stage that would end later is skipped. Results go to one directory (RESULTS_DATE, fixed at the first
# launch) and a relaunch resumes: stages with results are not rerun. The start time and date live in
# runs/vast/session-<name>.start (delete it to start over).
# Started detached by `taxameter run scripts/session.sh --env SESSION=<name>`; ends with SESSION_DONE in
# runs/vast/session-<name>.log.
exec 8> /root/tetraeder_session.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
name="${SESSION:-}"
log="runs/vast/session-$name.log"
# minutes for this script: the box cap less 20% (144 of 180 min; b2 has the 6 h proxy exception: 288), less about 25
# for provisioning before and fetch, verify and destroy after
case "$name" in
  b1-exit | b1b-levers | b3-planner | b3-detector) budget=120 ;;
  b2-proxy) budget=260 ;;
  *) echo "SESSION must be b1-exit, b1b-levers, b2-proxy, b3-planner or b3-detector, not '$name'" >&2; exit 2 ;;
esac
state="runs/vast/session-$name.start"
[ -s "$state" ] || echo "$(date +%s) $(date +%Y%m%d)" > "$state"
read -r start RESULTS_DATE < "$state"
DEADLINE=$((start + ${SESSION_MIN:-$budget} * 60))
export RESULTS_DATE DEADLINE RESUME=1
results="bench/results/$RESULTS_DATE"
mkdir -p "$results"
models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"

step() {
  echo "[$(date +%T)] start $1"
  shift
  "$@" 8>&-
  echo "[$(date +%T)] done rc=$?"
}
# left_min: minutes until the deadline
left_min() { echo $(((DEADLINE - $(date +%s)) / 60)); }
# fetch <hf spec> <file under models_dir>: taxameter's box downloader, once per box
fetch() {
  [ -s "$models_dir/$2" ] && return 0
  TAXAMETER_DIR="$PWD" bash "${TAXAMETER_HOME:-/root/.taxameter}/box.sh" models "$1" 8>&-
}
q30="Qwen/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf"
f30="Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
q32="Qwen/Qwen3-32B-GGUF:Qwen3-32B-Q4_K_M.gguf"
f32="Qwen__Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf"

{
  echo "[$(date +%T)] session $name: results $results, deadline $(date -d "@$DEADLINE" +%T) ($(left_min) min left)"
  case "$name" in
    b1-exit | b1b-levers)
      # the 7-layer -ot llama.cpp fits on a 24 GB card at 32K and the same expert bytes as builder slots (ADR 0045)
      export VRAM_SLOT_BYTES="${VRAM_SLOT_BYTES:-10966007808}"
      export OT_FIT="${OT_FIT:-blk\.[0-6]\.ffn_.*_exps=CUDA0;exps=CPU}"
      (bash scripts/fetch_exit_model.sh 8>&- &)
      share=0.5
      if [ "$name" = b1-exit ]; then
        # llama.cpp's CPU share of misses: MOE_CPU_SHARE, else the newest committed tuner result (B3), else 0.5
        share="${MOE_CPU_SHARE:-}"
        if [ -z "$share" ]; then
          share=$(python3 -I -c '
import glob, json
for f in sorted(glob.glob("bench/results/*/moe-cache-tune/tune.json"), reverse=True):
    s = json.load(open(f)).get("cpu_share")
    if s is not None:
        print(s)
        break')
        fi
        [ -n "$share" ] || { share=0.5; echo "no tuned CPU share (run b3-planner first): using $share"; }
        echo "llama.cpp cache: $((VRAM_SLOT_BYTES / 1048576)) MiB, cpu share $share"
        # code order in exit_run.sh: smoke, main-prefetch, ktransformers, llamacpp-cache (mmap), llamacpp-cache-dio,
        # main-repeat, ktransformers-repeat, then ktransformers-avx2 and llamacpp-cache-repeat if the deadline allows
        stages="smoke main-prefetch ktransformers llamacpp-cache llamacpp-cache-dio main-repeat ktransformers-repeat"
        stages="$stages ktransformers-avx2 llamacpp-cache-repeat"
      else
        # each lever stage pairs with main-prefetch on this box: kv-* right after it, the trace, the native ISA
        # main-iosvc and main-overlap (ADR 0054) follow the kv stages; main-overlap-ram runs if the deadline allows
        stages="smoke main-prefetch kv-q8 kv-q8-slots kv-q8q4 kv-q8q4-slots main-iosvc main-overlap trace main-native-repeat main-overlap-ram"
      fi
      # levers (ADR 0038): prefetch on in every builder stage, depth 2
      step exit_run env TETRAEDER_EXIT_PREFETCH_DEPTH=2 CPU_SHARES="$share" STAGES="$stages" bash scripts/exit_run.sh
      ;;
    b2-proxy)
      # the proxy run took 82 min with REPS=2 (2026-11-03); BUDGET_MIN is when the last block may start
      step proxy env REPS="${REPS:-2}" BUDGET_MIN="${BUDGET_MIN:-150}" STREAMING="${STREAMING:-1}" \
        bash scripts/claude_code_proxy.sh
      # KV-format KL acceptance (ADR 0050 b) on the 30B: fits the 48 GB card with every expert resident; fails loudly
      # when it writes no result
      if [ "$(left_min)" -ge 40 ]; then
        step kv_quality env KV_QUALITY_MIN=$(($(left_min) - 5)) bash scripts/kv_quality.sh "$PWD/$results/kv-quality.json"
      fi
      # hook for further A6000 stages (resident decode profiling, track-r's scripts/decode_gap.sh): each script runs if
      # present, with the time left as its limit
      for s in ${B2_STAGES:-scripts/decode_gap.sh}; do
        [ -f "$s" ] || { echo "skip $s: not in this checkout"; continue; }
        [ "$(left_min)" -ge 10 ] || { echo "skip $s: $(left_min) min left"; continue; }
        step "$s" timeout -k 30 "$(left_min)m" bash "$s"
      done
      ;;
    b3-planner)
      # the 32B downloads while the tuner runs on the 30B
      fetch "$q30" "$f30"
      (fetch "$q32" "$f32" > runs/vast/fetch_dense.log 2>&1 &)
      # the tuner first: its CPU share is B1's input; it resumes from its finished runs
      [ "$(left_min)" -ge 30 ] && step moe_cache_tune env TUNE_MIN="${TUNE_MIN:-25}" bash scripts/moe_cache_tune.sh
      # the planner campaign waits for the download (its own fetch skips a file that exists, even a partial one)
      while pgrep -f 'box[.]sh models' > /dev/null; do sleep 5; done
      # the fixture runs until the deadline (FIXTURE_MIN is only the upper bound)
      step planner env FIXTURES=dense FIXTURE_MIN="${FIXTURE_MIN:-240}" bash scripts/planner_campaign.sh
      ;;
    b3-detector)
      step detector bash scripts/detector_campaign.sh
      # CPU-only Q4_K kernels (mainline, ik, baseline) on the 30B the detector fetched; about 15 min
      if [ "$(left_min)" -ge 18 ]; then
        step kernels_cpu bash scripts/kernels_cpu.sh
      else
        echo "skip kernels-cpu: $(left_min) min left"
      fi
      ;;
  esac
  touch "$results/DONE"
  echo "[$(date +%T)] SESSION_DONE ($(left_min) min before the deadline)"
} >> "$log" 2>&1
