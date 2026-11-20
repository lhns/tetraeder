#!/usr/bin/env bash
# s3f confirmation on a 4-GPU box (ADR 0081, patch 0006: prefill steps run without CUDA graph capture).
#   SESSION=s3f-20261120 bash scripts/s3f_confirm_box.sh        (taxameter run --env SESSION=<s>)
# Stage A: scripts/phase5_box.sh <session> 4 with ROUNDS=1 (the s3c-style rehearsal): gates (now with the ggml-sys test
#   gpu_no_capture_flag_skips_the_cuda_graph), anchors, calibrate, predict, one round of every candidate and baseline,
#   pp-overlap. Results in bench/results/<s>/phase5/. If it ends with rc=3 (a gate failed), the
#   session stops after stage A (reason in stageA-stop.txt) and B does not run.
# Stage B: per model (30B first), tp2-pp2 and layer-4, both at prefill_micro_batches 4, two arms interleaved per
#   configuration as default, nograph, nograph, default (so two runs per arm):
#     default   as is: prefill steps run direct (patch 0006), decode keeps capture and replay
#     nograph   GGML_CUDA_DISABLE_GRAPHS=1 (no graphs at all)
#   Each run is `bench -p 2048 -n 128 -r 5` (one warm-up, then 5 timed reps: samples_ns has 5 values per row, pp and tg),
#   no GGML_CUDA_GRAPH_DEBUG, no trace. The 1 Hz gpu-samples.csv (throttle reasons) runs during B, and every run has a
#   <name>.window.txt (`start=<epoch s> end=<epoch s>`) to cut it. Names: <model>-<cfg>-mb4-<arm>-r<k>, cfg tp2pp2 or layer4.
#   The plans come from stage A's enumeration (phase5/plans/<model>-tp2-pp2.toml, -layer-4.toml) with the
#   prefill_micro_batches line set to 4 (plans/<model>-<cfg>-mb4.toml here).
# Run count B: 2 models x 2 configurations x 4 = 16.
# Wall time: provision with both models about 10 min (before this script); stage A about 40 min on a cold build (s3c: 25
# min with ROUNDS=2 on a cached build; here the cold build adds about 10 min for the box and about 5 for the tests);
# stage B about 40 to 60 s per run (s3e: about 40 s per -r 5 run; here pp and tg), 16 runs, about 15 min. A has
# BUDGET_A_MIN 50 (phase5_box BUDGET_MIN: runs that would start later are skipped, loudly, the verdict is then null), B
# has BUDGET_MIN 22 from its own start. Box time about 10 + 50 + 22 + 5 for fetch and destroy = 87 min at worst.
# Results: bench/results/<s>/ (<name>.json|.log|.rc|.window.txt, plans/, gpu-samples.csv, env.txt, stageA.txt), ends with
# S3F_CONFIRM_DONE rc=<n> in runs/vast/s3f_confirm.log and the DONE file. A rerun skips stage A when stageA.txt exists
# (phase5_box itself resumes otherwise) and every run whose .rc exists.
# Knobs: SESSION, BUDGET_MIN (22, stage B), BUDGET_A_MIN (50), MODELS, DRY_RUN=1 (print the commands only), OUT_DIR.
DRY="${DRY_RUN:-0}"
if [ "$DRY" != 1 ]; then
  exec 8> /root/tetraeder_s3f_confirm.lock
  flock -n 8 || exit 0
fi
cd "$(dirname "$0")/.." || exit 1
session="${SESSION:?SESSION required}"
out="${OUT_DIR:-$PWD/bench/results/$session}"
a_out="$out/phase5"
log=runs/vast/s3f_confirm.log
mkdir -p "$out/plans" "$(dirname "$log")"
[ "$DRY" = 1 ] || exec >> "$log" 2>&1
say() { echo "[$(date +%T)] $*"; }
[ "$DRY" = 1 ] || echo "=== RUN $(date +%T) s3f_confirm_box.sh ==="

finish() {
  if [ "$DRY" != 1 ]; then
    touch "$out/gpu-sampler.stop"
    # the fetch brings *.log and *.json, not *.rc: logs and the analysis are copied here (the analysis is rerun locally with s3c)
    cp runs/vast/s3f_confirm.log runs/vast/phase5_box.log "$out/" 2> /dev/null
    python3 -I scripts/s3f_analyze.py "$out" /nonexistent > /dev/null 2>&1 || say "analysis on the box failed (rerun it locally)"
  fi
  touch "$out/DONE"
  say "S3F_CONFIRM_DONE rc=${1:-0}"
}
die() {
  say "FATAL: $*"
  finish 1
  exit 1
}

models="${MODELS:-qwen3-30b-a3b qwen3-32b}"
say "s3f_confirm start (session=$session models=\"$models\" out=$out)"

# ---- stage A ----
say "STAGE A (phase5_box.sh $session 4, ROUNDS=1)"
if [ "$DRY" = 1 ]; then
  echo "PLAN SESSION=$session NGPUS=4 ROUNDS=1 BUDGET_MIN=${BUDGET_A_MIN:-50} bash scripts/phase5_box.sh $session 4"
elif [ -f "$out/stageA.txt" ]; then
  say "stage A: done before (rc=$(cat "$out/stageA.txt"))"
else
  SESSION="$session" NGPUS=4 ROUNDS=1 BUDGET_MIN="${BUDGET_A_MIN:-50}" bash scripts/phase5_box.sh "$session" 4 8>&-
  rc=$?
  echo "$rc" > "$out/stageA.txt"
  say "stage A done rc=$rc ($(tail -n 40 runs/vast/phase5_box.log 2> /dev/null | grep -E 'PHASE5_BOX_DONE|gate failed' | tail -n 2 | tr '\n' ' '))"
fi
if [ "$DRY" != 1 ]; then
  rc=$(cat "$out/stageA.txt")
  # only a failed gate ends the session; another rc is logged and B's own checks (models, binary, plans) decide
  [ "$rc" != 0 ] && [ "$rc" != 3 ] && say "stage A rc=$rc (not a gate failure): stage B tries anyway"
  if [ "$rc" = 3 ]; then
    {
      echo "stage A ended with rc=$rc; stage B did not run"
      grep -E 'gate failed|FAILED|FATAL' runs/vast/phase5_box.log 2> /dev/null | tail -n 10 | cut -c1-300
    } > "$out/stageA-stop.txt"
    say "stage A rc=$rc: session stops (see stageA-stop.txt)"
    finish "$rc"
    exit "$rc"
  fi
fi

# ---- stage B ----
say "STAGE B"
threads=8
[ "$DRY" = 1 ] || threads=$(bash scripts/cpu_quota.sh)
tet="${TET_BIN:-target/release/tetraeder}"
models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
declare -A FILE=(
  [qwen3-32b]="$models_dir/Qwen__Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf"
  [qwen3-30b-a3b]="$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
)
declare -A ENGINE=([qwen3-32b]=dense [qwen3-30b-a3b]=moe)
t_start=$(date +%s)
budget_s=$((${BUDGET_MIN:-22} * 60))

if [ "$DRY" != 1 ]; then
  for m in $models; do [ -s "${FILE[$m]}" ] || die "model $m missing: ${FILE[$m]}"; done
  # the box's environment as stage A left it (CUDA libraries for the release binary)
  if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
  export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
  # shellcheck source=scripts/box_libs.sh
  . scripts/box_libs.sh
  # shellcheck disable=SC1091
  . "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
  [ -x "$tet" ] || die "no $tet (stage A builds it)"
  OUT="$out" SAMPLER_PARENT=$$ SAMPLER_LOCK=/root/tetraeder_gpu_sampler_s3f.lock setsid nohup bash scripts/gpu_sampler.sh 8>&- > /dev/null 2>&1 &
  {
    date
    nvidia-smi --query-gpu=index,name,driver_version,power.limit,clocks.sm,clocks.max.sm,clocks.mem,pstate --format=csv
    nvidia-smi topo -m
    echo "cpu quota threads: $threads; host $(hostname) ${VAST_CONTAINERLABEL:-}"
    git rev-parse HEAD
  } > "$out/env.txt" 2>&1
fi

# plans: stage A's enumeration with prefill_micro_batches = 4 (the line must exist exactly once)
for m in $models; do
  for cfg in tp2-pp2 layer-4; do
    dst="$out/plans/$m-$cfg-mb4.toml"
    if [ "$DRY" = 1 ]; then
      echo "PLAN plans/$m-$cfg.toml of stage A with prefill_micro_batches = 4 -> $dst"
      continue
    fi
    [ -s "$dst" ] && continue
    # a plan that cannot be made skips its configuration (run_one), the others still run
    python3 - "$a_out/plans/$m-$cfg.toml" "$dst" << 'PYEOF' || say "WARNING: no mb4 plan for $m $cfg, its runs are skipped"
import re, sys
src, dst = sys.argv[1:3]
text = open(src).read()
pat = r"(?m)^prefill_micro_batches\s*=\s*(\d+)"
if len(re.findall(pat, text)) != 1:
    sys.exit(f"{src}: not exactly one prefill_micro_batches line")
print(f"{src}: prefill_micro_batches was {re.search(pat, text).group(1)}, now 4")
open(dst, "w").write(re.sub(pat, "prefill_micro_batches = 4", text))
PYEOF
  done
done

# run_one NAME PLAN MODEL [K=V...]: result to NAME.json, stderr to NAME.log, exit code to NAME.rc, window to NAME.window.txt
run_one() {
  local name=$1 plan=$2 m=$3 rc w0 w1
  shift 3
  local cmd=("$tet" bench -m "${FILE[$m]}" --engine "${ENGINE[$m]}" --plan "$plan" -p 2048 -n 128 -r 5 -t "$threads"
    --ngl -1 --ubatch 512 --flash-attn on -o json)
  if [ "$DRY" = 1 ]; then
    echo "PLAN env $* ${cmd[*]} > $name.json"
    return 0
  fi
  [ -f "$out/$name.rc" ] && { say "$name: done before"; return 0; }
  [ -s "$plan" ] || { say "$name: skipped, no plan $plan"; return 0; }
  if [ $(($(date +%s) - t_start)) -gt "$budget_s" ]; then say "BUDGET spent: skip $name"; return 0; fi
  say "$name start"
  w0=$(date +%s.%N)
  timeout -k 30 10m env "$@" "${cmd[@]}" < /dev/null > "$out/$name.json" 2> "$out/$name.log" 8>&-
  rc=$?
  w1=$(date +%s.%N)
  echo "start=$w0 end=$w1" > "$out/$name.window.txt"
  echo "$rc" > "$out/$name.rc"
  say "$name done rc=$rc"
}

for m in $models; do
  for cfg in tp2-pp2 layer-4; do
    short=${cfg//-/}
    plan="$out/plans/$m-$cfg-mb4.toml"
    base="$m-$short-mb4"
    # default, nograph, nograph, default: two runs per arm, paired in time
    run_one "$base-default-r1" "$plan" "$m"
    run_one "$base-nograph-r1" "$plan" "$m" GGML_CUDA_DISABLE_GRAPHS=1
    run_one "$base-nograph-r2" "$plan" "$m" GGML_CUDA_DISABLE_GRAPHS=1
    run_one "$base-default-r2" "$plan" "$m"
  done
done
finish 0
