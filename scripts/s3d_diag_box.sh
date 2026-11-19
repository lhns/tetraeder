#!/usr/bin/env bash
# s3d diagnostic on a 4-GPU box (ADR 0080, "Not decided here"): executor traces, no verdict.
#   SESSION=s3d-20261119 bash scripts/s3d_diag_box.sh        (taxameter run --env SESSION=<s>)
# Part A (tp2-pp2 prefill overlap): per model, pp2048 with prefill_micro_batches 4 and 1: untraced, host trace
#   (TETRAEDER_EXEC_TRACE), sync trace (+ TETRAEDER_EXEC_TRACE_SYNC=1, device time per op).
# Part B (decode excess): tg128 of single and layer-4 (decode toggles at their defaults, on) with trace, sync trace and
#   nvidia-smi clocks every 20 ms beside each run; on the 30B also ep-4 tg128 untraced, traced and synced (the ep-4
#   decode residual, ADR 0080 History); llama.cpp -sm layer and -sm none with the same clock log; 30B first, the 32B as
#   control.
# Stages: gpus (count and power gate), build, models, plans (hw, layout, candidates), A, B.
# Results: bench/results/<s>/ (<name>.json|.log|.rc, <name>.trace.jsonl, <name>.clocks.csv, gpu-samples.csv at 1 Hz, env.txt),
# ends with S3D_DIAG_DONE in runs/vast/s3d_diag.log and the DONE file. A rerun skips runs whose .rc exists.
# Knobs: BUDGET_MIN (50, measured from the start; runs that would start later are skipped), MODELS, PARTS ("A B"),
# DRY_RUN=1 (print the commands only), OUT_DIR.
DRY="${DRY_RUN:-0}"
if [ "$DRY" != 1 ]; then
  exec 8> /root/tetraeder_s3d_diag.lock
  flock -n 8 || exit 0
fi
cd "$(dirname "$0")/.." || exit 1
session="${SESSION:?SESSION required}"
out="${OUT_DIR:-$PWD/bench/results/$session}"
log=runs/vast/s3d_diag.log
mkdir -p "$out/plans" "$out/layouts" "$(dirname "$log")"
[ "$DRY" = 1 ] || exec >> "$log" 2>&1
say() { echo "[$(date +%T)] $*"; }
[ "$DRY" = 1 ] || echo "=== RUN $(date +%T) s3d_diag_box.sh ==="
if [ "$DRY" != 1 ]; then
  bash scripts/ensure_patches.sh 8>&- || exit 1
  if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
  export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
  # shellcheck source=scripts/box_libs.sh
  . scripts/box_libs.sh
  command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
  # shellcheck disable=SC1091
  . "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
  if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
    archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
    export TETRAEDER_CUDA_ARCHS="${archs:-86}"
  fi
fi
threads=8
[ "$DRY" = 1 ] || threads=$(bash scripts/cpu_quota.sh)
tet="${TET_BIN:-target/release/tetraeder}"
planbin="${PLAN_BIN:-target/release/tetraeder-plan}"
llama_bench="${LLAMA_BENCH:-/opt/llama.cpp-baseline/bin/llama-bench}"
models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
declare -A FILE=(
  [qwen3-32b]="$models_dir/Qwen__Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf"
  [qwen3-30b-a3b]="$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
)
declare -A ENGINE=([qwen3-32b]=dense [qwen3-30b-a3b]=moe)
models="${MODELS:-qwen3-30b-a3b qwen3-32b}"
parts="${PARTS:-A B}"
t_start=$(date +%s)
budget_s=$((${BUDGET_MIN:-50} * 60))
clock_pid=""

finish() {
  [ "$DRY" = 1 ] || touch "$out/gpu-sampler.stop"
  touch "$out/DONE"
  say "S3D_DIAG_DONE rc=${1:-0}"
}
die() {
  say "FATAL: $*"
  finish 1
  exit 1
}

say "s3d_diag start (session=$session threads=$threads models=\"$models\" parts=\"$parts\" out=$out)"
if [ "$DRY" != 1 ]; then
  # ---- gpus ----
  [ "$(nvidia-smi -L 2> /dev/null | grep -c '^GPU ')" = 4 ] || die "not a 4-GPU box"
  bash scripts/power_gate.sh 8>&- || die "power limit below default on a GPU"
  OUT="$out" SAMPLER_PARENT=$$ setsid nohup bash scripts/gpu_sampler.sh 8>&- > /dev/null 2>&1 &
  {
    date
    nvidia-smi --query-gpu=index,name,driver_version,power.limit,clocks.sm,clocks.max.sm,clocks.mem,pstate --format=csv
    nvidia-smi topo -m
    echo "cpu quota threads: $threads; host $(hostname) ${VAST_CONTAINERLABEL:-}"
    git rev-parse HEAD
  } > "$out/env.txt" 2>&1
  # ---- build ----
  say "build start"
  timeout -k 30 50m cargo build --release -p tetraeder --features cuda 8>&- || die "build failed"
  timeout -k 30 20m cargo build --release -p tetraeder-planner --bin tetraeder-plan 8>&- || die "build of tetraeder-plan failed"
  say "build done"
  # ---- models ----
  for m in $models; do [ -s "${FILE[$m]}" ] || die "model $m missing: ${FILE[$m]}"; done
  # ---- plans: single, layer-4, tp2-pp2 and its micro-batch variants ----
  [ -s "$out/hw.json" ] || "$tet" plan hw --out "$out/hw.json" 8>&- || die "no hw.json"
  for m in $models; do
    [ -s "$out/candidates-$m.json" ] && continue
    "$tet" plan layout "${FILE[$m]}" --out "$out/layouts/$m.json" 8>&- || die "layout of $m failed"
    "$planbin" candidates --multi --layout "$out/layouts/$m.json" --hw "$out/hw.json" --out "$out/candidates-$m.json" \
      --plans-dir "$out/plans" 8>&- || die "enumeration of $m failed"
    # plan files carry the GGUF's model name; the runs use the key
    python3 - "$out/plans" "$m" "$out/layouts/$m.json" << 'PYEOF' || die "plan names of $m"
import json, os, shutil, sys
d, key, layout = sys.argv[1:4]
name = json.load(open(layout))["model"]
for f in sorted(os.listdir(d)):
    if f.startswith(name + "-") and f.endswith(".toml") and name != key:
        shutil.copy(os.path.join(d, f), os.path.join(d, key + f[len(name):]))
PYEOF
  done
  for m in $models; do
    for mb in 1 4; do
      python3 - "$out/plans/$m-tp2-pp2.toml" "$out/plans/$m-tp2-pp2-mb$mb.toml" "$mb" << 'PYEOF' || die "mb plan of $m"
import re, sys
src, dst, mb = sys.argv[1:4]
text, n = re.subn(r"(?m)^prefill_micro_batches\s*=\s*\d+", f"prefill_micro_batches = {mb}", open(src).read())
if n != 1:
    sys.exit(f"{src}: {n} prefill_micro_batches lines")
open(dst, "w").write(text)
PYEOF
    done
  done
  say "plans done"
fi

clock_start() { # clock_start NAME: nvidia-smi every 20 ms into NAME.clocks.csv
  if [ "$DRY" = 1 ]; then
    echo "PLAN nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,pstate --format=csv -lms 20 > $1.clocks.csv"
    return 0
  fi
  setsid nohup nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,pstate --format=csv -lms 20 \
    > "$out/$1.clocks.csv" 2> /dev/null 8>&- &
  clock_pid=$!
  sleep 1
}
clock_stop() {
  [ -n "$clock_pid" ] || return 0
  kill "$clock_pid" 2> /dev/null
  wait "$clock_pid" 2> /dev/null
  clock_pid=""
}

# run_cmd NAME CLOCKS CMD...: result to NAME.json, exit code to NAME.rc; CLOCKS=1 logs the clocks beside it
run_cmd() {
  local name=$1 clocks=$2 rc
  shift 2
  [ -f "$out/$name.rc" ] && { say "$name: done before"; return 0; }
  if [ "$DRY" = 1 ]; then
    [ "$clocks" = 1 ] && clock_start "$name"
    echo "PLAN $* > $name.json"
    return 0
  fi
  if [ $(($(date +%s) - t_start)) -gt "$budget_s" ]; then say "BUDGET spent: skip $name"; return 0; fi
  say "$name start"
  [ "$clocks" = 1 ] && clock_start "$name"
  timeout -k 30 5m "$@" < /dev/null > "$out/$name.json" 2> "$out/$name.log" 8>&-
  rc=$?
  clock_stop
  echo "$rc" > "$out/$name.rc"
  say "$name done rc=$rc"
}

# tet_run NAME CLOCKS PLAN P N MODE MODEL: our engine; MODE plain|trace|sync (trace = host time, sync = device time per op)
tet_run() {
  local name=$1 clocks=$2 plan=$3 p=$4 n=$5 mode=$6 m=$7 envs=() r=2
  case $mode in
    trace | sync) envs+=("TETRAEDER_EXEC_TRACE=$out/$name.trace.jsonl") ;;
  esac
  [ "$mode" = sync ] && envs+=(TETRAEDER_EXEC_TRACE_SYNC=1)
  [ "$p" = 0 ] && r=5
  run_cmd "$name" "$clocks" env "${envs[@]}" "$tet" bench -m "${FILE[$m]}" --engine "${ENGINE[$m]}" --plan "$plan" \
    -p "$p" -n "$n" -r "$r" -t "$threads" --ngl -1 --ubatch 512 --flash-attn on -o json
}

has() { case " $parts " in *" $1 "*) return 0 ;; esac; return 1; }

if has A; then
  say "STAGE A"
  for m in $models; do
    for mb in 4 1; do
      for mode in plain trace sync; do
        tet_run "$m-tp2pp2-mb$mb-pp-$mode" 0 "$out/plans/$m-tp2-pp2-mb$mb.toml" 2048 0 "$mode" "$m"
      done
    done
  done
fi

if has B; then
  say "STAGE B"
  for m in $models; do
    sp="$out/plans/$m-single.toml"
    l4="$out/plans/$m-layer-4.toml"
    # synced runs first (the question), then the traced host-time runs of the same configuration, then llama.cpp
    tet_run "$m-layer4-tg-sync" 1 "$l4" 0 128 sync "$m"
    tet_run "$m-single-tg-sync" 1 "$sp" 0 128 sync "$m"
    tet_run "$m-layer4-tg-trace" 1 "$l4" 0 128 trace "$m"
    tet_run "$m-single-tg-trace" 1 "$sp" 0 128 trace "$m"
    if [ "${ENGINE[$m]}" = moe ]; then
      ep="$out/plans/$m-ep-4.toml"
      if [ "$DRY" = 1 ] || [ -s "$ep" ]; then
        tet_run "$m-ep4-tg-plain" 1 "$ep" 0 128 plain "$m"
        tet_run "$m-ep4-tg-trace" 1 "$ep" 0 128 trace "$m"
        tet_run "$m-ep4-tg-sync" 1 "$ep" 0 128 sync "$m"
      else
        say "no ep-4 plan for $m: skip its decode traces"
      fi
    fi
    run_cmd "$m-llama-layer-tg" 1 "$llama_bench" -m "${FILE[$m]}" -ngl 99 -fa 1 -p 0 -n 128 -r 5 -t "$threads" -sm layer -o json
    run_cmd "$m-llama-none-tg" 1 "$llama_bench" -m "${FILE[$m]}" -ngl 99 -fa 1 -p 0 -n 128 -r 5 -t "$threads" -sm none -mg 0 -o json
  done
fi

if [ "$DRY" != 1 ]; then
  for f in "$out"/*.trace.jsonl; do
    [ -f "$f" ] && python3 scripts/exec_trace_summary.py "$f" > "${f%.jsonl}.summary.txt" 2>&1
  done
fi
finish 0
