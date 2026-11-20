#!/usr/bin/env bash
# s3e diagnostic on a 4-GPU box (ADR 0080, s3d-20261120 README "Does not settle"): how fast is warm direct eval at mb4?
#   SESSION=s3e-20261120 bash scripts/s3e_diag_box.sh        (taxameter run --env SESSION=<s>)
# Background: per process and range uid, call 1 runs direct, call 2 captures a CUDA graph, call 3+ replays (ggml-cuda
# decides by call count). The judged protocol times the capture call; serving always runs call 1 (new uids per shape).
# Only tp2-pp2 pp2048 (no decode), both models (30B first), prefill_micro_batches 4 and 1, NCCL default ("nccl") and
# GGML_CUDA_ALLREDUCE=internal ("internal"). Every run has the host trace (TETRAEDER_EXEC_TRACE; s3d: no perturbation)
# and -r 5 timed reps (plus the bench's one warmup), so samples_ns has 5 values. Names: <model>-tp2pp2-mb<N>-pp-<ar>-<arm>.
# Arms, in run order per model (arms 2 and 1 for all four mb x ar cells first, the decisive pair; then 4, 3, 5):
#   nograph   GGML_CUDA_DISABLE_GRAPHS=1: warm direct-eval steady state (decisive). Graphs-off proof: the bench JSON
#             cannot show it (bench.rs fills `cuda_graphs` as `on_gpu` on the plan path used here, whatever the env;
#             only the llama path reads GGML_CUDA_DISABLE_GRAPHS). So the very first nograph run (30B, mb4, NCCL)
#             also sets GGML_CUDA_GRAPH_DEBUG=1 (it logs only on capture or replay, so a graphs-off run stays silent)
#             and its stderr is grepped for "CUDA graph warmup complete|reset" and "CUDA Graph id ... reused".
#             Verdict in nograph-check.txt: `ok` (run rc 0, no such line) or `void: <why>`; if void, the remaining
#             nograph runs are skipped (logged) and everything else continues. After the graphdebug arm the script adds
#             a `note:` line if that default-graphs run logged no capture line either (the `ok` is then unproven).
#   default   as is: call 1 direct, call 2 capture, call 3+ replay
#   varykey   (mb4, NCCL only) "every timed rep a different prompt length, so each is call 1". The bench takes ONE -p (u32) and -r repeats
#             that same length (crates/tetraeder/src/bench.rs), so this cannot be done in one process. Best equivalent
#             here: five processes, -p 2048, 2040, 2032, 2024, 2016, -r 1 each (name suffix varykey-<p>). In each, the
#             trace step 0 is the warmup = call 1 at that length (new uids, direct; a fresh process, so lazy init is
#             cold); samples_ns/step 1 is the capture call. A warm-process call 1 at a new shape needs a bench change
#             (a -p list); recommended if step 0 is not conclusive.
#   graphdebug  GGML_CUDA_GRAPH_DEBUG=1, stderr kept in <name>.graphdebug.log (NCCL only, mb4 and mb1)
#   eager     CUDA_MODULE_LOADING=EAGER NCCL_RUNTIME_CONNECT=0 (NCCL only, mb4 and mb1)
# Part C (cost model v5 calibration, 30B only, after all arms of both models): tg128 -r 5 (-p 0 -n 128, no trace) of
#   the model truncated to L/4 and L/2 layers (--layer-limit, as the s3c anchors in scripts/phase5_box.sh; L=48 from
#   the layout) as plans `single` and `layer-4`, plus the full model as `single` and `layer-4`. Names
#   <model>-<plan>-tg128 (full) and <model>-<plan>-L<k>-tg128. Each has the 20 ms clock log <name>.clocks.csv (as
#   s3d part B) and <name>.window.txt (`start=<epoch s> end=<epoch s>`, date +%s.%N before and after the whole
#   process, load included; the bench does not report the timed phase's wall clock) to cut the log. Layer-4 with a
#   layer limit is as yet untried on a box (the anchors ran single and tp-2 only).
# Run count: per model 4 nograph + 4 default + 5 varykey + 2 graphdebug + 2 eager = 17; 34 for the arms, 6 in part C.
# Wall time (s3d-20261120: about 30 s per pp2048 run incl. load; here about 40 s for -r 5 runs, 28 s for -r 1): per
# model 12 x 40 s + 5 x 28 s = 10 min, both 21 min; part C 6 x 40 s = about 5 min; build and plans about 5 min: about
# 31 min. BUDGET_MIN 40 leaves 9 min margin; runs that would start later are skipped (part C last, so it goes first).
# Stages: gpus (count and power gate), build, models, plans (hw, layout, candidates, mb plans), arms, part C.
# Results: bench/results/<s>/ (<name>.json|.log|.rc, <name>.trace.jsonl, nograph-check.txt, part C <name>.clocks.csv and
# <name>.window.txt, gpu-samples.csv at 1 Hz, env.txt), ends with
# S3E_DIAG_DONE in runs/vast/s3e_diag.log and the DONE file. A rerun skips runs whose .rc exists.
# Knobs: BUDGET_MIN (40, measured from the start; runs that would start later are skipped), MODELS, DRY_RUN=1 (print the
# commands only), OUT_DIR.
DRY="${DRY_RUN:-0}"
if [ "$DRY" != 1 ]; then
  exec 8> /root/tetraeder_s3e_diag.lock
  flock -n 8 || exit 0
fi
cd "$(dirname "$0")/.." || exit 1
session="${SESSION:?SESSION required}"
out="${OUT_DIR:-$PWD/bench/results/$session}"
log=runs/vast/s3e_diag.log
mkdir -p "$out/plans" "$out/layouts" "$(dirname "$log")"
[ "$DRY" = 1 ] || exec >> "$log" 2>&1
say() { echo "[$(date +%T)] $*"; }
[ "$DRY" = 1 ] || echo "=== RUN $(date +%T) s3e_diag_box.sh ==="
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
t_start=$(date +%s)
budget_s=$((${BUDGET_MIN:-40} * 60))

clock_pid=""
clock_start() { # clock_start NAME: nvidia-smi every 20 ms into NAME.clocks.csv (as s3d_diag_box.sh)
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

finish() {
  clock_stop
  [ "$DRY" = 1 ] || touch "$out/gpu-sampler.stop"
  touch "$out/DONE"
  say "S3E_DIAG_DONE rc=${1:-0}"
}
die() {
  say "FATAL: $*"
  finish 1
  exit 1
}

say "s3e_diag start (session=$session threads=$threads models=\"$models\" out=$out)"
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

# run_cmd NAME CMD...: result to NAME.json, stderr to NAME.${errext:-log}, exit code to NAME.rc.
# With clocks=1 in its environment: clock log beside it and NAME.window.txt (epoch start and end of the process).
run_cmd() {
  local name=$1 rc w0 w1
  shift
  [ -f "$out/$name.rc" ] && { say "$name: done before"; return 0; }
  if [ "$DRY" = 1 ]; then
    [ "${clocks:-0}" = 1 ] && clock_start "$name"
    echo "PLAN $* > $name.json"
    return 0
  fi
  if [ $(($(date +%s) - t_start)) -gt "$budget_s" ]; then say "BUDGET spent: skip $name"; return 0; fi
  say "$name start"
  [ "${clocks:-0}" = 1 ] && clock_start "$name"
  w0=$(date +%s.%N)
  timeout -k 30 5m "$@" < /dev/null > "$out/$name.json" 2> "$out/$name.${errext:-log}" 8>&-
  rc=$?
  w1=$(date +%s.%N)
  if [ "${clocks:-0}" = 1 ]; then
    clock_stop
    echo "start=$w0 end=$w1" > "$out/$name.window.txt"
  fi
  echo "$rc" > "$out/$name.rc"
  say "$name done rc=$rc"
}

# tet_run NAME PLAN P R MODEL [K=V...]: pp test with host trace; K=V are set in the run's environment
tet_run() {
  local name=$1 plan=$2 p=$3 r=$4 m=$5
  shift 5
  run_cmd "$name" env "$@" "TETRAEDER_EXEC_TRACE=$out/$name.trace.jsonl" "$tet" bench -m "${FILE[$m]}"     --engine "${ENGINE[$m]}" --plan "$plan" -p "$p" -n 0 -r "$r" -t "$threads" --ngl -1 --ubatch 512 --flash-attn on -o json
}

# nograph_void=1 skips the remaining nograph runs; nograph-check.txt holds the verdict (see header)
nograph_void=0
graph_lines='CUDA graph warmup (complete|reset)|CUDA Graph id.*reused'
check_nograph() { # check_nograph NAME: verdict on the first nograph run, which ran with GGML_CUDA_GRAPH_DEBUG=1
  [ "$DRY" = 1 ] && return 0
  local f="$out/nograph-check.txt" rc n
  if [ ! -f "$f" ]; then
    [ -f "$out/$1.rc" ] || { nograph_void=1; say "nograph check: $1 not run, nograph runs skipped"; return 0; }
    rc=$(cat "$out/$1.rc")
    n=$(grep -Ec "$graph_lines" "$out/$1.log" 2> /dev/null)
    if [ "$rc" != 0 ]; then echo "void: $1 failed rc=$rc" > "$f"
    elif [ "${n:-0}" != 0 ]; then echo "void: $1 logged $n CUDA graph capture or replay lines with graphs disabled" > "$f"
    else echo ok > "$f"; fi
  fi
  say "nograph check: $(cat "$f")"
  case $(cat "$f") in void*) nograph_void=1; say "nograph runs skipped" ;; esac
}
note_graphdebug() { # note_graphdebug NAME: positive control, the default-graphs run must have logged capture lines
  [ "$DRY" = 1 ] && return 0
  local f="$out/nograph-check.txt"
  [ -f "$f" ] && [ -f "$out/$1.rc" ] && ! grep -q '^note:' "$f" || return 0
  [ "$(grep -Ec "$graph_lines" "$out/$1.graphdebug.log" 2> /dev/null)" = 0 ] \
    && echo "note: $1 (graphs on, GRAPH_DEBUG=1) logged no capture line, so a silent nograph run proves nothing" >> "$f"
  return 0
}

first_nograph=1
for m in $models; do
  for arm in nograph default; do
    for mb in 4 1; do
      for ar in nccl internal; do
        envs=()
        [ "$ar" = internal ] && envs+=(GGML_CUDA_ALLREDUCE=internal)
        [ "$arm" = nograph ] && envs+=(GGML_CUDA_DISABLE_GRAPHS=1)
        name="$m-tp2pp2-mb$mb-pp-$ar-$arm"
        if [ "$arm" = nograph ] && [ "$nograph_void" = 1 ]; then
          say "$name: skipped, nograph check void"
          continue
        fi
        [ "$arm" = nograph ] && [ "$first_nograph" = 1 ] && envs+=(GGML_CUDA_GRAPH_DEBUG=1)
        tet_run "$name" "$out/plans/$m-tp2-pp2-mb$mb.toml" 2048 5 "$m" "${envs[@]}"
        if [ "$arm" = nograph ] && [ "$first_nograph" = 1 ]; then
          first_nograph=0
          check_nograph "$name"
        fi
      done
    done
  done
  for p in 2048 2040 2032 2024 2016; do
    tet_run "$m-tp2pp2-mb4-pp-nccl-varykey-$p" "$out/plans/$m-tp2-pp2-mb4.toml" "$p" 1 "$m"
  done
  for mb in 4 1; do
    errext=graphdebug.log tet_run "$m-tp2pp2-mb$mb-pp-nccl-graphdebug" "$out/plans/$m-tp2-pp2-mb$mb.toml" 2048 5 "$m" \
      GGML_CUDA_GRAPH_DEBUG=1
  done
  note_graphdebug "$m-tp2pp2-mb4-pp-nccl-graphdebug"
  for mb in 4 1; do
    tet_run "$m-tp2pp2-mb$mb-pp-nccl-eager" "$out/plans/$m-tp2-pp2-mb$mb.toml" 2048 5 "$m" \
      CUDA_MODULE_LOADING=EAGER NCCL_RUNTIME_CONNECT=0
  done
done

# ---- part C: cost model v5 calibration, 30B only (header) ----
m=qwen3-30b-a3b
case " $models " in
  *" $m "*)
    if [ "$DRY" = 1 ]; then n_layer=48; else n_layer=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["n_layer"])' "$out/layouts/$m.json"); fi
    for plan_id in single layer-4; do
      for lim in $((n_layer / 4)) $((n_layer / 2)) full; do
        cmd=("$tet" bench -m "${FILE[$m]}" --engine "${ENGINE[$m]}" --plan "$out/plans/$m-$plan_id.toml" -p 0 -n 128 -r 5
          -t "$threads" --ngl -1 --ubatch 512 --flash-attn on -o json)
        name="$m-$plan_id-tg128"
        if [ "$lim" != full ]; then
          cmd+=(--layer-limit "$lim")
          name="$m-$plan_id-L$lim-tg128"
        fi
        clocks=1 run_cmd "$name" "${cmd[@]}"
      done
    done
    ;;
esac

if [ "$DRY" != 1 ]; then
  for f in "$out"/*.trace.jsonl; do
    [ -f "$f" ] && python3 scripts/exec_trace_summary.py "$f" > "${f%.jsonl}.summary.txt" 2>&1
  done
fi
finish 0
