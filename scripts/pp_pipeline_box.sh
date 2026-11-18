#!/usr/bin/env bash
# Pipelined layer prefill on a 2-GPU box (ADR 0077, "Box confirmation"). Run after phase5_box.sh of the same session
# (it needs the release build, the models and the session's layer-2 plans):
#   SESSION=<s> bash scripts/pp_pipeline_box.sh        (taxameter run --env SESSION=<s>)
# Per model, two rounds (round 2 in reverse order) of three runs, each pp2048 only (-n 0), ubatch 512:
#   tetraeder bench with the planner's layer-2 plan (prefill_micro_batches = 4), the same plan with = 1, llama.cpp -sm layer.
# Results: bench/results/<s>-pp/ (pp-<model>-<kind>-r<k>.json, .rc, .log, plans/, gpu-samples.csv, env.txt); ends with
# PP_PIPELINE_DONE in runs/vast/pp_pipeline.log. A rerun skips runs whose .rc exists. Knobs: P5_DIR (the session's phase5
# dir), BUDGET_MIN (25), MODELS, DRY_RUN=1 (print the commands only).
DRY="${DRY_RUN:-0}"
if [ "$DRY" != 1 ]; then
  exec 8> /root/tetraeder_pp_pipeline.lock
  flock -n 8 || exit 0
fi
cd "$(dirname "$0")/.." || exit 1
session="${SESSION:?SESSION required}"
p5="${P5_DIR:-$PWD/bench/results/$session/phase5}"
out="$PWD/bench/results/$session-pp"
log=runs/vast/pp_pipeline.log
mkdir -p "$out/plans" "$(dirname "$log")"
exec >> "$log" 2>&1
say() { echo "[$(date +%T)] $*"; }
if [ "$DRY" != 1 ]; then
  if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
  export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
  # shellcheck disable=SC1091
  . "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
  # shellcheck source=scripts/box_libs.sh
  . scripts/box_libs.sh
fi
threads=8
[ "$DRY" = 1 ] || threads=$(bash scripts/cpu_quota.sh)
tet="${TET_BIN:-target/release/tetraeder}"
llama_bench="${LLAMA_BENCH:-/opt/llama.cpp-baseline/bin/llama-bench}"
models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
declare -A FILE=(
  [qwen3-32b]="$models_dir/Qwen__Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf"
  [qwen3-30b-a3b]="$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
)
declare -A ENGINE=([qwen3-32b]=dense [qwen3-30b-a3b]=moe)
t_start=$(date +%s)
budget_s=$((${BUDGET_MIN:-25} * 60))
say "pp_pipeline start (session=$session threads=$threads out=$out)"
if [ "$DRY" != 1 ]; then
  OUT="$out" SAMPLER_PARENT=$$ setsid nohup bash scripts/gpu_sampler.sh 8>&- > /dev/null 2>&1 &
  { date; nvidia-smi --query-gpu=index,name,driver_version,power.limit,power.default_limit --format=csv; } > "$out/env.txt" 2>&1
fi

# run_cmd NAME CMD...: result to $out/NAME.json, exit code to NAME.rc
run_cmd() {
  local name=$1 rc
  shift
  [ -f "$out/$name.rc" ] && { say "$name: done before"; return 0; }
  if [ "$DRY" = 1 ]; then echo "PLAN $* > $name.json"; return 0; fi
  if [ $(($(date +%s) - t_start)) -gt "$budget_s" ]; then say "BUDGET spent: skip $name"; return 0; fi
  say "$name start"
  timeout -k 30 15m "$@" < /dev/null > "$out/$name.json" 2> "$out/$name.log" 8>&-
  rc=$?
  echo "$rc" > "$out/$name.rc"
  say "$name done rc=$rc"
}

for m in ${MODELS:-qwen3-32b qwen3-30b-a3b}; do
  src="$p5/plans/$m-layer-2.toml"
  if [ "$DRY" != 1 ] && [ ! -f "$src" ]; then
    say "no plan $src"
    continue
  fi
  cp "$src" "$out/plans/$m-layer-2-m4.toml" 2> /dev/null
  sed 's/^\([[:space:]]*prefill_micro_batches[[:space:]]*=[[:space:]]*\)[0-9][0-9]*/\11/' "$src" > "$out/plans/$m-layer-2-m1.toml" 2> /dev/null
  grep -H prefill_micro_batches "$out/plans/$m-layer-2-m4.toml" "$out/plans/$m-layer-2-m1.toml" 2> /dev/null
  for k in 1 2; do
    kinds="m4 m1 llama"
    [ "$k" = 2 ] && kinds="llama m1 m4"
    for kind in $kinds; do
      case $kind in
        llama) run_cmd "pp-$m-llama-layer-r$k" "$llama_bench" -m "${FILE[$m]}" -ngl 99 -fa 1 -p 2048 -n 0 -r 1 -t "$threads" -ub 512 -sm layer -o json ;;
        *) run_cmd "pp-$m-$kind-r$k" "$tet" bench -m "${FILE[$m]}" --engine "${ENGINE[$m]}" --plan "$out/plans/$m-layer-2-$kind.toml" \
          -p 2048 -n 0 -r 1 -t "$threads" --ngl -1 --ubatch 512 --flash-attn on -o json ;;
      esac
    done
  done
done
[ "$DRY" = 1 ] || touch "$out/gpu-sampler.stop"
say "PP_PIPELINE_DONE"
