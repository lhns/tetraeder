#!/usr/bin/env bash
# Layer-N decode overhead diagnostic on a 4-GPU box (ADR 0069 History 2026-11-19). Run after phase5_box.sh of the same
# session (it needs the release build, the models and the session's single and layer-4 plans):
#   SESSION=<s> bash scripts/layer_decode_box.sh        (taxameter run --env SESSION=<s>)
# tg128 only (-p 0 -n 128 -r 5), per model: llama.cpp -sm none and -sm layer, our single and layer-4 with the executor
# toggles off (TETRAEDER_RANGE_UID=0 TETRAEDER_ASYNC_INPUTS=0; both are on by default), each alone and both, and traced runs (TETRAEDER_EXEC_TRACE, issue and
# sync mode). The 30B (the M4 gate) runs a second, reversed round of the untraced runs for interleaved pairs; the 32B
# runs one round of the key runs; then the GPU parity tests that cover the toggles run with both on (the kept-range
# replay with GGML_CUDA_GRAPH_DEBUG=1, the one-rank bit-equality and the gpu4_ shapes of both builders), whose
# result decides whether a toggle may become the default. About 15 minutes after the build.
# Results: bench/results/<s>-decode/ (<model>-<kind>-r<k>.json, .rc, .log, .trace.jsonl, gpu-samples.csv, env.txt);
# parity-toggles.log and .rc; ends with LAYER_DECODE_DONE in runs/vast/layer_decode.log. A rerun skips runs whose .rc
# exists. Knobs: P5_DIR (the session's phase5 dir), BUDGET_MIN (18), MODELS, DRY_RUN=1 (print the commands only).
DRY="${DRY_RUN:-0}"
if [ "$DRY" != 1 ]; then
  exec 8> /root/tetraeder_layer_decode.lock
  flock -n 8 || exit 0
fi
cd "$(dirname "$0")/.." || exit 1
session="${SESSION:?SESSION required}"
p5="${P5_DIR:-$PWD/bench/results/$session/phase5}"
case $session in *-decode) decode_name=$session ;; *) decode_name=$session-decode ;; esac
out="${OUT_DIR:-$PWD/bench/results/$decode_name}"
log=runs/vast/layer_decode.log
mkdir -p "$out" "$(dirname "$log")"
[ "$DRY" = 1 ] || exec >> "$log" 2>&1
say() { echo "[$(date +%T)] $*"; }
if [ "$DRY" != 1 ]; then
  if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
  export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
  # shellcheck disable=SC1091
  . "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
  # shellcheck source=scripts/box_libs.sh
  . scripts/box_libs.sh
  # the parity step builds ggml-sys; without the arch pin it recompiles for every arch and times out
  if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
    archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
    export TETRAEDER_CUDA_ARCHS="${archs:-86}"
  fi
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
budget_s=$((${BUDGET_MIN:-18} * 60))
say "layer_decode start (session=$session threads=$threads out=$out)"
if [ "$DRY" != 1 ]; then
  OUT="$out" SAMPLER_PARENT=$$ setsid nohup bash scripts/gpu_sampler.sh 8>&- > /dev/null 2>&1 &
  { date; nvidia-smi --query-gpu=index,name,driver_version,power.limit,clocks.sm,clocks.max.sm --format=csv; } > "$out/env.txt" 2>&1
fi

# run_cmd NAME CMD...: result to $out/NAME.json, exit code to NAME.rc
run_cmd() {
  local name=$1 rc
  shift
  [ -f "$out/$name.rc" ] && { say "$name: done before"; return 0; }
  if [ "$DRY" = 1 ]; then echo "PLAN $* > $name.json"; return 0; fi
  if [ $(($(date +%s) - t_start)) -gt "$budget_s" ]; then say "BUDGET spent: skip $name"; return 0; fi
  say "$name start"
  timeout -k 30 5m "$@" < /dev/null > "$out/$name.json" 2> "$out/$name.log" 8>&-
  rc=$?
  echo "$rc" > "$out/$name.rc"
  say "$name done rc=$rc"
}

# run_kind MODEL KIND ROUND: one run. KIND: llama-none, llama-layer, or <plan>[-uid][-async][-trace|-tracesync] with plan
# single or layer4.
run_kind() {
  local m=$1 kind=$2 k=$3 plan envs=()
  case $kind in
    llama-none) run_cmd "$m-$kind-r$k" "$llama_bench" -m "${FILE[$m]}" -ngl 99 -fa 1 -p 0 -n 128 -r 5 -t "$threads" -sm none -mg 0 -o json; return ;;
    llama-layer) run_cmd "$m-$kind-r$k" "$llama_bench" -m "${FILE[$m]}" -ngl 99 -fa 1 -p 0 -n 128 -r 5 -t "$threads" -sm layer -o json; return ;;
  esac
  case $kind in
    single*) plan="$p5/plans/$m-single.toml" ;;
    layer4*) plan="$p5/plans/$m-layer-4.toml" ;;
  esac
  case $kind in *-uid*) envs+=(TETRAEDER_RANGE_UID=1) ;; *) envs+=(TETRAEDER_RANGE_UID=0) ;; esac
  case $kind in *-async*) envs+=(TETRAEDER_ASYNC_INPUTS=1) ;; *) envs+=(TETRAEDER_ASYNC_INPUTS=0) ;; esac
  case $kind in *-trace*) envs+=("TETRAEDER_EXEC_TRACE=$out/$m-$kind-r$k.trace.jsonl") ;; esac
  case $kind in *-tracesync) envs+=(TETRAEDER_EXEC_TRACE_SYNC=1) ;; esac
  run_cmd "$m-$kind-r$k" env "${envs[@]}" "$tet" bench -m "${FILE[$m]}" --engine "${ENGINE[$m]}" --plan "$plan" \
    -p 0 -n 128 -r 5 -t "$threads" --ngl -1 --flash-attn on -o json
}

for m in ${MODELS:-qwen3-30b-a3b qwen3-32b}; do
  if [ "$DRY" != 1 ] && { [ ! -f "$p5/plans/$m-single.toml" ] || [ ! -f "$p5/plans/$m-layer-4.toml" ]; }; then
    say "no single or layer-4 plan for $m in $p5/plans"
    continue
  fi
  if [ "$m" = qwen3-30b-a3b ]; then
    pairs="llama-none llama-layer single layer4 layer4-uid layer4-async layer4-uid-async single-uid-async"
    traces="layer4-trace layer4-tracesync single-trace layer4-uid-async-trace"
    rounds="1 2"
  else
    pairs="llama-none llama-layer single layer4 layer4-uid-async single-uid-async"
    traces="layer4-trace"
    rounds="1"
  fi
  for k in $rounds; do
    kinds="$pairs"
    # round 2 reversed, so each pair is interleaved
    [ "$k" = 2 ] && kinds=$(echo "$pairs" | tr ' ' '\n' | tac | tr '\n' ' ')
    for kind in $kinds; do run_kind "$m" "$kind" "$k"; done
    [ "$k" = 1 ] && for kind in $traces; do run_kind "$m" "$kind" 1; done
  done
done
# parity with both toggles on: the test binaries are the gpu-parity stage's (same features, release), so nothing builds
toggles=(TETRAEDER_RANGE_UID=1 TETRAEDER_ASYNC_INPUTS=1 TETRAEDER_REQUIRE_GPU=1 TETRAEDER_REQUIRE_GPUS=4
  "TETRAEDER_MOE_GGUF=${FILE[qwen3-30b-a3b]}" "TETRAEDER_MODELS_DIR=$models_dir")
features=(--release --no-fail-fast --features tetraeder-ggml-sys/cuda)
if [ -f "$out/parity-toggles.rc" ]; then
  say "parity-toggles: done before"
elif [ "$DRY" = 1 ]; then
  echo "PLAN env ${toggles[*]} GGML_CUDA_GRAPH_DEBUG=1 cargo test ${features[*]} -p tetraeder-executor --test gpu -- gpu_cuda_graphs_replay_kept_ranges"
  echo "PLAN env ${toggles[*]} cargo test ${features[*]} -p tetraeder-model --test gpu_sharded --test gpu_sharded_moe -- gpu_one_rank gpu4_"
elif [ $(($(date +%s) - t_start)) -gt "$budget_s" ]; then
  say "BUDGET spent: skip parity-toggles"
else
  say "parity-toggles start"
  {
    env "${toggles[@]}" GGML_CUDA_GRAPH_DEBUG=1 timeout -k 30 10m cargo test "${features[@]}" -p tetraeder-executor --test gpu       -- gpu_cuda_graphs_replay_kept_ranges --nocapture 8>&- &&
      env "${toggles[@]}" timeout -k 30 15m cargo test "${features[@]}" -p tetraeder-model --test gpu_sharded         --test gpu_sharded_moe -- gpu_one_rank gpu4_ --nocapture 8>&-
  } < /dev/null > "$out/parity-toggles.log.full" 2>&1
  rc=$?
  # cargo counts a skipped test as passed
  grep -qE '(^|[ .])skip: ' "$out/parity-toggles.log.full" && rc=3
  cut -c1-300 "$out/parity-toggles.log.full" > "$out/parity-toggles.log"
  echo "$rc" > "$out/parity-toggles.rc"
  say "parity-toggles done rc=$rc"
fi
if [ "$DRY" != 1 ]; then
  touch "$out/gpu-sampler.stop"
  for f in "$out"/*.trace.jsonl; do
    [ -f "$f" ] && python3 scripts/exec_trace_summary.py "$f" > "${f%.jsonl}.summary.txt" 2>&1
  done
fi
say "LAYER_DECODE_DONE"
