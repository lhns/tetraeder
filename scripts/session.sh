#!/usr/bin/env bash
# Box session profiles (docs/sessions.md has box requirements, estimates and the fetch steps):
#   SESSION=b1-exit      Phase 2 exit verdict: builder with the ADR 0038 levers, KTransformers, llama.cpp's expert
#                        cache at the tuned CPU share (mmap and dio), a builder/KT repeat pair; ktransformers-avx2 and
#                        the cache repeat only if time is left
#   SESSION=b1b-levers   the lever stages on a second 235B box: builder, low-bit KV (kv-*), routing trace and replay
#   SESSION=b2-proxy     Claude Code proxy rerun (48 GB card), then KV quality on the 30B and the scripts in B2_STAGES
#   SESSION=b3-planner   llama.cpp's cache tuner on the 30B, the disk curve and routing trace of the builder fixture
#                        (ADR 0056), then the planner dense fixture
#   SESSION=b3-detector  detector campaign, then kernels-cpu
#   SESSION=gpu-smoke    weekly GPU smoke (scripts/gpu_smoke.sh drives it): the gpu_* tests on the synthetic fixtures,
#                        the Qwen3-0.6B golden, one A/A pair for the measured noise band; TETRAEDER_REQUIRE_GPU=1
# Every profile but b1-exit (the verdict session; NOISE_STAGE=1 opts in) ends with an A/A pair (noise_stage, NOISE_STAGE=0 turns it off) when 12 minutes are left: the sessions
# are the source of the measured noise bands (docs/sessions.md, "Noise bands").
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
# the log appends across relaunches: waits look only after the last header (scripts/log_done.sh)
echo "=== RUN $(date +%FT%T) session.sh ===" >> "$log"
# minutes for this script: the box cap less 20% (144 of 180 min; b2 has the 6 h proxy exception: 288), less about 25
# for provisioning before and fetch, verify and destroy after
case "$name" in
  b1-exit | b1b-levers | b3-planner | b3-detector) budget=120 ;;
  b2-proxy) budget=260 ;;
  gpu-smoke) budget=30 ;; # about 17 min of work (build 10, tests 3, A/A pair 4); the box lives 40 min with provisioning
  *) echo "SESSION must be b1-exit, b1b-levers, b2-proxy, b3-planner, b3-detector or gpu-smoke, not '$name'" >&2; exit 2 ;;
esac
state="runs/vast/session-$name.start"
[ -s "$state" ] || echo "$(date +%s) $(date +%Y%m%d)" > "$state"
read -r start RESULTS_DATE < "$state"
DEADLINE=$((start + ${SESSION_MIN:-$budget} * 60))
export RESULTS_DATE DEADLINE RESUME=1
# Provenance of every result (ADR 0056): the commit taxameter pushed (the box checkout may not be a git
# repo for later stages) and a session name that survives a relaunch (built from the start time).
TETRAEDER_GIT_COMMIT="${TETRAEDER_GIT_COMMIT:-$(git rev-parse HEAD 2> /dev/null)}"
host_id="${CONTAINER_ID:-$(hostname)}"
TETRAEDER_SESSION="$RESULTS_DATE-$host_id-$(date -u -d "@$start" +%H%M%S)"
export TETRAEDER_GIT_COMMIT TETRAEDER_SESSION
results="bench/results/$RESULTS_DATE"
mkdir -p "$results"
models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"

step() {
  echo "[$(date +%T)] start $1"
  shift
  "$@" 8>&-
  local rc=$?
  echo "[$(date +%T)] done rc=$rc"
  return $rc
}
# cap_min: minutes left less 2 for the finish, at least 1 (timeout 0 would mean no limit)
cap_min() { local t=$(($(left_min) - 2)); [ "$t" -ge 1 ] || t=1; echo "${t}m"; }
# left_min: minutes until the deadline
left_min() { echo $(((DEADLINE - $(date +%s)) / 60)); }
# fetch <hf spec> <file under models_dir>: taxameter's box downloader, once per box
fetch() {
  [ -s "$models_dir/$2" ] && return 0
  TAXAMETER_DIR="$PWD" bash "${TAXAMETER_HOME:-/root/.taxameter}/box.sh" models "$1" 8>&-
}
# box_env: the image toolchain under /opt/cargo (as nightly.sh), ggml CUDA built for this box's GPUs only
box_env() {
  if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
  export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
  # shellcheck source=scripts/box_libs.sh
  . scripts/box_libs.sh
  export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
  command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
  if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
    local archs
    archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
    if [ -n "$archs" ]; then export TETRAEDER_CUDA_ARCHS="$archs"; fi
  fi
}
q8="Qwen/Qwen3-8B-GGUF:Qwen3-8B-Q4_K_M.gguf"
f8="Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
# noise_stage <out dir>: the phase0 A/A pair noise-cuda-dense (llama-bench against itself, interleaved, the flags of the
# Phase 1 gate) on this box; `tetraeder-bench noise update` turns its results JSON into the measured band of the GPU
# class. Needs 12 minutes and an otherwise idle GPU. Exit 0 and 3 (a noisy box is data) pass.
noise_stage() {
  local out=$1 lb="${LLAMA_BENCH:-/opt/llama.cpp/bin/llama-bench}" rc
  [ "${NOISE_STAGE:-1}" = 1 ] || { echo "skip noise: NOISE_STAGE=0"; return 0; }
  [ "$(left_min)" -ge 12 ] || { echo "skip noise: $(left_min) min left"; return 0; }
  [ -x "$lb" ] || { echo "skip noise: no $lb"; return 0; }
  box_env
  mkdir -p "$out"
  fetch "$q8" "$f8" || { echo "noise: the 8B fetch failed"; return 1; }
  timeout -k 30 "$(cap_min)" cargo build --release -p tetraeder-bench || return 1
  timeout -k 30 "$(cap_min)" target/release/tetraeder-bench run bench/spec/phase0.toml --only noise-cuda-dense \
    --repeats "${NOISE_REPEATS:-8}" --strict --out-dir "$out" --llama-bench "$lb" --models-dir "$models_dir"
  rc=$?
  [ "$rc" = 0 ] || [ "$rc" = 3 ]
}
# smoke_test <log> <cargo args>: one cargo test run, output to the log
smoke_test() {
  local log=$1
  shift
  timeout -k 30 "$(cap_min)" cargo test --release "$@" > "$log" 2>&1
}
# gpu_smoke <out dir>: the GPU check between exit sessions. TETRAEDER_REQUIRE_GPU=1 makes a missing GPU a failure
# (Backend::gpu_available panics), and a test that skips for a missing model leaves a "skip:" line that fails the
# smoke. Writes <out>/summary.json, "ok": true only when every stage passed and at least GPU_SMOKE_MIN_TESTS ran.
gpu_smoke() {
  local out=$1 rc_tests=-1 rc_golden=-1 rc_noise=-1 passed=0 skips=0 cuda_ok=0 ok=false
  mkdir -p "$out"
  export TETRAEDER_REQUIRE_GPU=1 TETRAEDER_MODELS_DIR="$models_dir"
  # shellcheck disable=SC1091
  . "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
  echo "[$(date +%T)] cuda: $CUDA_ENV_STATUS"
  case "$CUDA_ENV_STATUS" in ok | ok_preload) cuda_ok=1 ;; esac
  if [ "$cuda_ok" = 1 ] && bash scripts/ensure_patches.sh 8>&-; then
    box_env
    fetch "Qwen/Qwen3-0.6B-GGUF:Qwen3-0.6B-Q8_0.gguf" "Qwen__Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf"
    # the gpu_ tests that need no model file; gpu_cuda_graphs_decode and gpu_real_moe_streaming need the 30B and
    # are not in these binaries
    step gpu_tests smoke_test "$out/gpu-tests.log" -p tetraeder-model --features tetraeder-ggml-sys/cuda \
      --test cpu_experts --test quantized --test slots --test parity -- gpu_ --nocapture --test-threads 1
    rc_tests=$?
    if [ "$(left_min)" -ge 8 ]; then
      step golden smoke_test "$out/golden.log" -p tetraeder-engine --features tetraeder-ggml-sys/cuda --test golden \
        -- --nocapture --test-threads 1
      rc_golden=$?
    else
      echo "skip golden: $(left_min) min left"
    fi
    step noise noise_stage "$out"
    rc_noise=$?
  fi
  passed=$(sed -n 's/^test result: ok\. \([0-9]*\) passed.*/\1/p' "$out/gpu-tests.log" 2> /dev/null | awk '{s += $1} END {print s + 0}')
  skips=$(cat "$out/gpu-tests.log" "$out/golden.log" 2> /dev/null | grep -c '^skip')
  if [ "$cuda_ok" = 1 ] && [ "$rc_tests" = 0 ] && [ "$rc_golden" = 0 ] && [ "$rc_noise" = 0 ] \
    && [ "$passed" -ge "${GPU_SMOKE_MIN_TESTS:-5}" ] && [ "$skips" = 0 ]; then
    ok=true
  fi
  q() { nvidia-smi --query-gpu="$1" --format=csv,noheader 2> /dev/null | head -1 | tr -d '"'; }
  cat > "$out/summary.json" << EOJ
{
  "schema": 1,
  "ok": $ok,
  "cuda_status": "$CUDA_ENV_STATUS",
  "rc": {"gpu_tests": $rc_tests, "golden": $rc_golden, "noise": $rc_noise},
  "gpu_tests_passed": $passed,
  "min_gpu_tests": ${GPU_SMOKE_MIN_TESTS:-5},
  "skipped_tests": $skips,
  "require_gpu": true,
  "host_id": "${CONTAINER_ID:-}",
  "gpu": "$(q name)",
  "driver": "$(q driver_version)",
  "clocks_sm_mhz": "$(q clocks.sm)",
  "clocks_max_sm_mhz": "$(q clocks.max.sm)",
  "throttle_reasons": "$(q clocks_throttle_reasons.active)",
  "finished": "$(date -u +%FT%TZ)"
}
EOJ
  echo "gpu-smoke ok=$ok passed=$passed skips=$skips rc: tests=$rc_tests golden=$rc_golden noise=$rc_noise"
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
        stages="smoke disk-curve main-prefetch ktransformers llamacpp-cache llamacpp-cache-dio main-repeat ktransformers-repeat"
        stages="$stages ktransformers-avx2 llamacpp-cache-repeat"
      else
        # each lever stage pairs with main-prefetch on this box: kv-* right after it, the trace
        # main-iosvc and main-overlap (ADR 0054) follow the kv stages; main-overlap-ram runs if the deadline allows
        stages="smoke disk-curve main-prefetch kv-q8 kv-q8-slots kv-q8q4 kv-q8q4-slots main-iosvc main-overlap trace main-overlap-ram"
        # the ledger directory of this session (ADR 0056): last, the deadline cuts it first
        stages="$stages anchors-builder-short"
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
      # ADR 0056, the builder fixture's session stages on the 30B: the disk curve (2 min) and the routing trace with its
      # replay (5 min), plus the campaign's extra build (3 min); the candidates themselves are not run here (planner-moe-builder is
      # 3.7 h of runs). Needs the deadline: skipped with fewer than 25 minutes left
      if [ "$(left_min)" -ge 25 ]; then
        step planner_builder env OUT_DIR="$PWD/$results/planner-moe-builder" STAGES="disk-curve trace" \
          bash scripts/planner_builder_campaign.sh
      else
        echo "skip planner_builder (disk-curve, trace): $(left_min) min left"
      fi
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
    gpu-smoke)
      gpu_smoke "$results/gpu-smoke"
      ;;
  esac
  # the A/A pair for the measured noise band; gpu-smoke ran it already
  # b1-exit is the verdict session: no extra GPU stage unless NOISE_STAGE=1 is passed for it
  if [ "$name" != gpu-smoke ]; then NOISE_STAGE="${NOISE_STAGE:-$([ "$name" = b1-exit ] && echo 0 || echo 1)}" step noise noise_stage "$results/noise"; fi
  touch "$results/DONE"
  echo "[$(date +%T)] SESSION_DONE ($(left_min) min before the deadline)"
} >> "$log" 2>&1
