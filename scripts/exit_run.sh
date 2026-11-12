#!/usr/bin/env bash
# Phase 2 exit run (ADR 0035, 0037): Qwen3-235B-A22B Q4_K_M, larger than the box RAM, builder with MissPolicy::Cpu
# (tests/exit_regime.rs) against llama.cpp with expert offload (-ot) on the same box and threads.
# Waits for scripts/fetch_exit_model.sh. Started detached by `taxameter run scripts/exit_run.sh`; ends with
# EXIT_RUN_DONE rc=N in runs/vast/exit_run.log. Every stage has a time limit; a failed smoke gate stops the run.
# Every builder, KTransformers and llama.cpp cache run is sampled by scripts/engine_sample.py (disk reads and queue depth,
# major faults, CPU and GPU busy, RSS and page cache, prefill and decode apart): "host" and the CPU flags in "env" of the
# result JSON. ggml is built once for the native ISA of the box (TETRAEDER_GGML_NATIVE=ON, ADR 0052 addendum
# 2026-11-09: a DL build would compile 14 CPU variants on a fresh box); "ggml_system_info" in each result records
# the ISA. ktransformers-avx2 forces kt-kernel's AVX2 kernels.
# Host and engine metrics also go to a collector (ADR 0060, scripts/otelcol_box.sh) started before the first stage and
# stopped before DONE: the sampler posts to it, the builder exports to it (OTEL_EXPORTER_OTLP_ENDPOINT), hostmetrics
# scrapes the box, all into runs/vast/otel/<start>/; then every result gets "host_otlp" next to "host". OTEL_COLLECTOR=0
# runs without it; ENGINE_OTLP=0 keeps the collector but turns the builder's export off (the overhead pair).
# Session profiles (scripts/session.sh) set RESULTS_DATE, RESUME=1 (skip stages with a result) and DEADLINE (skip stages
# that would end after it).
# Stage order: with STAGES set (session.sh passes B1_STAGES there) the stages run in the order of that list, once
# each; unset, in code order (all stages). The block below runs once per listed stage with ONLY set to its name.
order=""
for _s in ${STAGES:-}; do [[ " $order " == *" $_s "* ]] || order="$order $_s"; done
# DRY_RUN=1 prints the sequence and stops before any side effect
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "stage order:${order:- (code order, all stages)}"
  exit 0
fi
exec 8> /root/tetraeder_exit_run.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
bash scripts/ensure_patches.sh 8>&- || exit 1  # ggml patch queue (ADR 0052)
export TETRAEDER_GGML_NATIVE="${TETRAEDER_GGML_NATIVE:-ON}"
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# start the server with the lock fd closed: one started by cargo inherits fd 8 and keeps the flock (a relaunch
# then exits silently, b1b-levers 2026-11-10)
if [ "${RUSTC_WRAPPER:-}" = sccache ]; then sccache --start-server > /dev/null 2>&1 8>&- || true; fi
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
# RESULTS_DATE (scripts/session.sh) pins the results directory of a session that crosses midnight or resumes
date_dir="$PWD/bench/results/${RESULTS_DATE:-$(date +%Y%m%d)}"
mkdir -p "$date_dir" runs/vast
log=runs/vast/exit_run.log
stamp=$(date +%H%M%S)
# this box's GPU only, as moe_check.sh: another arch list rebuilds ggml-cuda for every arch (an hour, no cache hit)
if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
model="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}/Qwen__Qwen3-235B-A22B-GGUF/Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf"
threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"
threads=$(bash scripts/clamp_threads.sh "$threads")
# RAM the container may use: the cgroup limit if set, else MemTotal. The exit test needs a model larger than RAM, but
# Vast's cgroup limit can exceed the offer's RAM (183 GB on a "126 GB" offer, model 142 GB). So a balloon process
# holds the difference to EMULATE_RAM_GB (default 128) as touched anonymous memory (no swap: it stays resident), and
# both engines, page cache included, run within the rest. The RAM tier gets the emulated RAM minus RAM_RESERVE_GB.
# (cgroup v2 memory.max, else v1 limit_in_bytes, which reads as a huge number when unlimited; at most MemTotal)
total=$(awk '/MemTotal/ { print $2 * 1024 }' /proc/meminfo)
mem=$(cat /sys/fs/cgroup/memory.max 2> /dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2> /dev/null)
case "$mem" in '' | max) mem=$total ;; esac
[ "$mem" -gt "$total" ] && mem=$total
emulated=$((${EMULATE_RAM_GB:-128} * 1000000000))
balloon=$((mem - emulated))
# The balloon must outlive the OOM killer: engines get oom_score_adj 1000 (scripts/oom_engine.sh), the balloon -1000 when
# the container allows it. Before and after every stage balloon_pre/balloon_post check it (scripts/balloon_guard.py);
# a dead or shrunken balloon is restarted and the stage's result gets balloon_ok false and the reason, because the run
# had no memory pressure (20261112f: killed during the dio cache stage, later stages ran with the model resident).
start_balloon() {
  python3 -c "
import sys, time
n = int(sys.argv[1])
b = bytearray(n)
b[::4096] = b'\x01' * len(range(0, n, 4096))
print('balloon holds', n, 'B', flush=True)
time.sleep(10**9)
" "$balloon" >> runs/vast/exit_run.log 2>&1 &
  balloon_pid=$!
  if echo -1000 > /proc/$balloon_pid/oom_score_adj 2> /dev/null; then
    echo "[$(date +%T)] balloon pid $balloon_pid: oom_score_adj -1000"
  else
    echo "[$(date +%T)] balloon pid $balloon_pid: oom_score_adj -1000 not allowed (no CAP_SYS_RESOURCE); engines run at 1000"
  fi
  sleep 60  # touching the pages takes a while; the run must not start before
}
balloon_note=""
# balloon_pre: restart a balloon that died between stages; the next balloon_post records it
balloon_pre() {
  balloon_note=""
  [ "$balloon" -gt 0 ] || return 0
  local why
  why=$(python3 -I scripts/balloon_guard.py check --pid "$balloon_pid" --target "$balloon") && return 0
  balloon_note="before stage: $why"
  echo "[$(date +%T)] BALLOON PROBLEM $balloon_note; restarting"
  start_balloon
}
# balloon_post <stage> <json>: check after the stage; record balloon_ok in its JSON (a sidecar when it is no object)
balloon_post() {
  [ "$balloon" -gt 0 ] || return 0
  local why ok=true
  if ! why=$(python3 -I scripts/balloon_guard.py check --pid "$balloon_pid" --target "$balloon"); then
    balloon_note="${balloon_note:+$balloon_note; }after stage: $why"
    echo "[$(date +%T)] BALLOON PROBLEM in $1: $why; restarting"
    start_balloon
  fi
  [ -z "$balloon_note" ] || ok=false
  python3 -I scripts/balloon_guard.py merge --json "$2" --ok $ok --reason "$balloon_note" || true
  [ "$ok" = true ] || rc=31
}
if [ "$balloon" -gt 0 ]; then
  start_balloon
  trap 'kill $balloon_pid 2> /dev/null' EXIT
  mem=$emulated
fi
# The builder reads with direct I/O (no page cache): besides the tiers it holds the bounce buffers (1.3 GB) and about
# 1 GB of process memory (peak RSS 72.75 GB with a 70.8 GB RAM tier, 2026-11-02), so 8 GB is left for the rest.
ram_bytes=$((mem - ${RAM_RESERVE_GB:-8} * 1000000000))
# expert VRAM: VRAM_SLOT_BYTES if set, else what llama.cpp's tuned -ot puts on the GPU (stage llama-fit), else 5.5 GB
vram_bytes="${VRAM_SLOT_BYTES:-5500000000}"
# OT_FIT: the tuned llama.cpp -ot when the llama-fit stage is skipped
ot_fit="${OT_FIT:-blk\.[0-3]\.ffn_.*_exps=CUDA0;exps=CPU}"
ik_bench="${IK_BENCH:-/opt/ik_llama/bin/llama-bench}"
bench="${LLAMA_BENCH:-/opt/llama.cpp/bin/llama-bench}"
# natural text for the prompt (random token ids spread routing pessimistically), frozen: it was built from the
# checkout's AGENTS.md and ADRs, so every ADR edit changed the 32K prompt (ADR 0038, Results). Copied for the scripts
# that read /root/exit_prompt.txt (Colibri, KTransformers).
prompt=/root/exit_prompt.txt
cp bench/prompts/exit_prompt.txt "$prompt"
rc=0
# shellcheck source=scripts/stage_budget.sh
. scripts/stage_budget.sh
# Minutes a stage takes on the 91 GB / 3090 class (docs/sessions.md has the sources); the deadline guard uses them.
stage_min() {
  case "$1" in
    smoke) echo 3 ;;
    disk-curve) echo 2 ;;
    anchors-builder-short) echo 12 ;;
    gpu-tests) echo 20 ;;
    drift-ref*) echo 9 ;;
    drift-*) echo 4 ;;
    llama-fit) echo 10 ;;
    main-overlap* | main-prefetch* | main-repeat) budget_min 10 "$LAST_BUILDER_SEC" ;; # builder 32K stages: measured
    ktransformers) echo 25 ;;
    ktransformers-*) echo 20 ;;
    llamacpp-cache*) budget_min 30 "$LAST_CACHE_SEC" ;; # per share: 29 to 30 min with load and warmup on 20261111c and 20261111g
    moe-tune) echo 45 ;;
    reference | trace) echo 15 ;;
    kv-quality) echo 30 ;;
    exit-*) echo 40 ;;
    colibri-*) echo 60 ;;
    *) echo 10 ;;
  esac
}
# DEADLINE (epoch seconds, from scripts/session.sh): a stage that would end after it is skipped, so the box cap holds
fits() {
  [ -n "${DEADLINE:-}" ] || return 0
  local m
  m=$(stage_min "$1")
  [ $(($(date +%s) + m * 60)) -le "$DEADLINE" ] && return 0
  echo "[$(date +%T)] skip $1: needs about $m min, deadline $(date -d "@$DEADLINE" +%T)"
  return 1
}
# RESUME=1: a stage whose result file (exit-<name>-<stamp>.json) with numbers is already in date_dir is not rerun
stage_done() {
  [ "${RESUME:-0}" = 1 ] || return 1
  python3 -I - "$date_dir" "$1" << 'EOF'
import glob, json, os, re, sys
d, name = sys.argv[1:3]
for f in glob.glob(os.path.join(d, f"exit-{name}-*.json")):
    if not re.fullmatch(rf"exit-{re.escape(name)}-\d+\.json", os.path.basename(f)):
        continue
    try:
        j = json.load(open(f))
    except ValueError:
        continue
    if j.get("ok") is True or any(r.get("decode_tok_s") for r in j.get("runs") or []):
        sys.exit(0)
sys.exit(1)
EOF
}
# STAGES (space list of stage names, default all) lets a relaunch skip stages that already ran
want() {
  [ -z "${ONLY:-}" ] || [ "$ONLY" = "$1" ] || return 1
  [ -z "${STAGES:-}" ] || [[ " $STAGES " == *" $1 "* ]] || return 1
  if stage_done "$1"; then
    echo "[$(date +%T)] skip $1: result in $date_dir"
    return 1
  fi
  fits "$1"
}
# KTransformers installs while the model downloads (its own log and lock)
if want ktransformers || want ktransformers-avx2 || want ktransformers-repeat; then
  bash scripts/ktransformers_build.sh 8>&- &
  kt_build_pid=$!
fi
# Colibri builds while the model downloads (its own log and lock; fd 8 is this script's lock)
if want colibri-short || want colibri-deep; then
  bash scripts/colibri_build.sh 8>&- &
  colibri_build_pid=$!
fi
{
  echo "[$(date +%T)] start: threads $threads, memory $mem B (balloon ${balloon} B), ram tier $ram_bytes B, vram slots $vram_bytes B"
  nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active --format=csv
  # build while the model downloads
  cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda --test exit_regime --no-run || rc=21
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo)"
  echo "avx512 flags: $(grep -o -w 'avx512[a-z_0-9]*' /proc/cpuinfo | sort -u | tr '\n' ' ')"
  if want gpu-tests; then
    # GPU checks on the synthetic fixtures (no model needed): logits against RAM slots and prefetch (ADR 0045)
    echo "[$(date +%T)] start gpu-tests"
    TETRAEDER_REQUIRE_GPU=1 timeout -k 30 20m cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda --test cpu_experts       -- gpu_ --nocapture --test-threads 1 > "$date_dir/gpu-tests-$stamp.log" 2>&1
    # ADR 0054: the service and arrival-order compute give the serial logits on the GPU builder
    TETRAEDER_REQUIRE_GPU=1 timeout -k 30 20m cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda --test overlap       -- gpu_ --nocapture --test-threads 1 >> "$date_dir/gpu-tests-$stamp.log" 2>&1
    # ADR 0059: decode admission into VRAM slots stays within the GPU tolerance of the static tier
    TETRAEDER_REQUIRE_GPU=1 timeout -k 30 20m cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda --test admit       -- gpu_ --nocapture --test-threads 1 >> "$date_dir/gpu-tests-$stamp.log" 2>&1
    echo "[$(date +%T)] done gpu-tests rc=$?"
    tail -20 "$date_dir/gpu-tests-$stamp.log"
  fi
  echo "[$(date +%T)] waiting for the model"
  for _ in $(seq 1 1440); do bash scripts/log_done.sh runs/vast/fetch_exit.log FETCH_DONE && break; sleep 5; done
  grep FETCH_DONE runs/vast/fetch_exit.log | tail -1
  grep -q "FETCH_DONE rc=0" runs/vast/fetch_exit.log || { echo "model fetch failed"; rc=20; }
} >> "$log" 2>&1

# MAJ:MIN of the model's block device, probed once (a 2 GB direct read)
ensure_disk() {
  [ -n "${HOST_DISK_DEV:-}" ] || HOST_DISK_DEV=$(python3 scripts/engine_sample.py probe-disk "$model")
}

# membw_probe <name>: 10 s copy-bandwidth probe at the stage's threads, JSON to membw-<name>-<stamp>.json (MEMBW_PROBE=0
# skips it); sets $membw_json (empty when skipped or failed)
membw_probe() {
  membw_json=""
  [ "${MEMBW_PROBE:-1}" = 0 ] && return 0
  local f="$date_dir/membw-$1-$stamp.json"
  TETRAEDER_STAGE="$1" python3 scripts/engine_sample.py membw --seconds 10 --threads "$threads" > "$f" || { echo "membw probe failed for $1"; rm -f "$f"; return 0; }
  membw_json=$f
}

# exit_regime <name> <minutes> [ENV=VALUE...]: one harness run, JSON to exit-<name>-<stamp>.json, sampled alongside
exit_regime() {
  local name=$1 min=$2
  shift 2
  want "$name" || return 0
  # RANDOM_PROMPT=1: pseudo-random token ids instead of the prompt text, as llama-bench uses
  local text=(TETRAEDER_EXIT_PROMPT_FILE="$prompt")
  [[ " $* " == *" RANDOM_PROMPT=1 "* ]] && text=()
  # deep builder stages decode in 4 windows, so a warm-up after the prefill shows (run-level decode_tok_s, decode_io and
  # profile stay whole-run aggregates); a caller's own TETRAEDER_EXIT_DECODE_WINDOWS in "$@" comes later and wins
  case "$name" in main-overlap* | main-prefetch* | main-repeat) text+=(TETRAEDER_EXIT_DECODE_WINDOWS=4) ;; esac
  echo "[$(date +%T)] start $name"
  balloon_pre
  local t0=$SECONDS json="$date_dir/exit-$name-$stamp.json" samples="runs/vast/samples-$name-$stamp.jsonl"
  # the builder's own metrics to the collector every second (ADR 0060), with the stage as a resource attribute
  local otel=() attrs="tetraeder.stage=$name"
  if [ -n "${TETRAEDER_HOST_OTLP:-}" ] && [ "${ENGINE_OTLP:-1}" = 1 ]; then
    [ -n "${TETRAEDER_SESSION:-}" ] && attrs="$attrs,tetraeder.session=$TETRAEDER_SESSION"
    otel=(OTEL_EXPORTER_OTLP_ENDPOINT="$TETRAEDER_HOST_OTLP" OTEL_METRIC_EXPORT_INTERVAL=1000
      OTEL_RESOURCE_ATTRIBUTES="$attrs,host.id=${CONTAINER_ID:-$(hostname)}")
  fi
  ensure_disk
  membw_probe "$name"
  env "${text[@]}" "${otel[@]}" TETRAEDER_EXIT_GGUF="$model" TETRAEDER_EXIT_GPU=1 TETRAEDER_EXIT_THREADS="$threads" \
    TETRAEDER_EXIT_VRAM_SLOT_BYTES="$vram_bytes" TETRAEDER_EXIT_RAM_SLOT_BYTES="$ram_bytes" \
    TETRAEDER_EXIT_OUT="$json" \
    TETRAEDER_EXIT_MACHINE="vast $(hostname)" "$@" \
    bash scripts/oom_engine.sh timeout -k 30 "${min}m" cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda \
    --test exit_regime -- --ignored --nocapture &
  local pid=$!
  TETRAEDER_STAGE="$name" python3 scripts/engine_sample.py sample --pid "$pid" --disk "$HOST_DISK_DEV" --out "$samples" 8>&- &
  local spid=$!
  wait "$pid"
  local r=$?
  wait "$spid"
  if [ -s "$json" ]; then
    python3 scripts/engine_sample.py summarize --samples "$samples" --threads "$threads" --builder-json "$json" \
      --out "$samples.summary" --merge-into "$json" ${membw_json:+--membw "$membw_json"} > /dev/null || echo "host summary failed for $name"
  fi
  [ -s "$json" ] && record_stage_sec "$name" $((SECONDS - t0))
  balloon_post "$name" "$json"
  echo "[$(date +%T)] done $name rc=$r"
  return $r
}

# baseline <engine-bin> <name> <minutes> <args...>: llama-bench JSON to <name>-<stamp>.json
baseline() {
  local bin=$1 name=$2 min=$3
  shift 3
  want "$name" || return 0
  [ -x "$bin" ] || { echo "skip $name: $bin missing"; return 0; }
  echo "[$(date +%T)] start $name"
  membw_probe "$name"
  balloon_pre
  bash scripts/oom_engine.sh timeout -k 30 "${min}m" "$bin" -m "$model" -ngl 99 -fa 1 -t "$threads" -r 1 -o json "$@" > "$date_dir/$name-$stamp.json"
  local r=$?
  balloon_post "$name" "$date_dir/$name-$stamp.json"
  echo "[$(date +%T)] done $name rc=$r"
}

# colibri <short|deep>: the Colibri baseline (scripts/colibri_exit.sh), same model, threads and RAM as the builder
colibri() {
  want "colibri-$1" || return 0
  [ -n "${colibri_build_pid:-}" ] && wait "$colibri_build_pid"
  colibri_build_pid=
  echo "[$(date +%T)] start colibri-$1"
  balloon_pre
  bash scripts/oom_engine.sh bash scripts/colibri_exit.sh "$model" "$threads" "$ram_bytes" "$date_dir/exit-colibri-$1-$stamp.json" "$1" 8>&-
  local r=$?
  balloon_post "colibri-$1" "$date_dir/exit-colibri-$1-$stamp.json"
  echo "[$(date +%T)] done colibri-$1 rc=$r"
}

# ktransformers [avx2|repeat]: the KTransformers baseline (scripts/ktransformers_exit.sh), deep profile, same threads
# and RAM; with avx2 (stage ktransformers-avx2) kt-kernel is forced to its AVX2 kernels, as our portable ggml build
# runs; repeat (stage ktransformers-repeat) is the second default run of a noise-band pair
ktransformers() {
  local name=ktransformers${1:+-$1} variant=${1:-}
  [ "$variant" = repeat ] && variant=
  want "$name" || return 0
  [ -n "${kt_build_pid:-}" ] && wait "$kt_build_pid"
  kt_build_pid=
  echo "[$(date +%T)] start $name"
  ensure_disk
  balloon_pre
  # KT_KERNEL_CPU_VARIANT only for a variant: set but empty is not the same as kt-kernel's own pick
  TETRAEDER_STAGE="$name" HOST_DISK_DEV="$HOST_DISK_DEV" ${variant:+env KT_KERNEL_CPU_VARIANT=$variant} \
    bash scripts/oom_engine.sh bash scripts/ktransformers_exit.sh "$model" "$threads" "$ram_bytes" "$date_dir/exit-$name-$stamp.json" deep 8>&-
  local r=$?
  balloon_post "$name" "$date_dir/exit-$name-$stamp.json"
  echo "[$(date +%T)] done $name rc=$r"
}

# llamacpp_cache [suffix]: llama.cpp's tuned expert cache with a CPU share of the misses (#30966, baseline build, ADR
# 0021 pairing), one llama-server run per share in CPU_SHARES at the builder's expert bytes (vram_bytes as the cache
# size). Stage llamacpp-cache; with suffix "repeat", stage llamacpp-cache-repeat for the second pass of the pairs; with
# suffix "dio", stage llamacpp-cache-dio: the same shares with --load-mode dio, the pair that shows whether mmap
# streams or thrashes in this regime (m1-open item 1); "dio-repeat" is the second pass of that pair (opt in via STAGES).
llamacpp_cache() {
  local name=llamacpp-cache${1:+-$1} s mode=${LOAD_MODE:-auto} shares=${CPU_SHARES:-0 0.5 0.9} tuned t0 rc_c extra=(env)
  [[ "${1:-}" == dio* ]] && mode=dio
  # variants: poll0 passes POLL=0 (sleeping ggml threads); fp8 passes KV_TYPE (the cache script's knob; set
  # LLAMACPP_FP8_KV_TYPE, the default "fp8" is a placeholder until the real type name is confirmed)
  case "${1:-}" in
    poll0*) extra+=(POLL=0) ;;
    fp8*) extra+=(KV_TYPE="${LLAMACPP_FP8_KV_TYPE:-f8_e4m3}") ;;
  esac
  want "$name" || return 0
  local mib=$((vram_bytes / 1048576))
  # the tuner's pick from this session (stage moe-tune) replaces the session's shares and cache size
  tuned=$(python3 -I -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    sys.exit()
if d.get("cpu_share") is not None and d.get("cache_mib"):
    print(d["cache_mib"], d["cpu_share"])' "$date_dir/moe-cache-tune/tune.json")
  if [ -n "$tuned" ]; then
    read -r mib shares <<< "$tuned"
    echo "[$(date +%T)] $name: tuner picked cache $mib MiB, cpu share $shares"
  fi
  ensure_disk
  for s in $shares; do
    stage_done "llamacpp-cache-s$s${1:+-$1}" && { echo "[$(date +%T)] skip $name s$s: result in $date_dir"; continue; }
    fits "$name" || continue
    echo "[$(date +%T)] start $name s$s: cache $mib MiB, load mode $mode"
    membw_probe "llamacpp-cache-s$s${1:+-$1}"
    balloon_pre
    t0=$SECONDS
    TETRAEDER_STAGE="llamacpp-cache-s$s${1:+-$1}" HOST_DISK_DEV="$HOST_DISK_DEV" LOAD_MODE=$mode "${extra[@]}" bash scripts/oom_engine.sh timeout -k 30 75m bash scripts/llamacpp_cache.sh "$model" "$threads" "$mib" "$s" 33024 "$prompt" \
      "$date_dir/exit-llamacpp-cache-s$s${1:+-$1}-$stamp.json" 8>&-
    rc_c=$?
    [ -s "$date_dir/exit-llamacpp-cache-s$s${1:+-$1}-$stamp.json" ] && record_stage_sec "$name" $((SECONDS - t0))
    balloon_post "llamacpp-cache-s$s${1:+-$1}" "$date_dir/exit-llamacpp-cache-s$s${1:+-$1}-$stamp.json"
    echo "[$(date +%T)] done $name s$s rc=$rc_c"
  done
}

# moe_tune: llama.cpp's cache tuner (#30966) on the 235B under this script's balloon, cache capped at the builder's
# VRAM slot bytes; llamacpp_cache then runs the tuner's pick ("as its tuner picks them", m1-open item 1)
moe_tune() {
  want moe-tune || return 0
  [ "${RESUME:-0}" = 1 ] && [ -s "$date_dir/moe-cache-tune/tune.json" ] && {
    echo "[$(date +%T)] skip moe-tune: result in $date_dir"
    return 0
  }
  echo "[$(date +%T)] start moe-tune: max $((vram_bytes / 1048576)) MiB"
  # CTX 8192: the tuner's first probe takes 10% of the expert bytes (13 GB on the 235B) whatever --max-mib says,
  # which with the f16 KV of 32K OOMs a 24 GB card (20261111c); its pick stays capped at MAX_MIB and runs at 32K
  balloon_pre
  GGUF="$model" MAX_MIB=$((vram_bytes / 1048576)) THREADS="$threads" TUNE_MIN="${TUNE_MIN:-40}" CTX="${TUNE_CTX:-8192}" RESULTS_DATE="$(basename "$date_dir")" bash scripts/oom_engine.sh bash scripts/moe_cache_tune.sh 8>&-
  local r=$?
  balloon_post moe-tune "$date_dir/moe-cache-tune/tune.json"
  echo "[$(date +%T)] done moe-tune rc=$r"
}

# disk_curve: the model disk's read rate by queue depth, idle and beside quota-1 spinning threads (ADR 0056; about 2
# min). The largest shard, random 1 MiB chunks through the I/O service, direct I/O. disk-curve-loaded-<stamp>.json
# is the session's nvme0 curve for the graph; its decode sampler check: `read_mb_s` at `disk_queue_depth_mean` of a
# builder stage should lie within 20% of the loaded curve at that depth (reported, not gated)
disk_curve() {
  want disk-curve || return 0
  echo "[$(date +%T)] start disk-curve"
  local shard load=$((threads > 1 ? threads - 1 : 1))
  shard=$(ls -S "$(dirname "$model")"/*.gguf | head -1)
  cargo build --release -p tetraeder-bench --bin tetraeder-bench || { echo "bench build failed"; return 0; }
  target/release/tetraeder-bench disk-curve --file "$shard" --seconds 4 --out "$date_dir/disk-curve-idle-$stamp.json" \
    || echo "disk-curve idle failed"
  target/release/tetraeder-bench disk-curve --file "$shard" --seconds 4 --load-threads "$load" \
    --out "$date_dir/disk-curve-loaded-$stamp.json" || echo "disk-curve loaded failed"
  local k
  for k in idle loaded; do
    [ -s "$date_dir/disk-curve-$k-$stamp.json" ] && TETRAEDER_STAGE=disk-curve \
      python3 scripts/engine_sample.py emit-curve "$date_dir/disk-curve-$k-$stamp.json" --kind $k
  done
  echo "[$(date +%T)] done disk-curve"
}

# anchors_builder_short: the builder anchors that fit (two on the 91 GB class) at pp512 tg32 and this session's builder stages (main, main-prefetch,
# ...), written as a planner fixture directory ledger-<session>/ (layout.json, candidates.json, <variant>.r1.json) that
# `tetraeder-plan ledger` reads like a fixture (ADR 0056). Opt in: STAGES names it, or ANCHORS_BUILDER_SHORT=1. About
# 8 to 12 min on the 235B: two to four runs of 2 min and the planner build (4 min the first time)
anchors_builder_short() {
  [[ " ${STAGES:-} " == *" anchors-builder-short "* || "${ANCHORS_BUILDER_SHORT:-0}" = 1 ]] || return 0
  want anchors-builder-short || return 0
  local session=${TETRAEDER_SESSION:-$stamp} d id kv free_mib saved=${STAGES:-} st saved_only=${ONLY:-}
  d="$date_dir/ledger-$session"
  mkdir -p "$d"
  echo "[$(date +%T)] start anchors-builder-short: $d"
  cargo build --release -p tetraeder -p tetraeder-planner --features tetraeder-ggml-sys/cuda \
    || { echo "planner build failed"; return 0; }
  free_mib=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  target/release/tetraeder plan layout "$model" --out "$d/layout.json" || return 0
  local stage_args=()
  while read -r st; do
    [ -n "$st" ] && stage_args+=(--stage "$st")
  done < <(python3 scripts/ledger_stages.py "$date_dir" main main-prefetch main-repeat main-iosvc main-overlap \
    main-overlap-d2 main-overlap-ram --copy-to "$d")
  target/release/tetraeder-plan ledger-dir --layout "$d/layout.json" --gpu-bytes $((free_mib * 1048576)) \
    --ram-bytes "$ram_bytes" --threads "$threads" --prompt 512 --gen 32 --model-key moe235 \
    --hf "Qwen/Qwen3-235B-A22B-GGUF:Q4_K_M" "${stage_args[@]}" --out "$d/candidates.json" || return 0
  while IFS=$'\t' read -r id kv; do
    [ -s "$d/$id.r1.json" ] && continue
    # exit_regime asks want() for its name: let the anchor runs through a STAGES filter
    STAGES="${saved:+$saved }ledger-$id"
    ONLY="ledger-$id"
    # shellcheck disable=SC2086  # kv is a KEY=VALUE list without spaces inside
    exit_regime "ledger-$id" 10 $kv && cp "$date_dir/exit-ledger-$id-$stamp.json" "$d/$id.r1.json"
  done < <(python3 scripts/ledger_stages.py --anchors "$d/candidates.json")
  STAGES=$saved
  ONLY=$saved_only
  echo "[$(date +%T)] done anchors-builder-short"
}

# llama_fit: the most layers' experts llama.cpp keeps on the GPU at ubatch 2048 and llama-bench's context for
# -p 2048 -n 32 -d 32768 (34848; at 33024 eight layers fit llama-server but not llama-bench) (scripts/llama_fit.sh); the
# builder then gets the same expert bytes as VRAM slots (from the smoke run's per-layer expert sizes), so both engines
# use the same VRAM for experts
llama_fit() {
  want llama-fit || return 0
  echo "[$(date +%T)] start llama-fit"
  local f="$date_dir/exit-llama-fit-$stamp.json"
  FIT_MAX="${FIT_MAX:-9}" bash scripts/llama_fit.sh "$model" "$threads" 2048 34848 "$f" || return 0
  local n
  n=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['layers'])" "$f") || return 0
  [ "$n" -ge 0 ] || return 0
  ot_fit=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['ot'])" "$f")
  if [ -z "${VRAM_SLOT_BYTES:-}" ] && [ "$n" -gt 0 ]; then
    vram_bytes=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(sum(d['expert_bytes_per_layer'][:int(sys.argv[2])]) * int(sys.argv[3]))" "$(ls -t "$date_dir"/exit-smoke-*.json | head -1)" "$n" 128)
  fi
  echo "[$(date +%T)] done llama-fit: $n layers, -ot $ot_fit, builder vram slots $vram_bytes B"
}

run_stages() {
if [ $rc = 0 ]; then
  {
    # smoke gate: real header, the full KV (a VRAM OOM shows here), direct I/O, the backend split, memory
    exit_regime smoke 20 TETRAEDER_EXIT_PREFILL=8 TETRAEDER_EXIT_UBATCH=8 TETRAEDER_EXIT_DECODE=2 \
      TETRAEDER_EXIT_CTX=33024 TETRAEDER_EXIT_OP_OFFLOAD=1 || rc=22
    # the smoke summary prints with the smoke stage only
    if [ -z "${ONLY:-}" ] || [ "$ONLY" = smoke ]; then
    python3 - "$date_dir/exit-smoke-$stamp.json" << 'EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    sys.exit(f"smoke json: {e}")
print("smoke:", {k: d.get(k) for k in ("io_backend", "slot_bytes_all_layers", "options")})
for r in d.get("runs") or []:
    print("  run:", {k: r.get(k) for k in ("max_splits", "rss_bytes", "peak_rss_bytes", "decode_tok_s")})
    for m in r.get("memory") or []:
        print("  memory:", m)
EOF
    fi
    if [ $rc = 0 ]; then
      disk_curve
      llama_fit
      # ubatch 2048: a pair group passes one row per (token, expert) pair to CUDA's mul_mat_id, whose id helper
      # needs 4 B of shared memory per row (about 25k rows on Ampere); ubatch 4096 at top-8 is 32k (abort, 2026-11-02)
      deep=(TETRAEDER_EXIT_PREFILL=32768 TETRAEDER_EXIT_UBATCH=2048 TETRAEDER_EXIT_DECODE=64 TETRAEDER_EXIT_CTX=33024
        TETRAEDER_EXIT_OP_OFFLOAD=1)
      # logit drift against libllama at 2K (ADR 0045): prefetch off and on, a smaller RAM tier, CUDA fusion off
      short=(TETRAEDER_EXIT_PREFILL=2048 TETRAEDER_EXIT_UBATCH=2048 TETRAEDER_EXIT_DECODE=8 TETRAEDER_EXIT_CTX=4096
        TETRAEDER_EXIT_OP_OFFLOAD=1)
      exit_regime drift-ref 15 "${short[@]}" TETRAEDER_EXIT_ENGINE=libllama TETRAEDER_EXIT_REF_OT="$ot_fit"
      exit_regime drift-off 15 "${short[@]}" TETRAEDER_EXIT_PREFETCH=0
      exit_regime drift-on 15 "${short[@]}" TETRAEDER_EXIT_PREFETCH=1
      exit_regime drift-small 15 "${short[@]}" TETRAEDER_EXIT_PREFETCH=0 TETRAEDER_EXIT_RAM_SLOT_BYTES=30000000000
      exit_regime drift-nooffload 15 "${short[@]}" TETRAEDER_EXIT_PREFETCH=0 TETRAEDER_EXIT_OP_OFFLOAD=0
      exit_regime drift-nooffload-on 15 "${short[@]}" TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_OP_OFFLOAD=0
      # the RAM groups on the CPU (exact over slot counts on the synthetic fixtures) with the smaller RAM tier: equal
      # to drift-nooffload unless an expert's bytes depend on the slot count
      exit_regime drift-nooffload-small 15 "${short[@]}" TETRAEDER_EXIT_PREFETCH=0 TETRAEDER_EXIT_OP_OFFLOAD=0         TETRAEDER_EXIT_RAM_SLOT_BYTES=30000000000
      exit_regime drift-nofuse 15 "${short[@]}" TETRAEDER_EXIT_PREFETCH=0 GGML_CUDA_DISABLE_FUSION=1
      # libllama's own sensitivity to how the same tokens are batched: the yardstick for the builder's drift
      exit_regime drift-ref-ub512 15 "${short[@]}" TETRAEDER_EXIT_UBATCH=512 TETRAEDER_EXIT_ENGINE=libllama         TETRAEDER_EXIT_REF_OT="$ot_fit"
      exit_regime drift-ref-nofuse 15 "${short[@]}" TETRAEDER_EXIT_ENGINE=libllama TETRAEDER_EXIT_REF_OT="$ot_fit"         GGML_CUDA_DISABLE_FUSION=1
      # prefetch off and on (ADR 0038), paired
      exit_regime main 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=0 || rc=23
      exit_regime main-prefetch 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 || rc=26
      # Low-bit KV (ADR 0050 b), paired with `main-prefetch` right before: the same VRAM slot bytes, then the slot
      # bytes raised by the freed KV (94 layers x 33024 positions x the per-token bytes saved; F16/F16 is 2048 B,
      # Q8_0/Q8_0 1088 B, Q8_0/Q4_0 832 B at n_embd_kv 512)
      kv_free() { echo $((94 * 33024 * (2048 - $1))); }
      exit_regime kv-q8 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_KV=q8_0 || rc=29
      exit_regime kv-q8-slots 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_KV=q8_0         TETRAEDER_EXIT_VRAM_SLOT_BYTES=$((vram_bytes + $(kv_free 1088))) || rc=29
      exit_regime kv-q8q4 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_KV=q8_0/q4_0 || rc=29
      exit_regime kv-q8q4-slots 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_KV=q8_0/q4_0         TETRAEDER_EXIT_VRAM_SLOT_BYTES=$((vram_bytes + $(kv_free 832))) || rc=29
      # ADR 0054, paired with main-prefetch on the same box: the persistent I/O service alone, with experts computed
      # in arrival order, at prefetch depth 2, and with the RAM tier sized to the container (4 GB reserve instead of 8)
      exit_regime main-iosvc 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_IO_SERVICE=1 || rc=30
      exit_regime main-overlap 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_IO_SERVICE=1 TETRAEDER_EXIT_OVERLAP=1 || rc=30
      exit_regime main-overlap-d2 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_PREFETCH_DEPTH=2 TETRAEDER_EXIT_IO_SERVICE=1 TETRAEDER_EXIT_OVERLAP=1 || rc=30
      exit_regime main-overlap-ram 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_IO_SERVICE=1 TETRAEDER_EXIT_OVERLAP=1 TETRAEDER_EXIT_RAM_SLOT_BYTES=$((mem - 4000000000)) || rc=30
      # Hidden share levers (ADR 0054), each paired with main-overlap: misses submitted before the next layers'
      # prefetch; fewer workers, so deadline order applies (the disk curve is flat from depth 4); idle ggml threads
      # that sleep instead of spinning while the main thread waits for a read. Opt in: only when STAGES names them
      # (about 10 min each), never in a full run
      if [ -n "${STAGES:-}" ]; then
        ovl=(TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_IO_SERVICE=1 TETRAEDER_EXIT_OVERLAP=1)
        exit_regime main-overlap-uf 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_URGENT_FIRST=1 || rc=30
        exit_regime main-overlap-w4 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_IO_WORKERS=4 || rc=30
        exit_regime main-overlap-uf-w4 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_URGENT_FIRST=1 TETRAEDER_EXIT_IO_WORKERS=4 || rc=30
        exit_regime main-overlap-w8 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_IO_WORKERS=8 || rc=30
        exit_regime main-overlap-poll0 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_CPU_POLL=0 || rc=30
        exit_regime main-overlap-poll0-repeat 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_CPU_POLL=0 || rc=30
        # four threads fewer than the quota (min 1): leaves cores to the I/O workers; the repeat is the second pass
        tq4=$((threads > 4 ? threads - 4 : 1))
        exit_regime main-overlap-tq4 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_THREADS="$tq4" || rc=30
        exit_regime main-overlap-tq4-repeat 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_THREADS="$tq4" || rc=30
        # Huge pages on the RAM tier (ADR 0035 addendum), paired with main-overlap; the repeat is the second pass
        exit_regime main-overlap-thp 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_RAM_HUGEPAGES=1 || rc=30
        exit_regime main-overlap-thp-repeat 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_RAM_HUGEPAGES=1 || rc=30
        # Adaptive compute tier (ADR 0059), paired with main-overlap: decode uploads a share of the CPU misses into
        # VRAM slots (LRU seeded by the hot set); the repeat is the second pass of the pair. Other shares via VARIANTS
        exit_regime main-overlap-adapt 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_ADMIT_SHARE=0.1 || rc=30
        exit_regime main-overlap-adapt-repeat 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_ADMIT_SHARE=0.1 || rc=30
        # poll0 combined with the other levers (m1-open item 1): huge pages; the hot set rebuilt from the whole
        # prefill (replay: VRAM share 0.253 against 0.178, no upload cost); a 0.05 admit share. Each has a repeat
        exit_regime main-overlap-poll0-thp 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_CPU_POLL=0 TETRAEDER_EXIT_RAM_HUGEPAGES=1 || rc=30
        exit_regime main-overlap-poll0-thp-repeat 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_CPU_POLL=0 TETRAEDER_EXIT_RAM_HUGEPAGES=1 || rc=30
        exit_regime main-overlap-poll0-hot 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_CPU_POLL=0 TETRAEDER_EXIT_REFRESH_HOT=1 || rc=30
        exit_regime main-overlap-poll0-hot-repeat 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_CPU_POLL=0 TETRAEDER_EXIT_REFRESH_HOT=1 || rc=30
        exit_regime main-overlap-poll0-adapt 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_CPU_POLL=0 TETRAEDER_EXIT_ADMIT_SHARE=0.05 || rc=30
        exit_regime main-overlap-poll0-adapt-repeat 60 "${deep[@]}" "${ovl[@]}" TETRAEDER_EXIT_CPU_POLL=0 TETRAEDER_EXIT_ADMIT_SHARE=0.05 || rc=30
      fi
      # the same on random token ids, the pair of llama-bench (random ids route differently from text)
      exit_regime main-prefetch-random 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 RANDOM_PROMPT=1 || rc=26
      # routing trace for the RAM-tier replay (ADR 0050 (c)): the prefetch levers of ADR 0038 (depth 2, so the trace
      # carries both predictions), 256 decode tokens on the frozen prompt; then the replay on the box. The trace (tens
      # of MB) stays in runs/vast/ on the box; only the replay JSON goes to bench/results/<date>/replay-235b.json
      if want trace; then
        trace="$PWD/runs/vast/routing-235b-$stamp.trace"
        exit_regime trace 90 "${deep[@]}" TETRAEDER_EXIT_DECODE=256 TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_PREFETCH_DEPTH=2           TETRAEDER_EXIT_ROUTING_OUT="$trace" || rc=29
        cargo run --release -p tetraeder-bench --bin tetraeder-bench -- replay --trace "$trace"           --vram 10,12,13 --ram 48,64,77,79,81,83,96,110 --vram-policy static,prefill,lru,lru:0.5,freq --out "$date_dir/replay-235b.json" || rc=29
      fi
      # VARIANTS: extra builder runs at the same depth, "name:ENV=V,ENV=V name2:..." (stage names exit-<name>)
      for v in ${VARIANTS:-}; do
        IFS=',' read -ra extra <<< "${v#*:}"
        exit_regime "${v%%:*}" 60 "${deep[@]}" "${extra[@]}" || rc=27
      done
      # KL acceptance of the KV formats on Qwen3-30B-A3B Q4_K_M (scripts/kv_quality.sh fetches it if missing)
      if want kv-quality; then
        echo "[$(date +%T)] start kv-quality"
        bash scripts/kv_quality.sh "$date_dir/kv-quality-$stamp.json" 8>&- || rc=29
        echo "[$(date +%T)] done kv-quality"
      fi
      # llama.cpp on the same prompt tokens through libllama (exit_regime's reference engine): logit parity of the
      # builder at 32K (greedy, first-token top-10, margins) and the tuned -ot baseline on real text
      exit_regime reference 60 "${deep[@]}" TETRAEDER_EXIT_ENGINE=libllama TETRAEDER_EXIT_REF_OT="$ot_fit" || rc=28
      # The verdict: KTransformers and llama.cpp's expert cache (mmap, then dio: does mmap stream or thrash) after the
      # builder; then the builder and KTransformers again, a second pair for the noise band; the diagnosis stages
      # (KT on AVX2) and the cache repeat last, so a deadline cuts them first
      moe_tune
      ktransformers
      llamacpp_cache
      llamacpp_cache dio
      exit_regime main-repeat 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 || rc=26
      exit_regime main-overlap-repeat 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_IO_SERVICE=1 TETRAEDER_EXIT_OVERLAP=1 || rc=30
      ktransformers repeat
      ktransformers avx2
      llamacpp_cache repeat
      # the second pass of the dio pair, only when STAGES names it
      [[ " ${STAGES:-} " == *" llamacpp-cache-dio-repeat "* ]] && llamacpp_cache dio-repeat
      # opt-in cache variants: sleeping ggml threads, low-bit KV (see llamacpp_cache); each with its repeat
      for _v in poll0 poll0-repeat fp8 fp8-repeat; do
        [[ " ${STAGES:-} " == *" llamacpp-cache-$_v "* ]] && llamacpp_cache "$_v"
      done
      # the ledger's view of this session: the anchors at pp512 tg32 and the builder stages above (ADR 0056)
      anchors_builder_short
      # llama-bench, tuned -ot (random tokens; llama-bench takes ',' as alternative configs, so overrides in one config
      # are joined with ';'), then ik_llama.cpp with the same overrides (it was OOM-killed at 128 GB with mmap)
      baseline "$bench" exit-llama-ot-fit-d32k 40 -ub 2048 -p 2048 -n 32 -d 32768 -ot "$ot_fit"
      # (ik's llama-bench has no -d: -gp 32768,32 is decode after a 32K prompt)
      # (and its -ot takes ',' between overrides)
      baseline "$ik_bench" exit-ik-ot-fit-d32k 40 -ub 2048 -p 2048 -n 0 -gp 32768,32 -ot "${ot_fit//;/,}"
      # other tracks' stages (speculative decoding) go here
      # Colibri last; 32K deep is out of reach (2046 tokens took 895 s), so only the short profile by default
      colibri short
      [ -n "${STAGES:-}" ] && colibri deep
    fi
  } >> "$log" 2>&1
fi
}
# the collector for every stage alike (ADR 0060); without it the run goes on with the sampler's JSONL alone
unset TETRAEDER_HOST_OTLP
if [ $rc = 0 ] && [ "${OTEL_COLLECTOR:-1}" = 1 ]; then
  if bash scripts/otelcol_box.sh start "$PWD/runs/vast/otel/$(date +%Y%m%d)-$stamp" >> "$log" 2>&1 8>&-; then
    export TETRAEDER_HOST_OTLP=http://127.0.0.1:4318
  fi
fi
if [ -n "$order" ]; then
  for ONLY in $order; do run_stages; done
  ONLY=
else
  run_stages
fi
# stop before DONE so the files are complete when fetch packs them, then derive host_otlp from all of this box's files
if [ -n "${TETRAEDER_HOST_OTLP:-}" ]; then
  {
    bash scripts/otelcol_box.sh stop 8>&-
    python3 scripts/engine_sample.py otlp-merge --otel-dir runs/vast/otel --results "$date_dir" || echo "otlp-merge failed"
  } >> "$log" 2>&1
fi
touch "$date_dir/DONE"
echo "EXIT_RUN_DONE rc=$rc" >> "$log"
# the exit status is the session's, not sccache's (stop fails when no server runs: rc=2 on box 60072027)
command -v sccache > /dev/null 2>&1 && timeout 30 sccache --stop-server > /dev/null 2>&1
exit 0
