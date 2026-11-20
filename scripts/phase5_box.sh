#!/usr/bin/env bash
# Phase 5 verdict session on a multi-GPU box (ADR 0068): gates first, then calibration and predictions, then the
# rounds of candidates and baselines, then verdict.json (scripts/phase5_summary.py).
#   taxameter run scripts/phase5_box.sh <session> <2|4>      (or SESSION=... NGPUS=... in the environment)
# Started detached; ends with PHASE5_BOX_DONE in runs/vast/phase5_box.log.
# Results: bench/results/<session>/phase5/ (file names are fixed by ADR 0068; run-<run>.rc, cotenant-<run>.json,
# prepass-*.json, gates.json and timeline.jsonl are this script's, read by phase5_summary.py). A rerun resumes: a stage
# whose output exists and a run whose result exists are skipped.
#
# Stages, in order (STAGE <name> lines in the log):
#   gpus        nvidia-smi -L must list exactly <2|4> GPUs; taxameter 0.3.3 filters offers by `--num-gpus`; still,
#               a wrong box ends the session here (PHASE5_BOX_DONE rc=1) before anything is built or downloaded
#   env         env.json: host, GPU names, driver, clocks, power limits (power_limit_w, power_default_limit_w, power_max_limit_w), throttle reasons, PCIe gen and width, topo -m, peer-access matrix
#               (nvidia-smi topo -p2p r), NCCL version, CPU quota, commits; comm_impl is filled in by probe
#   build       release build with CUDA: tetraeder and tetraeder-plan
#   models      both GGUFs must be in .hf_cache (taxameter models); missing files end the session
#   gpu-parity  TETRAEDER_REQUIRE_GPU=1 cargo test gpu_ gpu2_ [gpu4_] on the executor, the sharded builders and
#               ggml-sys (only gpu_no_capture_flag_skips_the_cuda_graph, which must run and pass, ADR 0081); log in
#               gpu-parity.log (cut to 300 columns; the full log stays on the box as gpu-parity.log.full), result in
#               gates.json. Every package and test runs (--no-fail-fast); only the executor and ggml-sys packages
#               get GGML_CUDA_GRAPH_DEBUG=1. A failure does not skip discover to prepass (cheap, and the
#               diagnosis); the session then ends with rc=3 before anchors
#   discover    hw.json (the profiler's hardware graph; HW_CMD, default `tetraeder plan hw --out hw.json`)
#   enumerate   per model layouts/<model>.json, then `tetraeder-plan candidates --multi` -> candidates-<model>.json and
#               plans/<model>-<id>.toml; fixed before any calibration (ADR 0040)
#   probe       `tetraeder comm-probe` -> calib-comm.json, comm_impl into env.json; failure is a gate (rc=3 before anchors)
#   prepass     per model, a short llama-bench per baseline split mode (llama.cpp baseline: layer, row, tensor; ik_llama:
#               layer, row, graph if its help lists it): prepass-<model>-<engine>-<sm>.json says which modes the build
#               accepts (ok) and which all-reduce path its log names (comm); only accepted modes run in the rounds. A
#               model with no accepted llama-bench mode has no baseline gate: baseline_modes failed, rc=3 before anchors
#   anchors     per model the first L/4 and L/2 layers as `single` and `tp-2`, pp2048 and tg128, two rounds ->
#               anchors-<model>-r<k>.json (on a 4x box the tp-2 plan comes from a 2-GPU hardware graph); on a 4x box
#               the dense model also as `tp-4` (the session's plan), so the all-reduces span two group sizes (ADR 0078)
#   calibrate   per MoE model first the routing of the judged token stream: `tetraeder bench --engine moe
#               --routing-out` on GPU 0 without a plan (one pass of pp2048 and tg128) -> routing-<model>.json (ADR 0070);
#               then `tetraeder-plan calibrate` -> calib.json (one fit over both models)
#   predict     `tetraeder-plan predict` (MoE models with `--routing routing-<model>.json`) -> predict-<model>-<n>x.json,
#               its sha256 into predict-<model>-<n>x.sha256, the log and timeline.jsonl; no candidate runs before this ends
#   rounds      per model three rounds (summary --round-order: the case order rotated by k, round 2 reversed) of every
#               candidate and baseline; up to two more rounds rerun only configurations with fewer than two clean rounds
#               (an unclean run is rerun, never dropped); then the A/A run (the first candidate again) closes the model
#   pp-overlap  optional, report only (PP_OVERLAP=1; default on for 4x rehearsals, ROUNDS=1): per model the session's
#               tp2-pp2 plan with prefill_micro_batches 1 and 4, two rounds interleaved (mb1 mb4, then mb4 mb1) ->
#               pp-overlap-<model>.json (ADR 0078: does the pipeline of tp stages overlap at all); not judged
#   vllm        optional, report only: skipped when less than 25 minutes of the budget remain or no AWQ/GPTQ checkpoint
#   summary     phase5_summary.py -> verdict.json, verdict-notes.json
#
# Models (assumed downloaded, .hf_cache/<org>__<model>/): qwen3-32b (dense, --engine dense), qwen3-30b-a3b (MoE, --engine moe).
# Env knobs: SESSION, NGPUS, MODELS ("qwen3-32b qwen3-30b-a3b"; one model per session is allowed, ADR 0068), THREADS (CPU
# quota), BUDGET_MIN (150: runs that would start later are skipped, loudly, and the verdict is then null), ROUNDS (3), RERUNS (2),
# P5_OUT, P5_LOG_FILE, P5_LOCK, TET_BIN, PLAN_BIN, LLAMA_BENCH, IK_BENCH, HW_CMD, SKIP_GPU=1 (no nvidia-smi or CUDA: the
# rehearsal of the plumbing; gates are written as skipped and the verdict is null), DRY_RUN=1 (print the sequence and the
# commands, run nothing; needs no GPU, lock or build; writes a synthetic enumeration into P5_OUT).
# DRY_RUN only: DRY_POWER=fail plays a lowered power limit, DRY_GPUS=<n> plays an `nvidia-smi -L` count, DRY_GPU_PARITY=fail a failed gpu-parity, DRY_PROBE=fail a
# failed probe, DRY_PREPASS=fail a baseline build that accepts no split mode.
# ROUNDS=1 (S2 and S3 rehearsals) runs one round and no reruns (RERUNS=<n> overrides).
# PP_OVERLAP=1|0 forces the pp-overlap stage on or off (default: on when NGPUS=4 and ROUNDS=1; off under ONLY).
# ONLY="tp-2 llama-bench:tensor" (ADR 0069 step-0 check): the rounds run only these configurations (a candidate is its plan
# id, a baseline is <engine>:<sm>), anchors, calibrate, predict, vllm, reruns and the A/A run are skipped, and the summary
# writes step0.json (own executor against the baseline per model and round, ratios against 1 - band) instead of
# verdict.json. Unset, the behaviour is the full session. Order on the first 2x box (S2):
#   SESSION=<s>-step0 ROUNDS=2 ONLY="tp-2 llama-bench:tensor" bash scripts/phase5_box.sh <s>-step0 2   # round 2 reversed: two interleaved pairs
#   SESSION=<s> ROUNDS=1 bash scripts/phase5_box.sh <s> 2                                              # the rehearsal
# GPU throttle (ADR 0068 "Throttle"): scripts/gpu_sampler.sh runs for the whole session (1 Hz, every GPU,
# gpu-samples.csv; not with SKIP_GPU/DRY_RUN) and run_one stamps window-<run>.json (box-local start and end); the summary
# judges each run on the sampler's window samples per GPU.
# cargo and every long job run with fd 8 closed: a server they start would otherwise keep the lock.
DRY="${DRY_RUN:-0}"
skip_gpu="${SKIP_GPU:-0}"
if [ "$DRY" != 1 ]; then
  exec 8> "${P5_LOCK:-/root/tetraeder_phase5_box.lock}"
  flock -n 8 || exit 0
fi
cd "$(dirname "$0")/.." || exit 1
session="${1:-${SESSION:-$(date +%Y%m%d)}}"
ngpu="${2:-${NGPUS:-}}"
case "$ngpu" in 2 | 4) ;; *)
  echo "usage: phase5_box.sh <session> <2|4>" >&2
  exit 2
  ;;
esac
if [ "$DRY" != 1 ]; then
  if [ "$skip_gpu" != 1 ]; then bash scripts/ensure_patches.sh 8>&- || exit 1; fi # ggml patch queue (ADR 0052)
  if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
  export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
  # shellcheck source=scripts/box_libs.sh
  . scripts/box_libs.sh
  command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
  if [ "$skip_gpu" != 1 ]; then
    # shellcheck disable=SC1091
    . "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
    if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
      archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
      [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
    fi
  fi
fi
log="${P5_LOG_FILE:-runs/vast/phase5_box.log}"
out="${P5_OUT:-$PWD/bench/results/$session/phase5}"
mkdir -p "$out/plans" "$out/layouts" "$out/anchor-parts" "$(dirname "$log")"
# 1 Hz GPU sampler for the whole session (ADR 0068 "Throttle"; scripts/gpu_sampler.sh -> $out/gpu-samples.csv); not in the CPU
# rehearsal. It ends on $out/gpu-sampler.stop or when this script is gone.
if [ "$DRY" = 1 ]; then
  [ "$skip_gpu" = 1 ] || echo "PLAN gpu_sampler.sh -> gpu-samples.csv (stopped at the end)"
elif [ "$skip_gpu" != 1 ]; then
  OUT="$out" SAMPLER_PARENT=$$ setsid nohup bash scripts/gpu_sampler.sh 8>&- > /dev/null 2>&1 &
fi
stop_gpu_sampler() { [ "$DRY" = 1 ] || touch "$out/gpu-sampler.stop"; }
only="${ONLY:-}"
t_start=$(date +%s)
if [ "$DRY" = 1 ]; then threads="${THREADS:-8}"; else threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"; fi
budget_s=$((${BUDGET_MIN:-150} * 60))
models="${MODELS:-qwen3-32b qwen3-30b-a3b}"
n_rounds="${ROUNDS:-3}"
reruns="${RERUNS:-2}" # more rounds for configurations short of two clean ones; ROUNDS=1 (rehearsals) has none
[ "$n_rounds" -lt 2 ] && reruns=${RERUNS:-0}
pp_overlap="${PP_OVERLAP:-}"
if [ -z "$pp_overlap" ]; then
  pp_overlap=0
  [ "$ngpu" = 4 ] && [ "$n_rounds" = 1 ] && pp_overlap=1
fi
models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
tet="${TET_BIN:-target/release/tetraeder}"
planbin="${PLAN_BIN:-target/release/tetraeder-plan}"
llama_bench="${LLAMA_BENCH:-/opt/llama.cpp-baseline/bin/llama-bench}"
ik_bench="${IK_BENCH:-/opt/ik_llama/bin/llama-bench}"
py="${PY:-python3}"
box="${ngpu}x"
gpu_list=0,1
[ "$ngpu" = 4 ] && gpu_list=0,1,2,3
server_pid=""
sampler_pid=""

declare -A MODEL_FILE=(
  [qwen3-32b]="${MODEL_QWEN3_32B:-$models_dir/Qwen__Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf}"
  [qwen3-30b-a3b]="${MODEL_QWEN3_30B:-$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf}"
)
declare -A MODEL_ENGINE=([qwen3-32b]=dense [qwen3-30b-a3b]=moe)
# AWQ / GPTQ int4 checkpoints for the vLLM report (HF directories, not GGUF)
declare -A MODEL_VLLM=(
  [qwen3-32b]="$models_dir/Qwen__Qwen3-32B-AWQ"
  [qwen3-30b-a3b]="$models_dir/Qwen__Qwen3-30B-A3B-GPTQ-Int4"
)

say() { echo "[$(date +%T)] $*"; }
plan() { echo "PLAN $*"; }
summ() { "$py" scripts/phase5_summary.py "$@"; }

die() {
  say "FATAL: $*"
  stop_server
  stop_gpu_sampler
  touch "$out/DONE"
  say "PHASE5_BOX_DONE rc=1"
  exit 1
}

stop_server() {
  [ -n "$server_pid" ] || return 0
  kill "$server_pid" 2> /dev/null
  for _ in $(seq 1 30); do kill -0 "$server_pid" 2> /dev/null || break; sleep 1; done
  kill -9 "$server_pid" 2> /dev/null
  wait "$server_pid" 2> /dev/null
  server_pid=""
  sleep 3 # GPU memory is released after the process is gone
}
trap stop_server EXIT

over_budget() { [ $(($(date +%s) - t_start)) -gt "$budget_s" ]; }
left_s() { echo $((budget_s - ($(date +%s) - t_start))); }

# tl EVENT MODEL KIND RUN [SHA]: one line of timeline.jsonl (the order of events survives a copy of the files)
tl() {
  [ "$DRY" = 1 ] && return 0
  printf '{"t": %s, "event": "%s", "model": "%s", "kind": "%s", "run": "%s", "sha256": "%s"}\n' \
    "$(date +%s.%N)" "$1" "$2" "$3" "$4" "${5:-}" >> "$out/timeline.jsonl"
}

# gate NAME VALUE: gates.json
gate() {
  [ "$DRY" = 1 ] && { plan "gates.json $1=$2"; return 0; }
  "$py" - "$out/gates.json" "$1" "$2" << 'EOF'
import json, sys
p, k, v = sys.argv[1:4]
try:
    d = json.load(open(p))
except (OSError, ValueError):
    d = {}
d[k] = v
json.dump(d, open(p, "w"), indent=2)
EOF
}
gate_value() { # gate_value NAME: the recorded value or empty
  "$py" -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))
except (OSError, ValueError): print("")' "$out/gates.json" "$1"
}

# gpu_note RUN before|after: the throttle reasons bitmask (hex) of every GPU, one `<when>: <hex>` line each
gpu_note() {
  local f="$out/gpu-$1.txt" when=$2 v
  if [ "$DRY" = 1 ]; then
    plan "nvidia-smi throttle reasons $when -> gpu-$1.txt"
    return 0
  fi
  if [ "$skip_gpu" = 1 ]; then
    echo "$when: 0x0000000000000000" >> "$f" # no GPU: counts as idle
    return 0
  fi
  v=$(nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader 2> /dev/null) \
    || v=$(nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader 2> /dev/null)
  [ -n "$v" ] || say "WARNING: no throttle reasons from nvidia-smi for $1 $when (the run counts as unclean)"
  # shellcheck disable=SC2001
  echo "$v" | sed "s/^ *\([^ ]\)/$when: \1/" >> "$f"
  nvidia-smi --query-gpu=index,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used --format=csv,noheader \
    | sed "s/^/# $when /" >> "$f"
}

# ---- stage: gpus -------------------------------------------------------------------------------------------
gpus() {
  local n
  if [ "$DRY" = 1 ]; then
    plan "nvidia-smi -L | count == $ngpu (taxameter filters by --num-gpus; still stop on a wrong box)"
    n="${DRY_GPUS:-$ngpu}"
  elif [ "$skip_gpu" = 1 ]; then
    say "gpus skipped (SKIP_GPU=1)"
    return 0
  else
    n=$(nvidia-smi -L 2> /dev/null | grep -c '^GPU ')
  fi
  [ "$n" = "$ngpu" ] || die "this box has $n GPU(s), the session needs $ngpu: destroy it and rent another offer"
  say "gpus: $n"
  # power limit at default on every GPU (ADR 0068): a lowered cap changes the ranking of shapes
  if [ "$DRY" = 1 ]; then
    plan "scripts/power_gate.sh: power.limit == power.default_limit on every GPU (within 1 W)"
    [ "${DRY_POWER:-}" = fail ] && die "power limit below default on a GPU: destroy the box and rent another offer"
  else
    bash scripts/power_gate.sh 8>&- || die "power limit below default on a GPU: destroy the box and rent another offer"
  fi
}

# ---- stage: env --------------------------------------------------------------------------------------------
env_json() {
  if [ "$DRY" = 1 ]; then
    plan "env.json (host, gpu names, driver, clocks, power limits, throttle reasons, pcie gen and width, topo -m, topo -p2p r, nccl, cpu quota, commits)"
    return 0
  fi
  [ -f "$out/env.json" ] && return 0
  if [ "$skip_gpu" != 1 ]; then
    nvidia-smi topo -m > "$out/topo.txt" 2>&1
    nvidia-smi topo -p2p r > "$out/topo-p2p.txt" 2>&1
    nvidia-smi -q -d CLOCK,PERFORMANCE > "$out/nvidia-smi-clock-performance.txt" 2>&1
  fi
  P5_OUT="$out" P5_THREADS="$threads" P5_QUOTA="$(bash scripts/cpu_quota.sh 2> /dev/null)" P5_NGPU="$ngpu" \
    P5_COMMIT="$(git rev-parse HEAD 2> /dev/null)" P5_HOST="$(hostname)" P5_VAST="${VAST_CONTAINERLABEL:-} ${CONTAINER_ID:-}" \
    P5_LLAMA="$("$llama_bench" --version 2>&1 | head -3 | tr '\n' ' ')" P5_SKIP="$skip_gpu" \
    P5_P2P="$(summ --p2p "$out/topo-p2p.txt" 2> /dev/null)" \
    P5_GPU="$(nvidia-smi --query-gpu=index,name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,clocks_throttle_reasons.active,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader 2> /dev/null)"     P5_POWER="$(nvidia-smi --query-gpu=power.limit,power.default_limit,power.max_limit --format=csv,noheader,nounits 2> /dev/null)" \
    P5_NCCL="$(ldconfig -p 2> /dev/null | grep -m1 libnccl | sed 's/^ *//')${NCCL_VERSION:+ $NCCL_VERSION}" \
    "$py" - << 'EOF'
import json, os, re
e = os.environ
gpus = [[x.strip() for x in ln.split(",")] for ln in e["P5_GPU"].splitlines() if ln.strip()]
def col(i):
    return [g[i] if len(g) > i else "" for g in gpus]
def pw(i):
    out = []
    for ln in e.get("P5_POWER", "").splitlines():
        f = [x.strip().removesuffix(" W") for x in ln.split(",")]
        try:
            out.append(float(f[i]))
        except (IndexError, ValueError):
            out.append(None)
    return out
env = {
    "host": e["P5_HOST"], "vast_container": e["P5_VAST"].strip(), "gpu_count": len(gpus) or int(e["P5_NGPU"]),
    "gpu_names": col(1), "driver": (col(2) or [""])[0], "clock_sm_mhz": col(3), "clock_sm_max_mhz": col(4),
    "clock_mem_mhz": col(5), "throttle_reasons_active": col(6), "vram": col(7), "pcie_gen": col(8), "pcie_width": col(9),
    "power_limit_w": pw(0), "power_default_limit_w": pw(1), "power_max_limit_w": pw(2),
    "p2p": e["P5_P2P"] or None, "nccl": e["P5_NCCL"].strip() or None,
    "comm_impl": None, "cpu_quota_cores": e["P5_QUOTA"], "threads": e["P5_THREADS"], "git_commit": e["P5_COMMIT"],
    "baseline_llama_bench": e["P5_LLAMA"], "skip_gpu": e["P5_SKIP"] == "1",
    "see": ["topo.txt", "topo-p2p.txt", "nvidia-smi-clock-performance.txt"],
}
json.dump(env, open(os.path.join(e["P5_OUT"], "env.json"), "w"), indent=2)
EOF
}

set_comm_impl() {
  [ "$DRY" = 1 ] && { plan "env.json comm_impl <- calib-comm.json comm_path"; return 0; }
  "$py" - "$out" << 'EOF'
import json, os, sys
d = sys.argv[1]
try:
    comm = json.load(open(os.path.join(d, "calib-comm.json"))).get("comm_path")
    env = json.load(open(os.path.join(d, "env.json")))
except (OSError, ValueError):
    sys.exit(0)
env["comm_impl"] = comm
json.dump(env, open(os.path.join(d, "env.json"), "w"), indent=2)
EOF
}

# ---- stage: build ------------------------------------------------------------------------------------------
build() {
  if [ "$DRY" = 1 ]; then
    plan "cargo build --release -p tetraeder -p tetraeder-planner --features cuda"
    return 0
  fi
  say "build start"
  local features="--features cuda"
  [ "$skip_gpu" = 1 ] && features=""
  # shellcheck disable=SC2086
  timeout -k 30 50m cargo build --release -p tetraeder $features 8>&- || die "build failed"
  timeout -k 30 20m cargo build --release -p tetraeder-planner --bin tetraeder-plan 8>&- || die "build of tetraeder-plan failed"
  say "build done"
}

# ---- stage: models -----------------------------------------------------------------------------------------
models_check() {
  local m
  for m in $models; do
    if [ "$DRY" = 1 ]; then
      plan "model $m: ${MODEL_FILE[$m]}"
      continue
    fi
    [ -s "${MODEL_FILE[$m]}" ] || die "model $m missing: ${MODEL_FILE[$m]} (taxameter models)"
    ls -l "${MODEL_FILE[$m]}" >> "$out/models.txt"
  done
  # gpu-parity's dense tests read Qwen3-8B from the models dir and fail under REQUIRE_GPU without it (s3f-20261120:
  # the provision list lacked it, found only after the build). Provision it with
  # Qwen/Qwen3-8B-GGUF:Qwen3-8B-Q4_K_M.gguf.
  local parity_8b="$models_dir/Qwen__Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
  if [ "$DRY" = 1 ]; then
    plan "model qwen3-8b (gpu-parity): $parity_8b"
  elif [ "$skip_gpu" != 1 ]; then
    [ -s "$parity_8b" ] || die "model qwen3-8b missing (gpu-parity needs it): $parity_8b (taxameter models)"
    ls -l "$parity_8b" >> "$out/models.txt"
  fi
}

# ---- stage: gpu-parity -------------------------------------------------------------------------------------
# No timed run on a box whose executor or sharded builders disagree with the one-rank reference. gpu_ is one GPU, gpu2_
# two, gpu4_ four (ADR 0069, E5 to E7 tests).
# ggml-sys runs only its capture-flag test (ADR 0081): its other gpu_ tests are timing and kernel checks of other phases.
sys_filter=gpu_no_capture_flag_skips_the_cuda_graph
# The KV save timing tests (gpu_kv_save.rs, ADR 0074) judge host memory speed, not parity, and have their own sessions
# (gpu_kv_save2.sh). On host 20210 one 8B copy path ran at 50 to 60 ms instead of about 12 in s3c (straight, not asserted)
# and s3f (staged, over the 25 ms bound), which ended s3f at the gate. Skipped here.
model_skip="--skip save_through_pinned_staging"
gpu_parity() {
  local filters="gpu_ gpu2_" f pkg extra failed="" full="$out/gpu-parity.log.full"
  [ "$ngpu" = 4 ] && filters="gpu_ gpu2_ gpu4_"
  if [ "$DRY" = 1 ]; then
    for pkg in tetraeder-executor tetraeder-model tetraeder-ggml-sys; do
      extra=""
      [ "$pkg" != tetraeder-model ] && extra="GGML_CUDA_GRAPH_DEBUG=1 "
      f=$filters
      [ "$pkg" = tetraeder-ggml-sys ] && f=$sys_filter
      [ "$pkg" = tetraeder-model ] && f="$f $model_skip"
      plan "${extra}TETRAEDER_REQUIRE_GPU=1 TETRAEDER_REQUIRE_GPUS=$ngpu TETRAEDER_MOE_GGUF=${MODEL_FILE[qwen3-30b-a3b]} cargo test --release --no-fail-fast -p $pkg --features tetraeder-ggml-sys/cuda -- $f >> gpu-parity.log.full"
    done
    plan "gpu-parity fails if gpu_no_capture_flag_skips_the_cuda_graph did not run (ADR 0081)"
    plan "gpu-parity fails if the log has a skip: line (cargo counts a skipped test as passed)"
    plan "gpu-parity.log = gpu-parity.log.full cut to 300 columns (fetch takes *.log, not *.log.full)"
    if [ "${DRY_GPU_PARITY:-ok}" = fail ]; then gate gpu_parity failed; return 1; fi
    gate gpu_parity ok
    return 0
  fi
  if [ "$skip_gpu" = 1 ]; then
    say "gpu-parity skipped (SKIP_GPU=1)"
    gate gpu_parity skipped
    return 0
  fi
  [ "$(gate_value gpu_parity)" = ok ] && { say "gpu-parity: done before"; return 0; }
  say "gpu-parity start"
  : > "$full"
  # Every package runs and every test in it (--no-fail-fast): one box run shows all failures. The full log stays on
  # the box (raw logit dumps, 20 MB on S2); results get a copy cut to 300 columns. GGML_CUDA_GRAPH_DEBUG=1 is for the
  # executor's and ggml-sys's graph tests only and must never reach a timed run.
  # The MoE tests read the 30B file from TETRAEDER_MOE_GGUF and skip (a pass for cargo) without it; REQUIRE_GPUS makes
  # missing GPUs a failure; a skip: line in the log fails the gate below.
  for pkg in tetraeder-executor tetraeder-model tetraeder-ggml-sys; do
    extra=""
    [ "$pkg" != tetraeder-model ] && extra="GGML_CUDA_GRAPH_DEBUG=1"
    f=$filters
    [ "$pkg" = tetraeder-ggml-sys ] && f=$sys_filter
    [ "$pkg" = tetraeder-model ] && f="$f $model_skip"
    # shellcheck disable=SC2086
    env $extra TETRAEDER_REQUIRE_GPU=1 TETRAEDER_REQUIRE_GPUS="$ngpu" TETRAEDER_MOE_GGUF="$(realpath "${MODEL_FILE[qwen3-30b-a3b]}")" TETRAEDER_MODELS_DIR="$models_dir" timeout -k 30 40m cargo test --release \
      --no-fail-fast -p "$pkg" --features tetraeder-ggml-sys/cuda -- --nocapture --test-threads 1 $f >> "$full" 2>&1 8>&- || {
      say "gpu-parity FAILED in $pkg (see $out/gpu-parity.log)"
      failed="$failed $pkg"
    }
  done
  cut -c1-300 "$full" > "$out/gpu-parity.log"
  if [ -n "$failed" ]; then
    gate gpu_parity failed
    return 1
  fi
  if grep -qE '(^|[ .])skip: ' "$full"; then
    say "gpu-parity FAILED: tests skipped under TETRAEDER_REQUIRE_GPU: $(grep -E '(^|[ .])skip: ' "$full" | cut -c1-120 | tr '
' ';')"
    gate gpu_parity failed
    return 1
  fi
  # --nocapture lets ggml's debug lines land between "..." and "ok", so only the start line is matched here; a failure
  # of the package is caught above.
  if ! grep -qE '^test gpu_no_capture_flag_skips_the_cuda_graph \.\.\.' "$full"; then
    say "gpu-parity FAILED: gpu_no_capture_flag_skips_the_cuda_graph did not run (ADR 0081)"
    gate gpu_parity failed
    return 1
  fi
  if ! awk '/^test result: ok\. [0-9]+ passed/ { n += $4 } END { exit !(n > 0) }' "$full"; then
    say "gpu-parity FAILED: no test ran (the gpu_ / gpu2_ / gpu4_ filters matched nothing, see $out/gpu-parity.log)"
    gate gpu_parity failed
    return 1
  fi
  gate gpu_parity ok
  say "gpu-parity done"
}

# ---- stage: discover, enumerate ----------------------------------------------------------------------------
discover() {
  if [ "$DRY" = 1 ]; then
    plan "${HW_CMD:-$tet plan hw --out $out/hw.json}"
    return 0
  fi
  [ -s "$out/hw.json" ] && return 0
  if [ -n "${HW_CMD:-}" ]; then
    eval "$HW_CMD" 8>&- || die "discover failed ($HW_CMD)"
  else
    "$tet" plan hw --out "$out/hw.json" 8>&- || die "discover failed: no hw.json"
  fi
}

# DRY only: the ADR 0068 candidate sets and the baseline modes, so the round order can be printed without a box
dry_enumeration() {
  [ "$DRY" = 1 ] || return 0
  "$py" - "$out" "$ngpu" "$models" << 'EOF'
import json, os, sys
d, n, models = sys.argv[1], int(sys.argv[2]), sys.argv[3].split()
dense = ["single", f"layer-{n}", f"tp-{n}"] + (["tp2-pp2"] if n == 4 else [])
moe = ["single", f"layer-{n}", f"tp-{n}", f"ep-{n}"] + (["tp2-pp2"] if n == 4 else [])
for m in models:
    ids = moe if "a3b" in m else dense
    p = os.path.join(d, f"candidates-{m}.json")
    if not os.path.exists(p):
        json.dump({"model": m, "gpus": n, "candidates": [{"id": i} for i in ids], "pruned": []}, open(p, "w"))
EOF
}

# DRY only: the split modes both baselines accept, written after the prepass plan
dry_prepass_files() {
  [ "$DRY" = 1 ] || return 0
  "$py" - "$out" "$models" << 'EOF'
import json, os, sys
d, models = sys.argv[1], sys.argv[2].split()
for m in models:
    for eng, sms in (("llama-bench", ["layer", "row", "tensor"]), ("ik-llama-bench", ["layer", "row", "graph"])):
        for sm in sms:
            q = os.path.join(d, f"prepass-{m}-{eng}-{sm}.json")
            if not os.path.exists(q):
                json.dump({"ok": True, "rc": 0, "comm": None}, open(q, "w"))
EOF
}

# key_plans DIR KEY LAYOUT: tetraeder-plan names a plan file <layout model name>-<id>.toml, and that name is the GGUF's
# (spaces, case); the runs use <KEY>-<id>.toml
key_plans() {
  "$py" - "$1" "$2" "$3" << 'EOF'
import json, os, shutil, sys
d, key, layout = sys.argv[1:4]
name = json.load(open(layout))["model"]
for f in sorted(os.listdir(d)):
    if f.startswith(name + "-") and f.endswith(".toml") and name != key:
        shutil.copy(os.path.join(d, f), os.path.join(d, key + f[len(name):]))
EOF
}

enumerate_cands() {
  local m
  for m in $models; do
    if [ "$DRY" = 1 ]; then
      plan "$tet plan layout ${MODEL_FILE[$m]} --out layouts/$m.json"
      plan "$planbin candidates --multi --layout layouts/$m.json --hw hw.json --out candidates-$m.json --plans-dir plans"
      continue
    fi
    [ -s "$out/candidates-$m.json" ] && continue
    "$tet" plan layout "${MODEL_FILE[$m]}" --out "$out/layouts/$m.json" 8>&- || die "layout of $m failed"
    "$planbin" candidates --multi --layout "$out/layouts/$m.json" --hw "$out/hw.json" --out "$out/candidates-$m.json" \
      --plans-dir "$out/plans" 8>&- || die "enumeration of $m failed"
    key_plans "$out/plans" "$m" "$out/layouts/$m.json"
    say "enumerate $m: $("$py" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d["candidates"]), "candidates,", len(d["pruned"]), "pruned")' "$out/candidates-$m.json")"
  done
  dry_enumeration
}

# ---- stage: probe ------------------------------------------------------------------------------------------
probe() {
  if [ "$DRY" = 1 ]; then
    plan "$tet comm-probe --gpus $gpu_list --out $out/calib-comm.json --repeats 20"
    if [ "${DRY_PROBE:-ok}" = fail ]; then gate comm_probe failed; return 1; fi
    gate comm_probe ok
    set_comm_impl
    return 0
  fi
  if [ "$skip_gpu" = 1 ]; then
    gate comm_probe skipped
    return 0
  fi
  if [ ! -s "$out/calib-comm.json" ]; then
    timeout -k 30 20m "$tet" comm-probe --gpus "$gpu_list" --out "$out/calib-comm.json" --repeats 20 > "$out/comm-probe.log" 2>&1 8>&- || {
      say "comm-probe FAILED (see $out/comm-probe.log)"
      gate comm_probe failed
      return 1
    }
  fi
  gate comm_probe ok
  set_comm_impl
  say "comm-probe: $("$py" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("comm_path"))' "$out/calib-comm.json")"
}

# ---- benchmark runs ----------------------------------------------------------------------------------------
declare -A disk_dev

# sampler_start RUN MODEL PID: host sampler beside the run, best effort; its host block feeds cotenant_guard.py
sampler_start() {
  sampler_pid=""
  [ "$DRY" = 1 ] && return 0
  if [ -z "${disk_dev[$2]:-}" ]; then
    disk_dev[$2]=$(python3 scripts/engine_sample.py probe-disk "${MODEL_FILE[$2]}" 2> /dev/null)
  fi
  [ -n "${disk_dev[$2]}" ] || return 0
  samples="runs/vast/p5-$1.samples"
  python3 scripts/engine_sample.py sample --pid "$3" --disk "${disk_dev[$2]}" --out "$samples" > /dev/null 2>&1 8>&- &
  sampler_pid=$!
}

# sampler_finish RUN T_START T_END: needs the run finished (the sampler follows its pid); writes cotenant-<run>.json
sampler_finish() {
  local run=$1 t0=$2 t1=$3 host="$out/cotenant-$1.json"
  [ "$DRY" = 1 ] && return 0
  [ -n "$sampler_pid" ] && wait "$sampler_pid" 2> /dev/null
  if [ -n "$sampler_pid" ] && [ -s "$samples" ]; then
    printf '{"t_start": %s, "t_first": %s, "t_end": %s, "prompt_tokens": 1, "decode_tokens": 2}\n' "$t0" "$t0" "$t1" \
      > "runs/vast/p5-$run.window.json"
    python3 scripts/engine_sample.py summarize --samples "$samples" --threads "$threads" \
      --kt-json "runs/vast/p5-$run.window.json" --out "$host" > /dev/null || say "host summary failed for $run"
  fi
  [ -s "$host" ] && { python3 -I scripts/cotenant_guard.py merge --json "$host" || true; }
}

# run_one KIND MODEL RUN FILE LIMIT_MIN CMD...: KIND candidate|baseline|aa|anchor. Skipped when FILE exists (resume) or the
# budget is spent. stdout of CMD is FILE (llama-bench JSON rows); the exit code goes to run-<RUN>.rc, kept for every
# run: a failed run is evidence (a candidate that fails on the box fails the case), not a hole.
run_one() {
  local kind=$1 model=$2 run=$3 f=$4 limit=$5 rc t0 t1 w0 w1
  shift 5
  if [ -f "$out/run-$run.rc" ] && [ "$kind" != anchor ]; then
    say "run $run: done before"
    return 0
  fi
  if [ "$kind" = anchor ] && [ -f "$f" ]; then
    say "run $run: done before"
    return 0
  fi
  if over_budget || [ "$(left_s)" -lt "$limit" ]; then
    say "BUDGET spent: skip run $run (the verdict will be null)"
    return 0
  fi
  say "run $run start"
  if [ "$DRY" = 1 ]; then
    plan "$* > ${f##*/}"
    gpu_note "$run" before
    gpu_note "$run" after
    return 0
  fi
  tl run "$model" "$kind" "$run"
  : > "$out/gpu-$run.txt"
  gpu_note "$run" before
  t0=$(date +%s.%N)
  w0=$(date +%Y-%m-%dT%H:%M:%S)
  setsid nohup timeout -k 30 30m "$@" < /dev/null > "$f.tmp" 2> "$out/$run.log" 8>&- &
  server_pid=$!
  sampler_start "$run" "$model" "$server_pid"
  wait "$server_pid"
  rc=$?
  server_pid=""
  t1=$(date +%s.%N)
  w1=$(date +%Y-%m-%dT%H:%M:%S)
  gpu_note "$run" after
  # box-local start and end of the process, the clock of the sampler's timestamps (phase5_summary.py run_window)
  printf '{"start": "%s", "end": "%s"}\n' "$w0" "$w1" > "$out/window-$run.json"
  echo "$rc" > "$out/run-$run.rc"
  [ -s "$f.tmp" ] && mv "$f.tmp" "$f"
  rm -f "$f.tmp"
  sampler_finish "$run" "$t0" "$t1"
  [ "$rc" = 0 ] || say "run $run FAILED rc=$rc (see $run.log)"
  say "run $run done"
}

# Commands of the three kinds of run. $threads is the one explicit -t of both engines.
tet_cmd() { # tet_cmd MODEL PLAN_FILE [LAYER_LIMIT]
  local c=("$tet" bench -m "${MODEL_FILE[$1]}" --engine "${MODEL_ENGINE[$1]}" --plan "$2" -p 2048 -n 128 -r 1 -t "$threads"
    --ngl -1 --ubatch 512 --flash-attn on -o json)
  [ -n "${3:-}" ] && c+=(--layer-limit "$3")
  echo "${c[@]}"
}
base_cmd() { # base_cmd ENGINE MODEL SM [PROMPT GEN]
  local bin=$llama_bench
  [ "$1" = ik-llama-bench ] && bin=$ik_bench
  echo "$bin -m ${MODEL_FILE[$2]} -ngl 99 -fa 1 -p ${4:-2048} -n ${5:-128} -r 1 -t $threads -ub 512 -sm $3 -o json"
}

# ---- stage: prepass ----------------------------------------------------------------------------------------
# prepass_one MODEL ENGINE SM: a short llama-bench; ok when it exits 0 with rows. comm: the all-reduce path its log names
prepass_one() {
  local m=$1 e=$2 sm=$3 f="$out/prepass-$1-$2-$3.json" plog="$out/prepass-$1-$2-$3.log" rc cmd
  [ -f "$f" ] && return 0
  cmd=$(base_cmd "$e" "$m" "$sm" 64 8)
  if [ "$DRY" = 1 ]; then
    plan "$cmd > ${f##*/}"
    return 0
  fi
  over_budget && { say "BUDGET spent: skip prepass $m $e $sm"; return 0; }
  say "prepass $m $e $sm start"
  # shellcheck disable=SC2086
  timeout -k 30 10m $cmd > "$plog.out" 2> "$plog" 8>&-
  rc=$?
  "$py" - "$f" "$rc" "$plog" "$plog.out" << 'EOF'
import json, re, sys
f, rc, log, out = sys.argv[1:5]
text = open(log, errors="replace").read()
lines = [ln.strip() for ln in text.splitlines() if re.search(r"nccl|all.?reduce|peer", ln, re.I)][:10]
comm = None
if re.search(r"nccl", text, re.I) and not re.search(r"(no|without|disabled|not found)[^\n]{0,20}nccl", text, re.I):
    comm = "nccl"
elif lines:
    comm = "internal"
try:
    rows = json.load(open(out))
except (OSError, ValueError):
    rows = []
json.dump({"ok": int(rc) == 0 and bool(rows), "rc": int(rc), "comm": comm, "comm_lines": lines}, open(f, "w"), indent=2)
EOF
  say "prepass $m $e $sm: $("$py" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("ok" if d["ok"] else "rejected rc=%s" % d["rc"])' "$f")"
}

# prepass_gate: every model needs at least one llama-bench split mode the baseline build accepts
prepass_gate() {
  local m ok=1
  for m in $models; do
    if [ "$DRY" = 1 ] && [ "${DRY_PREPASS:-ok}" = fail ]; then
      say "prepass: no llama-bench split mode accepted for $m (DRY_PREPASS=fail)"
      ok=0
    elif [ "$DRY" != 1 ] && ! "$py" -c 'import glob,json,sys
sys.exit(0 if any(json.load(open(f)).get("ok") for f in glob.glob(sys.argv[1] + "/prepass-" + sys.argv[2] + "-llama-bench-*.json")) else 1)' "$out" "$m"; then
      say "prepass: no llama-bench split mode accepted for $m (see prepass-$m-llama-bench-*.log)"
      ok=0
    fi
  done
  if [ "$ok" = 1 ]; then gate baseline_modes ok; else gate baseline_modes failed; return 1; fi
}

prepass() {
  local m ik_modes
  for m in $models; do
    for sm in layer row tensor; do prepass_one "$m" llama-bench "$sm"; done
    ik_modes="layer row"
    if [ "$DRY" = 1 ] || "$ik_bench" --help 2>&1 | grep -A3 -e '-sm' | grep -qw graph; then ik_modes="layer row graph"; fi
    if [ "$DRY" != 1 ] && [ ! -x "$ik_bench" ]; then
      say "ik_llama.cpp: $ik_bench missing, skipped"
      continue
    fi
    for sm in $ik_modes; do prepass_one "$m" ik-llama-bench "$sm"; done
  done
}

# ---- stage: anchors ----------------------------------------------------------------------------------------
# Truncated runs; no anchor equals a candidate. tp-2 on a 4x box runs on GPUs 0 and 1 with a plan made from a 2-GPU graph.
anchor_plan() { # anchor_plan MODEL ID: the plan file of an anchor
  if [ "$2" = tp-2 ] && [ "$ngpu" = 4 ]; then echo "$out/plans-anchor/$1-tp-2.toml"; else echo "$out/plans/$1-$2.toml"; fi
}

anchor_plans_4x() { # a 2-GPU hardware graph (CUDA_VISIBLE_DEVICES=0,1) gives the tp-2 plan
  [ "$ngpu" = 4 ] || return 0
  local m
  if [ "$DRY" = 1 ]; then
    plan "CUDA_VISIBLE_DEVICES=0,1 $tet plan hw --out hw-2gpu.json; $planbin candidates --multi --hw hw-2gpu.json --plans-dir plans-anchor (tp-2 anchors)"
    return 0
  fi
  mkdir -p "$out/plans-anchor"
  [ -s "$out/hw-2gpu.json" ] || CUDA_VISIBLE_DEVICES=0,1 "$tet" plan hw --out "$out/hw-2gpu.json" 8>&- || die "2-GPU discover failed"
  for m in $models; do
    [ -s "$out/plans-anchor/$m-tp-2.toml" ] && continue
    "$planbin" candidates --multi --layout "$out/layouts/$m.json" --hw "$out/hw-2gpu.json" \
      --out "$out/plans-anchor/candidates-$m.json" --plans-dir "$out/plans-anchor" 8>&- || die "2-GPU enumeration of $m failed"
    key_plans "$out/plans-anchor" "$m" "$out/layouts/$m.json"
  done
}

# anchor_ids MODEL: the anchored shapes; on a 4x box the dense model adds tp-4, a second all-reduce group size
anchor_ids() {
  if [ "$ngpu" = 4 ] && [ "${MODEL_ENGINE[$1]}" = dense ]; then echo "single tp-2 tp-4"; else echo "single tp-2"; fi
}

anchors() {
  local m k plan_id lim n_layer f vis ids want
  anchor_plans_4x
  for m in $models; do
    if [ "$DRY" = 1 ]; then n_layer=64; else n_layer=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["n_layer"])' "$out/layouts/$m.json"); fi
    ids=$(anchor_ids "$m")
    want=$(($(wc -w <<< "$ids") * 2))
    for k in 1 2; do
      f="$out/anchors-$m-r$k.json"
      [ -f "$f" ] && continue
      for plan_id in $ids; do
        for lim in $((n_layer / 4)) $((n_layer / 2)); do
          vis=""
          [ "$plan_id" = tp-2 ] && [ "$ngpu" = 4 ] && vis="env CUDA_VISIBLE_DEVICES=0,1"
          # shellcheck disable=SC2046,SC2086
          run_one anchor "$m" "$m-anchor-$plan_id-L$lim-r$k" "$out/anchor-parts/$m-$plan_id-L$lim-r$k.json" 0 \
            $vis $(tet_cmd "$m" "$(anchor_plan "$m" "$plan_id")" "$lim")
        done
      done
      if [ "$DRY" = 1 ]; then
        plan "merge anchor-parts/$m-*-r$k.json ($want parts) -> anchors-$m-r$k.json (plan_id, layer_limit set on every row)"
      else
        "$py" - "$out" "$m" "$k" "$want" << 'EOF' || say "WARNING: anchors for $m round $k incomplete"
import glob, json, os, re, sys
d, m, k, want = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = []
parts = sorted(glob.glob(os.path.join(d, "anchor-parts", f"{m}-*-r{k}.json")))
if len(parts) < want:
    sys.exit(1)
for p in parts:
    mt = re.fullmatch(rf"{re.escape(m)}-(.+)-L(\d+)-r{k}\.json", os.path.basename(p))
    for r in json.load(open(p)):
        r["plan_id"] = mt.group(1)  # bench rows carry a content hash there; calibrate wants the shape id
        r.setdefault("layer_limit", int(mt.group(2)))
        rows.append(r)
json.dump(rows, open(os.path.join(d, f"anchors-{m}-r{k}.json"), "w"))
EOF
      fi
    done
  done
}

# ---- stage: calibrate, predict -----------------------------------------------------------------------------
# routing_cmd MODEL FILE: one pass of the judged pp2048 and tg128 token streams on GPU 0, the router's choices recorded
routing_cmd() {
  echo "env CUDA_VISIBLE_DEVICES=0 $tet bench -m ${MODEL_FILE[$1]} --engine moe -p 2048 -n 128 -r 1 -t $threads --ngl -1" \
    "--ubatch 512 --flash-attn on -o json --routing-out $2"
}

# routing: per MoE model routing-<model>.json (ADR 0070); a model without it would be priced at uniform routing, so a
# failure ends the session
routing() {
  local m f
  for m in $models; do
    [ "${MODEL_ENGINE[$m]}" = moe ] || continue
    f="$out/routing-$m.json"
    if [ "$DRY" = 1 ]; then
      plan "$(routing_cmd "$m" "${f##*/}")"
      continue
    fi
    [ -s "$f" ] && continue
    say "routing $m start"
    # shellcheck disable=SC2046
    timeout -k 30 15m $(routing_cmd "$m" "$f.tmp") < /dev/null > /dev/null 2> "$out/routing-$m.log" 8>&- \
      || die "routing record of $m failed (see routing-$m.log)"
    mv "$f.tmp" "$f"
    tail -n 3 "$out/routing-$m.log"
  done
}

calibrate() {
  local comm=()
  routing
  [ "$skip_gpu" != 1 ] && comm=(--comm "$out/calib-comm.json")
  if [ "$DRY" = 1 ]; then
    plan "$planbin calibrate --hw hw.json --comm calib-comm.json --layout-dir layouts --anchors . --out calib.json"
    return 0
  fi
  [ -s "$out/calib.json" ] && return 0
  "$planbin" calibrate --hw "$out/hw.json" "${comm[@]}" --layout-dir "$out/layouts" --anchors "$out" --out "$out/calib.json" \
    > "$out/calibrate.log" 2>&1 8>&- || die "calibrate failed (see calibrate.log)"
  cat "$out/calibrate.log"
}

predict() {
  local m comm=() f sha routing=()
  [ "$skip_gpu" != 1 ] && comm=(--comm "$out/calib-comm.json")
  for m in $models; do
    f="$out/predict-$m-$box.json"
    routing=()
    [ "${MODEL_ENGINE[$m]}" = moe ] && routing=(--routing "$out/routing-$m.json")
    if [ "$DRY" = 1 ]; then
      plan "$planbin predict --hw hw.json --comm calib-comm.json --layout layouts/$m.json --calib calib.json --candidates candidates-$m.json${routing:+ --routing routing-$m.json} --out ${f##*/}"
      plan "sha256 of ${f##*/} -> predict-$m-$box.sha256, timeline.jsonl"
      continue
    fi
    if [ -s "$f" ]; then
      say "predict $m: done before ($(cut -d' ' -f1 "${f%.json}.sha256" 2> /dev/null))" # no new timeline event on resume
      continue
    fi
    TETRAEDER_GIT_COMMIT="$(git rev-parse HEAD 2> /dev/null)" "$planbin" predict --hw "$out/hw.json" "${comm[@]}" \
      --layout "$out/layouts/$m.json" --calib "$out/calib.json" --candidates "$out/candidates-$m.json" "${routing[@]}" --out "$f" 8>&- \
      || die "predict failed for $m"
    sha=$(sha256sum "$f" | cut -d' ' -f1)
    echo "$sha  ${f##*/}" > "${f%.json}.sha256"
    tl predict "$m" prediction "${f##*/}" "$sha"
    say "PREDICT $m sha256 $sha (written before the first candidate run)"
  done
}

# ---- stage: rounds -----------------------------------------------------------------------------------------
# do_round MODEL K [only]: the configurations of summary --round-order, one run each
do_round() {
  local m=$1 k=$2 only=${3:-} line cfgs=() kind a b opt=() order
  [ -n "$only" ] && opt=(--only-need)
  [ -n "${ONLY:-}" ] && opt=(--only "$ONLY") # step-0: the ONLY configurations, no reruns
  order=$(summ "$out" --round-order "$m" "$k" "${opt[@]}") || die "phase5_summary.py --round-order $m $k failed"
  [ -n "$order" ] || [ -n "$only" ] || [ -n "${ONLY:-}" ] || die "round $m $k has no configurations (no candidates or baselines in $out)"
  order=${order//$'\r'/} # a python on Windows prints CRLF
  mapfile -t cfgs <<< "$order"
  say "round $m $k: ${#cfgs[@]} configurations"
  for line in "${cfgs[@]}"; do
    [ -n "$line" ] || continue
    read -r kind a b <<< "$line"
    if [ "$kind" = tet ]; then
      # shellcheck disable=SC2046
      run_one candidate "$m" "$m-$a-r$k" "$out/bench-$m-$a-r$k.json" 600 $(tet_cmd "$m" "$out/plans/$m-$a.toml")
    else
      # shellcheck disable=SC2046
      run_one baseline "$m" "$m-$a-$b-r$k" "$out/baseline-$m-$a-$b-r$k.json" 600 $(base_cmd "$a" "$m" "$b")
    fi
  done
}

rounds() {
  local m k first
  for m in $models; do
    for k in $(seq 1 "$n_rounds"); do do_round "$m" "$k"; done
    [ -n "$only" ] && continue # step-0 check: two interleaved pairs, no reruns, no A/A
    # an unclean run is rerun in a later round, never dropped; at most two more rounds, only what is short
    for k in $(seq $((n_rounds + 1)) $((n_rounds + reruns))); do
      if [ "$DRY" = 1 ] && ! ls "$out"/run-"$m"-*.rc > /dev/null 2>&1; then break; fi
      [ -n "$(summ "$out" --need-rerun "$m")" ] || break
      over_budget && { say "BUDGET spent: no rerun round $k for $m"; break; }
      say "rerun round $k for $m"
      do_round "$m" "$k" only
    done
    first=$(summ "$out" --first "$m")
    # A/A: the first candidate again, the last run of the case
    # shellcheck disable=SC2046
    run_one aa "$m" "$m-$first-aa" "$out/bench-$m-$first-aa.json" 300 $(tet_cmd "$m" "$out/plans/$m-$first.toml")
  done
}

# ---- stage: pp-overlap -------------------------------------------------------------------------------------
# Reported only (ADR 0078): the session's tp2-pp2 plan with prefill_micro_batches 1 and 4, two rounds interleaved. Run
# names start with ppov- (not the model), so the summary never takes them for candidates.
pp_overlap_stage() {
  local m src mb j order
  if [ "$pp_overlap" != 1 ] || [ -n "$only" ]; then
    say "pp-overlap skipped (PP_OVERLAP=$pp_overlap${only:+, ONLY})"
    return 0
  fi
  mkdir -p "$out/pp-overlap-parts"
  for m in $models; do
    src="$out/plans/$m-tp2-pp2.toml"
    if ! "$py" -c 'import json,sys; sys.exit(0 if any(c["id"] == "tp2-pp2" for c in json.load(open(sys.argv[1]))["candidates"]) else 1)' \
      "$out/candidates-$m.json" 2> /dev/null; then
      say "pp-overlap $m: no tp2-pp2 candidate, skipped"
      continue
    fi
    for mb in 1 4; do
      if [ "$DRY" = 1 ]; then
        plan "plans/$m-tp2-pp2.toml with prefill_micro_batches = $mb -> plans/$m-tp2-pp2-mb$mb.toml"
        continue
      fi
      "$py" - "$src" "$out/plans/$m-tp2-pp2-mb$mb.toml" "$mb" << 'EOF' || die "pp-overlap plan of $m failed"
import re, sys
src, dst, mb = sys.argv[1:4]
text, n = re.subn(r"(?m)^prefill_micro_batches\s*=\s*\d+", f"prefill_micro_batches = {mb}", open(src).read())
if n != 1:
    sys.exit(f"{src}: {n} prefill_micro_batches lines")
open(dst, "w").write(text)
EOF
    done
    for j in 1 2; do
      order="1 4"
      [ "$j" = 2 ] && order="4 1"
      for mb in $order; do
        # shellcheck disable=SC2046
        run_one overlap "$m" "ppov-$m-mb$mb-r$j" "$out/pp-overlap-parts/$m-mb$mb-r$j.json" 600 \
          $(tet_cmd "$m" "$out/plans/$m-tp2-pp2-mb$mb.toml")
      done
    done
    if [ "$DRY" = 1 ]; then
      plan "merge pp-overlap-parts/$m-mb*-r*.json -> pp-overlap-$m.json (reported, not judged)"
      continue
    fi
    "$py" - "$out" "$m" << 'EOF' || say "WARNING: pp-overlap-$m.json not written"
import glob, json, os, re, statistics, sys
d, m = sys.argv[1:3]
runs = []
for p in sorted(glob.glob(os.path.join(d, "pp-overlap-parts", f"{m}-mb*-r*.json"))):
    mt = re.fullmatch(rf"{re.escape(m)}-mb(\d+)-r(\d+)\.json", os.path.basename(p))
    if not mt:
        continue
    rate = {}
    for r in json.load(open(p)):
        if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0:
            rate["pp_tps"] = r["avg_ts"]
        elif r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0:
            rate["tg_tps"] = r["avg_ts"]
    runs.append({"micro_batches": int(mt.group(1)), "round": int(mt.group(2)), **rate})
med = {}
for mb in sorted({r["micro_batches"] for r in runs}):
    v = [r["pp_tps"] for r in runs if r["micro_batches"] == mb and "pp_tps" in r]
    if v:
        med[str(mb)] = statistics.median(v)
ratio = med["4"] / med["1"] if "1" in med and "4" in med else None
out = {"model": m, "plan": "tp2-pp2", "judged": False, "runs": runs, "median_pp_tps": med, "pp_ratio_mb4_mb1": ratio}
json.dump(out, open(os.path.join(d, f"pp-overlap-{m}.json"), "w"), indent=2)
print(f"pp-overlap {m}: median pp {med}, mb4 / mb1 {ratio if ratio is None else round(ratio, 3)}")
EOF
  done
}

# ---- stage: vllm -------------------------------------------------------------------------------------------
# Reported only (other weights and kernels). Skipped first when the session runs short.
vllm_stage() {
  local m ck f run url=http://127.0.0.1:18090 vpy=/root/p5venv/bin/python rc
  if [ "$DRY" = 1 ]; then
    plan "vllm serve <awq|gptq checkpoint> --tensor-parallel-size $ngpu; gateway_bench.py (prompt 2048, 128 tokens) -> baseline-<model>-vllm-tp-r1.json"
    return 0
  fi
  if [ "$(left_s)" -lt 1500 ]; then
    say "vllm skipped: $(($(left_s) / 60)) min of the budget left, under 25"
    return 0
  fi
  [ "$skip_gpu" = 1 ] && { say "vllm skipped (SKIP_GPU=1)"; return 0; }
  if [ ! -x "$vpy" ]; then
    command -v uv > /dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.local/bin sh > /dev/null 2>&1; export PATH="/root/.local/bin:$PATH"; }
    timeout -k 30 15m sh -c "uv venv /root/p5venv && uv pip install --python $vpy vllm httpx" > "$out/vllm-install.log" 2>&1 8>&- \
      || { say "vllm install failed or over 15 min (vllm-install.log): skipped"; return 0; }
  fi
  for m in $models; do
    ck=${MODEL_VLLM[$m]}
    run="$m-vllm-tp-r1"
    f="$out/baseline-$run.json"
    [ -f "$out/run-$run.rc" ] && continue
    [ -d "$ck" ] || { say "vllm $m: no checkpoint at $ck, skipped"; continue; }
    [ "$(left_s)" -lt 1500 ] && { say "vllm skipped: under 25 min left"; return 0; }
    say "run $run start"
    : > "$out/gpu-$run.txt"
    gpu_note "$run" before
    tl run "$m" baseline "$run"
    setsid nohup "$vpy" -m vllm.entrypoints.openai.api_server --model "$ck" --tensor-parallel-size "$ngpu" --max-model-len 4096 \
      --served-model-name vllm-m --port 18090 < /dev/null > "$out/server-$run.log" 2>&1 8>&- &
    server_pid=$!
    for _ in $(seq 1 450); do # 15 min to load
      kill -0 "$server_pid" 2> /dev/null || break
      curl -sf -o /dev/null "$url/v1/models" && break
      sleep 2
    done
    if curl -sf -o /dev/null "$url/v1/models"; then
      timeout -k 30 15m "$vpy" scripts/gateway_bench.py "$url" --model vllm-m --temperature 0 --no-keepalive --label "$run" \
        --prompt-tokens 2048 --max-tokens 128 --concurrency 1 --requests 4 --seed 51 --out "$f.tmp" >> "$out/bench.log" 2>&1
      rc=$?
    else
      rc=1
    fi
    gpu_note "$run" after
    echo "$rc" > "$out/run-$run.rc"
    [ -s "$f.tmp" ] && mv "$f.tmp" "$f"
    stop_server
    say "run $run done rc=$rc"
  done
}

# ---- stage: summary ----------------------------------------------------------------------------------------
summary() {
  if [ "$DRY" = 1 ]; then
    if [ -n "$only" ]; then plan "phase5_summary.py $out --only \"$only\" -> step0.json"; else plan "phase5_summary.py $out -> verdict.json, verdict-notes.json"; fi
    return 0
  fi
  if [ -n "$only" ]; then summ "$out" --only "$only"; else summ "$out"; fi
}

stop_session() { # a gate failed: nothing timed ran beyond the probe; the driver fetches and destroys the box
  stop_server
  stop_gpu_sampler
  summary || true
  touch "$out/DONE"
  say "PHASE5_BOX_DONE rc=3"
}

main() {
  say "phase5_box start (session=$session ngpu=$ngpu threads=$threads models=\"$models\" budget=${BUDGET_MIN:-150} min) out=$out"
  say "STAGE gpus"
  gpus
  say "STAGE env"
  env_json
  say "STAGE build"
  build
  say "STAGE models"
  models_check
  say "STAGE gpu-parity"
  local rc=0 parity="" probe_failed=""
  gpu_parity || parity=failed
  say "STAGE discover"
  discover
  say "STAGE enumerate"
  enumerate_cands
  say "STAGE probe"
  probe || probe_failed=failed
  say "STAGE prepass"
  prepass
  dry_prepass_files
  local prepass_failed=""
  prepass_gate || prepass_failed=1
  if [ -n "$parity" ] || [ -n "$probe_failed" ] || [ -n "$prepass_failed" ]; then
    say "gate failed (gpu-parity=${parity:-ok} comm-probe=${probe_failed:-ok}${prepass_failed:+ baseline-modes=failed}): the session ends before anchors"
    stop_session
    return 3
  fi
  say "STAGE anchors"
  if [ -n "$only" ]; then say "anchors skipped (ONLY)"; else anchors; fi
  say "STAGE calibrate"
  if [ -n "$only" ]; then say "calibrate skipped (ONLY)"; else calibrate; fi
  say "STAGE predict"
  if [ -n "$only" ]; then say "predict skipped (ONLY)"; else predict; fi
  say "STAGE rounds"
  [ "$DRY" = 1 ] && plan "host sampler beside every run (engine_sample.py) and cotenant_guard.py merge -> cotenant-<run>.json"
  rounds
  say "STAGE pp-overlap"
  pp_overlap_stage
  say "STAGE vllm"
  if [ -n "$only" ]; then say "vllm skipped (ONLY)"; else vllm_stage; fi
  say "STAGE summary"
  stop_gpu_sampler
  summary
  rc=$?
  touch "$out/DONE"
  say "PHASE5_BOX_DONE rc=$rc"
  return $rc
}

if [ "$DRY" = 1 ]; then
  main
else
  main >> "$log" 2>&1
fi
