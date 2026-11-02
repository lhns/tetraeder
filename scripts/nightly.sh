#!/usr/bin/env bash
# Nightly job, runs ON the Vast box (started detached by `taxameter nightly` via box.sh run).
# Stages: cuda check, cargo build, cargo test, parity (greedy golden check, parity.json), two optional stages (below),
# bench env snapshot (env.json), benchmark spec (bench/spec/phase0.toml; BENCH_SPEC=phase2-baseline selects the Phase 2
# llama.cpp baseline instead).
# Optional stages, off unless asked (taxameter's `nightly` cannot pass env; use `taxameter run scripts/nightly.sh --env ...`).
# Their exit codes are logged, never the night's:
#   gpu-moe  MOE_CHECK=1: the builder on the real Qwen3-30B-A3B Q4_K_M (tests/real_moe.rs: greedy equal to libllama, resident
#            and streaming) -> $date_dir/builder-real-<HHMMSS>.json. MOE_GGUF overrides the model path.
#   gap      GAP_PAIR=<pair id of phase0.toml>, e.g. gate-tetraeder-cuda-dense: that pair again with 10 repeats ->
#            $date_dir/gap/ (own parity.json, results json). Needs both models of phase0.toml (provisioned from it).
# Always ends with bench/results/<date>/DONE (content: rc=N) and a final `NIGHTLY_DONE rc=N` log line, even on
# failure or signal, so taxameter can verify the fetch and read the verdict.
#
# Exit codes (the first failing stage decides):
#   from the bench: 0 ok   1 regression beyond the noise band   2 error or skipped pair   3 inconclusive (noisy box)
#   own: 10 build failed   11 tests failed   12 env snapshot failed   13 CUDA unusable
#   124 a stage hit its time limit   143 killed by a signal
# Time limits in minutes. taxameter's nightly_wait_min (taxameter.toml, now 170) must cover the sum of the limits of the
# stages that run. Worst case with both optional stages: phase0 40+15+5+15+10+45 = 130, phase2-baseline
# 40+15+5+15+10+80 = 165 (+ a 30 s kill grace per stage). Without them 105 and 140.
BUILD_MIN="${BUILD_MIN:-40}"
TEST_MIN="${TEST_MIN:-15}"
PARITY_MIN="${PARITY_MIN:-5}"
MOE_MIN="${MOE_MIN:-15}"  # three loads of 18.6 GB from disk, libllama, the checks
GAP_MIN="${GAP_MIN:-10}"  # one pair, warmup + 10 repeats, plus a parity run when the night's spec is not phase0
BENCH_SPEC="${BENCH_SPEC:-phase0}"  # name of a spec in bench/spec/
case "$BENCH_SPEC" in phase0) bench_default=45;; phase2-baseline) bench_default=80;; *) bench_default=45;; esac
BENCH_MIN="${BENCH_MIN:-$bench_default}"  # the phase0 spec took 31 min on an RTX 3090 box; phase2-baseline is estimated at 60 with the ik_llama variants
exec 8> /root/tetraeder_nightly.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
[ -f "bench/spec/$BENCH_SPEC.toml" ] || { echo "no spec bench/spec/$BENCH_SPEC.toml"; exit 2; }
# Image toolchain lives in /opt/cargo (RUSTUP_HOME/CARGO_HOME); a rustup under $HOME is the fallback
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
date_dir="bench/results/$(date +%Y%m%d)"
mkdir -p "$date_dir" runs/vast
rm -f "$date_dir/DONE"
# Fresh log: an old NIGHTLY_DONE line must never satisfy taxameter's wait. The log is open in append mode.
log=runs/vast/nightly.log
if [ -s "$log" ]; then cp -f "$log" "$log.prev"; : > "$log"; fi
rc=0
trap 'rc=143; exit 143' TERM INT HUP

stage() {  # stage <name> <limit-min|0> <cmd...>: log "[HH:MM:SS] start|done <name>" (taxameter health parses these lines)
  local name="$1" limit="$2"; shift 2
  echo "[$(date +%T)] start $name"
  if [ "$limit" -gt 0 ]; then timeout -k 30 "${limit}m" "$@"; else "$@"; fi
  local r=$?
  echo "[$(date +%T)] done $name rc=$r"
  return $r
}

finish() {
  echo "rc=$rc" > "$date_dir/DONE"
  echo "NIGHTLY_DONE rc=$rc"
  # sccache inherits fd 8 and would keep the flock (and a stale server) alive after this script exits
  command -v sccache > /dev/null 2>&1 && timeout 30 sccache --stop-server > /dev/null 2>&1
  return 0
}
trap finish EXIT

# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
echo "[$(date +%T)] cuda: $CUDA_ENV_STATUS"
if [ "$CUDA_ENV_STATUS" = fail ]; then rc=13; echo "cuda unusable, aborting"; exit "$rc"; fi

# fail_stage <rc-if-failed> <stage rc>: a timeout (124) keeps its code
fail_stage() { if [ "$2" = 124 ]; then rc=124; else rc="$1"; fi; exit "$rc"; }
cuda_feat=tetraeder-ggml-sys/cuda
# Build ggml CUDA only for this box's GPUs (compute_cap 8.6 -> 86), not the 4-arch default (37 min cold). An explicit value wins.
if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  if [ -n "$archs" ]; then export TETRAEDER_CUDA_ARCHS="$archs"; fi
fi
echo "[$(date +%T)] cuda archs: ${TETRAEDER_CUDA_ARCHS:-default}"
stage build "$BUILD_MIN" cargo build --release --workspace --features "$cuda_feat" || fail_stage 10 $?
# --release reuses the ggml build of the stage above (a debug test build compiled ggml again, 5.5 min)
stage test "$TEST_MIN" cargo test --release --workspace --features "$cuda_feat" || fail_stage 11 $?

parity_check() {  # quality next to speed: greedy golden check of the Tetraeder build against llama-simple (ADR 0026).
  # Writes $date_dir/parity.json; the bench attaches it to every Tetraeder gate. Its exit code is not the night's:
  # a failed or missing check makes the gate pair an error there. A stale file from an earlier run must never count.
  rm -f "$date_dir/parity.json"
  timeout -k 30 "${PARITY_MIN}m" target/release/tetraeder-bench parity "bench/spec/$BENCH_SPEC.toml" --out-dir "$date_dir" \
    --llama-simple "${LLAMA_SIMPLE:-/opt/llama.cpp/bin/llama-simple}" --models-dir "${TETRAEDER_MODELS_DIR:-.hf_cache}"
}
stage parity 0 parity_check || echo "parity rc=$? (Tetraeder gates will report an error)"

abs() { case "$1" in /*) echo "$1";; *) echo "$PWD/$1";; esac; }  # cargo test runs in the crate dir: paths must be absolute
models_dir="$(abs "${TETRAEDER_MODELS_DIR:-.hf_cache}")"

if [ "${MOE_CHECK:-0}" = 1 ]; then
  # same path rule as the bench: <models-dir>/<org>__<repo>/<file>; same build as the stages above (release, same features)
  moe_gguf="${MOE_GGUF:-$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf}"
  if [ -f "$moe_gguf" ]; then
    stage gpu-moe "$MOE_MIN" env TETRAEDER_MOE_GGUF="$moe_gguf" TETRAEDER_REAL_OUT="$(abs "$date_dir")/builder-real-$(date +%H%M%S).json" \
      cargo test --release -p tetraeder-model --features "$cuda_feat" --test real_moe -- --nocapture \
      || echo "gpu-moe rc=$? (not the night's)"
  else
    echo "gpu-moe skipped: no model file $moe_gguf"
  fi
fi

gap_run() {  # run by bash -c under a timeout; the phase0 pair again with more repeats, own results dir and parity.json
  local dir="$date_dir/gap" spec=bench/spec/phase0.toml
  grep -q "^id = \"$GAP_PAIR\"" "$spec" || { echo "no pair $GAP_PAIR in $spec"; return 2; }
  mkdir -p "$dir"
  rm -f "$dir/parity.json"
  # The Tetraeder gate needs a parity.json for phase0. The night's own is that when the spec is phase0; otherwise it
  # is absent or from another spec (phase2-baseline has no Tetraeder engine: an empty file), so check again here.
  if [ "$BENCH_SPEC" = phase0 ] && [ -s "$date_dir/parity.json" ]; then
    cp "$date_dir/parity.json" "$dir/parity.json"
  else
    timeout -k 30 "${PARITY_MIN}m" target/release/tetraeder-bench parity "$spec" --out-dir "$dir" \
      --llama-simple "${LLAMA_SIMPLE:-/opt/llama.cpp/bin/llama-simple}" --models-dir "$models_dir" \
      || echo "gap parity rc=$? (the Tetraeder gate will report an error)"
  fi
  target/release/tetraeder-bench run "$spec" --only "$GAP_PAIR" --repeats 10 --strict --out-dir "$dir" \
    --llama-bench "${LLAMA_BENCH:-/opt/llama.cpp/bin/llama-bench}" --models-dir "$models_dir"
}
if [ -n "${GAP_PAIR:-}" ]; then
  export -f gap_run
  export date_dir BENCH_SPEC GAP_PAIR PARITY_MIN models_dir
  stage gap "$GAP_MIN" bash -c gap_run || echo "gap rc=$? (not the night's)"
fi

bench_env() {  # shell-side environment snapshot next to the harness results (the harness also records its own)
  q() { nvidia-smi --query-gpu="$1" --format=csv,noheader 2> /dev/null | head -1 | sed 's/"//g'; }
  # Container CPU quota in cores (cgroup v2 cpu.max, else v1 cfs), null if unlimited. nproc ignores the quota, so
  # available_parallelism is min(nproc, ceil(quota)), what Rust reports; the harness records the threads it passes.
  local quota=null cpus avail m1 m2
  cpus=$(nproc)
  if [ -r /sys/fs/cgroup/cpu.max ]; then read -r m1 m2 < /sys/fs/cgroup/cpu.max
  elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then m1=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); m2=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us); fi
  case "${m1:-max}" in max | -1 | 0) ;; *) quota=$(awk -v q="$m1" -v p="${m2:-100000}" 'BEGIN {printf "%.2f", q / p}');; esac
  avail=$cpus
  if [ "$quota" != null ]; then avail=$(awk -v c="$cpus" -v q="$quota" 'BEGIN {n = int(q) < q ? int(q) + 1 : int(q); print (n < c ? n : c)}'); fi
  cat > "$date_dir/env.json" <<EOJ
{
  "date": "$(date -u +%FT%TZ)",
  "host": "$(hostname)",
  "container_id": "${CONTAINER_ID:-}",
  "kernel": "$(uname -r)",
  "cpu": "$(sed -n 's/^model name[^:]*: *//p' /proc/cpuinfo | head -1)",
  "cpus": $cpus,
  "cpu_quota": $quota,
  "available_parallelism": $avail,
  "cuda_archs": "${TETRAEDER_CUDA_ARCHS:-default}",
  "ram_gb": $(awk '/MemTotal/ {printf "%d", $2 / 1048576}' /proc/meminfo),
  "disk_read_mbs": "$(cat /root/.disk_probe 2> /dev/null)",
  "gpu": "$(q name)",
  "driver": "$(q driver_version)",
  "clocks_sm_mhz": "$(q clocks.sm)",
  "clocks_mem_mhz": "$(q clocks.mem)",
  "clocks_max_sm_mhz": "$(q clocks.max.sm)",
  "throttle_reasons": "$(q clocks_throttle_reasons.active)",
  "power_w": "$(q power.draw)",
  "temperature_c": "$(q temperature.gpu)",
  "cuda_env": "$CUDA_ENV_STATUS"
}
EOJ
}
stage bench-env 0 bench_env || rc=12

bench_spec() {  # paired llama.cpp baseline run; writes $date_dir/$BENCH_SPEC-<HHMMSS>.json; exit codes above
  # llama-bench lives in /opt/llama.cpp/bin on the box (LLAMA_BENCH overrides). --strict: a pair skipped for a
  # missing binary or model fails the night, so a run that measured nothing never passes.
  timeout -k 30 "${BENCH_MIN}m" target/release/tetraeder-bench run "bench/spec/$BENCH_SPEC.toml" --strict --out-dir "$date_dir" \
    --llama-bench "${LLAMA_BENCH:-/opt/llama.cpp/bin/llama-bench}" --models-dir "${TETRAEDER_MODELS_DIR:-.hf_cache}"
}
stage bench 0 bench_spec  # the time limit is inside bench_spec (timeout cannot run a function)
r=$?
if [ "$r" -ne 0 ]; then rc=$r; fi  # 1 regression, 2 error, 3 noisy, 124 timeout

exit "$rc"
