#!/usr/bin/env bash
# Nightly job, runs ON the Vast box (started detached by `taxameter nightly` via box.sh run).
# Stages: cuda check, cargo build, cargo test, parity (greedy golden check, parity.json), bench env snapshot (env.json),
# benchmark spec (bench/spec/phase0.toml; BENCH_SPEC=phase2-baseline selects the Phase 2 llama.cpp baseline instead).
# Always ends with bench/results/<date>/DONE (content: rc=N) and a final `NIGHTLY_DONE rc=N` log line, even on
# failure or signal, so taxameter can verify the fetch and read the verdict.
#
# Exit codes (the first failing stage decides):
#   from the bench: 0 ok   1 regression beyond the noise band   2 error or skipped pair   3 inconclusive (noisy box)
#   own: 10 build failed   11 tests failed   12 env snapshot failed   13 CUDA unusable
#   124 a stage hit its time limit   143 killed by a signal
# Time limits (BUILD_MIN + TEST_MIN + PARITY_MIN + BENCH_MIN stay under taxameter's nightly_wait_min of 110 for phase0;
# phase2-baseline needs a larger nightly_wait_min).
BUILD_MIN="${BUILD_MIN:-40}"
TEST_MIN="${TEST_MIN:-15}"
PARITY_MIN="${PARITY_MIN:-5}"
BENCH_SPEC="${BENCH_SPEC:-phase0}"  # name of a spec in bench/spec/
case "$BENCH_SPEC" in phase0) bench_default=45;; phase2-baseline) bench_default=60;; *) bench_default=45;; esac
BENCH_MIN="${BENCH_MIN:-$bench_default}"  # the phase0 spec took 31 min on an RTX 3090 box; phase2-baseline is estimated at 42
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
