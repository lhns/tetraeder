#!/usr/bin/env bash
# Nightly job, runs ON the Vast box (started detached by `vastctl nightly` via vast_box.sh run).
# Stages: cuda check, cargo build, cargo test, bench env snapshot (env.json), benchmark spec (bench/spec/phase0.toml).
# Always ends with bench/results/<date>/DONE (content: rc=N) and a final `NIGHTLY_DONE rc=N` log line, even on
# failure or signal, so vastctl can verify the fetch and read the verdict.
#
# Exit codes (the first failing stage decides):
#   from the bench: 0 ok   1 regression beyond the noise band   2 error or skipped pair   3 inconclusive (noisy box)
#   own: 10 build failed   11 tests failed   12 env snapshot failed   13 CUDA unusable
#   124 a stage hit its time limit   143 killed by a signal
# Time limits (BUILD_MIN + TEST_MIN + BENCH_MIN stay well under vastctl's nightly_wait_min of 90).
BUILD_MIN="${BUILD_MIN:-40}"
TEST_MIN="${TEST_MIN:-15}"
BENCH_MIN="${BENCH_MIN:-30}"
exec 8> /root/tetraeder_nightly.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
# Image toolchain lives in /opt/cargo (RUSTUP_HOME/CARGO_HOME); a rustup under $HOME is the fallback
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
date_dir="bench/results/$(date +%Y%m%d)"
mkdir -p "$date_dir" runs/vast
rm -f "$date_dir/DONE"
# Fresh log: an old NIGHTLY_DONE line must never satisfy vastctl's wait. The log is open in append mode.
log=runs/vast/nightly.log
if [ -s "$log" ]; then cp -f "$log" "$log.prev"; : > "$log"; fi
rc=0
trap 'rc=143; exit 143' TERM INT HUP

stage() {  # stage <name> <limit-min|0> <cmd...>: log "[HH:MM:SS] start|done <name>" (vastctl health parses these lines)
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
. scripts/cuda_env.sh
echo "[$(date +%T)] cuda: $CUDA_ENV_STATUS"
if [ "$CUDA_ENV_STATUS" = fail ]; then rc=13; echo "cuda unusable, aborting"; exit "$rc"; fi

# fail_stage <rc-if-failed> <stage rc>: a timeout (124) keeps its code
fail_stage() { if [ "$2" = 124 ]; then rc=124; else rc="$1"; fi; exit "$rc"; }
cuda_feat=tetraeder-ggml-sys/cuda
stage build "$BUILD_MIN" cargo build --release --workspace --features "$cuda_feat" || fail_stage 10 $?
stage test "$TEST_MIN" cargo test --workspace --features "$cuda_feat" || fail_stage 11 $?

bench_env() {  # shell-side environment snapshot next to the harness results (the harness also records its own)
  q() { nvidia-smi --query-gpu="$1" --format=csv,noheader 2> /dev/null | head -1 | sed 's/"//g'; }
  cat > "$date_dir/env.json" <<EOJ
{
  "date": "$(date -u +%FT%TZ)",
  "host": "$(hostname)",
  "container_id": "${CONTAINER_ID:-}",
  "kernel": "$(uname -r)",
  "cpu": "$(sed -n 's/^model name[^:]*: *//p' /proc/cpuinfo | head -1)",
  "cpus": $(nproc),
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

bench_spec() {  # paired llama.cpp baseline run; writes $date_dir/phase0-<HHMMSS>.json; exit codes above
  # llama-bench lives in /opt/llama.cpp/bin on the box (LLAMA_BENCH overrides). --strict: a pair skipped for a
  # missing binary or model fails the night, so a run that measured nothing never passes.
  timeout -k 30 "${BENCH_MIN}m" target/release/tetraeder-bench run bench/spec/phase0.toml --strict --out-dir "$date_dir" \
    --llama-bench "${LLAMA_BENCH:-/opt/llama.cpp/bin/llama-bench}" --models-dir "${TETRAEDER_MODELS_DIR:-.hf_cache}"
}
stage bench 0 bench_spec  # the time limit is inside bench_spec (timeout cannot run a function)
r=$?
if [ "$r" -ne 0 ]; then rc=$r; fi  # 1 regression, 2 error, 3 noisy, 124 timeout

exit "$rc"
