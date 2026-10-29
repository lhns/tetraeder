#!/usr/bin/env bash
# Nightly job, runs ON the Vast box (started detached by `vastctl nightly` via vast_box.sh run).
# Stages: cuda check, cargo build, cargo test, bench env snapshot (env.json), benchmark spec (bench/spec/phase0.toml).
# Always ends with bench/results/<date>/DONE (content: rc=N) and a final `NIGHTLY_DONE rc=N` log line, even on
# failure, so vastctl can verify the fetch and read the verdict.
exec 8> /root/tetraeder_nightly.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
date_dir="bench/results/$(date +%Y%m%d)"
mkdir -p "$date_dir"
rc=0

stage() {  # stage <name> <cmd...>: log "[HH:MM:SS] start|done <name>" (vastctl health parses these lines)
  local name="$1"; shift
  echo "[$(date +%T)] start $name"
  "$@"
  local r=$?
  echo "[$(date +%T)] done $name rc=$r"
  return $r
}

finish() {
  echo "rc=$rc" > "$date_dir/DONE"
  echo "NIGHTLY_DONE rc=$rc"
}
trap finish EXIT

# shellcheck disable=SC1091
. scripts/cuda_env.sh
echo "[$(date +%T)] cuda: $CUDA_ENV_STATUS"
if [ "$CUDA_ENV_STATUS" = fail ]; then rc=2; echo "cuda unusable, aborting"; exit "$rc"; fi

stage build cargo build --release --workspace || { rc=1; exit "$rc"; }
stage test cargo test --workspace || { rc=1; exit "$rc"; }

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
stage bench-env bench_env || rc=1

bench_spec() {  # paired llama.cpp baseline run; writes $date_dir/phase0.json, exits 1 on a regression, 2 on an error
  # llama-bench lives in /opt/llama.cpp/bin on the box (LLAMA_BENCH overrides); missing binaries or models skip pairs
  # (status "skipped" in the JSON) instead of failing the night.
  target/release/tetraeder-bench run bench/spec/phase0.toml --out-dir "$date_dir" \
    --llama-bench "${LLAMA_BENCH:-/opt/llama.cpp/bin/llama-bench}" --models-dir "${TETRAEDER_MODELS_DIR:-.hf_cache}"
}
stage bench bench_spec || rc=1
exit "$rc"
