#!/usr/bin/env bash
# Session stream-predict: builds for the Qwen3-8B c4 gate pair (node bench-regression-gate): A = f68d2a0 and B = 41dcedd2 from
# tarballs of `git archive` put at /root/ab_A.tar.gz and /root/ab_B.tar.gz, each release with CUDA in its own target dir
# (/root/tetraeder-A, /root/tetraeder-B), as scripts/ab_moe_box.sh builds them. Also the python env of the bench. Run it during
# the download, never beside a timed run. Ends with SP_ABBUILD_DONE in runs/vast/sp_abbuild.log.
exec 8> /root/tetraeder_sp_abbuild.lock
flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
root=$PWD
name="${RESULTS_NAME:?set RESULTS_NAME}"
out="$root/bench/results/$name/ab"
mkdir -p "$out" runs/vast
say() { echo "[$(date +%T)] $*"; }
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:/root/.local/bin:$PATH"
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
[ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
quota=$(bash scripts/cpu_quota.sh 2> /dev/null || nproc)
jobs=$((quota / 4))
[ "$jobs" -lt 4 ] && jobs=4
[ "$jobs" -gt 16 ] && jobs=16
py=/root/gwvenv/bin/python

build_one() { # build_one TAG: unpack /root/ab_TAG.tar.gz to /root/tetraeder-TAG, patch queue, release build with CUDA
  local d=/root/tetraeder-$1
  if [ ! -d "$d/crates" ]; then
    [ -s "/root/ab_$1.tar.gz" ] || { say "no /root/ab_$1.tar.gz"; exit 1; }
    mkdir -p "$d" && tar -xzf "/root/ab_$1.tar.gz" -C "$d" || exit 1
  fi
  cd "$d" || exit 1
  bash scripts/ensure_patches.sh 8>&- || exit 1
  # shellcheck source=scripts/box_libs.sh
  . scripts/box_libs.sh
  CARGO_BUILD_JOBS=$jobs timeout -k 30 50m cargo build --release -p tetraeder --features cuda 8>&-
}

main() {
  say "sp_abbuild start (cargo jobs $jobs each)"
  if ! "$py" -c 'import httpx, tomllib' 2> /dev/null; then
    command -v uv > /dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/root/.local/bin sh > /dev/null 2>&1; }
    uv venv --allow-existing /root/gwvenv && uv pip install --python "$py" httpx pytest pyyaml || { say "python env failed"; echo "SP_ABBUILD_DONE rc=1"; exit 1; }
  fi
  (build_one A > "$out/build-A.log" 2>&1) &
  local pa=$!
  (build_one B > "$out/build-B.log" 2>&1) &
  local pb=$!
  wait "$pa" || { say "build A failed (see $out/build-A.log)"; echo "SP_ABBUILD_DONE rc=1"; exit 1; }
  wait "$pb" || { say "build B failed (see $out/build-B.log)"; echo "SP_ABBUILD_DONE rc=1"; exit 1; }
  for t in A B; do
    [ -x "/root/tetraeder-$t/target/release/tetraeder" ] || { say "no binary $t"; echo "SP_ABBUILD_DONE rc=1"; exit 1; }
  done
  say "SP_ABBUILD_DONE rc=0"
}
main >> runs/vast/sp_abbuild.log 2>&1
