#!/usr/bin/env bash
# Builds the Colibri baseline engine (github.com/JustVugg/colibri, pure C, expert streaming) for the exit run: its
# qwen36 engine reads Qwen3 MoE GGUFs (qwen3moe) in place. Pinned to the v2.1.0 commit; CUDA for this box's GPU only.
# Idempotent (skips when the binary was built from the same commit and arch). Started detached by
# `taxameter run scripts/colibri_build.sh` or called from another script; ends with COLIBRI_BUILD_DONE rc=N in
# runs/vast/colibri_build.log.
exec 8> /root/tetraeder_colibri_build.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
log=runs/vast/colibri_build.log
commit="${COLIBRI_COMMIT:-704a3f68184b78367a04cb5526e428dc2f90f05b}" # v2.1.0
src="${COLIBRI_SRC:-/opt/colibri-src}"
# jobs capped like other side builds (AGENTS.md CPU quota note); the exit run is not running during the build
jobs="${COLIBRI_BUILD_JOBS:-4}"
# shellcheck disable=SC1091
[ -f "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh" ] && . "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

build() {
  echo "[$(date +%T)] colibri build: commit $commit, src $src, jobs $jobs"
  # sm_86 from compute_cap 8.6: one arch, as scripts/exit_run.sh does for ggml-cuda
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | head -1)
  [ -n "$cap" ] || { echo "no GPU compute capability from nvidia-smi"; return 30; }
  arch="sm_$cap"
  command -v nvcc > /dev/null 2>&1 || { echo "nvcc not found (CUDA_HOME=$CUDA_HOME)"; return 31; }
  stamp="$commit $arch"
  if [ "$(cat "$src/c/.built" 2> /dev/null)" = "$stamp" ] && [ -x "$src/c/qwen36" ]; then
    echo "already built: $stamp"
    return 0
  fi
  if [ ! -d "$src/.git" ]; then
    mkdir -p "$src" 2> /dev/null || { src="$HOME/colibri-src"; mkdir -p "$src"; }
    git -C "$src" init -q || return 32
    git -C "$src" remote add origin https://github.com/JustVugg/colibri.git 2> /dev/null
  fi
  # fetch the one commit, not the history (the repo is large)
  timeout -k 10 600 git -C "$src" fetch -q --depth 1 origin "$commit" || return 33
  git -C "$src" checkout -q -f "$commit" || return 34
  [ "$(git -C "$src" rev-parse HEAD)" = "$commit" ] || return 35
  rm -f "$src/c/.built"
  # qwen36 only: the engine for Qwen3 MoE. Native CPU flags (AVX2/VNNI of this box), CUDA backend with the tier.
  timeout -k 30 1500 make -C "$src/c" -j"$jobs" qwen36 CUDA=1 CUDA_ARCH="$arch" ARCH=native || return 36
  # a build without the CUDA runtime linked would silently run CPU-only
  ldd "$src/c/qwen36" | grep -q cudart || { echo "qwen36 is not linked to cudart"; return 37; }
  echo "$stamp" > "$src/c/.built"
  echo "built $src/c/qwen36 ($stamp)"
}

rc=0
build >> "$log" 2>&1 || rc=$?
echo "COLIBRI_BUILD_DONE rc=$rc" >> "$log"
exit $rc
