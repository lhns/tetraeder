#!/usr/bin/env bash
# Local cargo for parallel agents and the 16 GB dev VM: bash scripts/dev-cargo.sh <cargo args>
#  - at most DEV_CARGO_SLOTS (default 2) cargo jobs run at once on this machine, each with CARGO_BUILD_JOBS
#    (default 4); a job waits for a free slot. Replaces the single shared lock.
#  - sccache (if installed) caches rustc and, with generators that support launchers (Ninja, Makefiles), the
#    ggml C/C++/CUDA objects, shared by all worktrees. Without sccache this is a no-op.
#  - each worktree keeps its own target/ (nothing here sets CARGO_TARGET_DIR).
# Slots are directories under DEV_CARGO_LOCK_DIR holding the owner's pid; a dead owner's slot is reclaimed.
set -u
slots="${DEV_CARGO_SLOTS:-2}"
dir="${DEV_CARGO_LOCK_DIR:-$HOME/.tetraeder-cargo-slots}"
mkdir -p "$dir"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"

if command -v sccache > /dev/null 2>&1; then
  export RUSTC_WRAPPER="${RUSTC_WRAPPER:-sccache}"
  export CMAKE_C_COMPILER_LAUNCHER="${CMAKE_C_COMPILER_LAUNCHER:-sccache}"
  export CMAKE_CXX_COMPILER_LAUNCHER="${CMAKE_CXX_COMPILER_LAUNCHER:-sccache}"
  export CMAKE_CUDA_COMPILER_LAUNCHER="${CMAKE_CUDA_COMPILER_LAUNCHER:-sccache}"
fi

slot=""
acquire() {
  local i pid
  for i in $(seq 1 "$slots"); do
    if mkdir "$dir/$i" 2> /dev/null; then
      echo $$ > "$dir/$i/pid"
      slot="$dir/$i"
      return 0
    fi
    pid=$(cat "$dir/$i/pid" 2> /dev/null)
    if [ -n "$pid" ] && ! kill -0 "$pid" 2> /dev/null; then
      rm -rf "$dir/$i" # owner died without releasing
    fi
  done
  return 1
}
release() { [ -n "$slot" ] && rm -rf "$slot"; slot=""; }
trap 'release' EXIT
trap 'release; exit 130' INT TERM

until acquire; do sleep 10; done
cargo "$@"
