#!/usr/bin/env bash
# taxameter provision hook, runs ON the box from the repo dir (/root/tetraeder).
# Installs only what is missing (the GHCR image has it all): C toolchain, curl, git, cmake, ninja, sccache,
# python3 (taxameter's cuda_env.sh probes libcuda through ctypes), rustup and the channel from rust-toolchain.toml.
set -uo pipefail
# the GHCR image keeps the toolchain in /opt/cargo; a rustup under $HOME is the fallback
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.." || exit 1

sudo="" need=""
[ "$(id -u)" -eq 0 ] || sudo="sudo"
command -v cc > /dev/null 2>&1 || need="$need build-essential"
command -v curl > /dev/null 2>&1 || need="$need curl ca-certificates"
command -v git > /dev/null 2>&1 || need="$need git"
command -v cmake > /dev/null 2>&1 || need="$need cmake"
command -v ninja > /dev/null 2>&1 || need="$need ninja-build"
command -v sccache > /dev/null 2>&1 || need="$need sccache"
command -v python3 > /dev/null 2>&1 || need="$need python3"
if [ -n "$need" ]; then
  echo "installing missing packages:$need"
  # sccache is optional: retry without it if apt does not know it
  { DEBIAN_FRONTEND=noninteractive $sudo apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive $sudo apt-get install -y -qq --no-install-recommends $need; } \
    || { need=${need// sccache/}; DEBIAN_FRONTEND=noninteractive $sudo apt-get install -y -qq --no-install-recommends $need; } \
    || exit 1
fi
if ! command -v rustup > /dev/null 2>&1; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none --profile minimal || exit 1
  export PATH="$HOME/.cargo/bin:$PATH"
fi
# the channel comes from the repo's rust-toolchain.toml; rustup does nothing when it is already installed
ch=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml 2>/dev/null | head -1)
rustup toolchain install "${ch:-stable}" --profile minimal -c rustfmt -c clippy || exit 1
command -v sccache > /dev/null 2>&1 || echo "note: sccache missing, builds will not be cached"
echo "deps ok: $(cargo --version) $(cmake --version | head -1) ninja $(ninja --version) sccache $(command -v sccache > /dev/null 2>&1 && echo yes || echo no)"
