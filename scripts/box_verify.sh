#!/usr/bin/env bash
# taxameter verify hook, runs ON the box from the repo dir: cargo works and cmake exists.
set -uo pipefail
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")/.." || exit 1
if cargo --version > /dev/null 2>&1 && command -v cmake > /dev/null 2>&1; then
  echo "verify_tools=1"
else
  echo "verify_tools=0"
  exit 1
fi
