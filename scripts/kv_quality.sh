#!/usr/bin/env bash
# KL acceptance of the low-bit KV formats (ADR 0050 b, crates/model/tests/kv_quality.rs) on Qwen3-30B-A3B Q4_K_M.
#   scripts/kv_quality.sh <out-json>
# Fetches the model if missing (TETRAEDER_KV_QUALITY_GGUF overrides the path). The test skips silently without a
# model, so a missing result JSON is a failure here (rc 30). Log next to the JSON. Called by exit_run.sh (stage
# kv-quality) and scripts/session.sh (b2-proxy). Env: KV_QUALITY_MIN (time limit, default 60).
exec 7> /root/tetraeder_kv_quality.lock; flock -n 7 || exit 0
cd "$(dirname "$0")/.." || exit 1
out=${1:?usage: $0 <out-json>}
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
gguf="${TETRAEDER_KV_QUALITY_GGUF:-$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf}"
if [ ! -s "$gguf" ]; then
  TAXAMETER_DIR="$PWD" bash "${TAXAMETER_HOME:-/root/.taxameter}/box.sh" models \
    "Qwen/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf" 7>&-
fi
[ -s "$gguf" ] || { echo "kv-quality: no model at $gguf"; exit 20; }
mkdir -p "$(dirname "$out")"
rm -f "$out"
TETRAEDER_KV_QUALITY_GGUF="$gguf" TETRAEDER_KV_QUALITY_OUT="$out" \
  timeout -k 30 "${KV_QUALITY_MIN:-60}m" cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda \
  --test kv_quality -- --ignored --nocapture > "${out%.json}.log" 2>&1 7>&-
rc=$?
[ "$rc" = 0 ] && [ ! -s "$out" ] && { echo "kv-quality: test passed but wrote no $out (skipped?)"; rc=30; }
echo "kv-quality rc=$rc"
exit "$rc"
