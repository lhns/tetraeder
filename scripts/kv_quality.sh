#!/usr/bin/env bash
# KL acceptance of the low-bit KV formats (ADR 0050 b, crates/model/tests/kv_quality.rs) on Qwen3-30B-A3B Q4_K_M.
#   scripts/kv_quality.sh <out-json>
# Fetches the model if missing (TETRAEDER_KV_QUALITY_GGUF overrides the path). The test skips silently without a
# model, so a missing result JSON is a failure here (rc 30). Log next to the JSON. Called by exit_run.sh (stage
# kv-quality) and scripts/session.sh (b2-proxy). Env: KV_QUALITY_MIN (time limit, default 60). Fetches wikitext-2 for
# the calibration set of ADR 0050 b (TETRAEDER_KV_QUALITY_TEXT overrides); a result without that set fails (rc 31).
exec 7> /root/tetraeder_kv_quality.lock; flock -n 7 || exit 0
cd "$(dirname "$0")/.." || exit 1
bash scripts/ensure_patches.sh 8>&- || exit 1  # ggml patch queue (ADR 0052)
out=${1:?usage: $0 <out-json>}
if [ -d /opt/cargo ]; then export RUSTUP_HOME="${RUSTUP_HOME:-/opt/rustup}" CARGO_HOME="${CARGO_HOME:-/opt/cargo}"; fi
export PATH="/opt/cargo/bin:$HOME/.cargo/bin:$PATH" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
# this box's GPU only (the default arch list rebuilds ggml-cuda for four archs)
if [ -z "${TETRAEDER_CUDA_ARCHS:-}" ]; then
  archs=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' .' | grep -E '^[0-9]+$' | sort -un | paste -sd';')
  [ -n "$archs" ] && export TETRAEDER_CUDA_ARCHS="$archs"
fi
# shellcheck source=scripts/box_libs.sh
. scripts/box_libs.sh
command -v sccache > /dev/null 2>&1 && export RUSTC_WRAPPER=sccache
# shellcheck disable=SC1091
. "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
gguf="${TETRAEDER_KV_QUALITY_GGUF:-$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf}"
if [ ! -s "$gguf" ]; then
  TAXAMETER_DIR="$PWD" bash "${TAXAMETER_HOME:-/root/.taxameter}/box.sh" models \
    "Qwen/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf" 7>&-
fi
[ -s "$gguf" ] || { echo "kv-quality: no model at $gguf"; exit 20; }
text="${TETRAEDER_KV_QUALITY_TEXT:-/root/q/wikitext-2-raw/wiki.train.raw}"
if [ ! -s "$text" ]; then
  mkdir -p /root/q
  (cd /root/q && curl -fsSL -o wt.zip https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip \
    && python3 -c "import zipfile; zipfile.ZipFile('wt.zip').extractall('.')") || { echo "kv-quality: wikitext fetch failed"; exit 21; }
fi
mkdir -p "$(dirname "$out")"
rm -f "$out"
TETRAEDER_KV_QUALITY_GGUF="$gguf" TETRAEDER_KV_QUALITY_OUT="$out" TETRAEDER_KV_QUALITY_TEXT="$text" \
  timeout -k 30 "${KV_QUALITY_MIN:-60}m" cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda \
  --test kv_quality -- --ignored --nocapture > "${out%.json}.log" 2>&1 7>&-
rc=$?
[ "$rc" = 0 ] && [ ! -s "$out" ] && { echo "kv-quality: test passed but wrote no $out (skipped?)"; rc=30; }
[ -s "$out" ] && ! grep -q '"calibration_512"' "$out" && { echo "kv-quality: no calibration_512 set in $out"; rc=31; }
echo "kv-quality rc=$rc"
exit "$rc"
