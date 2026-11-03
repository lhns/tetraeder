#!/usr/bin/env bash
# Installs the KTransformers baseline for the exit run (github.com/kvcache-ai/ktransformers): kt-kernel (CPU experts,
# llamafile backend reads GGUF in place) plus the kvcache-ai SGLang fork (GPU attention). Prebuilt wheels, no source
# build: kt-kernel 0.7.2 (cp312, static CUDA runtime, sm_80/86/89/90) and sglang-kt 0.7.2 (tag v0.7.2). The classic
# local_chat path has no Qwen3Moe rule and its Qwen3 path needs a source build of balance_serve, so it is not used.
# Also fetches the HF config and tokenizer files (no weights) of Qwen/Qwen3-235B-A22B. Idempotent (skips when the venv was
# built for the same versions and arch). Started detached by `taxameter run scripts/ktransformers_build.sh` or called from
# another script; ends with KTRANSFORMERS_BUILD_DONE rc=N in runs/vast/ktransformers_build.log.
exec 8> /root/tetraeder_ktransformers_build.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast .hf_cache
log=runs/vast/ktransformers_build.log
version="${KT_VERSION:-0.7.2}" # tag v0.7.2 = a02efe36b5802e7ca79a6b29146e01cf3b97d305
venv="${KT_VENV:-/opt/ktransformers-venv}"
cfg_rev=8efa61729e24bd65b1d152b5ab5409052aa80e65 # Qwen/Qwen3-235B-A22B
cfg_dir=.hf_cache/Qwen__Qwen3-235B-A22B-config
export PATH="$HOME/.local/bin:$PATH"

build() {
  echo "[$(date +%T)] ktransformers build: kt-kernel/sglang-kt $version, venv $venv"
  cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' ' | grep -E '^[0-9]+\.[0-9]+$' | sort -un | head -1)
  [ -n "$cap" ] || { echo "no GPU compute capability from nvidia-smi"; return 30; }
  case "$cap" in 8.0 | 8.6 | 8.9 | 9.0) ;; *) echo "compute capability $cap is not in the kt-kernel wheel (8.0 8.6 8.9 9.0)"; return 31 ;; esac
  stamp="kt-kernel $version sglang-kt $version ktransformers $version cap $cap"
  mkdir -p "$venv" 2> /dev/null || { venv="$HOME/ktransformers-venv"; mkdir -p "$venv"; }
  echo "$venv" > runs/vast/ktransformers_venv.path
  if [ "$(cat "$venv/.built" 2> /dev/null)" = "$stamp" ] && [ -x "$venv/bin/python" ]; then
    echo "already built: $stamp"
  else
    command -v uv > /dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh || return 32; }
    rm -f "$venv/.built"
    # torch 2.9.1 (the PyPI linux wheel is cu128) comes in as a pinned dependency of both packages
    timeout -k 10 300 uv venv --python 3.12 --seed --allow-existing "$venv" || return 33
    # The ktransformers wheel (depends on kt-kernel only) carries the last part of the sgl_kernel CUDA payload: kt-kernel
    # holds the python files, sglang-kt, transformers-kt and accelerate-kt hold other parts. Without it sgl_kernel finds
    # no common_ops (the sm100 object, which also runs on sm_86) and sglang does not start.
    UV_HTTP_TIMEOUT=300 timeout -k 30 1800 uv pip install --python "$venv/bin/python" \
      "kt-kernel==$version" "sglang-kt==$version" "ktransformers==$version" "huggingface_hub" || return 34
    # llamafile (GGUF) support must be compiled into the wheel
    "$venv/bin/python" -I -c "
import kt_kernel
from kt_kernel.utils import llamafile
print('kt_kernel cpu variant:', kt_kernel.__cpu_variant__, 'llamafile:', llamafile._HAS_LLAMAFILE_SUPPORT)
assert llamafile._HAS_LLAMAFILE_SUPPORT
import sglang, torch
# materializes the sgl_kernel payload (about 4 GB written under the venv) and loads common_ops for this GPU
from sgl_kernel import load_utils
print('sgl_kernel common_ops:', load_utils._load_architecture_specific_ops().__file__)
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
assert torch.cuda.is_available()" || return 35
    echo "$stamp" > "$venv/.built"
  fi
  # config and tokenizer only, from a pinned revision
  mkdir -p "$cfg_dir"
  for f in config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    [ -s "$cfg_dir/$f" ] || curl -fsSL --retry 3 -o "$cfg_dir/$f" "https://huggingface.co/Qwen/Qwen3-235B-A22B/resolve/$cfg_rev/$f" || return 36
  done
  echo "built $venv ($stamp), config in $cfg_dir"
}

rc=0
build >> "$log" 2>&1 || rc=$?
echo "KTRANSFORMERS_BUILD_DONE rc=$rc" >> "$log"
exit $rc
