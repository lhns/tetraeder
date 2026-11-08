#!/usr/bin/env bash
# KTransformers baseline for the Phase 2 exit run: Qwen3-235B-A22B Q4_K_M (split GGUF) with GPU attention (sglang-kt) and
# CPU experts read in place from the GGUF by kt-kernel (llamafile backend, np.memmap), greedy decode.
#   scripts/ktransformers_exit.sh <model-first-shard> <threads> <ram-limit-bytes> <out-json> [deep|short]
# deep (default): prefill about 32768 tokens, then 64 decoded. short: 2048 and 32, the llama-bench `-p 2048 -n 32` size
# (the decode then starts at depth 2048). KT_PREFILL / KT_DECODE override the sizes. Needs scripts/ktransformers_build.sh
# first. Writes the JSON even on failure (ok=false) and exits non-zero then. The RAM limit is only recorded: the
# container enforces it and the mmap'd experts are page cache that can be evicted, so a model above the limit runs
# slower instead of failing. Time limit: 40 min to load, 50 min for the request.
# The sglang-kt GPU side needs HF-format weights for everything except the experts, so the first run writes a stub
# checkpoint with the non-expert tensors dequantized from the GGUF to bf16 (about 16 GB, no experts). It does not fit
# the 24 GB card with a bf16 KV cache at 32K, so the KV cache is fp8 (KT_KV_DTYPE=auto to change).
# Same prompt text as scripts/exit_run.sh (/root/exit_prompt.txt), repeated and cut to the token count with the model's
# tokenizer.
# Host sampling (scripts/engine_sample.py, same columns as the builder in exit_run.sh): disk reads, major faults, CPU,
# GPU, RSS and page cache, split into prefill and decode, written under "host" in the result JSON next to
# kt_cpu_variant (the variant kt-kernel picked). KT_KERNEL_CPU_VARIANT=avx2 forces the AVX2 kernels (kt-kernel's
# _cpu_detect.py reads it); HOST_DISK_DEV (MAJ:MIN) skips the disk probe.
exec 8> /root/tetraeder_ktransformers_exit.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 4 ] || { echo "usage: $0 <model-first-shard> <threads> <ram-limit-bytes> <out-json> [deep|short]" >&2; exit 2; }
model=$(realpath "$1" 2> /dev/null || echo "$1") threads=$2 ram_bytes=$3 out_json=$4 profile=${5:-deep}
case "$profile" in
  deep) prefill="${KT_PREFILL:-32768}" decode="${KT_DECODE:-64}" ;;
  short) prefill="${KT_PREFILL:-2048}" decode="${KT_DECODE:-32}" ;;
  *) echo "profile must be deep or short" >&2; exit 2 ;;
esac
venv="${KT_VENV:-$(cat runs/vast/ktransformers_venv.path 2> /dev/null || echo /opt/ktransformers-venv)}"
py="$venv/bin/python"
cfg=$(realpath .hf_cache/Qwen__Qwen3-235B-A22B-config 2> /dev/null)
[ -x "$py" ] || { echo "missing $py: run scripts/ktransformers_build.sh" >&2; exit 3; }
[ -s "$cfg/config.json" ] || { echo "missing $cfg: run scripts/ktransformers_build.sh" >&2; exit 3; }
[ -f "$model" ] || { echo "missing model $model" >&2; exit 3; }
mkdir -p runs/vast "$(dirname "$out_json")"
log=runs/vast/ktransformers_exit.log
work="${KT_WORK:-/root/ktransformers_work}"
mkdir -p "$work"
stub="$work/qwen3-235b-a22b-attn-bf16"
port="${KT_PORT:-30011}"
# shellcheck disable=SC1091
[ -f "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh" ] && . "${TAXAMETER_HOME:-/root/.taxameter}/cuda_env.sh"
export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# flashinfer JIT builds for this card only
cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | tr -d ' ' | grep -E '^[0-9]+\.[0-9]+$' | sort -un | head -1)
export TORCH_CUDA_ARCH_LIST="${cap:-8.6}"
version=$("$py" -I -c "import importlib.metadata as m; print('kt-kernel', m.version('kt-kernel'), 'sglang-kt', m.version('sglang-kt'))" 2> /dev/null)
commit="v0.7.2 a02efe36b5802e7ca79a6b29146e01cf3b97d305 ($version)"

srv_log="$work/server-$profile.log"
res="$work/result-$profile.json"
rss_file="$work/peak_rss_kb-$profile"
samples="$work/samples-$profile.jsonl"
summary="$work/host-$profile.json"
variant_file="$work/cpu_variant-$profile"
cmd_file="$work/command-$profile.txt"
rm -f "$res" "$rss_file" "$samples" "$summary" "$variant_file"
: > "$srv_log"
rc=0
{
  echo "[$(date +%T)] ktransformers $profile: prefill $prefill, decode $decode, threads $threads, ram $ram_bytes B, commit $commit"

  # stub checkpoint: non-expert tensors from the GGUF as bf16 safetensors with HF names, plus the HF config and tokenizer
  if [ ! -f "$stub/.done" ]; then
    cat > "$work/make_stub.py" << 'EOF'
import glob, json, os, re, shutil, sys
import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize
from safetensors.torch import save_file

first, cfg, out = sys.argv[1:4]
m = re.match(r"(.*)-\d+-of-(\d+)\.gguf$", os.path.basename(first))
shards = sorted(glob.glob(os.path.join(os.path.dirname(first), f"{m[1]}-*-of-{m[2]}.gguf"))) if m else [first]
top = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight", "output.weight": "lm_head.weight"}
sub = {
    "attn_norm.weight": "input_layernorm.weight", "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
}
def hf(name):
    if name in top:
        return top[name]
    b = re.match(r"blk\.(\d+)\.(.*)$", name)
    return f"model.layers.{b[1]}.{sub[b[2]]}" if b and b[2] in sub else None

tmp = out + ".tmp"
shutil.rmtree(tmp, ignore_errors=True)
os.makedirs(tmp)
weight_map, cur, cur_bytes, total, n_files = {}, {}, 0, 0, 0
def flush():
    global cur, cur_bytes, n_files
    if not cur:
        return
    fn = f"model-{n_files:05d}.safetensors"
    save_file(cur, os.path.join(tmp, fn), metadata={"format": "pt"})
    weight_map.update({k: fn for k in cur})
    n_files += 1
    cur, cur_bytes = {}, 0
skipped = []
for s in shards:
    for t in GGUFReader(s).tensors:
        name = hf(t.name)
        if name is None:
            if "_exps" not in t.name:
                skipped.append(t.name)
            continue
        a = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape([int(x) for x in t.shape[::-1]])
        w = torch.from_numpy(a).to(torch.bfloat16).contiguous()
        cur[name] = w
        cur_bytes += w.numel() * 2
        total += w.numel() * 2
        if cur_bytes >= 4 << 30:
            flush()
    print("stub: done", s, "tensors", len(weight_map) + len(cur), flush=True)
flush()
print("stub: skipped non-expert tensors:", skipped)
json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, open(os.path.join(tmp, "model.safetensors.index.json"), "w"))
for f in os.listdir(cfg):
    shutil.copy(os.path.join(cfg, f), tmp)
open(os.path.join(tmp, ".done"), "w").write("ok\n")
shutil.rmtree(out, ignore_errors=True)
os.rename(tmp, out)
print("stub: wrote", out, total / 1e9, "GB")
EOF
    echo "[$(date +%T)] writing stub checkpoint $stub"
    timeout -k 10 2400 "$py" -I "$work/make_stub.py" "$model" "$cfg" "$stub" || rc=40
  fi

  base=/root/exit_prompt.txt
  [ -f "$base" ] || cat AGENTS.md docs/adr/*.md > "$base"

  cat > "$work/client.py" << 'EOF'
import json, sys, time
import requests
from transformers import AutoTokenizer

cfg, base, url, prefill, decode, out, tmo = sys.argv[1:8]
prefill, decode, tmo = int(prefill), int(decode), float(tmo)
tok = AutoTokenizer.from_pretrained(cfg)
text = open(base, errors="replace").read()
ids = tok.encode(text, add_special_tokens=False)
ids = tok.encode(text * (prefill // max(len(ids), 1) + 1), add_special_tokens=False)[:prefill]
print("prompt tokens", len(ids), flush=True)

def gen(ids, n, stream):
    body = {"input_ids": ids, "stream": stream,
            "sampling_params": {"temperature": 0, "max_new_tokens": n, "ignore_eos": True}}
    t0 = time.time()
    first = end = meta = None
    with requests.post(url + "/generate", json=body, stream=stream, timeout=tmo) as r:
        r.raise_for_status()
        if not stream:
            return r.json()["meta_info"], t0, None, time.time()
        for line in r.iter_lines():
            if not line.startswith(b"data:"):
                continue
            s = line[5:].strip()
            if s == b"[DONE]":
                break
            d = json.loads(s)
            meta = d["meta_info"]
            end = time.time()
            if first is None and meta.get("completion_tokens", 0) >= 1:
                first = end
    return meta, t0, first, end

gen(ids[:16], 2, False)  # warm up kernels, graphs and JIT on a tiny request
meta, t0, first, end = gen(ids, decode, True)
n = meta["completion_tokens"]
res = {"prompt_tokens": meta.get("prompt_tokens"), "decode_tokens": n, "ttft_s": first - t0, "total_s": end - t0,
       "prefill_tok_s": meta.get("prompt_tokens", len(ids)) / (first - t0),
       "decode_tok_s": (n - 1) / (end - first) if n > 1 and end > first else None, "cached_tokens": meta.get("cached_tokens"),
       "t_start": t0, "t_first": first, "t_end": end}
json.dump(res, open(out, "w"))
print(json.dumps(res), flush=True)
EOF

  if [ $rc = 0 ]; then
    # --kt-cpuinfer: CPU threads of the expert kernels. One thread pool (no NUMA split of the mmap'd weights).
    # --kt-num-gpu-experts 0: all experts stay on the CPU. No deferred experts (they change the numerics).
    # temperature 0 per request is greedy; ignore_eos keeps the decode count exact; no radix cache, so nothing is reused.
    # --watchdog-timeout: the default 300 s would kill a slow CPU prefill chunk.
    # OMP/MKL capped so side threads do not eat the container's CPU quota.
    srv=("$py" -m sglang.launch_server --model-path "$stub" --host 127.0.0.1 --port "$port" --trust-remote-code
      --kt-method LLAMAFILE --kt-weight-path "$(dirname "$model")" --kt-cpuinfer "$threads" --kt-threadpool-count 1
      --kt-num-gpu-experts 0 --kt-max-deferred-experts-per-token 0
      --attention-backend "${KT_ATTN:-flashinfer}" --kv-cache-dtype "${KT_KV_DTYPE:-fp8_e5m2}"
      --mem-fraction-static "${KT_MEM_FRACTION:-0.88}" --chunked-prefill-size "${KT_CHUNK:-4096}"
      --context-length $((prefill + decode + 256)) --max-total-tokens $((prefill + decode + 256))
      --max-running-requests 1 --cuda-graph-max-bs 1 --tensor-parallel-size 1 --disable-radix-cache
      --disable-shared-experts-fusion --watchdog-timeout 7200 --served-model-name qwen3)
    echo "${srv[*]}" > "$cmd_file"
    echo "[$(date +%T)] server: ${srv[*]}"
    setsid env HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${srv[@]}" > "$srv_log" 2>&1 &
    pid=$!
    # peak RSS of the whole process group (scheduler holds the mmap'd experts), polled every 5 s
    (
      peak=0
      while kill -0 "$pid" 2> /dev/null; do
        cur=$(ps -o rss= -g "$pid" 2> /dev/null | awk '{ t += $1 } END { print t + 0 }')
        [ "$cur" -gt "$peak" ] && { peak=$cur; echo "$peak" > "$rss_file"; }
        sleep 5
      done
    ) &
    mon=$!
    up=0
    for _ in $(seq 240); do
      curl -fs "http://127.0.0.1:$port/health" > /dev/null 2>&1 && { up=1; break; }
      kill -0 "$pid" 2> /dev/null || break
      sleep 10
    done
    if [ $up = 1 ]; then
      echo "[$(date +%T)] server up, request"
    # the variant kt-kernel picks on this CPU (and with KT_KERNEL_CPU_VARIANT, if set)
    "$py" -I -c "import kt_kernel; print(kt_kernel.__cpu_variant__)" > "$variant_file" 2>> "$log"
    echo "[$(date +%T)] kt_kernel cpu variant: $(cat "$variant_file") (KT_KERNEL_CPU_VARIANT=${KT_KERNEL_CPU_VARIANT:-unset})"
    disk="${HOST_DISK_DEV:-$(python3 scripts/engine_sample.py probe-disk "$model")}"
    echo "[$(date +%T)] sampling disk $disk"
    python3 scripts/engine_sample.py sample --pid "$pid" --disk "$disk" --out "$samples" 8>&- &
    spid=$!
      timeout -k 30 50m "$py" -I "$work/client.py" "$cfg" "$base" "http://127.0.0.1:$port" "$prefill" "$decode" "$res" 3000
      rc=$?
      echo "[$(date +%T)] client rc=$rc"
      kill "$spid" 2> /dev/null
      python3 scripts/engine_sample.py summarize --samples "$samples" --threads "$threads" --kt-json "$res" --out "$summary" \
        || echo "host summary failed"
    else
      echo "[$(date +%T)] server did not come up"
      rc=43
    fi
    kill -TERM -- "-$pid" 2> /dev/null
    sleep 15
    kill -KILL -- "-$pid" 2> /dev/null
    kill "$mon" "${spid:-}" 2> /dev/null
    wait 2> /dev/null
  fi
} >> "$log" 2>&1

# JSON from the client result, the server log tail and the polled RSS
python3 -I - "$out_json" "$res" "$srv_log" "$rss_file" "$cmd_file" "$rc" "$profile" "$commit" "$threads" "$ram_bytes" "$model" "$prefill" "$decode" "$summary" "$variant_file" "${KT_KERNEL_CPU_VARIANT:-}" << 'EOF'
import json, os, subprocess, sys
out, res, srv, rss, cmd, rc, profile, commit, threads, ram, model, want_pre, want_dec, summary, variant, variant_env = sys.argv[1:17]
rc = int(rc)
def read(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""
r = {}
try:
    r = json.loads(read(res))
except ValueError:
    pass
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
except OSError:
    gpu = None
rss_kb = read(rss).strip()
try:
    host = json.loads(read(summary))
except ValueError:
    host = {}
d = {
    "engine": "ktransformers",
    "engine_arch": "kt-kernel LLAMAFILE (GGUF experts in place, CPU) + sglang-kt (GPU attention, bf16 from GGUF, fp8 KV)",
    "commit": commit,
    "profile": profile,
    "model": os.path.basename(model),
    "command": read(cmd).strip(),
    "threads": int(threads),
    "ram_budget_bytes": int(ram),
    "prompt_tokens_wanted": int(want_pre),
    "prompt_tokens": r.get("prompt_tokens"),
    "decode_tokens": r.get("decode_tokens"),
    "decode_depth": r.get("prompt_tokens"),
    "ttft_s": r.get("ttft_s"),
    "total_s": r.get("total_s"),
    "prefill_tok_s": r.get("prefill_tok_s"),
    "decode_tok_s": r.get("decode_tok_s"),
    "cached_tokens": r.get("cached_tokens"),
    "peak_rss_gb": int(rss_kb) / 1e6 if rss_kb.isdigit() else None,
    "kt_cpu_variant": read(variant).strip() or None,
    "kt_cpu_variant_forced": variant_env or None,
    "host": host.get("host"),
    "env": host.get("env"),
    "engine_rc": rc,
    "gpu": gpu,
    "raw_tail": read(srv).splitlines()[-40:],
}
# a missing speed number is a failed baseline, not a slow one
d["ok"] = rc == 0 and d["prefill_tok_s"] is not None and d["decode_tok_s"] is not None
json.dump(d, open(out, "w"), indent=1)
print(json.dumps({k: d[k] for k in ("ok", "prompt_tokens", "decode_tokens", "prefill_tok_s", "decode_tok_s", "engine_rc")}))
sys.exit(0 if d["ok"] else 1)
EOF
r=$?
echo "KTRANSFORMERS_EXIT_DONE profile=$profile rc=$r" >> "$log"
exit $r
