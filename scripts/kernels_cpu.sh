#!/usr/bin/env bash
# Stage kernels-cpu (ADR 0045, "Diagnosing the KTransformers decode gap"): CPU-only decode of the Q4_K kernels on
# Qwen3-30B-A3B Q4_K_M, same threads, no GPU: mainline llama.cpp (the pin), ik_llama.cpp (iqk kernels) and the mainline
# baseline build (the newer commit). All three are portable AVX2 builds (docker/box/Dockerfile: GGML_NATIVE=OFF), so a
# gap between ik and mainline is the kernel, not the ISA. Records the CPU flags, each binary's PIN and how many AVX-512
# (zmm) instructions it contains. Result: bench/results/<date>/kernels-cpu/kernels-cpu-<stamp>.json.
#   scripts/kernels_cpu.sh            # started detached by `taxameter run`; ends with KERNELS_CPU_DONE rc=N
# Env: THREADS (default: the CPU quota), REPEATS (3), DECODE (64).
exec 8> /root/tetraeder_kernels_cpu.lock; flock -n 8 || exit 0
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs/vast
log=runs/vast/kernels_cpu.log
out_dir="$PWD/bench/results/${RESULTS_DATE:-$(date +%Y%m%d)}/kernels-cpu"
mkdir -p "$out_dir"
stamp=$(date +%H%M%S)
threads="${THREADS:-$(bash scripts/cpu_quota.sh)}"
reps="${REPEATS:-3}"
decode="${DECODE:-64}"
models_dir="$PWD/${TETRAEDER_MODELS_DIR:-.hf_cache}"
gguf="$models_dir/Qwen__Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
rc=0
{
  echo "[$(date +%T)] kernels-cpu: threads $threads, repeats $reps, decode $decode"
  if [ ! -s "$gguf" ]; then
    TAXAMETER_DIR="$PWD" bash "${TAXAMETER_HOME:-/root/.taxameter}/box.sh" models \
      "Qwen/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf" 8>&-
  fi
  [ -s "$gguf" ] || { echo "no model"; rc=20; }
  declare -A bins=(
    [mainline]=/opt/llama.cpp/bin/llama-bench
    [ik]=/opt/ik_llama/bin/llama-bench
    [baseline]=/opt/llama.cpp-baseline/bin/llama-bench
  )
  declare -A pins=([mainline]=/opt/llama.cpp/PIN [ik]=/opt/ik_llama/PIN [baseline]=/opt/llama.cpp-baseline/PIN)
  # page cache warm before the interleaved loop, so the first binary does not pay the disk read
  [ $rc = 0 ] && cat "$gguf" > /dev/null
  # a bench run per binary, one repeat per call and interleaved, so box drift hits all three alike
  if [ $rc = 0 ]; then
    for r in $(seq "$reps"); do
      for name in mainline ik baseline; do
        bin=${bins[$name]}
        [ -x "$bin" ] || { echo "skip $name: $bin missing"; continue; }
        echo "[$(date +%T)] $name repeat $r"
        CUDA_VISIBLE_DEVICES='' GGML_VK_VISIBLE_DEVICES='' timeout -k 30 15m "$bin" -m "$gguf" -ngl 0 -fa 1 -t "$threads" \
          -p 0 -n "$decode" -r 1 -o json > "$out_dir/raw-$name-$r-$stamp.json" 2> "$out_dir/raw-$name-$r-$stamp.err" \
          || echo "$name repeat $r rc=$?"
      done
    done
  fi
  # AVX-512 code in each binary (objdump is slow on the CUDA builds: capped)
  for name in mainline ik baseline; do
    bin=${bins[$name]}
    [ -x "$bin" ] || continue
    # the count is only valid when objdump finished: a timeout (rc 124) or failure records the rc, not a count
    res=$(timeout 180 objdump -d --no-show-raw-insn "$bin" 2> /dev/null | grep -c zmm; echo "${PIPESTATUS[0]}")
    z=${res%%$'\n'*} orc=${res##*$'\n'}
    [ "$orc" = 0 ] || z="objdump rc $orc"
    echo "$name pin: $(cat "${pins[$name]}" 2> /dev/null); zmm instructions: $z"
    echo "$z" > "$out_dir/zmm-$name-$stamp.txt"
  done
  python3 -I - "$out_dir" "$stamp" "$threads" "$decode" "$gguf" << 'EOF'
import glob, json, os, re, sys
d, stamp, threads, decode, gguf = sys.argv[1:6]
def read(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""
flags = sorted({w for ln in read("/proc/cpuinfo").splitlines() if ln.startswith("flags") for w in ln.split()
                if w.startswith(("avx", "fma", "f16c", "amx"))})
model = next((ln.split(":", 1)[1].strip() for ln in read("/proc/cpuinfo").splitlines() if ln.startswith("model name")), None)
pins = {"mainline": "/opt/llama.cpp/PIN", "ik": "/opt/ik_llama/PIN", "baseline": "/opt/llama.cpp-baseline/PIN"}
res = {"model": os.path.basename(gguf), "threads": int(threads), "decode_tokens": int(decode),
       "cpu_model": model, "cpu_flags": flags,
       "build": "portable AVX2 (GGML_NATIVE=OFF, docker/box/Dockerfile)", "engines": {}}
for name, pin in pins.items():
    tps = []
    for f in sorted(glob.glob(f"{d}/raw-{name}-*-{stamp}.json")):
        try:
            for row in json.loads(read(f)):
                if row.get("n_gen"):
                    tps.append(row["avg_ts"])
        except ValueError:
            pass
    if not tps and not os.path.exists(pin):
        continue
    z = read(f"{d}/zmm-{name}-{stamp}.txt").strip()
    tps_sorted = sorted(tps)
    res["engines"][name] = {"pin": read(pin).strip(), "zmm_instructions": int(z) if z.isdigit() else None,
                            "zmm_error": None if z.isdigit() else (z or "no count"),
                            "decode_tok_s": tps, "median_tok_s": tps_sorted[len(tps) // 2] if tps else None}
json.dump(res, open(f"{d}/kernels-cpu-{stamp}.json", "w"), indent=1)
print(json.dumps({k: v["median_tok_s"] for k, v in res["engines"].items()}))
EOF
} >> "$log" 2>&1
echo "KERNELS_CPU_DONE rc=$rc" >> "$log"
exit $rc
