#!/usr/bin/env bash
# How many layers' experts llama.cpp can keep on the GPU at the exit run's context and ubatch: the tuned `-ot` baseline
# (plan.md, Phase 2 exit). Starts llama-server with experts of layers 0..N-1 on CUDA0 and the rest on the CPU, from
# FIT_MAX layers down, until the server comes up (KV, weights and compute buffers allocated, warmup decoded).
#   scripts/llama_fit.sh <model-first-shard> <threads> <ubatch> <ctx> <out-json>
# The JSON has the largest N that fit (layers), the -ot string for llama-bench (`;` joins overrides) and per try the
# result and the CUDA buffer sizes from the log. Called by exit_run.sh (stage llama-fit); about 30 s per try.
[ $# -ge 5 ] || { echo "usage: $0 <model> <threads> <ubatch> <ctx> <out-json>" >&2; exit 2; }
model=$1 threads=$2 ub=$3 ctx=$4 out=$5
server="${LLAMA_SERVER:-/opt/llama.cpp/bin/llama-server}"
port="${FIT_PORT:-18087}"
work=/root/llama_fit
mkdir -p "$work"
[ -x "$server" ] || { echo "missing $server" >&2; exit 3; }
# newer llama.cpp fits tensors to free memory by itself (-fit); off, so a try that does not fit fails
fit_off=()
"$server" --help 2>&1 | grep -q -- '--fit ' && fit_off=(-fit off)

ot_for() {
  local n=$1
  [ "$n" -gt 0 ] || { echo 'exps=CPU'; return; }
  echo "blk\\.($(seq -s '|' 0 $((n - 1))))\\.ffn_.*_exps=CUDA0;exps=CPU"
}

try() {
  local n=$1 log="$work/fit-$1.log" ot args=() pid ok=0
  ot=$(ot_for "$n")
  IFS=';' read -ra parts <<< "$ot"
  for p in "${parts[@]}"; do args+=(-ot "$p"); done
  timeout -k 10 600 "$server" -m "$model" -ngl 99 -fa on -c "$ctx" -b "$ub" -ub "$ub" -t "$threads" \
    --port "$port" --host 127.0.0.1 -np 1 "${fit_off[@]}" "${args[@]}" > "$log" 2>&1 &
  pid=$!
  for _ in $(seq 1 600); do
    kill -0 "$pid" 2> /dev/null || break
    curl -sf "http://127.0.0.1:$port/health" > /dev/null 2>&1 && { ok=1; break; }
    sleep 1
  done
  kill "$pid" 2> /dev/null
  wait "$pid" 2> /dev/null
  echo "$n $ok"
}

results=()
best=-1
for n in $(seq "${FIT_MAX:-12}" -1 0); do
  r=$(try "$n")
  echo "[$(date +%T)] llama fit: $n layers of experts on CUDA0 -> $([ "${r#* }" = 1 ] && echo fits || echo fails)"
  results+=("$r")
  [ "${r#* }" = 1 ] && { best=$n; break; }
done
python3 -I - "$out" "$best" "$(ot_for "$best")" "$ub" "$ctx" "$work" "${results[@]}" << 'EOF'
import json, re, sys
out, best, ot, ub, ctx, work = sys.argv[1:7]
tries = []
for r in sys.argv[7:]:
    n, ok = r.split()
    try:
        text = open(f"{work}/fit-{n}.log", errors="replace").read()
    except OSError:
        text = ""
    bufs = re.findall(r"(CUDA0 (?:model|KV|compute) buffer size)\s*=\s*([\d.]+) MiB", text)
    tries.append({"layers": int(n), "fits": ok == "1", "buffers_mib": [[k, float(v)] for k, v in bufs],
                  "tail": text.splitlines()[-5:] if ok != "1" else []})
json.dump({"layers": int(best), "ot": ot, "ubatch": int(ub), "ctx": int(ctx), "tries": tries}, open(out, "w"), indent=1)
print("llama fit:", best, ot)
EOF
