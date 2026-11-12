#!/usr/bin/env bash
# llama.cpp's best tuned setup for the exit regime: GPU expert cache with a CPU share of the misses
# (llama.cpp #30966, docs/moe-cache.md), run through llama-server from the baseline build (/opt/llama.cpp-baseline).
#   scripts/llamacpp_cache.sh <model-first-shard> <threads> <cache-mib> <cpu-share> <ctx> <prompt-file> <out-json>
# Env: LOAD_MODE (default auto = mmap; the docs say none, mlock and dio pin the experts in RAM, which a model larger
# than RAM cannot do, so dio is opt-in), PREFILL (32768), DECODE (64), UBATCH (2048), LLAMA_BASELINE_SERVER, PORT.
# llama-server has no llama-bench-style depth run, so decode speed is the /completion timings.predicted_per_second of
# a DECODE-token greedy generation right after a PREFILL-token prompt (the same token ids as exit_regime: the prompt
# file through the model's vocab, repeated to length). Called by exit_run.sh (stages llamacpp-cache*). The server is
# sampled by scripts/engine_sample.py like the other engines ("host" and "env" in the result; HOST_DISK_DEV is the
# model's MAJ:MIN, probed when unset); the loaded disk curve is looked up next to <out-json>.
[ $# -ge 7 ] || { echo "usage: $0 <model> <threads> <cache-mib> <cpu-share> <ctx> <prompt-file> <out-json>" >&2; exit 2; }
# one llama-server at a time (fixed port, 142 GB model): a duplicate launch exits
exec 9> /root/tetraeder_llamacpp_cache.lock; flock -n 9 || { echo "llamacpp_cache.sh already running" >&2; exit 4; }
model=$1 threads=$2 mib=$3 share=$4 ctx=$5 prompt=$6 out=$7
server="${LLAMA_BASELINE_SERVER:-/opt/llama.cpp-baseline/bin/llama-server}"
[ -x "$server" ] || { echo "missing $server" >&2; exit 3; }
# server log and host samples go under runs/vast (fetched with the results), one pair per result file
work="${LLAMACPP_CACHE_WORK:-$PWD/runs/vast/llamacpp_cache}"
mkdir -p "$work"
# newer llama.cpp fits tensors to free memory by itself (-fit); off, so the flags below are what runs
fit_off=()
"$server" --help 2>&1 | grep -q -- '--fit ' && fit_off=(-fit off)
SCRIPTS_DIR=$(cd "$(dirname "$0")" && pwd) HOST_DISK_DEV="${HOST_DISK_DEV:-}" python3 -I - "$server" "$work/$(basename "$out" .json).log" "${PORT:-18088}" "$model" "$threads" "$mib" "$share" "$ctx" "$prompt" "$out" \
  "${LOAD_MODE:-auto}" "${PREFILL:-32768}" "${DECODE:-64}" "${UBATCH:-2048}" "${fit_off[@]}" << 'EOF'
import json, os, re, signal, subprocess, sys, time, urllib.request

sys.path.insert(0, os.environ["SCRIPTS_DIR"])
import engine_sample as es

(server, log, port, model, threads, mib, share, ctx, prompt, out, load_mode, prefill, decode, ub) = sys.argv[1:15]
fit_off, port, prefill, decode = sys.argv[15:], int(port), int(prefill), int(decode)
flags = ["-ngl", "99", "-fa", "on", "-c", ctx, "-b", ub, "-ub", ub, "-t", threads, "-np", "1", "-lm", load_mode,
         "-cmoe", "--moe-cache-mib", mib, "--moe-cache-cpu-share", share, "-lv", "4", *fit_off]
cmd = [server, "-m", model, *flags, "--host", "127.0.0.1", "--port", str(port)]


def moe_cache_stats(path):
    """llama.cpp's expert-cache summary lines (#30966) as numbers; decode counts include the warm-up tokens."""
    out = {}
    for l in open(path, errors="replace"):
        m = re.search(r"ubatch <= (\d+): hits = (\d+), misses = (\d+) \((\d+) on the CPU\), hit rate = ([\d.]+)%, uploaded = ([\d.]+) MiB", l)
        if m:
            out["decode"] = {"ubatch_max": int(m[1]), "hits": int(m[2]), "misses": int(m[3]), "misses_cpu": int(m[4]),
                             "hit_rate": float(m[5]) / 100, "uploaded_mib": float(m[6])}
        m = re.search(r"large batches: (\d+) experts copied from the cache, ([\d.]+) MiB", l)
        if m:
            out["large_batches"] = {"experts_copied": int(m[1]), "copied_mib": float(m[2])}
    return out or None
def http(endpoint, body=None, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{endpoint}", data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def proc_stat(pid):
    """major page faults and RSS bytes of the server (a model larger than RAM streams or thrashes here)"""
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return {"majflt": int(f[9]), "rss_bytes": int(f[21]) * os.sysconf("SC_PAGE_SIZE")}
    except OSError:
        return {"majflt": -1, "rss_bytes": -1}


def tokens(n):
    # as exit_regime's prompt(): the vocab's own BOS rule (add_special), special-token text not parsed, cycled to n
    t = http("/tokenize", {"content": open(prompt).read(), "add_special": True, "parse_special": False})["tokens"]
    assert t, "empty prompt file"
    return [t[i % len(t)] for i in range(n)]


def gen(toks, n):
    return http("/completion", {"prompt": toks, "n_predict": n, "ignore_eos": True, "cache_prompt": False,
                                "temperature": 0, "top_k": 1, "seed": 1})


sampler = None
res = {"engine": f"llamacpp-cache-s{share}", "cache_mib": int(mib), "cpu_share": float(share), "load_mode": load_mode,
       "flags": flags, "threads": int(threads), "ctx": int(ctx), "ubatch": int(ub), "prefill": prefill, "decode": decode,
       "server": server, "machine": os.uname().nodename, "ok": False}
try:
    res["baseline_pin"] = open(os.path.join(os.path.dirname(server), "..", "PIN")).read().strip()
except OSError:
    pass
with open(log, "w") as lf:
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
try:
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited ({proc.returncode}), see {log}")
        try:
            if http("/health", timeout=5).get("status") == "ok":
                break
        except OSError:
            pass
        time.sleep(2)
    res["load_s"] = round(time.time() - t0, 1)
    try:  # host sampling must never fail the run
        sample_py = os.path.join(os.environ["SCRIPTS_DIR"], "engine_sample.py")
        disk = os.environ.get("HOST_DISK_DEV") or subprocess.run(
            [sys.executable, sample_py, "probe-disk", model], capture_output=True, text=True).stdout.strip() or "none"
        samples = os.path.splitext(log)[0] + ".samples.jsonl"
        sampler = subprocess.Popen([sys.executable, sample_py, "sample", "--pid", str(proc.pid), "--disk", disk,
                                    "--out", samples])
    except Exception as e:
        res["host_error"] = repr(e)
    # is the model streamed or thrashed: load mode, RSS and major faults around a short warmup
    before = proc_stat(proc.pid)
    t1 = time.time()
    w = gen(tokens(256), 8)
    after = proc_stat(proc.pid)
    res["warmup"] = {"before": before, "after": after, "s": round(time.time() - t1, 1),
                     "majflt": after["majflt"] - before["majflt"], "decode_tok_s": w["timings"]["predicted_per_second"]}
    res["log_load_mode"] = [l.strip() for l in open(log, errors="replace") if re.search(r"mmap|direct|dio|mlock|pinned", l, re.I)][:8]
    print("warmup:", json.dumps(res["warmup"]), flush=True)
    # measured run: PREFILL-token prompt, then DECODE-token greedy generation
    before = proc_stat(proc.pid)
    toks = tokens(prefill)
    # the builder's hash of its token ids (exit_regime's tokens_hash, JSON "prompt.fnv1a64"): equal means both
    # engines saw the same prompt (c743c72b8ffceadd for the frozen 32K prompt)
    h = 0xCBF29CE484222325
    for b in b"".join(t.to_bytes(4, "little", signed=True) for t in toks):
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    res["prompt"] = {"tokens": len(toks), "fnv1a64": f"{h:016x}"}
    t_a = time.time()
    g = gen(toks, decode)
    t_b = time.time()
    after = proc_stat(proc.pid)
    t = g["timings"]
    res["timings"] = {k: t.get(k) for k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms",
                                            "predicted_per_second")}
    res["prefill_tok_s"], res["decode_tok_s"] = t["prompt_per_second"], t["predicted_per_second"]
    res["run_stat"] = {"before": before, "after": after, "majflt": after["majflt"] - before["majflt"]}
    res["ok"] = t["prompt_n"] >= prefill - 1 and t["predicted_n"] == decode
    try:
        if sampler is not None:
            time.sleep(2)  # one more sample after the run
            sampler.terminate()
            sampler.wait(timeout=30)
            t_first = t_a + t["prompt_ms"] / 1e3
            curve_file = es.find_curve(os.path.dirname(os.path.abspath(out)))
            curve = es.load_curve(curve_file) if curve_file else []
            res["host"] = es.host_summary(es.load_samples(samples), (t_a, t_first, t["prompt_n"]),
                                          (t_first, t_b, t["predicted_n"]), int(threads), curve)
            res["env"] = {**es.env_info(), "disk_curve": os.path.basename(curve_file) if curve else None}
            # the same windows for the OTLP path (engine_sample.py otlp-merge, ADR 0059)
            res["host_windows"] = es.host_windows((t_a, t_first, t["prompt_n"]), (t_first, t_b, t["predicted_n"]),
                                                  int(threads))
    except Exception as e:
        res["host_error"] = repr(e)
except Exception as e:
    res["error"] = repr(e)
finally:
    if sampler is not None and sampler.poll() is None:
        sampler.terminate()
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)  # the cache logs its hit statistics at exit
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
    res["cache_log"] = [l.strip() for l in open(log, errors="replace") if "llama_moe_cache:" in l][-6:]
    res["moe_cache"] = moe_cache_stats(log)
    json.dump({**res, "runs": [dict(res)]}, open(out, "w"), indent=1)
print(res["engine"], "decode tok/s:", res.get("decode_tok_s"), "ok:", res["ok"], res.get("error", ""))
sys.exit(0 if res["ok"] else 1)
EOF
