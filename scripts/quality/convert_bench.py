#!/usr/bin/env python3
"""Conversion speed and byte parity against llama-quantize (ADR 0047), paired on one box.

llama-quantize converts the source to a preset; the engine's converter (`tetraeder quant export --formats-from`)
then does the same per-tensor work with the same thread count and an empty cache. Repeats interleave the two.
Also times a load-path rerun with a warm cache, and compares every tensor's bytes.
"""

import argparse
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

import qlib


def run(cmd, log):
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    Path(log).write_text(" ".join(cmd) + "\n" + p.stdout + p.stderr)
    if p.returncode != 0:
        raise SystemExit(f"failed: {' '.join(cmd)} (see {log})")
    return time.time() - t, p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--preset", default="Q4_K_M")
    ap.add_argument("--work", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--llama-bin", default="/opt/llama.cpp/bin")
    ap.add_argument("--tetraeder", default="target/release/tetraeder")
    ap.add_argument("--gpu", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ref, ours, cache = work / "cb-llama.gguf", work / "cb-ours.gguf", work / "cb-cache"
    src_bytes = qlib.tensor_bytes(a.source)
    rows = []
    for r in range(a.repeats):
        for tool in ("llama-quantize", "tetraeder"):
            if tool == "llama-quantize":
                ref.unlink(missing_ok=True)
                dt, _ = run([str(Path(a.llama_bin) / "llama-quantize"), a.source, str(ref), a.preset, str(a.threads)],
                            work / f"cb-llama-{r}.log")
                stats = {}
            else:
                shutil.rmtree(cache, ignore_errors=True)
                ours.unlink(missing_ok=True)
                dt, out = run([a.tetraeder, "quant", "export", "--model", a.source, "--formats-from", str(ref),
                               "--out", str(ours), "--threads", str(a.threads), "--cache", str(cache)],
                              work / f"cb-ours-{r}.log")
                stats = json.loads(out.strip().splitlines()[-1])
            rows.append({"tool": tool, "repeat": r, "seconds": dt, "source_bytes": src_bytes,
                         "gb_per_s_in": src_bytes / dt / 1e9, **stats})
            print(f"[{time.strftime('%T')}] {tool} #{r}: {dt:.1f} s", flush=True)
    # warm cache: the load path's cost once every tensor is converted
    dt, out = run([a.tetraeder, "quant", "export", "--model", a.source, "--formats-from", str(ref), "--out",
                   str(ours), "--threads", str(a.threads), "--cache", str(cache)], work / "cb-ours-warm.log")
    rows.append({"tool": "tetraeder-warm-cache", "repeat": 0, "seconds": dt, **json.loads(out.strip().splitlines()[-1])})
    a_d, b_d = qlib.data_digest(ref), qlib.data_digest(ours)
    diff = sorted(n for n in a_d if a_d[n] != b_d.get(n))
    res = {"schema": 1, "source": Path(a.source).name, "preset": a.preset, "threads": a.threads, "gpu": a.gpu,
           "host": socket.gethostname(), "rows": rows, "tensors": len(a_d), "tensors_differing": diff,
           "byte_identical": not diff and set(a_d) == set(b_d)}
    qlib.write_json(a.out, res)
    shutil.rmtree(cache, ignore_errors=True)
    print(f"byte identical: {res['byte_identical']} ({len(diff)} differ)")
    print("CONVERT_BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
