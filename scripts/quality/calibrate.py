#!/usr/bin/env python3
"""Calibration job (ADR 0048): measures the quality profile of one model.

For each format the source is converted once by the engine (`tetraeder quant export --all FMT --groups`) into a
split GGUF with one shard per group. A sample is a directory of symbolic links (one group in the sample's format,
everything else in the base format), evaluated with `llama-perplexity --kl-divergence` against reference logits
of the calibration text. Results are saved after every sample; a rerun resumes.

Example (box):
  python3 scripts/quality/calibrate.py --source .hf_cache/unsloth__Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf \
    --work /root/q/qwen3-8b --text /root/q/wiki.train.raw --base BF16 --formats Q6_K Q5_K Q4_K IQ4_XS Q3_K Q2_K \
    --span 4 --out bench/quality/qwen3-8b.quality.json
"""

import argparse
import json
import random
import shutil
import socket
import subprocess
import time
from pathlib import Path

import qlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="first shard of the most precise GGUF")
    ap.add_argument("--reference", help="model for the reference logits (default: the source)")
    ap.add_argument("--work", required=True, help="scratch directory (converted shards, logits, logs)")
    ap.add_argument("--text", required=True, help="calibration text file")
    ap.add_argument("--text-name", default="wikitext-2-raw-v1 wiki.train.raw")
    ap.add_argument("--base", default="BF16", help="format of every tensor outside the measured group")
    ap.add_argument("--formats", nargs="+", required=True)
    ap.add_argument("--span", type=int, required=True, help="layers per group")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--ppl-args", default="-ngl 99 -fa on", help="extra llama-perplexity args (offload)")
    ap.add_argument("--ref-args", default=None, help="llama-perplexity args for the reference run")
    ap.add_argument("--checks", type=int, default=4, help="random whole-mix additivity checks")
    ap.add_argument("--llama-bin", default="/opt/llama.cpp/bin")
    ap.add_argument("--tetraeder", default="target/release/tetraeder")
    ap.add_argument("--gpu", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t_start = time.time()
    work = Path(a.work)
    (work / "logs").mkdir(parents=True, exist_ok=True)
    kv, tensors = qlib.read_model(a.source)
    arch, n_layer, n_expert = qlib.arch_layers(kv)
    groups = qlib.build_groups(tensors, n_layer, a.span)
    count = len(groups) + 1
    groups_file = work / "groups.json"
    groups_file.write_text(json.dumps([{"id": g["id"], "tensors": g["tensors"]} for g in groups]))
    print(f"[{time.strftime('%T')}] {arch}: {n_layer} layers, {len(groups)} groups, span {a.span}", flush=True)

    # 1. One converted split model per format, shard i+1 = group i.
    convert = {}
    for fmt in [a.base] + [f for f in a.formats if f != a.base]:
        d = work / "shards" / fmt
        done = d / "DONE"
        if not done.exists():
            d.mkdir(parents=True, exist_ok=True)
            cmd = [a.tetraeder, "quant", "export", "--model", a.source, "--all", fmt, "--groups", str(groups_file),
                   "--out", str(d / "m"), "--threads", str(a.threads), "--cache", str(work / "cache")]
            t = time.time()
            p = subprocess.run(cmd, capture_output=True, text=True)
            (work / "logs" / f"export-{fmt}.log").write_text(p.stdout + p.stderr)
            if p.returncode != 0:
                raise SystemExit(f"export {fmt} failed, see logs/export-{fmt}.log")
            stats = json.loads(p.stdout.strip().splitlines()[-1])
            stats["wall_seconds"] = time.time() - t
            # the converted tensors are now in the shards; the cache would double the disk use
            shutil.rmtree(work / "cache", ignore_errors=True)
            done.write_text(json.dumps(stats))
            print(f"[{time.strftime('%T')}] exported {fmt} in {stats['wall_seconds']:.0f} s", flush=True)
        convert[fmt] = json.loads(done.read_text())
    shard_dirs = {f: work / "shards" / f for f in convert}

    def group_bytes(fmt):
        _, ts = qlib.read_model(shard_dirs[fmt] / qlib.shard_name("m", 0, count))
        by = {t.name: t.nbytes for t in ts}
        return [sum(by[n] for n in g["tensors"]) for g in groups], sum(by.values())

    # 2. Reference logits of the calibration text.
    ref_logits = work / "ref.kld"
    if not ref_logits.exists():
        ref = a.reference or a.source
        args = ["-f", a.text, "-c", str(a.ctx), "--chunks", str(a.chunks), "-t", str(a.threads),
                "--kl-divergence-base", str(ref_logits)] + (a.ref_args or a.ppl_args).split()
        _, dt = qlib.perplexity(a.llama_bin, ref, args, work / "logs" / "reference.log")
        print(f"[{time.strftime('%T')}] reference logits in {dt:.0f} s", flush=True)

    def measure(name, choice):
        """KL of a mix: choice[i] is the format of shard i (shard 0 = metadata and untouched tensors)."""
        first = qlib.assemble(work / "mix", shard_dirs, choice, count)
        args = ["--kl-divergence-base", str(ref_logits), "--kl-divergence", "-c", str(a.ctx),
                "--chunks", str(a.chunks), "-t", str(a.threads)] + a.ppl_args.split()
        text, dt = qlib.perplexity(a.llama_bin, first, args, work / "logs" / f"{name}.log")
        r = qlib.parse_kld(text)
        r["seconds"] = dt
        return r

    out = Path(a.out)
    prof = json.loads(out.read_text()) if out.exists() else None
    if prof is None:
        bb, total = group_bytes(a.base)
        prof = {
            "schema": 1,
            "model": {"name": kv.get("general.name", arch), "arch": arch, "source": Path(a.source).name,
                      "source_format": _main_type(tensors),
                      "fingerprint": _fingerprint(a.source), "layers": n_layer, "experts": n_expert},
            "ggml": _pin(),
            "calibration": {"text": a.text_name, "sha256": qlib.sha256_file(a.text), "ctx": a.ctx,
                            "chunks": a.chunks, "scored_tokens": a.chunks * (a.ctx - 1 - a.ctx // 2)},
            "imatrix": None,
            "reference": {"format": _main_type(qlib.read_model(a.reference or a.source)[1])},
            "base": {"format": a.base, "kl": None, "bytes": total},
            "layer_span": a.span,
            "groups": [{"id": g["id"], "units": g["units"], "roles": g["roles"], "params": g["params"],
                        "entries": []} for g in groups],
            "checks": [],
            "cost": {"seconds": 0.0, "gpu": a.gpu, "host": socket.gethostname(), "threads": a.threads},
            "convert": convert,
        }
    elapsed0 = prof["cost"]["seconds"]

    def save():
        prof["cost"]["seconds"] = elapsed0 + time.time() - t_start
        qlib.write_json(out, prof)

    base_choice = [a.base] * count
    if prof["base"]["kl"] is None:
        r = measure("base", base_choice)
        prof["base"]["kl"] = r["kl"]
        prof["base"]["ppl"] = r.get("ppl")
        bb, _ = group_bytes(a.base)
        for g, b in zip(prof["groups"], bb):
            g["entries"].append({"format": a.base, "bytes": b, "kl": r["kl"], "ppl": r.get("ppl")})
        save()
        print(f"[{time.strftime('%T')}] base {a.base}: KL {r['kl']:.6f}", flush=True)

    # 3. One sample per (group, format).
    for fmt in a.formats:
        if fmt == a.base:
            continue
        fb, _ = group_bytes(fmt)
        for i, g in enumerate(prof["groups"]):
            if any(e["format"] == fmt for e in g["entries"]):
                continue
            choice = list(base_choice)
            choice[i + 1] = fmt
            r = measure(f"{g['id']}-{fmt}", choice)
            g["entries"].append({"format": fmt, "bytes": fb[i], "kl": r["kl"], "ppl": r.get("ppl"),
                                 "kl_err": r.get("kl_err"), "same_top_p": r.get("same_top_p")})
            save()
            print(f"[{time.strftime('%T')}] {g['id']} {fmt}: KL {r['kl']:.6f} ({r['seconds']:.0f} s)", flush=True)

    # 4. Additivity checks on whole mixes: uniform formats and random mixes.
    rng = random.Random(42)
    fmts = [f for f in a.formats if f != a.base]
    wanted = [(f"uniform-{f}", {g["id"]: f for g in groups}) for f in fmts[:2]]
    for k in range(a.checks):
        wanted.append((f"random-{k}", {g["id"]: rng.choice(fmts + [a.base]) for g in groups}))
    have = {c["name"] for c in prof["checks"]}
    for name, mix in wanted:
        if name in have:
            continue
        choice = [a.base] + [mix[g["id"]] for g in groups]
        r = measure(f"check-{name}", choice)
        prof["checks"].append({"name": name, "formats": mix, "estimate": qlib.estimate(prof, mix),
                               "measured": r["kl"], "ppl": r.get("ppl")})
        save()
        c = prof["checks"][-1]
        print(f"[{time.strftime('%T')}] check {name}: estimate {c['estimate']:.5f} measured {c['measured']:.5f}",
              flush=True)
    save()
    print("CALIBRATE_DONE", flush=True)


def _main_type(tensors):
    """Type holding the most weights (the source format of the model)."""
    by = {}
    for t in tensors:
        by[t.type] = by.get(t.type, 0) + t.n
    return max(by, key=by.get)


def _fingerprint(first):
    """Profile fingerprint (ADR 0048): SHA-256 over each shard's header bytes and length (no mtime, so a copy
    of the same file keeps its profile)."""
    import hashlib
    import os

    h = hashlib.sha256()
    for p in qlib.split_paths(first):
        _, ts = qlib.read_header(p)
        start = min((t.offset for t in ts), default=0)
        with open(p, "rb") as f:
            h.update(f.read(start))
        h.update(str(os.path.getsize(p)).encode())
    return h.hexdigest()


def _pin():
    try:
        return Path("vendor/PIN").read_text().split()[0].split("=", 1)[1]
    except (OSError, IndexError):
        return "unknown"


if __name__ == "__main__":
    main()
