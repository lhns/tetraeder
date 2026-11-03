#!/usr/bin/env python3
"""Phase 3a exit-test harness (ADR 0048): a format mix against llama.cpp presets at the same byte budget.

For each preset, llama-quantize builds it from the source; a mix within the preset's bytes is either given
(`--mix`, tensor name to format: the planner's output) or picked from the quality profile by a reference greedy
knapsack (a stand-in until the Phase 3a planner exists). Both are evaluated with `llama-perplexity --kl-divergence`
against reference logits of held-out text (not the calibration text). Optional ik_llama.cpp quant types are
measured the same way with ik_llama's own tools, each against a mix at its bytes.

Example (box):
  python3 scripts/quality/compare_mix.py --source <bf16.gguf> --profile bench/quality/qwen3-8b.quality.json \
    --work /root/q/qwen3-8b --text /root/q/wiki.test.raw --presets Q4_K_M Q3_K_M --threads 11 \
    --out bench/results/20261103/quality-qwen3-8b.json
"""

import argparse
import json
import socket
import subprocess
import time
from pathlib import Path

import qlib


def hull(points):
    """Lower convex hull of (bytes, kl, format), by bytes ascending."""
    h = []
    for p in sorted(points):
        if h and p[1] >= h[-1][1]:
            continue  # dominated: more bytes, no less KL
        # pop while the last point lies on or above the segment from h[-2] to p
        while len(h) >= 2:
            (x1, y1, _), (x2, y2, _) = h[-2], h[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) <= 0:
                h.pop()
            else:
                break
        h.append(p)
    return h


def greedy_mix(profile, budget):
    """Group formats with summed KL as low as the reference greedy finds within `budget` bytes: start every group
    at its smallest format, then take the hull step with the best KL drop per byte while it fits."""
    base = profile["base"]
    base_group_bytes = sum(next(e["bytes"] for e in g["entries"] if e["format"] == base["format"])
                           for g in profile["groups"])
    rest = base["bytes"] - base_group_bytes
    hulls = {g["id"]: hull([(e["bytes"], e["kl"], e["format"]) for e in g["entries"]]) for g in profile["groups"]}
    pos = {gid: 0 for gid in hulls}
    total = rest + sum(h[0][0] for h in hulls.values())
    if total > budget:
        raise SystemExit(f"budget {budget} below the smallest mix ({total} bytes)")
    while True:
        best = None
        for gid, h in hulls.items():
            i = pos[gid]
            if i + 1 < len(h):
                db, dk = h[i + 1][0] - h[i][0], h[i][1] - h[i + 1][1]
                if total + db <= budget and (best is None or dk / db > best[0]):
                    best = (dk / db, gid, db)
        if best is None:
            break
        _, gid, db = best
        pos[gid] += 1
        total += db
    # spend what is left on any upgrade that still fits (the hull steps can skip affordable points)
    choice = {gid: hulls[gid][pos[gid]] for gid in hulls}
    for g in profile["groups"]:
        cur = choice[g["id"]]
        for e in sorted(g["entries"], key=lambda e: e["kl"]):
            if e["kl"] < cur[1] and total - cur[0] + e["bytes"] <= budget:
                total += e["bytes"] - cur[0]
                choice[g["id"]] = cur = (e["bytes"], e["kl"], e["format"])
                break
    return {gid: c[2] for gid, c in choice.items()}, total


def tensor_map(source, profile, choice):
    """Tensor name -> format for a whole-group choice, from the groups the calibration used."""
    kv, tensors = qlib.read_model(source)
    _, n_layer, _ = qlib.arch_layers(kv)
    groups = qlib.build_groups(tensors, n_layer, profile["layer_span"])
    return {t: choice[g["id"]] for g in groups for t in g["tensors"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--reference", help="model for the reference logits (default: the source)")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--text", required=True, help="held-out evaluation text")
    ap.add_argument("--text-name", default="wikitext-2-raw-v1 wiki.test.raw")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=100)
    ap.add_argument("--presets", nargs="+", default=["Q4_K_M"])
    ap.add_argument("--mix", help="JSON {tensor: format} to evaluate against every preset at or above its bytes")
    ap.add_argument("--ik-bin", help="ik_llama.cpp bin dir with llama-quantize and llama-perplexity")
    ap.add_argument("--ik-types", nargs="*", default=[])
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--ppl-args", default="-ngl 99 -fa on")
    ap.add_argument("--ref-args", default=None)
    ap.add_argument("--llama-bin", default="/opt/llama.cpp/bin")
    ap.add_argument("--tetraeder", default="target/release/tetraeder")
    ap.add_argument("--gpu", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    work = Path(a.work)
    (work / "logs").mkdir(parents=True, exist_ok=True)
    profile = json.loads(Path(a.profile).read_text())
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {
        "schema": 1, "source": Path(a.source).name, "profile": Path(a.profile).name,
        "eval": {"text": a.text_name, "sha256": qlib.sha256_file(a.text), "ctx": a.ctx, "chunks": a.chunks},
        "host": socket.gethostname(), "gpu": a.gpu, "threads": a.threads, "ggml": profile.get("ggml"),
        "rows": [],
    }
    done = {r["name"] for r in res["rows"]}

    def save():
        qlib.write_json(out, res)

    def ref_logits(bin_dir, tag):
        p = work / f"eval-ref{tag}.kld"
        if not p.exists():
            args = ["-f", a.text, "-c", str(a.ctx), "--chunks", str(a.chunks), "-t", str(a.threads),
                    "--kl-divergence-base", str(p)] + (a.ref_args or a.ppl_args).split()
            qlib.perplexity(bin_dir, a.reference or a.source, args, work / "logs" / f"eval-ref{tag}.log")
        return p

    def evaluate(name, model, bin_dir, ref, extra):
        args = ["--kl-divergence-base", str(ref), "--kl-divergence", "-c", str(a.ctx), "--chunks", str(a.chunks),
                "-t", str(a.threads)] + a.ppl_args.split()
        text, dt = qlib.perplexity(bin_dir, model, args, work / "logs" / f"eval-{name}.log")
        r = {"name": name, "bytes": qlib.tensor_bytes(model), **qlib.parse_kld(text), "eval_seconds": dt, **extra}
        res["rows"].append(r)
        save()
        print(f"[{time.strftime('%T')}] {name}: {r['bytes'] / 1e9:.3f} GB, KL {r['kl']:.5f}, PPL {r.get('ppl')}",
              flush=True)
        return r

    def quantize(bin_dir, kind, out_path):
        if not out_path.exists():
            t = time.time()
            cmd = [str(Path(bin_dir) / "llama-quantize"), a.source, str(out_path), kind, str(a.threads)]
            p = subprocess.run(cmd, capture_output=True, text=True)
            (work / "logs" / f"quantize-{out_path.stem}.log").write_text(p.stdout + p.stderr)
            if p.returncode != 0:
                raise SystemExit(f"llama-quantize {kind} failed")
            (work / f"{out_path.stem}.seconds").write_text(str(time.time() - t))
        return float((work / f"{out_path.stem}.seconds").read_text())

    def export(name, fmap):
        path = work / f"{name}.gguf"
        if not path.exists():
            mp = work / f"{name}.formats.json"
            mp.write_text(json.dumps(fmap))
            cmd = [a.tetraeder, "quant", "export", "--model", a.source, "--formats", str(mp), "--out", str(path),
                   "--threads", str(a.threads), "--cache", str(work / "cache")]
            p = subprocess.run(cmd, capture_output=True, text=True)
            (work / "logs" / f"export-{name}.log").write_text(p.stdout + p.stderr)
            if p.returncode != 0:
                raise SystemExit(f"export {name} failed")
        return path

    ref = ref_logits(a.llama_bin, "")
    budgets = []
    for preset in a.presets:
        pp = work / f"preset-{preset}.gguf"
        secs = quantize(a.llama_bin, preset, pp)
        b = qlib.tensor_bytes(pp)
        budgets.append((preset, b))
        if f"preset-{preset}" not in done:
            evaluate(f"preset-{preset}", pp, a.llama_bin, ref, {"tool": "llama.cpp", "quantize_seconds": secs})
        name = f"greedy-at-{preset}"
        if name not in done:
            choice, est_bytes = greedy_mix(profile, b)
            mp = export(name, tensor_map(a.source, profile, choice))
            evaluate(name, mp, a.llama_bin, ref, {"tool": "tetraeder", "budget": b, "formats": choice,
                                                   "estimate": qlib.estimate(profile, choice)})

    if a.mix:
        fmap = json.loads(Path(a.mix).read_text())
        if "mix" not in done:
            evaluate("mix", export("mix", fmap), a.llama_bin, ref, {"tool": "tetraeder", "mix_file": a.mix})

    if a.ik_bin and a.ik_types:
        ik_ref = ref_logits(a.ik_bin, "-ik")
        for kind in a.ik_types:
            pp = work / f"ik-{kind}.gguf"
            secs = quantize(a.ik_bin, kind, pp)
            if f"ik-{kind}" not in done:
                evaluate(f"ik-{kind}", pp, a.ik_bin, ik_ref, {"tool": "ik_llama.cpp", "quantize_seconds": secs})
            name = f"greedy-at-ik-{kind}"
            if name not in done:
                b = qlib.tensor_bytes(pp)
                choice, _ = greedy_mix(profile, b)
                mp = export(name, tensor_map(a.source, profile, choice))
                # evaluated with ik_llama's tools too, so both sides share one reference file
                evaluate(name, mp, a.ik_bin, ik_ref, {"tool": "tetraeder", "budget": b, "formats": choice,
                                                      "estimate": qlib.estimate(profile, choice)})

    # Verdict rows for the exit test: the mix at or below each preset's bytes, KL no higher.
    res["verdicts"] = []
    rows = {r["name"]: r for r in res["rows"]}
    for preset, b in budgets:
        p, g = rows.get(f"preset-{preset}"), rows.get(f"greedy-at-{preset}")
        if p and g:
            res["verdicts"].append({"preset": preset, "preset_bytes": p["bytes"], "preset_kl": p["kl"],
                                    "mix": "greedy", "mix_bytes": g["bytes"], "mix_kl": g["kl"],
                                    "pass": g["bytes"] <= p["bytes"] and g["kl"] <= p["kl"]})
    save()
    print("COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
