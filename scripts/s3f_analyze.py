"""Analysis of session s3f (ADR 0081 confirmation): the P1 to P3 criteria and the report-only tables.

    python -I scripts/s3f_analyze.py [<session dir> [<s3c dir>]]

Defaults: bench/results/s3f-20261120 and bench/results/s3c-20261119. Writes <session dir>/analysis.txt and prints it.
Standard library only, so it runs under `python -I`. The criteria are in the session README ("Criteria (written
before measuring)"); the rules are repeated in the output.
"""

import glob
import json
import os
import re
import statistics
import sys

MODELS = ("qwen3-30b-a3b", "qwen3-32b")
CONFIGS = (("tp2pp2", "tp2-pp2"), ("layer4", "layer-4"))
NAME = "ggml-sys test gpu_no_capture_flag_skips_the_cuda_graph"
TEST = "gpu_no_capture_flag_skips_the_cuda_graph"
BAND = 0.05
FLAG = 0.15
THERMAL = 0x20
MOE_TG_TOL = 0.02


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def rates(path):
    """(pp reps, tg reps) in tok/s of a bench JSON; empty lists when missing."""
    pp, tg = [], []
    for r in load(path) or []:
        reps = list(r.get("samples_ts") or [])
        if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0:
            pp = reps
        elif r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0:
            tg = reps
    return pp, tg


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def rel(a, b):
    return a / b - 1 if b else float("nan")


# ---- P1 ----
def p1(d, out):
    out.append("P1: gpu-parity passed, and the ggml-sys test ran and passed")
    gates = load(os.path.join(d, "phase5", "gates.json")) or {}
    gate = gates.get("gpu_parity", "missing")
    path = os.path.join(d, "phase5", "gpu-parity.log")
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        text = ""
    # under --nocapture ggml's debug lines can sit between "..." and "ok": look from the start line to the next test
    m = re.search(r"(?m)^test " + TEST + r" \.\.\.(.*?)(?=^test |^test result|\Z)", text, re.S)
    ran = m is not None
    ok = bool(m) and re.search(r"(?m)(^|\s)ok$", m.group(1).strip()) is not None and "FAILED" not in m.group(1)
    skips = re.findall(r"(?m)(?:^|[ .])skip: .*$", text)
    out.append(f"  gates.json gpu_parity: {gate}")
    out.append(f"  {NAME} in gpu-parity.log: ran={ran} ok={ok}")
    out.append(f"  skip: lines in the log: {len(skips)}")
    verdict = gate == "ok" and ran and ok and not skips
    out.append(f"  P1 {'PASS' if verdict else 'FAIL'}")
    return verdict


# ---- B runs ----
def b_runs(d):
    """{(model, cfg short): {arm: [(rep k, pp reps, tg reps, name)]}}"""
    res = {}
    for m in MODELS:
        for short, _ in CONFIGS:
            arms = {"default": [], "nograph": []}
            for arm in arms:
                for k in (1, 2):
                    name = f"{m}-{short}-mb4-{arm}-r{k}"
                    pp, tg = rates(os.path.join(d, name + ".json"))
                    if pp or tg:
                        arms[arm].append((k, pp, tg, name))
            res[(m, short)] = arms
    return res


def p2_p3(d, out):
    runs = b_runs(d)
    out.append("")
    out.append("B runs: pp2048 and tg128, tok/s per timed rep (rep 1 first)")
    for (m, short), arms in runs.items():
        for arm, lst in arms.items():
            for k, pp, tg, name in lst:
                out.append(f"  {name}: pp " + " ".join(f"{x:.0f}" for x in pp) + " | tg " + " ".join(f"{x:.1f}" for x in tg))
    out.append("")
    out.append(f"P2 (flag live): default rep 1 within {BAND:.0%} of the median of reps 2 to 5 (each default run), and the")
    out.append(f"  default pp median within {BAND:.0%} of the nograph pp median (median of all reps pooled over the arm's runs)")
    p2_all, p3_all = True, True
    p3_lines = []
    for (m, short), arms in runs.items():
        de, ng = arms["default"], arms["nograph"]
        label = f"{m} {short}-mb4"
        ok = len(de) == 2 and len(ng) == 2 and all(len(r[1]) == 5 and len(r[2]) == 5 for r in de + ng)
        if not ok:
            out.append(f"  {label}: INCOMPLETE (default runs {len(de)}, nograph runs {len(ng)}, or fewer than 5 reps): FAIL")
            p2_all = p3_all = False
            continue
        first_ok = True
        for k, pp, tg, name in de:
            r1, base = pp[0], med(pp[1:])
            good = abs(rel(r1, base)) <= BAND
            first_ok &= good
            out.append(f"  {label} {name}: rep 1 {r1:.0f} vs median reps 2-5 {base:.0f} ({rel(r1, base):+.1%}) {'ok' if good else 'FAIL'}")
        dpp = med([x for r in de for x in r[1]])
        npp = med([x for r in ng for x in r[1]])
        pair_ok = abs(rel(dpp, npp)) <= BAND
        out.append(f"  {label}: default pp median {dpp:.0f} vs nograph {npp:.0f} ({rel(dpp, npp):+.1%}) {'ok' if pair_ok else 'FAIL'}")
        p2_all &= first_ok and pair_ok
        dtg = med([x for r in de for x in r[2]])
        ntg = med([x for r in ng for x in r[2]])
        # 32B: default must be faster. 30B (MoE): ggml-cuda can disable graphs itself (mul_mat_id_needs_sync), so replay
        # may not help: pass if default is not slower than nograph by more than 2%
        good = rel(dtg, ntg) >= -MOE_TG_TOL if m == MODELS[0] else dtg > ntg
        p3_all &= good
        p3_lines.append(f"  {label}: default tg median {dtg:.1f} vs nograph {ntg:.1f} ({rel(dtg, ntg):+.1%}) {'ok' if good else 'FAIL'}")
    out.append(f"  P2 {'PASS' if p2_all else 'FAIL'}")
    out.append("")
    out.append(f"P3 (decode keeps replay): 32B default tg median > nograph; 30B (MoE) default tg not slower than nograph by more than {MOE_TG_TOL:.0%}")
    out += p3_lines
    out.append(f"  P3 {'PASS' if p3_all else 'FAIL'}" + ("" if p3_all else " (the flag may have leaked onto decode)"))
    return p2_all, p3_all


# ---- stage A report ----
def stage_a(d):
    """{(model, id): {"pp": [..], "tg": [..]}} from phase5/bench-<model>-<id>-r<k>.json (the -aa runs are not rounds)."""
    res = {}
    for p in sorted(glob.glob(os.path.join(d, "phase5", "bench-*-r*.json"))):
        base = os.path.basename(p)[len("bench-"):-len(".json")]
        for m in MODELS:
            if base.startswith(m + "-"):
                cid = re.sub(r"-r\d+$", "", base[len(m) + 1:])
                pp, tg = rates(p)
                e = res.setdefault((m, cid), {"pp": [], "tg": []})
                e["pp"] += pp
                e["tg"] += tg
    return res


def predictions(d):
    res = {}
    for m in MODELS:
        pr = load(os.path.join(d, "phase5", f"predict-{m}-4x.json")) or {}
        for c in pr.get("candidates", []):
            res[(m, c["id"])] = (c["pp_tps"], c["tg_tps"])
    return res


def flag(x):
    return "*" if abs(x - 1) > FLAG else " "


def report_a(d, c3, out):
    out.append("")
    out.append(f"Report only: stage A (ROUNDS=1) measured / predicted, next to s3c ({os.path.basename(c3)}); * marks |ratio-1| > {FLAG:.0%}")
    a, pa = stage_a(d), predictions(d)
    c, pc = stage_a(c3), predictions(c3)
    out.append("  model          candidate  | s3f pp ratio  tg ratio  | s3c pp ratio  tg ratio  | s3f pp tok/s (pred)  tg tok/s (pred)")
    past = []
    for key in sorted(set(pa) | set(pc)):
        cells = []
        for runs, pred in ((a, pa), (c, pc)):
            if key in runs and key in pred and runs[key]["pp"] and runs[key]["tg"]:
                rp, rt = statistics.mean(runs[key]["pp"]) / pred[key][0], statistics.mean(runs[key]["tg"]) / pred[key][1]
                cells.append((rp, rt))
            else:
                cells.append(None)
        s = lambda t: "      n/a       n/a" if t is None else f"  {t[0]:6.3f}{flag(t[0])}  {t[1]:6.3f}{flag(t[1])}"  # noqa: E731
        detail = ""
        if key in a and key in pa and a[key]["pp"] and a[key]["tg"]:
            detail = f"  {statistics.mean(a[key]['pp']):7.0f} ({pa[key][0]:.0f})  {statistics.mean(a[key]['tg']):6.1f} ({pa[key][1]:.1f})"
        out.append(f"  {key[0]:14s} {key[1]:9s} |{s(cells[0])}  |{s(cells[1])}  |{detail}")
        if cells[0] and (abs(cells[0][0] - 1) > FLAG or abs(cells[0][1] - 1) > FLAG):
            past.append(f"{key[0]} {key[1]}")
    out.append("  s3f candidates past 15%: " + (", ".join(past) if past else "none"))


def anchors(d):
    res = {}
    for p in sorted(glob.glob(os.path.join(d, "phase5", "anchors-*-r*.json"))):
        base = os.path.basename(p)
        m = next((x for x in MODELS if base.startswith(f"anchors-{x}-")), None)
        for r in load(p) or []:
            kind = "pp" if r.get("n_prompt", 0) > 0 else "tg"
            key = (m, r.get("plan_id"), r.get("layer_limit"), kind)
            res.setdefault(key, []).append(r["avg_ts"])
    return {k: statistics.mean(v) for k, v in res.items()}


def report_anchors(d, c3, out):
    out.append("")
    out.append("Report only: anchors, s3f against s3c on the same host (expectation, not judged); pp and tg tok/s, mean over the rounds")
    a, c = anchors(d), anchors(c3)
    out.append("  model          plan   layers  kind |   s3f      s3c    shift")
    for key in sorted(set(a) | set(c), key=lambda k: tuple(str(x) for x in k)):
        x, y = a.get(key), c.get(key)
        sh = f"{rel(x, y):+.1%}" if x and y else "n/a"
        out.append(f"  {key[0]!s:14s} {key[1]!s:6s} {key[2]!s:6s} {key[3]} | {x if x else float('nan'):8.1f} {y if y else float('nan'):8.1f}  {sh:>7s}")


# ---- throttle ----
def throttle(d, out):
    out.append("")
    out.append("Report only: 0x20 throttle samples (1 Hz sampler, per GPU) inside each B run window; other reason bits seen in brackets")
    path = os.path.join(d, "gpu-samples.csv")
    samples = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 8:
                    try:
                        samples.append((float(parts[0]), int(parts[2], 16), int(parts[7])))
                    except ValueError:
                        continue
    except OSError:
        out.append("  no gpu-samples.csv")
        return
    total = 0
    for p in sorted(glob.glob(os.path.join(d, "*.window.txt"))):
        name = os.path.basename(p)[: -len(".window.txt")]
        mt = re.search(r"start=([\d.]+) end=([\d.]+)", open(p).read())
        if not mt:
            continue
        t0, t1 = float(mt.group(1)), float(mt.group(2))
        win = [s for s in samples if t0 <= s[0] <= t1]
        hot = {}
        other = set()
        for _, reasons, gpu in win:
            if reasons & THERMAL:
                hot[gpu] = hot.get(gpu, 0) + 1
            rest = reasons & ~0x1
            if rest and not reasons & THERMAL:
                other.add(hex(reasons))
        n = sum(hot.values())
        total += n
        out.append(f"  {name}: {len(win)} samples, 0x20 on " + (", ".join(f"GPU {g}: {c}" for g, c in sorted(hot.items())) if hot else "none")
                   + (f" [{' '.join(sorted(other))}]" if other else ""))
    out.append(f"  0x20 samples in all B runs: {total}")
    notes = load(os.path.join(d, "phase5", "verdict-notes.json")) or {}
    unclean = notes.get("unclean_runs", [])
    out.append(f"  stage A unclean runs (verdict-notes.json): {len(unclean)}")
    for u in unclean:
        out.append("    " + "; ".join(u.get("reasons", [u.get("run", "?")])))


def report_c(d, e3, out):
    out.append("")
    out.append("Report only: part C (30B, tg128, -r 5), median tok/s of the 5 reps; s3e layer-4 numbers beside")
    m = MODELS[0]
    rows = [(f"{m}-layer-4-L36-tg128", d), (f"{m}-single-gpu0-tg128", d), (f"{m}-single-gpu1-tg128", d),
            (f"{m}-single-gpu2-tg128", d), (f"{m}-single-gpu3-tg128", d),
            (f"{m}-layer-4-L12-tg128", e3), (f"{m}-layer-4-L24-tg128", e3), (f"{m}-layer-4-tg128", e3)]
    for name, src in rows:
        _, tg = rates(os.path.join(src, name + ".json"))
        tag = "s3f" if src == d else "s3e"
        out.append(f"  {tag} {name}: " + (f"median {med(tg):.1f} (" + " ".join(f"{x:.1f}" for x in tg) + ")" if tg else "missing"))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "bench/results/s3f-20261120"
    c3 = sys.argv[2] if len(sys.argv) > 2 else "bench/results/s3c-20261119"
    e3 = sys.argv[3] if len(sys.argv) > 3 else "bench/results/s3e-20261120"
    out = [f"s3f analysis of {d}", ""]
    v1 = p1(d, out)
    v2, v3 = p2_p3(d, out)
    report_a(d, c3, out)
    report_anchors(d, c3, out)
    report_c(d, e3, out)
    throttle(d, out)
    out.append("")
    out.append(f"Verdicts: P1 {'PASS' if v1 else 'FAIL'}, P2 {'PASS' if v2 else 'FAIL'}, P3 {'PASS' if v3 else 'FAIL'}")
    text = "\n".join(out) + "\n"
    with open(os.path.join(d, "analysis.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
