#!/usr/bin/env python3
"""Offline check of ADR 0076: per-shape step-time lines from truncated anchors, against measured candidates.

    python scripts/costmodel_lines.py bench/results/<session>/phase5                    fit and judge one session
    python scripts/costmodel_lines.py FIT_DIR --eval EVAL_DIR                           anchors of one session, candidates of another
    python scripts/costmodel_lines.py DIR --holdout                                     fit on all but the largest truncation, predict it
    python scripts/costmodel_lines.py DIR --json OUT.json                               also write the rows as JSON

For every (model, shape) with anchors at two or more layer counts L', and for each rate, the time of the judged
workload (2048 prompt tokens, 128 generated; the anchor rows' own n_prompt and n_gen) is fitted as a line in L':
`T(L') = t_step + L' * t_layer` (least squares). The prediction at the model's full L is `n / T(L)`. A candidate
without anchors of its own shape is reported as unanchored, never predicted from another shape.

Input files in DIR (ADR 0068 names): anchors-<model>-r<k>.json (rows with plan_id = shape id and layer_limit),
bench-<model>-<id>-r<k>.json (candidate rows; -aa runs are ignored), layouts/<model>.json (n_layer),
predict-<model>-<n>x.json (the session's committed prediction, shown for comparison). Run cleanliness (ADR 0068) is not
judged here: every bench-*-r<k>.json counts. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

BAND = 0.15  # ADR 0068 "Within 15%"


def _rows(path: Path) -> list[dict]:
    v = json.loads(path.read_text())
    return v if isinstance(v, list) else v.get("rows") or v.get("results") or []


def _rate(r: dict) -> str | None:
    if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0:
        return "pp"
    if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0:
        return "tg"
    return None


def _tokens(r: dict) -> int:
    return int(r["n_prompt"] or r["n_gen"])


def anchors(d: Path) -> dict:
    """{(model, shape): {rate: {L': (median tok/s, tokens)}}} from anchors-<model>-r<k>.json."""
    acc: dict = {}
    for f in sorted(d.glob("anchors-*-r*.json")):
        m = re.fullmatch(r"anchors-(.+)-r\d+\.json", f.name)
        if not m:
            continue
        for r in _rows(f):
            rate, lim = _rate(r), r.get("layer_limit")
            if rate is None or lim is None or "plan_id" not in r:
                continue
            e = acc.setdefault((m.group(1), r["plan_id"]), {}).setdefault(rate, {}).setdefault(int(lim), [[], 0])
            e[0].append(float(r["avg_ts"]))
            e[1] = _tokens(r)
    return {k: {q: {lim: (statistics.median(v[0]), v[1]) for lim, v in x.items()} for q, x in s.items()}
            for k, s in acc.items()}


def measured(d: Path) -> dict:
    """{(model, shape): {rate: median tok/s}} over bench-<model>-<id>-r<k>.json (not the A/A run)."""
    acc: dict = {}
    for f in sorted(d.glob("bench-*-r*.json")):
        m = re.fullmatch(r"bench-(.+)-r\d+\.json", f.name)
        if not m:
            continue
        stem = m.group(1)
        for model in models(d):
            if stem.startswith(model + "-"):
                for r in _rows(f):
                    rate = _rate(r)
                    if rate:
                        acc.setdefault((model, stem[len(model) + 1:]), {}).setdefault(rate, []).append(float(r["avg_ts"]))
    return {k: {q: statistics.median(v) for q, v in x.items()} for k, x in acc.items()}


def models(d: Path) -> list[str]:
    # longest first, so a model whose name extends another's is matched first
    return sorted((f.stem for f in (d / "layouts").glob("*.json")), key=len, reverse=True)


def n_layer(d: Path, model: str) -> int:
    return int(json.loads((d / "layouts" / f"{model}.json").read_text())["n_layer"])


def committed(d: Path) -> dict:
    """{(model, shape): {"pp": tok/s, "tg": tok/s}} of predict-<model>-<n>x.json."""
    out = {}
    for f in sorted(d.glob("predict-*.json")):
        m = re.fullmatch(r"predict-(.+)-\d+x\.json", f.name)
        if not m:
            continue
        for c in json.loads(f.read_text()).get("candidates", []):
            out[(m.group(1), c["id"])] = {"pp": c["pp_tps"], "tg": c["tg_tps"]}
    return out


def fit_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares (t_step, t_layer) through (L', seconds); needs two distinct L'."""
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    if n < 2 or sxx == 0:
        raise ValueError("a line needs anchors at two distinct layer counts")
    b = sum((p[0] - mx) * (p[1] - my) for p in points) / sxx
    return my - b * mx, b


def predict(pts: dict, at: int, holdout: bool = False) -> tuple[float, float, float, int] | None:
    """(predicted tok/s at L' = at, t_step, t_layer, number of points) from {L': (tok/s, tokens)}."""
    lims = sorted(pts)
    if holdout:
        lims = [x for x in lims if x != at]
    if len(lims) < 2:
        return None
    tokens = pts[lims[0]][1]
    a, b = fit_line([(lim, pts[lim][1] / pts[lim][0]) for lim in lims])
    t = a + b * at
    return (tokens / t if t > 0 else float("nan")), a, b, len(lims)


def evaluate(fit_dir: Path, eval_dir: Path, holdout: bool = False) -> list[dict]:
    anc = anchors(fit_dir)
    rows = []
    if holdout:
        # the largest truncation of every anchored shape is the target
        for (model, shape), rates in sorted(anc.items()):
            for rate, pts in sorted(rates.items()):
                top = max(pts)
                p = predict(pts, top, holdout=True)
                r = row(model, shape, rate, f"L{top}", p, pts[top][0], None)
                r["reason"] = None if p else "needs 3 truncations"
                rows.append(r)
        return rows
    meas, prior = measured(eval_dir), committed(eval_dir)
    for (model, shape), rates in sorted(meas.items()):
        for rate, m in sorted(rates.items()):
            pts = anc.get((model, shape), {}).get(rate, {})
            p = predict(pts, n_layer(eval_dir, model)) if pts else None
            r = row(model, shape, rate, "full", p, m, prior.get((model, shape), {}).get(rate))
            r["reason"] = None if p else ("unanchored" if not pts else "needs 2 truncations")
            rows.append(r)
    return rows


def row(model, shape, rate, at, p, meas, prior) -> dict:
    r = {"model": model, "shape": shape, "rate": rate, "at": at, "measured": meas, "committed": prior,
         "committed_err": (prior / meas - 1) if prior else None, "predicted": None, "err": None, "within": None,
         "t_step_s": None, "t_layer_s": None, "points": 0}
    if p:
        r.update(predicted=p[0], err=p[0] / meas - 1, t_step_s=p[1], t_layer_s=p[2], points=p[3])
        r["within"] = abs(r["err"]) <= BAND
    return r


def table(rows: list[dict]) -> str:
    out = ["model           shape    rate  at     committed   err    lines       err    measured  t_step ms  t_layer ms"]
    for r in rows:
        c = f"{r['committed']:9.1f}  {r['committed_err']:+.3f}" if r["committed"] else f"{'-':>9}  {'':6}"
        if r["predicted"] is None:
            p = f"{r.get('reason') or 'none':>16}"
            t = ""
        else:
            p = f"{r['predicted']:9.1f}  {r['err']:+.3f}"
            t = f"{r['t_step_s'] * 1e3:9.2f}  {r['t_layer_s'] * 1e3:10.3f}"
        out.append(f"{r['model']:15} {r['shape']:8} {r['rate']:4}  {r['at']:5}  {c}  {p}  {r['measured']:9.1f}  {t}")
    judged = [r for r in rows if r["predicted"] is not None]
    if judged:
        worst = max(judged, key=lambda r: abs(r["err"]))
        out.append(f"worst |err| {abs(worst['err']):.3f} ({worst['model']} {worst['shape']} {worst['rate']}); "
                   f"{sum(r['within'] for r in judged)}/{len(judged)} within {BAND}; "
                   f"{sum(r['predicted'] is None for r in rows)} without a line")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path, help="phase5 results directory with the anchors")
    ap.add_argument("--eval", type=Path, help="directory with the candidates (default: dir)")
    ap.add_argument("--holdout", action="store_true", help="predict each shape's largest truncation from the others")
    ap.add_argument("--json", type=Path, help="write the rows here")
    a = ap.parse_args(argv)
    rows = evaluate(a.dir, a.eval or a.dir, a.holdout)
    print(table(rows))
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
