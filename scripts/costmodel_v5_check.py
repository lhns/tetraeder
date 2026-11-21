#!/usr/bin/env python3
"""The offline check of cost model 5 (ADR 0082, "The check"): version 4 and version 5 `e_pp`, `e_tg` of every candidate
of the committed sessions, one fit per session (the fit is the same in both versions; the layer anchor is held out).

    python scripts/costmodel_v5_check.py [--v2-dir DIR_OF_V2_PHASE5] [--out bench/results/s3c-20261119/costmodel-v5]

Clocks: host 20210 (`decode-clocks-host20210.json` beside the output) for the 4-GPU sessions s3, s3b, s3c; 1590 MHz for
both anchors of s3f2 (its own gpu-samples.csv); none for the 2-GPU sessions s2f, s2g and v2-20261120. Writes
tables.md, results.json and the version-5 graph of s3b (`fit-s3b.json`, for `crates/planner/tests/costmodel_v2.rs`).
Standard library only; the fits take about a minute each.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("costmodel_replica", ROOT / "scripts" / "costmodel_replica.py")
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

R = ROOT / "bench" / "results"
OUT = R / "s3c-20261119" / "costmodel-v5"
M30, M32 = "Qwen3 30B Gptq Fp16", "Qwen3 32B Awq Compatible Instruct"
S3F2_CLOCKS = {"low_mhz": 1590.0, "busy_mhz": {M30: 1590.0, M32: 1590.0}, "duty_threshold": 0.32}


def sessions(v2_dir: Path | None) -> list:
    host = json.loads((OUT / "decode-clocks-host20210.json").read_text())
    clocks = {k: host[k] for k in ("low_mhz", "busy_mhz", "duty_threshold")}
    out = [("s2f", R / "s2f-20261118" / "phase5", None), ("s2g", R / "s2g-20261118" / "phase5", None),
           ("s3", R / "s3-20261119" / "phase5", clocks), ("s3b", R / "s3b-20261119" / "phase5", clocks),
           ("s3c", R / "s3c-20261119" / "phase5", clocks), ("s3f2", R / "s3f2-20261120" / "phase5", S3F2_CLOCKS)]
    if v2_dir is not None:
        out.append(("v2", v2_dir, None))
    return out


def load_routing(ev: Path) -> None:
    cr.ROUTING.clear()
    for model in cr.models(ev):
        layout = cr.load_json(ev / "layouts" / f"{model}.json")
        f = ev / f"routing-{model}.json"
        if layout["n_expert"] > 0 and f.exists():
            cr.ROUTING[model] = cr.routing_tables(cr.load_json(f))


def run(name: str, d: Path, clocks: dict | None) -> dict:
    load_routing(d)
    anchors = cr.session_anchors(d, 5)
    hw, resid = cr.fit(cr.prior_hw(d), [a for a in anchors if not a.get("held")], 5)
    v4 = cr.predict_session(hw, d, 4)
    hw5 = json.loads(json.dumps(hw))
    if clocks:
        hw5["decode_clocks"] = clocks
    v5 = cr.predict_session(hw5, d, 5)
    meas = cr.measured(d)
    rows = []
    for a, b in zip(v4, v5):
        m = meas.get((a["model"], a["id"]), {})
        e = lambda x, q: (x[q] / m[q] - 1.0) if q in m else None  # noqa: E731
        rows.append({"model": a["model"], "id": a["id"], "pp_meas": m.get("pp"), "tg_meas": m.get("tg"),
                     "pp4": a["pp"], "tg4": a["tg"], "pp5": b["pp"], "tg5": b["tg"],
                     "e_pp4": e(a, "pp"), "e_pp5": e(b, "pp"), "e_tg4": e(a, "tg"), "e_tg5": e(b, "tg")})
    extra = {}
    if name == "s3f2":
        # out of sample: layer-4 truncated to 36 layers, tg128, measured by the session
        mf = d.parent / "qwen3-30b-a3b-layer-4-L36-tg128.json"
        layout = cr.load_json(d / "layouts" / "qwen3-30b-a3b.json")
        plan = cr.plans(d, "qwen3-30b-a3b")["layer-4"]
        meas36 = next(r["avg_ts"] for r in cr.load_json(mf) if r.get("n_gen", 0) > 0)
        f = cr.features(layout, plan, 36, 5)
        extra = {"l36_meas": meas36, "l36_v4": cr.predict(hw, f, cr.PROMPT, cr.GEN, 4)["tg"],
                 "l36_v5": cr.predict(hw5, f, cr.PROMPT, cr.GEN, 5)["tg"]}
    return {"name": name, "dir": d.as_posix(), "clocks": clocks, "rows": rows, "extra": extra, "hw": hw5,
            "residuals": resid}


def fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:+.{nd}f}"


def table(res: list) -> str:
    out = ["| session | model | candidate | pp meas | tg meas | e_pp v4 | e_pp v5 | e_tg v4 | e_tg v5 | moved |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in res:
        for r in s["rows"]:
            moved = max(abs((r["pp5"] / r["pp4"]) - 1.0), abs((r["tg5"] / r["tg4"]) - 1.0))
            short = "30B" if r["model"] == "qwen3-30b-a3b" else "32B"
            out.append(f"| {s['name']} | {short} | {r['id']} | {r['pp_meas'] or 0:.0f} | {r['tg_meas'] or 0:.1f} | "
                       f"{fmt(r['e_pp4'])} | {fmt(r['e_pp5'])} | {fmt(r['e_tg4'])} | {fmt(r['e_tg5'])} | "
                       f"{moved:.4f} |")
    return "\n".join(out) + "\n"


def ratios(res: list) -> str:
    out = ["| session | measured layer-4 / single tg | v4 predicted | v5 predicted | v5 error | e_tg v4 | e_tg v5 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for s in res:
        if not s["clocks"] or s["name"] == "s3f2":
            continue
        by = {r["id"]: r for r in s["rows"] if r["model"] == "qwen3-30b-a3b"}
        l4, one = by["layer-4"], by["single"]
        meas = l4["tg_meas"] / one["tg_meas"]
        p4, p5 = l4["tg4"] / one["tg4"], l4["tg5"] / one["tg5"]
        out.append(f"| {s['name']} | {meas:.3f} | {p4:.3f} | {p5:.3f} | {p5 - meas:+.3f} | {fmt(l4['e_tg4'])} | "
                   f"{fmt(l4['e_tg5'])} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v2-dir", type=Path, help="bench/results/v2-20261120/phase5 of branch box-v2-20261120")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    res = [run(n, d, c) for n, d, c in sessions(a.v2_dir)]
    for s in res:
        print(s["name"], "done", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    text = "## v4 and v5 per candidate\n\n" + table(res) + "\n## 30B layer-4 against single\n\n" + ratios(res)
    for s in res:
        if s["extra"]:
            x = s["extra"]
            text += (f"\n## {s['name']} out of sample: 30B layer-4 truncated to 36 layers, tg128\n\n"
                     f"measured {x['l36_meas']:.1f}, v4 {x['l36_v4']:.1f}, v5 {x['l36_v5']:.1f}\n")
    (a.out / "tables.md").write_text(text, newline="\n")
    (a.out / "results.json").write_text(json.dumps(
        [{k: v for k, v in s.items() if k != "hw"} for s in res], indent=1) + "\n", newline="\n")
    s3b = next(s for s in res if s["name"] == "s3b")
    # the version-5 graph and rows of s3b for Rust (`predict` must reproduce them)
    rows = [{"model": r["model"], "id": r["id"], "pp": r["pp5"], "tg": r["tg5"]} for r in s3b["rows"]]
    (a.out / "fit-s3b.json").write_text(json.dumps(
        {"version": 5, "ar_form": cr.AR_FORM, "fit": s3b["dir"].replace(str(ROOT).replace("\\", "/") + "/", ""),
         "eval": s3b["dir"].replace(str(ROOT).replace("\\", "/") + "/", ""), "rows": rows, "hw": s3b["hw"]},
        indent=1) + "\n", newline="\n")
    print(text)
    bad = [r for s in res for r in s["rows"] if not all(math.isfinite(r[k]) for k in ("pp5", "tg5"))]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
