#!/usr/bin/env python3
"""Summarise scripts/poll_bench.sh: per-arm medians and paired ratios against poll50 (ADR 0061).

usage: poll_bench_summary.py DIR [--out summary.json]
DIR holds manifest.jsonl ({id, round, arm, quota}) and per run harness-ID.json (exit_regime output), cg-ID.txt
("cpu.max ..." and "pre|post KEY VALUE" lines of the scope's cpu.stat) and time-ID.txt (GNU time "%w %c %e").
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

BASE = "poll50"
ARMS = ("poll50", "poll0", "pause")
METRICS = ("decode_tok_s", "compute_ms", "throttled_share", "ctxsw_voluntary_per_s")


def read_cg(path: Path) -> dict:
    if not path.exists():
        return {"quota": None, "throttled_share": None}
    snap: dict[str, dict[str, int]] = {"pre": {}, "post": {}}
    quota = None
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts[:1] == ["cpu.max"]:
            quota = " ".join(parts[1:]) or None
        elif len(parts) == 3 and parts[0] in snap:
            snap[parts[0]][parts[1]] = int(parts[2])
    d = {k: snap["post"].get(k, 0) - snap["pre"].get(k, 0) for k in snap["post"]}
    periods = d.get("nr_periods", 0)
    share = d.get("nr_throttled", 0) / periods if periods > 0 else None
    return {"quota": quota, "throttled_share": share}


def read_ctxsw(path: Path) -> float | None:
    """Voluntary context switches per second from GNU time's "%w %c %e" (the last non-empty line)."""
    if not path.exists():
        return None
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    try:
        vol, _invol, wall = lines[-1].split()[:3]
        return int(vol) / float(wall) if float(wall) > 0 else None
    except (IndexError, ValueError):
        return None


def load_run(d: Path, m: dict) -> dict:
    h = json.loads((d / f"harness-{m['id']}.json").read_text())["runs"][-1]
    cg = read_cg(d / f"cg-{m['id']}.txt")
    return {
        "id": m["id"],
        "round": m["round"],
        "arm": m["arm"],
        "quota": m["quota"],
        "decode_tok_s": h["decode_tok_s"],
        "compute_ms": h.get("decode_profile_per_token", {}).get("compute_ms"),
        "throttled_share": cg["throttled_share"],
        "ctxsw_voluntary_per_s": read_ctxsw(d / f"time-{m['id']}.txt"),
        "cpu_max": cg["quota"],
    }


def med(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def summarize(runs: list[dict]) -> dict:
    out: dict = {}
    for q in (True, False):
        sel = [r for r in runs if r["quota"] == q]
        if not sel:
            continue
        arms = {}
        for a in ARMS:
            ra = [r for r in sel if r["arm"] == a]
            if ra:
                arms[a] = {"n": len(ra), **{f"{k}_median": med([r[k] for r in ra]) for k in METRICS}}
        by_round: dict[int, dict[str, dict]] = {}
        for r in sel:
            by_round.setdefault(r["round"], {})[r["arm"]] = r
        ratios = {}
        for a in ARMS:
            if a == BASE:
                continue
            rs = [
                rd[a]["decode_tok_s"] / rd[BASE]["decode_tok_s"]
                for rd in by_round.values()
                if a in rd and BASE in rd and rd[BASE]["decode_tok_s"] > 0
            ]
            if rs:
                ratios[a] = {
                    "n": len(rs),
                    "median": statistics.median(rs),
                    "min": min(rs),
                    "max": max(rs),
                    "stdev": statistics.stdev(rs) if len(rs) > 1 else 0.0,
                    "rounds_above_1": sum(1 for x in rs if x > 1),
                }
        out["quota" if q else "no_quota"] = {"arms": arms, "paired_ratio_vs_" + BASE: ratios}
    return out


def render(s: dict) -> str:
    def f(x, p=3):
        return "-" if x is None else f"{x:.{p}f}"

    lines = []
    for grp in ("quota", "no_quota"):
        g = s.get(grp)
        if not g:
            continue
        lines.append(f"## {grp}")
        lines.append("| arm | n | tok/s | compute ms | throttled share | vol ctxsw/s |")
        lines.append("|---|---|---|---|---|---|")
        for a, v in g["arms"].items():
            lines.append(
                f"| {a} | {v['n']} | {f(v['decode_tok_s_median'])} | {f(v['compute_ms_median'], 1)} | "
                f"{f(v['throttled_share_median'])} | {f(v['ctxsw_voluntary_per_s_median'], 0)} |"
            )
        for a, r in g["paired_ratio_vs_" + BASE].items():
            lines.append(
                f"- {a} / {BASE}: median {f(r['median'])}x, range {f(r['min'])} to {f(r['max'])}, "
                f"stdev {f(r['stdev'])}, above 1 in {r['rounds_above_1']} of {r['n']} rounds"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    manifest = [json.loads(ln) for ln in (args.dir / "manifest.jsonl").read_text().splitlines() if ln.strip()]
    runs = [load_run(args.dir, m) for m in manifest]
    summary = {"runs": runs, **summarize(runs)}
    arms = summary.get("quota", {}).get("arms", {})
    shares = [v["throttled_share_median"] for v in arms.values() if v["throttled_share_median"] is not None]
    summary["quota_bound"] = bool(shares) and max(shares) >= 0.05
    if not summary["quota_bound"]:
        print("WARNING: the quota did not bind (no arm throttled in 5% of periods); the comparison says nothing", file=sys.stderr)
    (args.out or args.dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(render(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
