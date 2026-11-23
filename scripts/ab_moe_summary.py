#!/usr/bin/env python3
"""Summary of scripts/ab_moe_box.sh: B/A ratios per round, medians and the reading by the rule fixed in
bench/results/ab30b-20261122/README.md. Usage: ab_moe_summary.py <results dir> [--rounds N] [--models q30b,q8b].
Writes ab-summary.json (no --models: files ab-<side>-<test>-r<n>.json) or ab-summary-<model>.json per model (files
ab-<model>-<side>-...), and ab-ttft.json when the TTFT replays ab-ttft-A.json and ab-ttft-B.json exist."""

import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("phase4_summary", HERE / "phase4_summary.py")
p4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p4)

NO_REGRESSION = 0.98  # median B/A at least this for both metrics
REGRESSION = 0.97  # median B/A below this for either metric, every clean round below 1
TTFT_LINE = 1.05  # TTFT B/A of the median and of the p90 at most this: no regression, else reported
MIN_CLEAN = 2  # clean rounds needed for any reading but "inconclusive"
# metric -> (test, summary field)
METRICS = {"c4": ("c4", "throughput_tps"), "c1": ("c1", "decode_tps_median")}


def read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def snapshot(d: Path, side: str, rnd: int, prefix: str = "") -> dict:
    try:
        text = (d / f"gpu-{prefix}{side}-r{rnd}.txt").read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict = {}
    for line in text.splitlines():
        k, _, v = line.partition(":")
        if k in ("before", "after") and v.strip():
            out.setdefault(k, []).append(v.strip())
    return out


def run_record(d: Path, side: str, rnd: int, samples, prefix: str = "") -> dict:
    # ab30b names the files ab-<side>-<test>-r<n>.json, ab_moe_box.sh with models ab-<model>-<side>-r<n>-<test>.json
    res = {t: read(d / f"ab-{prefix}{side}-{t}-r{rnd}.json") or read(d / f"ab-{prefix}{side}-r{rnd}-{t}.json") for t in ("c4", "c1")}
    reasons = []
    vals = {}
    for t, j in res.items():
        if j is None:
            reasons.append(f"no {t} result")
            continue
        s = j.get("summary") or {}
        if s.get("errors") != 0:
            reasons.append(f"{t}: errors={s.get('errors')}")
        if j.get("cotenant_ok") is False:
            reasons.append(f"{t}: co-tenant ({j.get('cotenant_reason', '?')})")
        vals[t] = s.get(METRICS[t][1])
        if not isinstance(vals[t], (int, float)) or vals[t] <= 0:
            reasons.append(f"{t}: no {METRICS[t][1]}")
    win = p4.window_stats(samples, [p4.result_window(j) for j in res.values()])
    if win is None:
        reasons.append("no sampler data in the run's window")
    elif win["disqualifying"]:
        reasons.append(f"{win['disqualifying']} window sample(s) with throttle bits {win['bad_bits']}")
    return {"clean": not reasons, "reasons": reasons, "c4_tps": vals.get("c4"), "c1_decode_tps": vals.get("c1"),
            "sampler": win, "snapshots": snapshot(d, side, rnd, prefix)}


def reading(ratios: dict[str, list[float]], n_clean: int) -> tuple[str, str]:
    """(reading, why) from the clean rounds' ratios per metric."""
    if n_clean < MIN_CLEAN:
        return "inconclusive", f"{n_clean} clean round(s), at least {MIN_CLEAN} needed"
    med = {m: statistics.median(v) for m, v in ratios.items()}
    if all(x >= NO_REGRESSION for x in med.values()):
        return "no regression evidence", f"both medians at least {NO_REGRESSION}"
    low = [m for m, x in med.items() if x < REGRESSION and all(r < 1 for r in ratios[m])]
    if low:
        return "regression", f"median below {REGRESSION} with every clean round below 1 for {', '.join(low)}"
    return "inconclusive", "between the lines, or a metric below the line with a round at or above 1"


def metric_reading(ratios: list[float]) -> str:
    """The rule for one metric (model and concurrency): the median of its clean rounds."""
    if len(ratios) < MIN_CLEAN:
        return "inconclusive"
    med = statistics.median(ratios)
    if med >= NO_REGRESSION:
        return "no regression evidence"
    if med < REGRESSION and all(r < 1 for r in ratios):
        return "regression"
    return "inconclusive"


def summarize(d: Path, rounds: int = 3, prefix: str = "") -> dict:
    samples = p4.load_samples(d)
    out_rounds, runs = [], {}
    for n in range(1, rounds + 1):
        rec = {s: run_record(d, s, n, samples, prefix) for s in "AB"}
        runs.update({f"{s}-r{n}": r for s, r in rec.items()})
        row = {"round": n, "clean": rec["A"]["clean"] and rec["B"]["clean"]}
        for m, key in (("c4", "c4_tps"), ("c1", "c1_decode_tps")):
            a, b = rec["A"][key], rec["B"][key]
            row[f"{m}_A"], row[f"{m}_B"] = a, b
            row[f"{m}_ratio"] = b / a if a and b else None
        sa, sb = rec["A"]["sampler"], rec["B"]["sampler"]
        row["board_share_A"] = sa and sa["share"]
        row["board_share_B"] = sb and sb["share"]
        row["why_unclean"] = [f"{s}: {r}" for s in "AB" for r in rec[s]["reasons"]]
        out_rounds.append(row)
    clean = [r for r in out_rounds if r["clean"] and r["c4_ratio"] and r["c1_ratio"]]
    ratios = {m: [r[f"{m}_ratio"] for r in clean] for m in METRICS}
    med = {m: (statistics.median(v) if v else None) for m, v in ratios.items()}
    read_, why = reading(ratios, len(clean))
    return {"rounds": out_rounds, "clean_rounds": [r["round"] for r in clean],
            "excluded_rounds": [r["round"] for r in out_rounds if r not in clean],
            "median_ratio": med, "metric_readings": {m: metric_reading(v) for m, v in ratios.items()}, "reading": read_, "reading_why": why, "runs": runs}


def pctl(v: list[float], q: float) -> float:
    """Nearest rank, as gateway_replay.summarize."""
    v = sorted(v)
    return v[min(max(math.ceil(q * len(v)) - 1, 0), len(v) - 1)]


def ttft_summary(d: Path) -> dict | None:
    """B/A of the median and p90 TTFT of the request-log replays, over the streamed requests that returned 200 on both
    sides (same requests), plus the replays' own summaries (all ok requests)."""
    res = {s: read(d / f"ab-ttft-{s}.json") for s in "AB"}
    if not all(res.values()):
        return None
    by = {s: {r["index"]: r for r in res[s]["results"] if r.get("status") == 200 and r.get("ttft_s") is not None} for s in "AB"}
    common = sorted(set(by["A"]) & set(by["B"]))
    out: dict = {"common_requests": len(common), "own": {s: res[s]["summary"] for s in "AB"}}
    if common:
        t = {s: [by[s][i]["ttft_s"] for i in common] for s in "AB"}
        for s in "AB":
            out[s] = {"median_s": statistics.median(t[s]), "p90_s": pctl(t[s], 0.9)}
        out["ratio_median"] = out["B"]["median_s"] / out["A"]["median_s"]
        out["ratio_p90"] = out["B"]["p90_s"] / out["A"]["p90_s"]
        errs = {s: res[s]["summary"]["errors"] for s in "AB"}
        ok = not any(errs.values()) and max(out["ratio_median"], out["ratio_p90"]) <= TTFT_LINE
        out["reading"] = "no regression" if ok else "report"
        out["reading_why"] = (f"median and p90 B/A at most {TTFT_LINE}, no errors" if ok else
                              f"median {out['ratio_median']:.3f}, p90 {out['ratio_p90']:.3f} (line {TTFT_LINE}), errors {errs}")
    return out


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    rounds = int(a[a.index("--rounds") + 1]) if "--rounds" in a else 3
    models = a[a.index("--models") + 1].split(",") if "--models" in a else None
    d = Path(a[0])
    brief = {}
    for m in models or [None]:
        s = summarize(d, rounds, f"{m}-" if m else "")
        name = f"ab-summary-{m}.json" if m else "ab-summary.json"
        (d / name).write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        brief[m or "all"] = {k: s[k] for k in ("clean_rounds", "excluded_rounds", "median_ratio", "metric_readings", "reading", "reading_why")}
    t = ttft_summary(d)
    if t:
        (d / "ab-ttft.json").write_text(json.dumps(t, indent=2) + "\n", encoding="utf-8")
        brief["ttft"] = {k: t.get(k) for k in ("common_requests", "ratio_median", "ratio_p90", "reading")}
    print(json.dumps(brief if models else brief["all"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
