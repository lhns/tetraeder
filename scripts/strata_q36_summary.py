#!/usr/bin/env python3
"""Summary of scripts/strata_q36_box.sh (bench-strata-q36, ADR 0085): per round b/a and c/a for c4 throughput and c1
decode tok/s, medians over all rounds, the throttle share per run, and the A/A of the own 30B engine.
Usage: strata_q36_summary.py <results dir> [--rounds N]. Writes summary.json. No verdict: no threshold applies."""

import importlib.util
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("phase4_summary", HERE / "phase4_summary.py")
p4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p4)

ENGINES = {"a": "llama-server", "b": "tetraeder auto", "c": "strata"}
METRICS = {"c4": "throughput_tps", "c1": "decode_tps_median"}


def read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run_record(d: Path, run: str, samples) -> dict:
    res = {t: read(d / f"bench-{run}-{t}.json") for t in METRICS}
    notes, vals = [], {}
    for t, j in res.items():
        if j is None:
            notes.append(f"no {t} result")
            continue
        s = j.get("summary") or {}
        if s.get("errors"):
            notes.append(f"{t}: errors={s['errors']}")
        vals[t] = s.get(METRICS[t])
        vals[t + "_tokens_per_request"] = (s.get("completion_tokens") or 0) / max(s.get("requests") or 1, 1)
        if not isinstance(vals[t], (int, float)) or vals[t] <= 0:
            notes.append(f"{t}: no {METRICS[t]}")
    win = p4.window_stats(samples, [p4.result_window(j) for j in res.values()])
    if win is None:
        notes.append("no sampler data in the run's window")
    elif win["disqualifying"]:
        notes.append(f"{win['disqualifying']} window sample(s) with throttle bits {win['bad_bits']}")
    snaps = {}
    try:
        for line in (d / f"gpu-{run}.txt").read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition(":")
            if k in ("before", "after") and v.strip():
                snaps.setdefault(k, []).append(v.strip())
    except OSError:
        pass
    return {"c4_tps": vals.get("c4"), "c1_decode_tps": vals.get("c1"),
            "c4_tokens_per_request": vals.get("c4_tokens_per_request"),
            "c1_tokens_per_request": vals.get("c1_tokens_per_request"),
            "sampler": win, "snapshots": snaps, "notes": notes}


def ratio(x, y):
    return x / y if x and y else None


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def summarize(d: Path, rounds: int) -> dict:
    samples = p4.load_samples(d)
    rows, runs = [], {}
    for n in range(1, rounds + 1):
        rec = {e: run_record(d, f"{e}-r{n}", samples) for e in ENGINES}
        runs.update({f"{e}-r{n}": r for e, r in rec.items()})
        row = {"round": n}
        for m, key in (("c4", "c4_tps"), ("c1", "c1_decode_tps")):
            a, b, c = (rec[e][key] for e in "abc")
            row.update({f"{m}_a": a, f"{m}_b": b, f"{m}_c": c, f"{m}_b_over_a": ratio(b, a), f"{m}_c_over_a": ratio(c, a)})
        row["board_share"] = {e: rec[e]["sampler"] and rec[e]["sampler"]["share"] for e in ENGINES}
        rows.append(row)
    medians = {k: med([r[k] for r in rows]) for k in ("c4_b_over_a", "c4_c_over_a", "c1_b_over_a", "c1_c_over_a")}
    aa = {}
    for t in METRICS:
        v = [(read(d / f"bench-aa-r{r}-{t}.json") or {}).get("summary", {}).get(METRICS[t]) for r in (1, 2)]
        if all(x for x in v):
            aa[t] = {"r1": v[0], "r2": v[1], "r2_over_r1": v[1] / v[0]}
    aa_runs = {f"aa-r{r}": run_record(d, f"aa-r{r}", samples) for r in (1, 2)} if aa else {}
    return {"rounds": rows, "median_ratio": medians, "runs": runs, "aa_30b": aa, "aa_runs": aa_runs,
            "baseline_variant": (d / "baseline-variant.txt").read_text().strip() if (d / "baseline-variant.txt").exists() else None}


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    rounds = int(a[a.index("--rounds") + 1]) if "--rounds" in a else 3
    d = Path(a[0])
    s = summarize(d, rounds)
    (d / "summary.json").write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    f = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    g = lambda x: "-" if x is None else f"{x:.1f}"  # noqa: E731
    print("round  c4 a / b / c tok/s      c4 b/a  c4 c/a   c1 a / b / c decode tok/s   c1 b/a  c1 c/a")
    for r in s["rounds"]:
        print(f"{r['round']:>5}  {g(r['c4_a'])} / {g(r['c4_b'])} / {g(r['c4_c'])}   {f(r['c4_b_over_a'])}  {f(r['c4_c_over_a'])}   "
              f"{g(r['c1_a'])} / {g(r['c1_b'])} / {g(r['c1_c'])}   {f(r['c1_b_over_a'])}  {f(r['c1_c_over_a'])}")
    m = s["median_ratio"]
    print(f"median                                   {f(m['c4_b_over_a'])}  {f(m['c4_c_over_a'])}                                  "
          f"{f(m['c1_b_over_a'])}  {f(m['c1_c_over_a'])}")
    print("aa_30b:", json.dumps(s["aa_30b"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
