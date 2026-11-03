#!/usr/bin/env python3
"""Summarize a gateway box run (scripts/gateway_box.sh): paired ratios tetraeder / llama-server and API overhead.

Reads, from the results directory:
  bench-<model>-<engine>-<test>-r<rep>.json   scripts/gateway_bench.py output (test: c1 c4 p4096 mt)
  overhead-<model>-<tool>.json                `tetraeder bench` or `llama-bench -o json` rows
Writes summary.json. Ratios are taken per repetition (same box, same session) and then the median is reported;
never compare absolute tok/s across nights (AGENTS.md).

    python scripts/gateway_summary.py bench/results/20260101/gateway
"""

import json
import pathlib
import re
import statistics
import sys

NAME = re.compile(r"^bench-(?P<model>[^-]+)-(?P<engine>.+)-(?P<test>c1|c4|p4096|mt)-r(?P<rep>\d+)\.json$")
BASELINE = "llama-server"
METRICS = ("ttft_median_s", "decode_tps_median", "throughput_tps")


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def ratio(a, b):
    return None if a is None or not b else a / b


def load_runs(d: pathlib.Path) -> dict:
    """{model: {engine: {test: {rep: summary}}}}"""
    runs: dict = {}
    for p in sorted(d.glob("bench-*.json")):
        m = NAME.match(p.name)
        if not m:
            continue
        try:
            summary = json.loads(p.read_text())["summary"]
        except (ValueError, KeyError):
            continue
        g = m.groupdict()
        runs.setdefault(g["model"], {}).setdefault(g["engine"], {}).setdefault(g["test"], {})[int(g["rep"])] = summary
    return runs


def per_test(reps: dict) -> dict:
    """Median over repetitions of each metric of one engine and test."""
    out = {"reps": len(reps), "errors": sum(s.get("errors", 0) for s in reps.values())}
    keys = set().union(*(s.keys() for s in reps.values())) - {"requests", "errors", "wall_s", "completion_tokens"}
    for k in sorted(keys):
        vals = [s.get(k) for s in reps.values()]
        if all(not isinstance(v, list) for v in vals):
            out[k] = med(vals)
    # multi-turn: median TTFT per turn index
    turns = [s["ttft_per_turn_s"] for s in reps.values() if s.get("ttft_per_turn_s")]
    if turns:
        n = min(len(t) for t in turns)
        out["ttft_per_turn_s"] = [med([t[i] for t in turns]) for i in range(n)]
    return out


def paired_ratios(runs_a: dict, runs_b: dict) -> dict:
    """{test: {metric: median over common reps of a/b}}"""
    out = {}
    for test in sorted(set(runs_a) & set(runs_b)):
        reps = sorted(set(runs_a[test]) & set(runs_b[test]))
        row = {}
        for metric in METRICS + ("ttft_first_s", "ttft_later_median_s"):
            r = med([ratio(runs_a[test][i].get(metric), runs_b[test][i].get(metric)) for i in reps])
            if r is not None:
                row[metric] = r
        if row:
            out[test] = {**row, "pairs": len(reps)}
    return out


def bench_rate(rows, key: str):
    """avg_ts of the pp (n_prompt > 0) or tg (n_gen > 0) row of llama-bench JSON."""
    for r in rows if isinstance(rows, list) else []:
        if key == "pp" and r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0:
            return r.get("avg_ts")
        if key == "tg" and r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0:
            return r.get("avg_ts")
    return None


def load_json(p: pathlib.Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def overhead(model: str, engines: dict, d: pathlib.Path) -> dict:
    """Server decode rate at c=1 over the same engine's bench rate (1.0 = no API cost)."""
    out = {}
    for engine, tool in (("tetraeder", "tetraeder-bench"), ("tetraeder-llama", "tetraeder-bench"), (BASELINE, "llama-bench")):
        rows = load_json(d / f"overhead-{model}-{tool}.json")
        c1 = engines.get(engine, {}).get("c1")
        if rows is None or not c1:
            continue
        tg, pp = bench_rate(rows, "tg"), bench_rate(rows, "pp")
        entry = {"bench_tg_tps": tg, "bench_pp_tps": pp, "serve_decode_tps": c1.get("decode_tps_median")}
        entry["decode_ratio"] = ratio(entry["serve_decode_tps"], tg)
        # prefill through the API includes the template, tokenization and the first sampled token
        if c1.get("ttft_median_s") and c1.get("prompt_tokens_median"):
            entry["serve_prefill_tps"] = c1["prompt_tokens_median"] / c1["ttft_median_s"]
            entry["prefill_ratio"] = ratio(entry["serve_prefill_tps"], pp)
        out[engine] = entry
    return out


def summarize(d: pathlib.Path) -> dict:
    out = {"dir": str(d), "env": load_json(d / "env.json"), "models": {}}
    for model, engines in load_runs(d).items():
        eng = {e: {t: per_test(reps) for t, reps in tests.items()} for e, tests in engines.items()}
        raw = engines
        m = {"engines": eng, "ratios_vs_" + BASELINE: {}}
        if BASELINE in raw:
            for e in raw:
                if e != BASELINE:
                    m["ratios_vs_" + BASELINE][e] = paired_ratios(raw[e], raw[BASELINE])
        m["api_overhead"] = overhead(model, eng, d)
        out["models"][model] = m
    return out


def main(argv=None) -> int:
    d = pathlib.Path((argv or sys.argv[1:])[0])
    out = summarize(d)
    (d / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    for model, m in out["models"].items():
        for e, tests in m["ratios_vs_" + BASELINE].items():
            for t, row in tests.items():
                print(model, e, t, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
