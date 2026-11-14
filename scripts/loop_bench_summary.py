#!/usr/bin/env python3
"""Summary of scripts/loop_bench.sh: per engine and test the median over repeats of throughput_tps and
decode_tps_median, and the ratio to llama-server. Writes summary.json and prints a table.

    python scripts/loop_bench_summary.py <results-dir>
"""

import glob
import json
import os
import re
import statistics
import sys

FIELDS = ("throughput_tps", "decode_tps_median")
BASE = "llama-server"
NAME = re.compile(r"^(?P<engine>.+)-(?P<test>c\d+)-r(?P<rep>\d+)\.json$")


def load_runs(d: str) -> dict:
    """{(engine, test): [summary dict per repeat]}."""
    runs: dict = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        m = NAME.match(os.path.basename(f))
        if not m:
            continue
        with open(f) as fh:
            s = json.load(fh).get("summary", {})
        runs.setdefault((m["engine"], m["test"]), []).append(s)
    return runs


def summarize(runs: dict) -> dict:
    out: dict = {}
    for (engine, test), ss in runs.items():
        e = out.setdefault(test, {}).setdefault(engine, {"runs": len(ss), "errors": sum(s.get("errors", 0) for s in ss)})
        for f in FIELDS:
            vals = [s[f] for s in ss if s.get(f) is not None]
            e[f] = statistics.median(vals) if vals else None
            e[f + "_all"] = vals
    for engines in out.values():
        base = engines.get(BASE)
        for engine, e in engines.items():
            if engine != BASE:
                e["ratio"] = {
                    f: (e[f] / base[f] if base and base.get(f) and e.get(f) is not None else None) for f in FIELDS
                }
    return out


def table(summary: dict) -> str:
    def fmt(v, p=2):
        return "n/a" if v is None else f"{v:.{p}f}"

    lines = [
        "| test | engine | runs | errors | throughput_tps | decode_tps_median | ratio thr | ratio dec |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for test in sorted(summary):
        for engine, e in sorted(summary[test].items(), key=lambda kv: (kv[0] != BASE, kv[0])):
            r = e.get("ratio", {})
            lines.append(
                f"| {test} | {engine} | {e['runs']} | {e['errors']} | {fmt(e['throughput_tps'])} | "
                f"{fmt(e['decode_tps_median'])} | {fmt(r.get('throughput_tps'), 3)} | {fmt(r.get('decode_tps_median'), 3)} |"
            )
    return "\n".join(lines)


def main(argv=None) -> int:
    d = (argv if argv is not None else sys.argv[1:])[0]
    summary = summarize(load_runs(d))
    if not summary:
        print("no results")
        return 1
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(table(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
