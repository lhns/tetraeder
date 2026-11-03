"""Rebuilds the samples of a tetraeder-bench run from its log lines, for runs cut off before the harness wrote JSON.

Usage: python bench_log_to_results.py <log> --since HH:MM:SS [--until HH:MM:SS] --out results.json
Lines: `[HH:MM:SSZ] pair <id> rep <i>/<n> <a|b> <variant>: pp Some(x) tg Some(y) t/s` (None when the test failed).
Writes {"source": "log", "pairs": [{"id", "samples": [{"rep", "side", "variant", "pp_ts", "tg_ts"}]}]}, the subset of
the harness results the planner scorer reads.
"""
import argparse
import json
import re

LINE = re.compile(r"^\[(\d\d:\d\d:\d\d)Z\] pair (\S+) rep (\d+)/\d+ (a|b) ([^:]+): pp (Some\(([\d.e+-]+)\)|None) "
                  r"tg (Some\(([\d.e+-]+)\)|None)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--since", required=True)
    ap.add_argument("--until", default="99:99:99")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pairs = {}
    for line in open(a.log, encoding="utf-8", errors="replace"):
        m = LINE.match(line)
        if not m or not (a.since <= m.group(1) <= a.until):
            continue
        num = lambda g: float(g) if g else None
        pairs.setdefault(m.group(2), []).append({"rep": int(m.group(3)), "side": m.group(4), "variant": m.group(5),
                                                 "pp_ts": num(m.group(7)), "tg_ts": num(m.group(9))})
    json.dump({"source": "log", "pairs": [{"id": k, "samples": v} for k, v in pairs.items()]}, open(a.out, "w"), indent=1)
    print(sum(len(v) for v in pairs.values()), "samples in", len(pairs), "pairs")


if __name__ == "__main__":
    main()
