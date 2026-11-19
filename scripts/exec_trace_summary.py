"""Summarizes an executor trace (TETRAEDER_EXEC_TRACE, ADR 0069 History 2026-11-19): the steps of the most common
event sequence (decode steps of a tg run), the median host microseconds of each event, and the median per kind.

    python3 scripts/exec_trace_summary.py trace.jsonl
"""

import collections
import json
import statistics
import sys


def summarize(lines):
    steps = [json.loads(ln) for ln in lines if ln.strip()]
    by_sig = collections.defaultdict(list)
    for s in steps:
        by_sig[tuple((e[0], e[1], e[2]) for e in s["ev"])].append(s)
    if not by_sig:
        return {"steps": 0, "events": [], "kinds": {}, "total_us": 0.0}
    sig, group = max(by_sig.items(), key=lambda kv: len(kv[1]))
    events = []
    for i, (kind, rank, peer) in enumerate(sig):
        us = statistics.median(s["ev"][i][3] for s in group) / 1e3
        events.append({"kind": kind, "rank": rank, "peer": peer, "us": us})
    kinds = collections.defaultdict(float)
    for e in events:
        kinds[e["kind"]] += e["us"]
    return {"steps": len(group), "sync": group[0]["sync"], "events": events, "kinds": dict(kinds),
            "total_us": sum(kinds.values())}


def main(path):
    with open(path) as f:
        s = summarize(f.readlines())
    print(f"{path}: {s['steps']} steps of the most common sequence, sync={s.get('sync')}")
    for e in s["events"]:
        peer = f">{e['peer']}" if e["peer"] >= 0 else ""
        print(f"  {e['kind']:10} rank {e['rank']}{peer:3} {e['us']:9.1f} us")
    for k, us in s["kinds"].items():
        print(f"  sum {k:10} {us:9.1f} us")
    print(f"  sum all        {s['total_us']:9.1f} us")


if __name__ == "__main__":
    main(sys.argv[1])
