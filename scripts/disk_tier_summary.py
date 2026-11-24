#!/usr/bin/env python3
"""TTFT and restore numbers of the prefix disk tier runs (scripts/sp_post.sh, stage disk).

    disk_tier_summary.py OUT_DIR SESSIONS
reads OUT_DIR/disk-<mode>.json (gateway_replay.py) and OUT_DIR/server-disk-<mode>.log (the `admit` lines of
tetraeder::step at debug level) for the modes `ram` (recompute) and `disk`, prints and writes OUT_DIR/disk-summary.json.
Requests are in turn-major order, so request i is turn i // SESSIONS + 1; turn 1 never has anything to restore.
"""

from __future__ import annotations

import json
import math
import re
import sys

ADMIT = re.compile(r"admit seq=\d+ .*?restore_ms=([0-9.]+) prompt=(\d+) common=(\d+) source=(\w+)")


def pct(v: list[float], q: float) -> float | None:
    v = sorted(v)
    return v[min(max(math.ceil(q * len(v)) - 1, 0), len(v) - 1)] if v else None


def one(out: str, mode: str, sessions: int) -> dict:
    rep = json.load(open(f"{out}/disk-{mode}.json", encoding="utf-8"))
    res = rep["results"]
    ok = [r for r in res if r.get("status") == 200 and r.get("ttft_s") is not None]
    later = [r["ttft_s"] for r in ok if r["index"] // sessions >= 1]
    first = [r["ttft_s"] for r in ok if r["index"] // sessions == 0]
    admits = []
    try:
        for line in open(f"{out}/server-disk-{mode}.log", encoding="utf-8", errors="replace"):
            m = ADMIT.search(re.sub(r"\x1b\[[0-9;]*m", "", line))
            if m:
                admits.append({"restore_ms": float(m[1]), "prompt": int(m[2]), "common": int(m[3]), "source": m[4]})
    except FileNotFoundError:
        pass
    by_source: dict[str, list[float]] = {}
    for a in admits:
        by_source.setdefault(a["source"], []).append(a["restore_ms"])
    return {
        "requests": len(res),
        "errors": len(res) - len(ok),
        "wall_s": rep["wall_s"],
        "ttft_all_median_s": pct([r["ttft_s"] for r in ok], 0.5),
        "ttft_all_p90_s": pct([r["ttft_s"] for r in ok], 0.9),
        "ttft_turn1_median_s": pct(first, 0.5),
        "ttft_later_turns_median_s": pct(later, 0.5),
        "ttft_later_turns_p90_s": pct(later, 0.9),
        "admits_by_source": {k: len(v) for k, v in sorted(by_source.items())},
        "restore_ms_by_source": {k: {"p50": pct(v, 0.5), "p90": pct(v, 0.9)} for k, v in sorted(by_source.items())},
    }


def main() -> None:
    out, sessions = sys.argv[1], int(sys.argv[2])
    doc = {mode: one(out, mode, sessions) for mode in ("ram", "disk")}
    json.dump(doc, open(f"{out}/disk-summary.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(doc, indent=1))


if __name__ == "__main__":
    main()
