#!/usr/bin/env python3
"""Summary of the q36 stage of scripts/gate_extra_box.sh: per run and test the bench summary, requests with fewer than
128 tokens and the distinct error texts. Usage: gate_extra_summary.py <dir> <rounds>; prints JSON."""

import json
import sys
from pathlib import Path


def summarize(d: Path, rounds: int, max_tokens: int = 128) -> dict:
    out: dict = {}
    for n in range(1, rounds + 1):
        for t in ("c4", "c1"):
            try:
                j = json.loads((d / f"bench-q36-default-r{n}-{t}.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                out[f"r{n}-{t}"] = None
                continue
            rs = j["results"]
            out[f"r{n}-{t}"] = {
                **j["summary"],
                "short_requests": sum(1 for r in rs if not r.get("error") and r.get("completion_tokens", 0) < max_tokens),
                "error_samples": sorted({r["error"][:120] for r in rs if r.get("error")}),
            }
    return out


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1]), int(sys.argv[2])), indent=2))
