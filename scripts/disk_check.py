#!/usr/bin/env python3
"""Reports a builder stage's decode disk rate against the session's loaded disk curve (ADR 0056).

  disk_check.py <exit-<stage>-<stamp>.json> <disk-curve-loaded-<stamp>.json>

The sampler's decode window has `read_mb_s` at `disk_queue_depth_mean`; the curve (`tetraeder-bench
disk-curve --load-threads`) gives the rate the disk delivers at that depth, interpolated as
`profiler::curve_rate` does. Prints both and whether the measured rate lies within 20% of the curve's. Reported,
not gated: the curve is one box's disk under one load, the stage is a whole engine.
"""
import json
import sys


def rate(curve, depth):
    pts = sorted(curve, key=lambda p: p["depth"])
    if depth <= pts[0]["depth"]:
        return pts[0]["bytes_per_s"] * depth / max(pts[0]["depth"], 1)
    if depth >= pts[-1]["depth"]:
        return pts[-1]["bytes_per_s"]
    for a, b in zip(pts, pts[1:]):
        if depth <= b["depth"]:
            w = (depth - a["depth"]) / (b["depth"] - a["depth"])
            return a["bytes_per_s"] + (b["bytes_per_s"] - a["bytes_per_s"]) * w
    return pts[-1]["bytes_per_s"]


def check(stage, curve_doc):
    d = (stage.get("host") or {}).get("decode") or {}
    mb_s, depth = d.get("read_mb_s"), d.get("disk_queue_depth_mean")
    if mb_s is None or depth is None:
        return {"ok": None, "why": "the stage has no sampler decode window (host.decode)"}
    want = rate(curve_doc["queue_curve"], depth) / 1e6
    ratio = mb_s / want if want else None
    return {
        "decode_read_mb_s": mb_s,
        "disk_queue_depth_mean": depth,
        "curve_mb_s": want,
        "ratio": ratio,
        "within_20_percent": ratio is not None and abs(ratio - 1) <= 0.2,
    }


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = check(json.load(open(argv[0])), json.load(open(argv[1])))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
