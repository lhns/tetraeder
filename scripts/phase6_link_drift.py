#!/usr/bin/env python3
"""Drift of the link probes per size (ADR 0091 answer 10, report only), standard library only.

    phase6_link_drift.py start=probe-start.json mid-1=probe-mid-1.json end=probe-end.json [--out drift.json]

Prints, per direction and probe size, the one-way copy time of each probe and the change of every later probe against the
first, then the same at 32 MiB read as the cost model does (linear between points). Nothing is gated here.
"""

import argparse
import json


def curve_time(points, nbytes):
    pts = sorted((float(a), float(b)) for a, b in points)
    if nbytes <= pts[0][0]:
        return pts[0][1]
    if nbytes >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        return y1 + max((y1 - y0) / (x1 - x0), 0.0) * (nbytes - x1)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if nbytes <= x1:
            return y0 + (y1 - y0) * (nbytes - x0) / (x1 - x0)
    raise AssertionError("unreachable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("probes", nargs="+", help="NAME=FILE, the first is the reference")
    ap.add_argument("--out")
    a = ap.parse_args()
    probes = []
    for spec in a.probes:
        name, path = spec.split("=", 1)
        probes.append((name, json.load(open(path))))
    dirs = {}
    for name, p in probes:
        for c in p["curves"]:
            if c.get("op") == "copy" and len(c.get("ranks", [])) == 2:
                d = "->".join(r.split("/", 1)[0] for r in c["ranks"])
                dirs.setdefault(d, {})[name] = c["points"]
    rows = []
    for d, per in sorted(dirs.items()):
        ref = probes[0][0]
        sizes = sorted({int(x) for x, _ in per[ref]} | {32 * 1024 * 1024})
        for s in sizes:
            vals = {n: curve_time(per[n], s) for n in per}
            row = {"direction": d, "bytes": s, "interpolated": s not in {int(x) for x, _ in per[ref]}}
            for n in per:
                row[f"{n}_s"] = vals[n]
                if n != ref:
                    row[f"{n}_vs_{ref}"] = vals[n] / vals[ref] - 1.0
            rows.append(row)
    names = [n for n, _ in probes]
    print("direction  bytes        " + "  ".join(f"{n:>12}" for n in names) + "   drift vs " + names[0])
    for r in rows:
        t = "  ".join(f"{r[n + '_s'] * 1e3:10.2f}ms" for n in names if n + "_s" in r)
        dr = "  ".join(f"{n} {r[n + '_vs_' + names[0]] * 100:+6.1f}%" for n in names[1:] if n + "_vs_" + names[0] in r)
        print(f"{r['direction']:<9} {r['bytes']:>10}{'*' if r['interpolated'] else ' '}  {t}   {dr}")
    print("* interpolated (not a probe point)")
    empty = {n: p.get("empty_rtt_s") for n, p in probes}
    print("empty_rtt_s", json.dumps(empty))
    if a.out:
        json.dump({"rows": rows, "empty_rtt_s": empty}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
