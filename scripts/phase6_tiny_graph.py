#!/usr/bin/env python3
"""LOOPBACK=1 only (the CI dry run of Case S, scripts/phase6_exit_s.sh): the committed two-node cluster graph (two RTX 3090 nodes,
crates/costmodel/fixtures/cluster/cluster_two_3090_lan.toml) as JSON, without its network link, so that `tetraeder cluster merge --swap-link`
puts the real loopback probe's link in its place.

    phase6_tiny_graph.py FIXTURE.toml OUT.json

Why: a CI runner has no GPU, so `cluster profile` gives a graph without gpu0 and the pinned plan of the tiny model (a builder placement with VRAM
slots) cannot be priced on it. This graph is the third stand-in of the loopback run, with the made-up RAM budget and the GPU throttle line. The driver
never uses it on a box: there a missing GPU or prior must fail the dry run.
"""

import json
import sys
import tomllib


def main() -> int:
    src, dst = sys.argv[1:3]
    with open(src, "rb") as f:
        g = tomllib.load(f)
    g["links"] = [x for x in g.get("links", []) if x.get("kind") != "network"]
    g.setdefault("collectives", [])
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(g, f, indent=1)
    print(f"wrote {dst}: {len(g['devices'])} devices, {len(g['links'])} links, the network link left to the probe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
