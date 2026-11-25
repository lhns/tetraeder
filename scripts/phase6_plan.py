#!/usr/bin/env python3
"""Writes the cluster plan (TOML) of one replica in two pipeline stages across two nodes (ADR 0091, split parity).

Layers 0..CUT on FIRST (the embedding too), CUT..N on SECOND (the head too); device ids are those of the merged
cluster graph (`a/gpu0`). Same shape as `two_stage_plan` of the pipeline tests.

    python3 scripts/phase6_plan.py --model qwen3-30b-a3b --layers 48 --cut 24 --first a/gpu0 --second b/gpu0 -o plan.toml
"""

import argparse


def stage(lo, hi, dev, kinds):
    s = [
        "[[replicas.stages]]",
        f"layers = [{lo}, {hi}]",
        f'devices = ["{dev}"]',
        "tp = 1",
        "ep = 1",
        "",
        "[[replicas.stages.assign]]",
        f"units = {{ layers = [{lo}, {hi}] }}",
        f'home = ["{dev}"]',
        f'compute = ["{dev}"]',
        "",
    ]
    for k in kinds:
        s += ["[[replicas.stages.assign]]", f'units = {{ kinds = ["{k}"] }}', f'home = ["{dev}"]', f'compute = ["{dev}"]', ""]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--cut", type=int, required=True)
    ap.add_argument("--first", required=True)
    ap.add_argument("--second", required=True)
    ap.add_argument("-o", required=True)
    a = ap.parse_args()
    out = [f'model = "{a.model}"', "", "[[replicas]]", ""]
    out += stage(0, a.cut, a.first, ["embed"])
    out += stage(a.cut, a.layers, a.second, ["head"])
    open(a.o, "w", newline="\n").write("\n".join(out))


if __name__ == "__main__":
    main()
