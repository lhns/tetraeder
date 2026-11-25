#!/usr/bin/env python3
"""Puts each node's fitted single-node graph into the cluster graph for `tetraeder-plan predict --cluster` (Case S, ADR 0091).

`cluster profile` writes the merged graph (ids `<node>/<id>`, JSON); `tetraeder-plan segment-predict --hw-out` writes each node's
anchor-fitted graph (ids `gpu0`, `cpu0`, `nvme0`, TOML) with the balloon's measured RAM budget in `cpu0.capacity_bytes`. No Rust tool
joins them, so this does, by the rule below. It is not checked by a Rust reader in CI: the first reader is `predict --cluster` on the
box, and the output keeps its input's keys, so a schema change fails there, not here.

    phase6_exit_s_merge.py HW.json --builder-disk PRIOR.toml --ram-budget a=BYTES --ram-budget b=BYTES -o hw-builder.json
    phase6_exit_s_merge.py hw-builder.json --node a=fitted-a.toml --node b=fitted-b.toml -o hw-cluster.json

The first form prepares the graph the candidates are enumerated on (what a judged session adds to a discovered graph, as
crates/planner/tests/cluster_discovered_ram_experts.rs does): discovery names disks `disk0..` and the planner reads a builder's experts
from `nvme0`, and without it homes every expert in RAM, which the 235B does not fit; so each node's discovered disks are replaced by an
`nvme0` taken from the committed NVMe prior (`crates/planner/fixtures/rtx3090_prior.toml`), and each node's CPU capacity is set to the
node's measured RAM budget (ADR 0092 section 4). The second form puts the fitted node graphs in, after the anchors.

Rule: a fitted device `X` replaces the fields of the cluster device `<node>/X` except its identity (`id`, `name`, `kind`, `tier`,
`node`, `index`, `numa_nodes`, `cores`) and, for a GPU, its capacity (the discovered one stays); the fitted disk `nvme0` goes to the
node's first disk device. A fitted link replaces alpha and beta of the cluster link of the same kind between the same two devices.
A device or link that has no counterpart is reported and left out. Everything the fit did not touch (the network links, the probe's
curves, the other nodes) is kept.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tomllib

KEEP = {"id", "name", "kind", "tier", "node", "index", "numa_nodes", "cores"}


def prepare(hw: dict, prior: dict, ram: dict[str, int]) -> tuple[dict, list[str]]:
    """Builder disks from the prior in place of each node's discovered disks; RAM budgets on the CPUs."""
    notes: list[str] = []
    nvme = next(d for d in prior["devices"] if d["id"] == "nvme0")
    link = next(x for x in prior["links"] if x["id"] == "nvme-cpu0")
    nodes = sorted({d["node"] for d in hw["devices"] if d.get("node")})
    for n in nodes:
        hw["devices"] = [d for d in hw["devices"] if not d["id"].startswith(f"{n}/disk")]
        hw["links"] = [x for x in hw["links"] if not x["id"].startswith(f"{n}/storage-disk")]
        disk = copy.deepcopy(nvme)
        disk.update(id=f"{n}/nvme0", node=n)
        hw["devices"].append(disk)
        lk = copy.deepcopy(link)
        lk.update(id=f"{n}/nvme-cpu0", a=f"{n}/nvme0", b=f"{n}/cpu0")
        hw["links"].append(lk)
    for d in hw["devices"]:
        if d.get("kind") == "cpu" and d.get("node") in ram:
            d["capacity_bytes"] = ram[d["node"]]
    notes += [f"{n}: no RAM budget given, the discovered capacity stays" for n in nodes if n not in ram]
    return hw, notes


def merge(hw: dict, fitted: dict[str, dict]) -> tuple[dict, list[str]]:
    notes: list[str] = []
    devs = {d["id"]: d for d in hw["devices"]}
    for node, f in fitted.items():
        disks = [d["id"] for d in hw["devices"] if d.get("node") == node and d.get("kind") == "disk"]
        idmap: dict[str, str | None] = {}
        for d in f.get("devices", []):
            target = f"{node}/{d['id']}"
            if target not in devs and d.get("kind") == "disk" and disks:
                target = disks[0]
            if target not in devs:
                notes.append(f"{node}: no device for fitted {d['id']}")
                idmap[d["id"]] = None
                continue
            idmap[d["id"]] = target
            dst = devs[target]
            for k, v in d.items():
                if k in KEEP or (k == "capacity_bytes" and dst.get("kind") == "gpu"):
                    continue
                dst[k] = v
        for lk in f.get("links", []):
            a, b = idmap.get(lk["a"]), idmap.get(lk["b"])
            hit = next((x for x in hw["links"] if x.get("kind") == lk.get("kind") and {x["a"], x["b"]} == {a, b}), None) if a and b else None
            if hit is None:
                notes.append(f"{node}: no link for fitted {lk.get('id')}")
                continue
            for k in ("alpha_s", "beta_bytes_per_s"):
                if k in lk:
                    hit[k] = lk[k]
    return hw, notes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hw")
    ap.add_argument("--node", action="append", default=[], metavar="ID=FITTED.toml")
    ap.add_argument("--builder-disk", metavar="PRIOR.toml", help="replace each node's discovered disks by the prior's nvme0")
    ap.add_argument("--ram-budget", action="append", default=[], metavar="ID=BYTES")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    if not a.node and not a.builder_disk:
        ap.error("--node or --builder-disk")
    with open(a.hw, encoding="utf-8") as f:
        hw = json.load(f)
    notes: list[str] = []
    if a.builder_disk:
        with open(a.builder_disk, "rb") as f:
            hw, notes = prepare(hw, tomllib.load(f), {k: int(v) for k, v in (x.split("=", 1) for x in a.ram_budget)})
    fitted = {}
    for spec in a.node:
        n, p = spec.split("=", 1)
        with open(p, "rb") as f:
            fitted[n] = tomllib.load(f)
    if fitted:
        hw, more = merge(hw, fitted)
        notes += more
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(hw, f, indent=2)
    for n in notes:
        print("note:", n)
    print(f"wrote {a.out}: builder disks {'yes' if a.builder_disk else 'no'}, {len(fitted)} fitted node graph(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
