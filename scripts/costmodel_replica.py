#!/usr/bin/env python3
"""Offline replica of the Phase 5 cost model (`costmodel::predict` ranks path, `planner::multi::features_multi`,
`costmodel::calib::fit`), for checks on committed session files without a Rust build (ADR 0070).

    python scripts/costmodel_replica.py DIR                       refit on DIR's anchors, predict DIR's candidates
    python scripts/costmodel_replica.py DIR --eval EVAL_DIR       fit on DIR's anchors, judge EVAL_DIR's candidates
    python scripts/costmodel_replica.py DIR --version 1           the model of MODEL_VERSION 1 (as committed in S2f)
    python scripts/costmodel_replica.py DIR --committed           version 1 on DIR's own calib.json (no refit)
    python scripts/costmodel_replica.py DIR --json OUT.json       also write the fitted graph and the rows
    python scripts/costmodel_replica.py DIR --ar-form scale       the all-reduce time times one factor (not adopted)
    python scripts/costmodel_replica.py DIR --drop roofline       version 2 without one fix (per-fix findings)
    python scripts/costmodel_replica.py DIR --skew 0.68           synthetic routing (offline only, see --skew)

Routing (ADR 0070): expert parallel is priced from DIR's (or EVAL_DIR's) routing-<model>.json when it exists, as
`tetraeder-plan predict --routing` does: `prefill` sets the busiest rank's share of a prefill step's routed rows,
`decode` the Monte Carlo of a decode step. Anchors never read it (they cut experts by columns).

Version 1 matches the session's `predict-*.json` to the printed digit (tests/test_costmodel_replica.py). Version 2 is
MODEL_VERSION 2 (ADR 0070, History 2026-11-18): expert matmuls at their own rate, the head (and the dense builder's
last FFN) on output rows, every expert read at prefill, a roofline per op class, `single` on the ranks path, and an
all-reduce in a step paying the fitted share of the probe's zero-byte time. Version 3 is MODEL_VERSION 3 (ADR 0078):
the all-reduce in a step pays the curve's byte part plus a fitted in-step latency a + 2 (p - 1) b over p ranks, b only
when the anchors' all-reduces span two group sizes. Version 4 (the default) is MODEL_VERSION 4 (ADR 0080): version 3
plus the routed slots' activation bytes at the memory rate, every slot on every expert-parallel rank; Rust prices them
on graphs that carry `expert_slot_traffic`, which a version-4 fit writes. Pipelined prefill (ADR 0077) is priced as `predict_ranks` does; version 2 on a session's own calib.json
gives its predict-*.json bit for bit (S2g, S3, S3b). Rust `predict` reproduces version 2 on the graphs this script
writes (`crates/planner/tests/costmodel_v2.rs`). The fit is the same Levenberg-Marquardt and coordinate search as
`calib.rs`, not bit-identical to it. Input files (ADR 0068 names): hw.json, calib-comm.json, calib.json,
layouts/<model>.json, candidates-<model>.json, anchors-<model>-r<k>.json, bench-<model>-<id>-r<k>.json, and on a
4-GPU box plans-anchor/candidates-<model>.json (the anchors' plans). Standard library only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import statistics
from pathlib import Path

PROMPT, GEN = 2048, 128
HOST = "cpu0"
OPS_ATTN, OPS_FFN, OPS_EXPERT, OPS_OUTPUT = 14.0, 10.0, 10.0, 4.0
# score::features (the pre-ranks single path of version 1)
OPS_DENSE_LAYER, OPS_EXPERT_LAYER = 24.0, 10.0
EP_SEED, EP_STEPS = 20_261_115, 1000
RANGE_PRIOR = {"c_range_graph_s": 3.0e-5, "c_range_eager_s": 1.0e-5, "c_step_s": 2.0e-4, "allreduce_latency_share": 1.0}
# calib.rs priors of the all-reduce's in-step latency a and its latency per ring step b (version 3, ADR 0078)
ALLREDUCE_LATENCY_PRIOR_S, ALLREDUCE_STEP_PRIOR_S = 1.2e-5, 1e-6
FORMATS = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q8_0": (32, 34), "Q6_K": (256, 210), "Q5_K": (256, 176),
           "Q4_K": (256, 144), "Q4_0": (32, 18)}
M64 = (1 << 64) - 1
# "offset" (MODEL_VERSION 2) or "scale": the brief's literal form, the whole all-reduce time times s (--ar-form)
AR_FORM = "offset"
# Version 2 with single fixes left out (--drop), for the per-fix findings: expert_rate, head_rows, prefill_experts,
# roofline, single_ranks, ar_share, link
DROP: set = set()
# hw.json writes `all_reduce` and `copy`, calib-comm.json `AllReduce` and `Copy`
# Per model key: {"prefill": [[f]], "decode": [[f]]}, each layer scaled to mean 1 (`multi::Routing::from_counts`)
ROUTING: dict = {}
# Monte Carlo results by (frequencies, top_k, ranks): a synthetic table repeats one row per layer
_MC: dict = {}
OPS = {"all_reduce": "AllReduce", "copy": "Copy"}
# multi.rs: activation bytes of a routed slot per n_ff_exp column and per n_embd row, the sum per added slot
SLOT_FF_BYTES, SLOT_EMBD_BYTES, SLOT_SUM_BYTES = 25.125, 22.25, 12.0


# ---- hardware graph ---------------------------------------------------------------------------------------------

def device(hw: dict, i: str) -> dict:
    return next(d for d in hw["devices"] if d["id"] == i)


def link_between(hw: dict, a: str, b: str) -> tuple[float, float]:
    for l in hw["links"]:
        if (l["a"] == a and l["b"] == b) or (l["a"] == b and l["b"] == a):
            beta = l["beta_bytes_per_s"]["value"]
            g = l.get("lane_group")
            if g:
                for lg in hw.get("lane_groups", []):
                    if lg["id"] == g:
                        beta = min(beta, lg["capacity_bytes_per_s"])
            return l["alpha_s"]["value"], beta
    raise KeyError(f"no link {a}-{b}")


def curve_time(points: list, n: float) -> float:
    pts = sorted(points, key=lambda p: p[0])
    first, last = pts[0], pts[-1]
    if n <= first[0]:
        return first[1]
    if n >= last[0]:
        if len(pts) == 1:
            return last[1] * n / last[0]
        prev = pts[-2]
        slope = max((last[1] - prev[1]) / (last[0] - prev[0]), 0.0)
        return last[1] + slope * (n - last[0])
    i = next(k for k, p in enumerate(pts) if n <= p[0])
    a, b = pts[i - 1], pts[i]
    return a[1] + (b[1] - a[1]) * (n - a[0]) / (b[0] - a[0])


def collective_time(hw: dict, op: str, ranks: list, n: float) -> float | None:
    for c in hw.get("collectives", []):
        if OPS.get(c["op"], c["op"]) != op:
            continue
        same = (len(c["ranks"]) == len(ranks) and all(r in c["ranks"] for r in ranks)) if op == "AllReduce" \
            else c["ranks"] == ranks
        if same:
            return curve_time(c["points"], n)
    return None


def range_costs(hw: dict) -> dict:
    rc = dict(RANGE_PRIOR)
    rc.update(hw.get("range_costs") or {})
    return rc


def rates(hw: dict, dev: str, graphs: bool = True) -> tuple[float, float, float, float]:
    """(F, F_exp, B, c0) of a GPU; F_exp falls back to F."""
    d = device(hw, dev)
    c = d["compute"]
    f = c["flops_per_s"]
    return f, c.get("expert_flops_per_s") or f, d["bandwidth_bytes_per_s"]["value"], \
        c["c0_graph_s"] if graphs else c["c0_eager_s"]


# ---- layout -------------------------------------------------------------------------------------------------------

def main_format(layout: dict) -> tuple[int, int]:
    name = max(layout["tensor_types"].items(), key=lambda kv: kv[1]["bytes"])[0]
    return FORMATS.get(name, (1, 2))


def bpp(layout: dict) -> float:
    n, b = main_format(layout)
    return ((b * 8) / n) / 8.0


def block_slice(length: int, block: int, n: int, i: int, rot: int) -> tuple[int, int]:
    assert length % block == 0
    blocks = length // block
    base, extra = blocks // n, blocks % n
    assert base > 0, f"{blocks} blocks cannot give {n} slices one block each"
    rot = rot % n

    def blocks_of(r: int) -> int:
        return base + (1 if (r + n - rot) % n < extra else 0)

    start = sum(blocks_of(r) for r in range(i))
    return start * block, (start + blocks_of(i)) * block


def splitmix(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & M64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return state, z ^ (z >> 31)


def expected_max_local(freq: list, top_k: int, ranks: int, steps: int, seed: int) -> float:
    e = len(freq)
    per = max(e // ranks, 1)
    state, total_sum = seed, 0.0
    for _ in range(steps):
        w = list(freq)
        total = 0.0
        for x in w:
            total += x
        count = [0] * ranks
        for _ in range(min(top_k, e)):
            state, r = splitmix(state)
            u = (r >> 11) * (1.0 / 9_007_199_254_740_992.0)
            target = u * total
            pick = None
            for i, x in enumerate(w):
                if x <= 0.0:
                    continue
                if target < x:
                    pick = i
                    break
                target -= x
            if pick is None:
                pos = [i for i, x in enumerate(w) if x > 0.0]
                if not pos:
                    break
                pick = pos[-1]
            total -= w[pick]
            w[pick] = 0.0
            count[min(pick // per, ranks - 1)] += 1
        total_sum += float(max(count))
    return total_sum / steps


def expected_max_local_cached(freq: list, top_k: int, ranks: int) -> float:
    key = (tuple(freq), top_k, ranks)
    if key not in _MC:
        _MC[key] = expected_max_local(freq, top_k, ranks, EP_STEPS, EP_SEED)
    return _MC[key]


def routing_tables(counts: dict) -> dict:
    """A routing-<model>.json as frequencies: each layer scaled to mean 1, equal counts giving exactly 1.0."""
    def scale(t):
        out = []
        for row in t:
            total = sum(row)
            if total == 0:
                raise ValueError("a routing layer without selections")
            out.append([c * len(row) / total for c in row])
        return out
    return {"prefill": scale(counts["prefill"]), "decode": scale(counts["decode"])}


def synthetic_routing(layout: dict, share: float) -> dict:
    """Offline only, no recorded routing: the first n_expert / 4 experts (rank 0 of ep-4, inside rank 0 of ep-2) hold
    `share` of every layer's selections, the rest the remainder evenly; prefill and decode alike."""
    e = layout["n_expert"]
    hot = e // 4
    row = [share / hot if i < hot else (1.0 - share) / (e - hot) for i in range(e)]
    row = [x * e for x in row]
    t = [list(row) for _ in range(layout["n_layer"])]
    return {"prefill": t, "decode": [list(row) for _ in t]}


def max_mass(freq: list, ranks: int) -> float:
    per = max(len(freq) // ranks, 1)
    total = sum(freq)
    return max(sum(freq[min(r * per, len(freq)):min((r + 1) * per, len(freq))]) for r in range(ranks)) / total


# ---- features -----------------------------------------------------------------------------------------------------

def part(dev, w, ops, kv, attn, b, ew=0.0, ep=None, sb=0.0):
    """A DevicePart; `ew`/`ep` are the expert bytes and params (version 2), `ep` defaults to `ew / bpp`; `sb` the
    routed slots' activation bytes per token (version 4)."""
    return {"device": dev, "w": w, "p": w / b, "res": w, "ops": ops, "kv": kv, "attn": attn,
            "ew": ew, "ep": ew / b if ep is None else ep, "sb": sb}


def slot_bytes(layout: dict, ff_share: float, version: int) -> float:
    """`multi::slot_bytes` (version 4, ADR 0080): activation bytes per token of one MoE layer's k routed slots on a
    rank whose experts have `ff_share` of the n_ff_exp columns. Per slot and n_ff_exp column: gate and up written (8),
    swiglu read and written (12), the down input read and quantized to Q8_1 (5.125); per slot and n_embd row: the
    down output written (4), weighted (8), the gate and up inputs quantized (10.25); the sum over k slots 12 (k - 1)
    per n_embd row. Expert parallel pays every slot, local or not; 0 before version 4."""
    if version < 4:
        return 0.0
    k = layout["n_expert_used"]
    return (k * (SLOT_FF_BYTES * layout["n_ff_exp"] * ff_share + SLOT_EMBD_BYTES * layout["n_embd"])
            + SLOT_SUM_BYTES * (k - 1) * layout["n_embd"])


def push_phase(v: list, p: dict) -> None:
    for q in v:
        if q["parts"] == p["parts"] and q.get("pre") == p.get("pre") and q["ranges"] == p["ranges"] \
                and q["after"] == p["after"] and q.get("out", False) == p.get("out", False):
            q["repeat"] += p["repeat"]
            return
    v.append(p)


def stages_of(plan: dict) -> list:
    return [(tuple(s["layers"]), s["devices"], s["tp"], s["ep"]) for s in plan["replicas"][0]["stages"]]


def features_single_v1(layout: dict, lim: int, dev: str) -> dict:
    """score::features for all layers on the GPU (the pre-ranks path of version 1)."""
    b = bpp(layout)
    moe = layout["n_expert"] > 0
    r = layout["n_expert_used"] / layout["n_expert"] if moe else 0.0
    kv_layer = 2.0 * layout["n_head_kv"] * layout["head_dim"] * 2.0
    attn_layer = 4.0 * layout["n_head"] * layout["head_dim"]
    p = {"device": dev, "w": 0.0, "p": 0.0, "res": 0.0, "ops": 0.0, "kv": 0.0, "attn": 0.0}

    def add(bytes_, active, ops, kv, attn):
        p["w"] += active
        p["p"] += active / b
        p["res"] += bytes_
        p["ops"] += ops
        p["kv"] += kv
        p["attn"] += attn

    for lb in layout["layers"][:lim]:
        d = float(lb["attn_bytes"] + lb["ffn_bytes"])
        add(d, d, OPS_DENSE_LAYER, kv_layer, attn_layer)
        if lb["expert_bytes"] > 0:
            e = float(lb["expert_bytes"])
            add(e, r * e, OPS_EXPERT_LAYER, 0.0, 0.0)
    o = float(layout["output_bytes"])
    add(o, o, OPS_OUTPUT, 0.0, 0.0)
    return {"single_v1": p}


def features(layout: dict, plan: dict, lim: int | None, version: int, routing: dict | None = None) -> dict:
    """features_multi: {"phases": [...]} (ranks path) or {"single_v1": part}. `routing`: ROUTING's entry, or None for
    uniform routing."""
    n_layer = layout["n_layer"]
    lim = min(lim or n_layer, n_layer)
    st_all = stages_of(plan)
    if (version == 1 or "single_ranks" in DROP) and len(st_all) == 1 and st_all[0][2] == 1:
        return features_single_v1(layout, lim, st_all[0][1][0])
    b = bpp(layout)
    moe = layout["n_expert"] > 0
    r = layout["n_expert_used"] / layout["n_expert"] if moe else 0.0
    kv_layer = 2.0 * layout["n_head_kv"] * layout["head_dim"] * 2.0
    attn_layer = 4.0 * layout["n_head"] * layout["head_dim"]
    block = main_format(layout)[0]
    act = 4.0 * layout["n_embd"]
    stages = [(st, (st[0][0], min(st[0][1], lim))) for st in st_all if st[0][0] < min(st[0][1], lim)]
    phases: list = []
    uniform_ep = None
    v2 = version >= 2
    # The dense builder selects the output rows before the last layer's FFN (`qwen3/ranked.rs`); the MoE shard
    # builder only before the head (`qwen3moe_shard.rs`).
    last_ffn_out = v2 and not moe
    for s, (st, span) in enumerate(stages):
        last = s + 1 == len(stages)
        devs, tp, ep = st[1], st[2], st[3]
        hop = None if last else {"copy": list(zip(devs, stages[s + 1][0][1])), "bpt": act, "per_step": False}
        readback = {"copy": [(x, HOST) for x in devs], "bpt": 4.0 * layout["n_vocab"] / tp, "per_step": True}
        if tp == 1:
            if not v2:
                w = ops = kv = attn = 0.0
                for lb in layout["layers"][span[0]:span[1]]:
                    w += (lb["attn_bytes"] + lb["ffn_bytes"]) + r * lb["expert_bytes"]
                    ops += OPS_ATTN + OPS_FFN + (OPS_EXPERT if lb["expert_bytes"] > 0 else 0.0)
                    kv += kv_layer
                    attn += attn_layer
                if last:
                    w += layout["output_bytes"]
                    ops += OPS_OUTPUT
                phases.append({"repeat": 1, "parts": [part(devs[0], w, ops, kv, attn, b)], "pre": None, "ranges": 1,
                               "after": readback if last else hop})
                continue
            # Version 2: one range per step, priced per op class; the head (and the dense last FFN) on output rows.
            w = ops = kv = attn = ex = sb = 0.0
            out_w = out_ops = 0.0
            for l in range(span[0], span[1]):
                lb = layout["layers"][l]
                if last and last_ffn_out and l + 1 == span[1]:
                    w += lb["attn_bytes"]
                    out_w += lb["ffn_bytes"]
                    ops += OPS_ATTN
                    out_ops += OPS_FFN
                else:
                    w += lb["attn_bytes"] + lb["ffn_bytes"]
                    ops += OPS_ATTN + OPS_FFN
                ex += lb["expert_bytes"]
                sb += slot_bytes(layout, 1.0, version) if lb["expert_bytes"] > 0 else 0.0
                ops += OPS_EXPERT if lb["expert_bytes"] > 0 else 0.0
                kv += kv_layer
                attn += attn_layer
            dec = part(devs[0], w, ops, kv, attn, b, ew=r * ex, sb=sb)
            pre = part(devs[0], w, ops, kv, attn, b, ew=ex, ep=r * ex / b, sb=sb) if ex > 0 else None
            phases.append({"repeat": 1, "parts": [dec], "pre": [pre] if pre else None, "ranges": 1,
                           "after": None if last else hop})
            if last:
                out_w += layout["output_bytes"]
                out_ops += OPS_OUTPUT
                phases.append({"repeat": 1, "parts": [part(devs[0], out_w, out_ops, 0.0, 0.0, b)], "pre": None,
                               "ranges": 0, "after": readback, "out": True})
            continue
        group = {"ar": list(devs), "bpt": act}
        for l in range(span[0], span[1]):
            lb = layout["layers"][l]
            rot = l % tp
            wo = layout["n_head"] * layout["head_dim"]
            for i in range(tp):
                block_slice(wo, block, tp, i, rot)
            attn_parts = [part(x, lb["attn_bytes"] / tp, OPS_ATTN, kv_layer / tp, attn_layer / tp, b) for x in devs]
            push_phase(phases, {"repeat": 1, "parts": attn_parts, "pre": None, "ranges": 1, "after": group})
            ffn = ffn_phase(layout, tp, ep, devs, l, rot, b, r, block, version, uniform_ep, routing)
            uniform_ep = ffn.pop("uniform_ep")
            ffn["after"] = group
            if last and last_ffn_out and l + 1 == span[1]:
                ffn["out"] = True
            push_phase(phases, ffn)
        if last:
            parts = [part(x, layout["output_bytes"] / tp, OPS_OUTPUT, 0.0, 0.0, b) for x in devs]
            ph = {"repeat": 1, "parts": parts, "pre": None, "ranges": 1, "after": readback}
            if v2:
                ph["out"] = True
            phases.append(ph)
        elif hop:
            phases.append({"repeat": 1, "parts": [part(x, 0.0, 0.0, 0.0, 0.0, b) for x in devs], "pre": None,
                           "ranges": 0, "after": hop})
    micro = max(plan.get("knobs", {}).get("prefill_micro_batches", 1), 1) if len(st_all) > 1 else 1
    return {"phases": phases, "micro": micro}


def ffn_phase(layout, tp, ep, devs, l, rot, b, r, block, version, uniform_ep, routing=None):
    lb = layout["layers"][l]
    v2 = version >= 2

    def width(length):
        out = []
        for i in range(tp):
            s, e = block_slice(length, block, tp, i, rot)
            out.append((e - s) / length)
        return out

    def mk(parts, pre):
        return {"repeat": 1, "parts": parts, "pre": pre, "ranges": 1, "after": None, "uniform_ep": uniform_ep}

    if layout["n_expert"] == 0:
        ff = layout["n_ff"]
        return mk([part(x, lb["ffn_bytes"] * s, OPS_FFN, 0.0, 0.0, b) for x, s in zip(devs, width(ff))], None)
    ops = OPS_FFN + OPS_EXPERT
    dense, ex = float(lb["ffn_bytes"]), float(lb["expert_bytes"])
    if ep == 1:
        shares = width(layout["n_ff_exp"])
        if not v2:
            return mk([part(x, dense + r * ex * s, ops, 0.0, 0.0, b) for x, s in zip(devs, shares)], None)
        dec = [part(x, dense, ops, 0.0, 0.0, b, ew=r * ex * s, sb=slot_bytes(layout, s, version))
               for x, s in zip(devs, shares)]
        pre = [part(x, dense, ops, 0.0, 0.0, b, ew=ex * s, ep=r * ex * s / b, sb=slot_bytes(layout, s, version))
               for x, s in zip(devs, shares)]
        return mk(dec, pre)
    e, g = layout["n_expert"], ep
    if uniform_ep is None:
        uniform_ep = (expected_max_local([1.0] * e, layout["n_expert_used"], g, EP_STEPS, EP_SEED),
                      max_mass([1.0] * e, g))
    dec_max, mass = uniform_ep

    def row(name):
        t = (routing or {}).get(name) or []
        return t[l] if l < len(t) and len(t[l]) == e else None

    if row("decode") is not None:
        dec_max = expected_max_local_cached(row("decode"), layout["n_expert_used"], g)
    if row("prefill") is not None:
        mass = max_mass(row("prefill"), g)
    eb = ex / e
    k = float(layout["n_expert_used"])
    local = (e // g) * eb
    if not v2:
        decode = [part(x, dense + dec_max * eb, ops, 0.0, 0.0, b) for x in devs]
        prefill = []
        for x in devs:
            p = part(x, dense + k * mass * eb, ops, 0.0, 0.0, b)
            p["w"] = dense + local
            prefill.append(p)
    else:
        sb = slot_bytes(layout, 1.0, version)
        decode = [part(x, dense, ops, 0.0, 0.0, b, ew=dec_max * eb, sb=sb) for x in devs]
        prefill = [part(x, dense, ops, 0.0, 0.0, b, ew=local, ep=k * mass * eb / b, sb=sb) for x in devs]
    out = mk(decode, prefill)
    out["uniform_ep"] = uniform_ep
    return out


# ---- predict ------------------------------------------------------------------------------------------------------

def add(v: list, label: str, s: float) -> None:
    for e in v:
        if e[0] == label:
            e[1] += s
            return
    v.append([label, s])


def total(v: list) -> float:
    t = 0.0
    for e in v:
        t += e[1]
    return t


def part_time(p: dict, rt: tuple, m: float, ctx: float, version: int) -> float:
    f, fe, bw, c0 = rt
    if version < 2:
        return c0 * p["ops"] + max((2.0 * p["p"] * m + p["attn"] * m * ctx) / f, (p["w"] + p["kv"] * ctx) / bw)
    if "roofline" in DROP:
        return c0 * p["ops"] + max((2.0 * p["p"] * m + p["attn"] * m * ctx) / f + 2.0 * p["ep"] * m / fe,
                                   (p["w"] + p["ew"] + p["kv"] * ctx) / bw)
    return (c0 * p["ops"] + max(2.0 * p["p"] * m / f, p["w"] / bw) + max(2.0 * p["ep"] * m / fe, p["ew"] / bw)
            + max(p["attn"] * m * ctx / f, p["kv"] * ctx / bw) + p.get("sb", 0.0) * m / bw)


def ar_latency(hw: dict, group: list) -> float:
    """The all-reduce time at zero bytes: the curve's first point, else the ring prior's `2 (p - 1) alpha`."""
    t = collective_time(hw, "AllReduce", group, 0.0)
    if t is not None:
        return t
    alpha = 0.0
    for d in group[1:]:
        alpha = max(alpha, link_between(hw, group[0], d)[0])
    return 2.0 * (len(group) - 1) * alpha


def coll_time(hw: dict, c: dict, m: float, version: int) -> tuple[str, float]:
    if "ar" in c:
        n = c["bpt"] * m
        label = "allreduce:" + "+".join(c["ar"])
        t = collective_time(hw, "AllReduce", c["ar"], n)
        if t is None:
            alpha, beta = 0.0, math.inf
            for d in c["ar"][1:]:
                a, bb = link_between(hw, c["ar"][0], d)
                alpha, beta = max(alpha, a), min(beta, bb)
            p = len(c["ar"])
            steps = 2.0 * (p - 1)
            t = steps * alpha + steps / p * n / beta
        lat = range_costs(hw).get("allreduce_latency_s")
        if version >= 2 and lat is not None and "ar_share" not in DROP:
            # Version 3: the byte part as measured plus the fitted in-step latency a + 2 (p - 1) b.
            b = range_costs(hw).get("allreduce_step_s") or 0.0
            t = max(t - ar_latency(hw, c["ar"]), 0.0) + lat + 2.0 * (len(c["ar"]) - 1) * b
        elif version >= 2:
            # The probe syncs every rank around each call; in a step only the share `s` of its small-message
            # time is paid, the byte-proportional part (the BF16 switch included) stays as measured.
            s = range_costs(hw)["allreduce_latency_share"]
            if "ar_share" in DROP:
                pass
            elif AR_FORM == "scale":
                t *= s
            else:
                t = max(t - (1.0 - s) * ar_latency(hw, c["ar"]), 0.0)
        return label, t
    n = c["bpt"] * (1.0 if c["per_step"] else m)
    label = f"copy:{c['copy'][0][0]}-{c['copy'][0][1]}"
    t = 0.0
    for a, bb in c["copy"]:
        one = collective_time(hw, "Copy", [a, bb], n)
        if one is None:
            alpha, beta = link_between(hw, a, bb)
            one = alpha + n / beta
        t = max(t, one)
    return label, t


def predict_single_v1(hw: dict, p: dict, prompt: int, gen: int) -> dict:
    f, _, bw, c0 = rates(hw, p["device"])
    ctx = gen / 2.0
    t_tok = c0 * p["ops"] + max((2.0 * p["p"] + p["attn"] * ctx) / f, (p["w"] + p["kv"] * ctx) / bw)
    t_pp, done = 0.0, 0
    while done < prompt:
        m_tok = min(512, prompt - done)
        m = float(m_tok)
        ctx = done + m / 2.0
        t_pp += c0 * p["ops"] + max((2.0 * p["p"] * m + p["attn"] * m * ctx) / f, (p["w"] + p["kv"] * ctx) / bw)
        done += m_tok
    return {"pp": prompt / t_pp, "tg": 1.0 / t_tok, "t": t_pp + gen * t_tok, "tg_bd": [], "pp_bd": []}


def micro_batch_sizes(n: int, m: int) -> list:
    """`plan::micro_batch_sizes`: `min(m, n)` chunks, sizes differ by at most one, the first ones larger."""
    k = max(min(n, max(m, 1)), 1)
    return [n // k + (1 if c < n % k else 0) for c in range(k)]


def pipeline_stages(phases: list) -> list:
    """`predict::pipeline_stages`: indices of the phases without output rows, cut after each per-token copy."""
    out: list = [[]]
    for i, ph in enumerate(phases):
        if ph.get("out"):
            continue
        out[-1].append(i)
        a = ph["after"]
        if a and "copy" in a and not a["per_step"]:
            out.append([])
    return [s for s in out if s]


def predict(hw: dict, feats: dict, prompt: int = PROMPT, gen: int = GEN, version: int = 2, ubatch: int = 512) -> dict:
    if "single_v1" in feats:
        return predict_single_v1(hw, feats["single_v1"], prompt, gen)
    rc = range_costs(hw)
    rt_of: dict = {}
    for ph in feats["phases"]:
        for p in ph["parts"] + (ph["pre"] or []):
            if p["device"] not in rt_of:
                rt_of[p["device"]] = rates(hw, p["device"])

    def phase(ph, m, ctx, prefill, with_after, out):
        # one output row per step (the bench and the judged request take the last row's logits)
        out_rows = ph.get("out") and "head_rows" not in DROP
        mm = 1.0 if (prefill and out_rows) else m
        pre = ph["pre"] if "prefill_experts" not in DROP else None
        parts = pre if (prefill and pre and not out_rows) else ph["parts"]
        rep = float(ph["repeat"])
        slow = ("", 0.0)
        for p in parts:
            t = part_time(p, rt_of[p["device"]], mm, ctx, version)
            if t > slow[1] or slow[0] == "":
                slow = (p["device"], t)
        add(out, "compute:" + slow[0], rep * slow[1])
        if ph["ranges"] > 0:
            add(out, "ranges", rep * ph["ranges"] * rc["c_range_graph_s"])
        if ph["after"] and with_after:
            label, t = coll_time(hw, ph["after"], mm, version)
            add(out, label, rep * t)

    def step(m, ctx, prefill, out):
        for ph in feats["phases"]:
            phase(ph, m, ctx, prefill, True, out)
        add(out, "step", rc["c_step_s"])

    tg: list = []
    step(1.0, gen / 2.0, False, tg)
    t_tok = total(tg)
    pp: list = []
    done = 0
    stages = pipeline_stages(feats["phases"])
    micro = feats.get("micro", 1)
    if micro > 1 and len(stages) > 1:
        # Pipelined prefill (ADR 0077), as `predict_ranks`: chunks of a `micro x ubatch` step in GPipe order, hops
        # serialized per boundary; the output-row phases and `c_step` once per step.
        n_st = len(stages)
        while done < prompt:
            n = min(ubatch * micro, prompt - done)
            free, link = [0.0] * n_st, [0.0] * n_st
            at = done
            for c in micro_batch_sizes(n, micro):
                m = float(c)
                ctx = at + m / 2.0
                arrive = 0.0
                for s, st in enumerate(stages):
                    terms: list = []
                    last = st[-1]
                    for i in st:
                        hop = s + 1 < n_st and i == last
                        phase(feats["phases"][i], m, ctx, True, not hop, terms)
                    free[s] = max(arrive, free[s]) + total(terms)
                    if s + 1 < n_st and feats["phases"][last]["after"]:
                        link[s] = max(free[s], link[s]) + coll_time(hw, feats["phases"][last]["after"], m, version)[1]
                        arrive = link[s]
                at += c
            add(pp, "pipeline", free[n_st - 1])
            m = float(n)
            for ph in feats["phases"]:
                if ph.get("out"):
                    phase(ph, m, done + m / 2.0, True, True, pp)
            add(pp, "step", rc["c_step_s"])
            done += n
    while done < prompt:
        m_tok = min(ubatch, prompt - done)
        m = float(m_tok)
        step(m, done + m / 2.0, True, pp)
        done += m_tok
    t_pp = total(pp)
    return {"pp": prompt / t_pp, "tg": 1.0 / t_tok, "t": t_pp + gen * t_tok, "tg_bd": tg, "pp_bd": pp}


# ---- fit (calib.rs) ------------------------------------------------------------------------------------------------

LAMBDA, LAMBDA_RANGES = 1e-3, 1e-9


class Problem:
    """The parameters of calib.rs for a GPU-only fit: GPU F, B, c0 graph and eager, CPU c0 (placeholder), PCIe alpha
    and beta of the first GPU's host link, then (version 2) F_exp when an anchor has expert work, the range costs
    when an anchor has ranks, and (version 2) the all-reduce latency share when an anchor has an all-reduce. Version
    2 holds the CPU c0 and the link at the prior (GPU-only anchors)."""

    def __init__(self, prior: dict, anchors: list, version: int):
        self.prior, self.version = prior, version
        g = next(d for d in prior["devices"] if d["kind"] == "gpu")
        self.gpu = g["id"]
        self.twins = [d["id"] for d in prior["devices"] if d["kind"] == "gpu" and d["id"] != g["id"]
                      and d["name"] == g["name"]]
        self.link = next(l["id"] for l in prior["links"]
                         if {l["a"], l["b"]} == {g["id"], HOST})
        gc = g["compute"]
        lk = next(l for l in prior["links"] if l["id"] == self.link)
        cpu = device(prior, HOST)
        cpu_c0 = (cpu.get("compute") or {}).get("c0_eager_s", gc["c0_eager_s"])
        ln = lambda v: math.log(max(v, 1e-30))
        self.x0 = [ln(gc["flops_per_s"]), ln(g["bandwidth_bytes_per_s"]["value"]), ln(gc["c0_graph_s"]),
                   ln(gc["c0_eager_s"]), ln(cpu_c0), ln(lk["alpha_s"]["value"]), ln(lk["beta_bytes_per_s"]["value"])]
        self.lam = [LAMBDA] * 7
        # Version 2: the CPU c0 and the host link are fitted only when an anchor computes on the CPU; on GPU-only
        # anchors the link carries just the logit readback, and fitting it turned it into a step cost.
        self.frozen = {4, 5, 6} if version >= 2 and "link" not in DROP else set()
        has_exp = version >= 2 and "expert_rate" not in DROP and any(
            p["ew"] > 0 for a in anchors for ph in a["f"].get("phases", []) for p in ph["parts"])
        self.exp_at = None
        if has_exp:
            self.exp_at = len(self.x0)
            self.x0.append(ln(gc.get("expert_flops_per_s") or gc["flops_per_s"]))
            self.lam.append(LAMBDA)
        self.ranges_at = None
        if any("phases" in a["f"] for a in anchors):
            self.ranges_at = len(self.x0)
            rc = range_costs(prior)
            self.x0 += [ln(rc["c_range_graph_s"]), ln(rc["c_range_eager_s"]), ln(rc["c_step_s"])]
            self.lam += [LAMBDA_RANGES] * 3
        self.ar_at = None
        if version >= 2 and "ar_share" not in DROP and any("ar" in (ph["after"] or {}) for a in anchors for ph in a["f"].get("phases", [])):
            self.ar_at = len(self.x0)
            rc = range_costs(prior)
            self.x0.append(ln(rc.get("allreduce_latency_s") or ALLREDUCE_LATENCY_PRIOR_S) if version >= 3
                           else ln(rc["allreduce_latency_share"]))
            # version 3: a fixed cost of a few percent of the step, pulled as weakly as the range costs
            self.lam.append(LAMBDA_RANGES if version >= 3 else LAMBDA)
        # b only when the anchors' all-reduces span two or more group sizes
        sizes = {len(ph["after"]["ar"]) for a in anchors for ph in a["f"].get("phases", []) if "ar" in (ph["after"] or {})}
        self.step_at = None
        if version >= 3 and self.ar_at is not None and len(sizes) > 1:
            self.step_at = len(self.x0)
            self.x0.append(ln(range_costs(prior).get("allreduce_step_s") or ALLREDUCE_STEP_PRIOR_S))
            self.lam.append(LAMBDA_RANGES)

    def apply(self, x: list) -> dict:
        hw = copy.deepcopy(self.prior)
        moved = lambda i: x[i] != self.x0[i]
        val = lambda v, i: math.exp(x[i]) if moved(i) else v
        for d in hw["devices"]:
            if d["id"] == self.gpu or d["id"] in self.twins:
                c = d["compute"]
                c["flops_per_s"] = val(c["flops_per_s"], 0)
                c["c0_graph_s"] = val(c["c0_graph_s"], 2)
                c["c0_eager_s"] = val(c["c0_eager_s"], 3)
                if self.exp_at is not None and moved(self.exp_at):
                    c["expert_flops_per_s"] = math.exp(x[self.exp_at])
                d["bandwidth_bytes_per_s"]["value"] = val(d["bandwidth_bytes_per_s"]["value"], 1)
        rc = range_costs(self.prior)
        new = dict(hw.get("range_costs") or {})
        changed = False
        if self.ranges_at is not None and any(moved(i) for i in range(self.ranges_at, self.ranges_at + 3)):
            i = self.ranges_at
            new.update({"c_range_graph_s": val(rc["c_range_graph_s"], i),
                        "c_range_eager_s": val(rc["c_range_eager_s"], i + 1),
                        "c_step_s": val(rc["c_step_s"], i + 2)})
            changed = True
        if self.ar_at is not None and (moved(self.ar_at) or self.version >= 3):
            new.update({k: rc[k] for k in ("c_range_graph_s", "c_range_eager_s", "c_step_s") if k not in new})
            if self.version >= 3:
                new.setdefault("allreduce_latency_share", rc["allreduce_latency_share"])
                new["allreduce_latency_s"] = math.exp(x[self.ar_at])
                if self.step_at is not None:
                    new["allreduce_step_s"] = math.exp(x[self.step_at])
            else:
                new["allreduce_latency_share"] = math.exp(x[self.ar_at])
            changed = True
        if self.version >= 4 and self.ranges_at is not None:
            # graphs fitted by version 4 price the routed slots' activation bytes (ADR 0080)
            new.update({k: rc[k] for k in ("c_range_graph_s", "c_range_eager_s", "c_step_s") if k not in new})
            new.setdefault("allreduce_latency_share", rc["allreduce_latency_share"])
            new["expert_slot_traffic"] = True
            changed = True
        if changed:
            hw["range_costs"] = new
        for l in hw["links"]:
            if l["id"] == self.link:
                l["alpha_s"]["value"] = val(l["alpha_s"]["value"], 5)
                l["beta_bytes_per_s"]["value"] = val(l["beta_bytes_per_s"]["value"], 6)
        return hw


def residuals(hw: dict, anchors: list, version: int) -> list:
    out = []
    for a in anchors:
        p = predict(hw, a["f"], a["prompt"], a["gen"], version)
        if a["pp"] is not None:
            out.append((a["id"] + ":pp", math.log(p["pp"] / a["pp"])))
        if a["tg"] is not None:
            out.append((a["id"] + ":tg", math.log(p["tg"] / a["tg"])))
    return out


def solve(a: list, b: list) -> list | None:
    n = len(b)
    a = [row[:] for row in a]
    b = b[:]
    for c in range(n):
        p = max(range(c, n), key=lambda i: abs(a[i][c]))
        if abs(a[p][c]) < 1e-300:
            return None
        a[c], a[p] = a[p], a[c]
        b[c], b[p] = b[p], b[c]
        for i in range(c + 1, n):
            k = a[i][c] / a[c][c]
            for j in range(c, n):
                a[i][j] -= k * a[c][j]
            b[i] -= k * b[c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = sum(a[r][j] * x[j] for j in range(r + 1, n))
        x[r] = (b[r] - s) / a[r][r]
    return x


def levenberg_marquardt(res, x0: list) -> list:
    n = len(x0)
    cost = lambda r: sum(v * v for v in r)
    x = x0[:]
    r = res(x)
    c = cost(r)
    mu = 1e-2
    for _ in range(300):
        h = 1e-5
        cols = []
        for i in range(n):
            hi, lo = x[:], x[:]
            hi[i] += h
            lo[i] -= h
            rh, rl = res(hi), res(lo)
            cols.append([(p - q) / (2.0 * h) for p, q in zip(rh, rl)])
        dot = lambda u, v: sum(p * q for p, q in zip(u, v))
        g = [dot(j, r) for j in cols]
        accepted = False
        while mu < 1e12:
            a = [[dot(cols[i], cols[j]) + (mu * max(dot(cols[i], cols[j]), 1e-12) if i == j else 0.0)
                  for j in range(n)] for i in range(n)]
            d = solve(a, [-v for v in g])
            if d is None:
                mu *= 10.0
                continue
            xn = [p + q for p, q in zip(x, d)]
            rn = res(xn)
            cn = cost(rn)
            if cn < c:
                tiny = c - cn < 1e-18 * max(c, 1e-30)
                x, r, c = xn, rn, cn
                mu = max(mu * 0.3, 1e-9)
                accepted = not tiny
                break
            mu *= 10.0
        if not accepted:
            break
    return x


def coordinate_search(f, x0: list) -> list:
    x = x0[:]
    best = f(x)
    step = 0.05
    for _ in range(10_000):
        if step < 1e-7:
            break
        improved = False
        for i in range(len(x)):
            for d in (1.0, -1.0):
                moved = False
                while True:
                    old = x[i]
                    x[i] = old + d * step
                    v = f(x)
                    if v < best - 1e-15:
                        best = v
                        moved = True
                    else:
                        x[i] = old
                        break
                if moved:
                    improved = True
                    break
        if not improved:
            step *= 0.5
    return x


def fit(prior: dict, anchors: list, version: int) -> tuple[dict, list]:
    prob = Problem(prior, anchors, version)
    free = [i for i in range(len(prob.x0)) if i not in prob.frozen]

    def full(y):
        x = prob.x0[:]
        for i, v in zip(free, y):
            x[i] = v
        return x

    def res(y):
        x = full(y)
        try:
            v = [e[1] for e in residuals(prob.apply(x), anchors, version)]
        except (ZeroDivisionError, OverflowError, ValueError):
            v = [1e3] * (2 * len(anchors))
        v += [math.sqrt(prob.lam[i]) * (x[i] - prob.x0[i]) for i in free]
        return v

    y = levenberg_marquardt(res, [prob.x0[i] for i in free])
    y = coordinate_search(lambda z: sum(v * v for v in res(z)), y)
    hw = prob.apply(full(y))
    return hw, residuals(hw, anchors, version)


# ---- session files ------------------------------------------------------------------------------------------------

def load_json(p: Path):
    return json.loads(p.read_text())


def prior_hw(d: Path) -> dict:
    hw = load_json(d / "hw.json")
    comm = d / "calib-comm.json"
    if comm.exists():
        cs = load_json(comm)["curves"]
        hw.setdefault("collectives", [])
        for c in cs:
            hw["collectives"] = [o for o in hw["collectives"]
                                 if not (OPS.get(o["op"], o["op"]) == OPS.get(c["op"], c["op"]) and o["ranks"] == c["ranks"])]
            hw["collectives"].append(c)
    return hw


def models(d: Path) -> list:
    return sorted(p.stem for p in (d / "layouts").glob("*.json"))


def rows(p: Path) -> list:
    v = load_json(p)
    return v if isinstance(v, list) else v.get("rows") or v.get("results") or []


def rate_of(r: dict) -> str | None:
    if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0:
        return "pp"
    if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0:
        return "tg"
    return None


def plans(d: Path, model: str) -> dict:
    return {c["id"]: c["plan"] for c in load_json(d / f"candidates-{model}.json")["candidates"]}


def anchor_plans(d: Path, model: str) -> dict:
    """The anchors' plans: on a 4-GPU box single and tp-2 come from a 2-GPU graph (plans-anchor/), tp-4 is the
    session's own plan."""
    pl = plans(d, model)
    if (d / "plans-anchor" / f"candidates-{model}.json").exists():
        pl.update(plans(d / "plans-anchor", model))
    return pl


def session_anchors(d: Path, version: int) -> list:
    out = []
    for model in models(d):
        layout = load_json(d / "layouts" / f"{model}.json")
        pl = anchor_plans(d, model)
        acc: dict = {}
        for f in sorted(d.glob(f"anchors-{model}-r*.json")):
            for r in rows(f):
                q = rate_of(r)
                if q is None or "plan_id" not in r:
                    continue
                e = acc.setdefault((r["plan_id"], r.get("layer_limit")), {"pp": [], "tg": [], "np": None, "ng": None})
                e[q].append(float(r["avg_ts"]))
                if q == "pp":
                    e["np"] = r["n_prompt"]
                else:
                    e["ng"] = r["n_gen"]
        for (pid, lim), e in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            out.append({"id": f"{layout['model']}/{pid}/{'L' + str(lim) if lim else 'all'}",
                        "f": features(layout, pl[pid], lim, version), "prompt": e["np"] or PROMPT,
                        "gen": e["ng"] or GEN, "pp": statistics.median(e["pp"]) if e["pp"] else None,
                        "tg": statistics.median(e["tg"]) if e["tg"] else None})
    return out


def measured(d: Path) -> dict:
    acc: dict = {}
    for f in sorted(d.glob("bench-*-r*.json")):
        stem = re.fullmatch(r"bench-(.+)-r\d+\.json", f.name).group(1)
        for model in models(d):
            if stem.startswith(model + "-"):
                for r in rows(f):
                    q = rate_of(r)
                    if q:
                        acc.setdefault((model, stem[len(model) + 1:]), {}).setdefault(q, []).append(float(r["avg_ts"]))
    return {k: {q: statistics.median(v) for q, v in s.items()} for k, s in acc.items()}


def predict_session(hw: dict, d: Path, version: int) -> list:
    out = []
    for model in models(d):
        layout = load_json(d / "layouts" / f"{model}.json")
        for cid, plan in plans(d, model).items():
            p = predict(hw, features(layout, plan, None, version, ROUTING.get(model)), PROMPT, GEN, version)
            out.append({"model": model, "id": cid, "pp": p["pp"], "tg": p["tg"], "t": p["t"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--eval", type=Path)
    ap.add_argument("--version", type=int, default=4)
    ap.add_argument("--committed", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--ar-form", choices=["offset", "scale"], default="offset")
    ap.add_argument("--drop", action="append", default=[], help="leave one version-2 fix out (repeatable)")
    ap.add_argument("--no-routing", action="store_true", help="uniform routing even where routing-<model>.json exists")
    ap.add_argument("--skew", type=float, help="synthetic routing for MoE models without a routing file (offline only)")
    a = ap.parse_args()
    global AR_FORM
    AR_FORM = a.ar_form
    DROP.update(a.drop)
    ev = a.eval or a.dir
    for model in models(ev):
        layout = load_json(ev / "layouts" / f"{model}.json")
        f = ev / f"routing-{model}.json"
        if layout["n_expert"] == 0 or a.no_routing:
            continue
        if f.exists():
            ROUTING[model] = routing_tables(load_json(f))
        elif a.skew is not None:
            ROUTING[model] = synthetic_routing(layout, a.skew)
    if ROUTING:
        print("routing: " + ", ".join(f"{m} ({'file' if (ev / f'routing-{m}.json').exists() else f'skew {a.skew}'})"
                                      for m in sorted(ROUTING)))
    if a.committed:
        hw = load_json(a.dir / "calib.json")["hw"]
        version, resid = 1, []
    else:
        version = a.version
        hw, resid = fit(prior_hw(a.dir), session_anchors(a.dir, version), version)
    g = next(d for d in hw["devices"] if d["kind"] == "gpu")
    c = g["compute"]
    rc = range_costs(hw)
    print(f"version {version}, fit {a.dir.as_posix()}: F {c['flops_per_s'] / 1e12:.1f} TFLOP/s, "
          f"F_exp {(c.get('expert_flops_per_s') or c['flops_per_s']) / 1e12:.1f}, "
          f"B {g['bandwidth_bytes_per_s']['value'] / 1e9:.0f} GB/s, c0 {c['c0_graph_s'] * 1e6:.2f} us, "
          f"c_range {rc['c_range_graph_s'] * 1e6:.2f} us, c_step {rc['c_step_s'] * 1e6:.1f} us, "
          f"ar_share {rc['allreduce_latency_share']:.3f}, ar_latency {(rc.get('allreduce_latency_s') or 0) * 1e6:.2f} us, "
          f"ar_step {(rc.get('allreduce_step_s') or 0) * 1e6:.2f} us")
    for k, v in resid:
        print(f"  anchor {k:<50} {v:+.3f}")
    meas = measured(ev)
    out = []
    for r in predict_session(hw, ev, version):
        m = meas.get((r["model"], r["id"]), {})
        e = {q: (r[q] / m[q] - 1.0) if q in m else None for q in ("pp", "tg")}
        out.append({**r, "pp_meas": m.get("pp"), "tg_meas": m.get("tg"), "err_pp": e["pp"], "err_tg": e["tg"]})
        f = lambda v: "   n/a" if v is None else f"{v:+.3f}"
        print(f"  {r['model']:<14} {r['id']:<8} pp {r['pp']:8.1f} / {m.get('pp') or 0:8.1f} ({f(e['pp'])})  "
              f"tg {r['tg']:6.1f} / {m.get('tg') or 0:6.1f} ({f(e['tg'])})")
    if a.json:
        a.json.write_text(json.dumps({"version": version, "ar_form": AR_FORM, "fit": a.dir.as_posix(),
                                      "eval": ev.as_posix(), "residuals": resid, "rows": out, "hw": hw},
                                     indent=1) + "\n", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
