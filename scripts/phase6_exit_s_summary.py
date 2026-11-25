#!/usr/bin/env python3
"""Verdict of Case S, the Phase 6 model-larger-than-a-node exit test (ADR 0091 "Case S", answers 6 to 8, ADR 0092), every
rule as code. Two Linux boxes, Qwen3-235B-A22B, one segment per node.

    phase6_exit_s_summary.py summary RESULTS_DIR      writes RESULTS_DIR/verdict-s.json, prints the tables
    phase6_exit_s_summary.py pick CANDIDATES.json PLANS_DIR OUT.toml [--n-expert N]    the analytic pick among the spanning (and covering) plans
    phase6_exit_s_summary.py layout-same COMMITTED.json MODEL.json   the model's own layout is the committed one the plan was made on
    phase6_exit_s_summary.py span PLAN.toml           the plan computes on both nodes, one node per stage
    phase6_exit_s_summary.py slots PLAN.toml [--n-expert N]   `node V R` per stage from the plan's expert cache (what the anchors run); with N: every stage covers
    phase6_exit_s_summary.py link START.json END.json [--band B]         the link did not drift (16 KiB and 32 MiB)
    phase6_exit_s_summary.py gate RESULTS_DIR         the link gate before the download: writes link/gate.json, exit 1 when it fails
    phase6_exit_s_summary.py chunk PLAN.toml [--prompt N]    tokens of one micro-batch of the pick's first prefill step (what the anchors run)
    phase6_exit_s_summary.py dryrun RESULTS_DIR      the dry run passed on both boxes of this session (exit 1 otherwise): the download stage asks
    phase6_exit_s_summary.py spans PLAN.toml [--n-layer N]   `node first..last` per stage, in order; with N: they partition the layers
    phase6_exit_s_summary.py precheck RESULTS_DIR calib|freeze           the rules that must hold before calibration is trusted (calib) or
                                                      before the first judged round (freeze); exit 1 and the reasons otherwise
    phase6_exit_s_summary.py round ROUND_DIR [--frozen-sha S]            is the judged round clean (reasons, exit 1)
    phase6_exit_s_summary.py next-round RESULTS_DIR   exit 0 while another judged round is due, 1 when done (3 clean or 8)

Layout (scripts/phase6_exit_s.sh writes it): session.json, gates/<box>/, env/<box>/env.json, trace/{routing,hidden}.sha256,
hidden/<box>.sha256, calib/<node>/{anchors/a{1,2}-r{1,2}.json, pred.json}, link/probe-{start,end}.json, plans/judged-plan.toml
and candidates.json, predictions/{predict-cluster.json, SHA256SUMS, FROZEN.sha256, frozen.json}, judged/r<k>/ and
judged/aa-last/ each with cluster.json (rates of the pipelined request), seg-<node>.json (+ .gpu.txt) and meta.json.

Rules (every one a line of `rules` in verdict-s.json; all must hold or there is no verdict):
  gates_<box>           one NUMA node, co-tenant reading of engine_sample.py under the limit, power and idle gates (before downloads)
  ram_budget_<box>      the measured RAM budget under the balloon is in env.json and not above the emulated 64 GB
  trace_hidden          the hidden rows of the trace run have the same sha256 on both boxes and in trace/hidden.sha256
  anchors_<node>        anchors ran on real input: hidden rows of the dump for a span after layer 0, the frozen prompt hash for
                        a span from layer 0 (which takes tokens); two rounds of both truncations
  anchors_step_<node>   the anchors ran the batch the judged plan runs: its micro-batch (the first prefill step, ubatch x prefill_micro_batches
                        tokens, cut as the pipeline cuts it: 512 for 2048 x 4 over a 2048-token prompt), stamped `exit_ubatch` in every anchor
                        file by sc_box.sh, not the whole prompt as one batch (ADR 0091 History 2026-11-26)
  fit_band_<node>       each node's fit band within 0.10 (sc_gates.FIT_BAND_LIMIT)
  dryrun                dryrun.json: the dry run on the tiny sharded MoE passed on both nodes of this session's boxes before the download
  ram_budget_remeasured the balloon went back after the install and each node's RAM budget was measured again (env/<n>/env.json; env-plan.json is
                        the plan's reading, session.json budget names both), and the covering pick made again on them is the judged plan
                        (plans/repick.json); a pick that no longer fits aborts the session before the freeze
  prediction_judged     predict-cluster.json is routing `trace`, judged, priced from the sha256 of the trace on disk, for the
                        frozen prompt and the judged plan
  freeze                FROZEN.sha256 and every frozen file match; every judged round ran under that hash, started after t_frozen
  segments_covering     every stage of the judged plan holds all experts of its layers in VRAM plus RAM slots (layout.json n_expert) and the anchors
                        ran the pick's slots per node (session.json segments; ADR 0091 History 2026-11-25)
  span_both_nodes       the judged plan is the planner's analytic pick (fastest predicted) among plans that compute on both
                        nodes, and no candidate that does not is considered
  link_gate             the link gate before the 142 GB download (advisor, delegated by the owner, 2026-11-25): two full probes (frames to
                        64 MiB) at least 10 minutes apart (link/probe-gate{1,2}.json, times in link/gate-times.json), the judged pick
                        itself priced with each (link/gate/predict-gate{1,2}.json: the plan's graph with both nodes' compute rates from the
                        committed fitted graph of segment-confirm-20261125b, not discovered priors (link/gate/fitted.toml, hw-gate{1,2}.json),
                        the committed 235B layout and only the gate probe's link curves swapped in; the plan hash of both prices is the judged
                        plan's); predicted pp and tg each move
                        by at most 0.08, else the pair is aborted before the download. Recomputed from the files here; missing evidence fails
  link_reprice          the link rule (advisor, delegated by the owner, 2026-11-25; docs/notes/phase6-rule-changes.md): the frozen judged plan
                        is priced again with the end probe's link curves and nothing else changed (predictions/predict-cluster-end.json);
                        predicted pp and tg each move by at most 0.08. Raw probe points (start, before each round, end) are reported, not gated
  rounds_clean          three clean rounds within eight (every round reported, unclean ones rerun on the same prediction)
  prompt_hash           every judged round ran the frozen prompt
  routing_gate          per segment, medians of the clean rounds against the replay: vram share 0.02, full hit 0.02, RAM hit 0.005
  aa_early              judged r1 against r2 within 0.08 on T
  aa                    judged r1 against judged/aa-last the last run, within 0.08 on T
A round is unclean when a segment's warm block says not covering, timed prefill read disk bytes, the balloon or co-tenant
flag is false, the GPU throttled or ran below its default power limit, a rate is missing, or the prompt differs, or the card is over
24 GiB (seg-<node>.json gpu_mem_total_mib above 24,576) and the peak GPU memory in use over the round (gpu_mem_peak_mib, sampled by
phase6_box.sh gpumem, the warm request included) is not under 24,576 MiB (ADR 0091 History 2026-11-26).
Then, only when clean: e_pp = pp_pred / pp_meas - 1 and e_tg on the medians of the clean rounds, against the frozen prediction and
against the end-priced one; pass only when both are within 0.15 against both, a miss when it is outside against both, else no verdict.
Report only: the errors against a prediction priced from a second prompt's routing trace (predictions/predict-cluster-p2.json).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
import tomllib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc_gates as sg  # noqa: E402
import sc_summary as ss  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_FITTED_REL = "bench/results/segment-confirm-20261125b/predictions/fitted.toml"  # the confirmed node fit, one 3090 box
CARD_MIB = 24576  # a card over 24 GiB must peak under this after the warm request: it stands in for the 24 GiB card
LINE = 0.15
BAND = sg.AA_BAND  # 0.08, bench/spec/phase0.toml: no two-node noise class exists
GATE_SHARE, GATE_RAM = ss.GATE_SHARE, ss.GATE_RAM
CLEAN_NEEDED, MAX_ROUNDS = 3, 8
EMULATED_RAM_B = 64_000_000_000
MIN_GATE_GAP_S = 600.0  # the two gate probes are at least 10 minutes apart; only the loopback dry run shortens it (P6_MIN_GAP_S)
FULL_PROBE_BYTES = 64 * 1024 * 1024  # a full probe reaches 64 MiB frames: no --max-bytes
PROBE_POINTS = (16 * 1024, 32 * 1024 * 1024)  # 32 MiB is no probe point (4 KiB x 4^k): interpolated as the cost model does


def load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha_file(path: str) -> str:
    return sg.sha256_file(path)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


# ---- the plan: both nodes -------------------------------------------------------------------------------------------------


def plan_nodes(plan_toml: str) -> tuple[list[str], list[str]]:
    """(nodes the stages compute on, problems): the node of a device id is the part before '/'. One node per stage."""
    plan = tomllib.loads(open(plan_toml, encoding="utf-8").read())
    nodes: list[str] = []
    why: list[str] = []
    stages = [s for r in plan.get("replicas", []) for s in r.get("stages", [])]
    for i, st in enumerate(stages):
        ns = sorted({d.split("/", 1)[0] for d in st.get("devices", []) if "/" in d})
        if len(ns) != 1:
            why.append(f"stage {i} names nodes {ns}")
        nodes += ns
    return sorted(set(nodes)), why + ([] if len(stages) >= 2 else [f"{len(stages)} stage(s): a spanning plan has one per node"])


def plan_spans(plan_toml: str) -> list[tuple[str, int, int]]:
    """(node, first layer, end layer) of every stage, in order."""
    plan = tomllib.loads(open(plan_toml, encoding="utf-8").read())
    out = []
    for r in plan.get("replicas", []):
        for st in r.get("stages", []):
            nodes = sorted({d.split("/", 1)[0] for d in st.get("devices", []) if "/" in d})
            lo, hi = st["layers"]
            out.append((nodes[0] if nodes else "?", int(lo), int(hi)))
    return out


def pick_chunk(plan_toml: str, prompt: int) -> int:
    """Tokens of one micro-batch of the judged plan's first prefill step: the batch every segment computes at once. A step is
    `prefill_micro_batches x ubatch` tokens cut as `plan::micro_batch_sizes` cuts it. A plan without knobs runs the prompt in one batch."""
    k = tomllib.loads(open(plan_toml, encoding="utf-8").read()).get("knobs") or {}
    ub, micro = int(k.get("ubatch") or prompt), max(int(k.get("prefill_micro_batches") or 1), 1)
    step = min(prompt, ub * micro)
    n = min(step, micro)
    return max(step // n + (1 if c < step % n else 0) for c in range(n))


def dryrun_check(res: str, sess: dict) -> list[str]:
    """The dry run on the tiny sharded MoE (stage `dryrun`) passed on both boxes of this session before the download."""
    try:
        d = load(os.path.join(res, "dryrun.json"))
    except (OSError, ValueError):
        return ["no dryrun.json: the dry run did not pass before the download"]
    why = []
    if d.get("passed") is not True:
        why.append("the dry run did not pass")
    nodes = sorted(s["node"] for s in sess.get("segments", []))
    if sorted(d.get("nodes") or []) != nodes:
        why.append(f"the dry run covers nodes {d.get('nodes')}, the session has {nodes}")
    boxes = {k: (v or {}).get("id") for k, v in (sess.get("boxes") or {}).items()}
    if (d.get("boxes") or {}) != boxes:
        why.append(f"the dry run ran on boxes {d.get('boxes')}, the session on {boxes}")
    return why


def budget_check(res: str, sess: dict) -> list[str]:
    """ADR 0091 History 2026-11-26: the balloon is released for the download and back after the model install, each node's RAM budget is
    measured again then (env/<n>/env.json is that reading, env-plan.json the one the plan was made on) and the covering pick is made again on
    it: the re-pick is the judged plan, or the session aborted before the freeze."""
    why: list[str] = []
    b = sess.get("budget") or {}
    for s in sess.get("segments", []):
        n = s["node"]
        try:
            now = load(os.path.join(res, "env", n, "env.json"))["ram_budget"]["ram_budget_bytes"]
            first = load(os.path.join(res, "env", n, "env-plan.json"))["ram_budget"]["ram_budget_bytes"]
        except (OSError, ValueError, KeyError, TypeError):
            why.append(f"{n}: no budget before and after the model install (env.json, env-plan.json)")
            continue
        if (b.get(n) or {}).get("plan") != first or (b.get(n) or {}).get("installed") != now:
            why.append(f"{n}: session.json budget {b.get(n)} is not the files' {first} and {now}")
    try:
        rp = load(os.path.join(res, "plans", "repick.json"))
        if rp.get("plan_sha256") != sha_file(os.path.join(res, "plans", "judged-plan.toml")) or rp.get("same") is not True:
            why.append("the pick made again on the measured budgets is not the judged plan")
    except (OSError, ValueError):
        why.append("no plans/repick.json: the pick was not made again on the budgets measured after the install")
    return why


def gate_fitted_path() -> str:
    """The committed node fit the link gate prices on (ADR 0091 History 2026-11-26); P6_GATE_FITTED replaces it in tests."""
    return os.environ.get("P6_GATE_FITTED") or os.path.join(ROOT, GATE_FITTED_REL)


def fitted_rates(path: str) -> dict[str, float]:
    """`device id -> flops_per_s` of a fitted node graph (TOML)."""
    with open(path, "rb") as f:
        d = tomllib.load(f)
    return {dev["id"]: dev["compute"]["flops_per_s"] for dev in d.get("devices", []) if (dev.get("compute") or {}).get("flops_per_s")}


def gate_fitted_check(res: str) -> list[str]:
    """The gate prices on the fitted rates of both nodes, not on discovered priors: the fitted file the driver put on node a is the
    committed one, and each priced graph (link/gate/hw-gate{1,2}.json, the one the price names by hash) carries its rates on both nodes."""
    why: list[str] = []
    try:
        want = fitted_rates(gate_fitted_path())
        used = os.path.join(res, "link", "gate", "fitted.toml")
        if sha_file(used) != sha_file(gate_fitted_path()):
            why.append("link/gate/fitted.toml is not the committed fitted graph")
        for k in ("gate1", "gate2"):
            hwp = os.path.join(res, "link", "gate", f"hw-{k}.json")
            hw = load(hwp)
            by_id = {d["id"]: d for d in hw.get("devices", [])}
            for node in ("a", "b"):
                for dev, rate in want.items():
                    got = ((by_id.get(f"{node}/{dev}") or {}).get("compute") or {}).get("flops_per_s")
                    if got != rate:
                        why.append(f"hw-{k}.json: {node}/{dev} rate {got} is not the fitted {rate}")
            priced = load(os.path.join(res, "link", "gate", f"predict-{k}.json")).get("hw_sha256")
            if priced != sha_file(hwp):
                why.append(f"predict-{k}.json priced graph {str(priced)[:12]}, not hw-{k}.json")
    except (OSError, ValueError, KeyError, TypeError) as e:
        why.append(f"no fitted-rate evidence of the gate ({e})")
    return why


def gate_check(res: str) -> tuple[dict, list[str]]:
    """The link gate from its files: both probes full, their times, and the two prices of the frozen candidate. Returns (facts, reasons)."""
    why: list[str] = []
    info: dict = {"band": BAND}
    min_gap = float(os.environ.get("P6_MIN_GAP_S", MIN_GATE_GAP_S))
    info["min_gap_s"] = min_gap
    try:
        t = load(os.path.join(res, "link", "gate-times.json"))
        info["gap_s"] = t["gate2"] - t["gate1"]
        if info["gap_s"] < min_gap:
            why.append(f"the probes are {info['gap_s']:.0f} s apart, at least {min_gap:.0f} s are needed")
    except (OSError, ValueError, KeyError, TypeError) as e:
        why.append(f"no probe times ({e})")
    for k in ("gate1", "gate2"):
        try:
            pr = load(os.path.join(res, "link", f"probe-{k}.json"))
            top = max((x for c in pr.get("curves", []) if c.get("op") == "copy" for x, _ in c["points"]), default=0)
            if top < FULL_PROBE_BYTES:
                why.append(f"probe-{k}.json stops at {top} B frames, a full probe reaches {FULL_PROBE_BYTES}")
        except (OSError, ValueError, KeyError, TypeError) as e:
            why.append(f"no probe-{k}.json ({e})")
    why += gate_fitted_check(res)
    try:
        # the gate prices the judged pick itself (ADR 0091 History 2026-11-25), not a nominal candidate
        want = sha_file(os.path.join(res, "plans", "judged-plan.toml"))
        for k in ("gate1", "gate2"):
            got = load(os.path.join(res, "link", "gate", f"predict-{k}.json")).get("plan_sha256")
            if got != want:
                why.append(f"predict-{k}.json priced plan {str(got)[:12]}, not the judged pick {want[:12]}")
        p1 = load(os.path.join(res, "link", "gate", "predict-gate1.json"))["prediction"]
        p2 = load(os.path.join(res, "link", "gate", "predict-gate2.json"))["prediction"]
        info["pp_move"], info["tg_move"] = (p2[k] / p1[k] - 1.0 for k in ("pp_tps", "tg_tps"))
        info["pp_tps"], info["tg_tps"] = (p1["pp_tps"], p2["pp_tps"]), (p1["tg_tps"], p2["tg_tps"])
        for k in ("pp_move", "tg_move"):
            if abs(info[k]) > BAND + 1e-9:
                why.append(f"{k[:2]} moves {info[k]:+.3f} between the probes against {BAND}")
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as e:
        why.append(f"no re-priced predictions ({e})")
    return info, why


def stage_slots(plan_toml: str) -> list[tuple[str, int, int]]:
    """(node, VRAM slots, RAM slots) of every stage from its `expert_cache` (gpu0: compute tier, cpu0: RAM tier). The workers run these
    slots and the calibration anchors run the same (ADR 0091 History 2026-11-25: slots come from the pick, not from a global V and R)."""
    plan = tomllib.loads(open(plan_toml, encoding="utf-8").read())
    out = []
    for st in (s_ for r in plan.get("replicas", []) for s_ in r.get("stages", [])):
        got = {c["device"].rsplit("/", 1)[-1]: c["slots"] for c in st.get("expert_cache", [])}
        nodes = sorted({d.split("/", 1)[0] for d in st.get("devices", []) if "/" in d})
        out.append((nodes[0] if nodes else "?", int(got.get("gpu0", 0)), int(got.get("cpu0", 0))))
    return out


def layout_same(a: str, b: str) -> list[str]:
    """The model's own layout (`tetraeder plan layout`, after the download) is the committed one the plan was made on: the same layer count,
    expert count and per-layer and embedding bytes, for every key both files have."""
    x, y = load(a), load(b)
    why = []
    for k in ("n_layer", "n_expert", "n_embd", "embed_bytes", "output_bytes", "layers"):
        if k in x and k in y and x[k] != y[k]:
            why.append(f"{k} differs ({str(x[k])[:60]} against {str(y[k])[:60]})")
    if not any(k in x and k in y for k in ("n_layer", "layers")):
        why.append("the layouts share no key to compare")
    return why


def check_covering(plan_toml: str, n_expert: int) -> list[str]:
    """Every stage's VRAM plus RAM slots hold all `n_expert` experts of each of its layers: the regime cost model 6 is confirmed on
    (segment-confirm-20261125b), which reads no expert from disk in prefill. A stage without a cache holds none."""
    why = []
    for i, (node, v, r) in enumerate(stage_slots(plan_toml)):
        if v + r < n_expert:
            why.append(f"stage {i} on {node} holds {v} + {r} = {v + r} of {n_expert} experts: not a covering segment")
    return why


# what must hold before the calibration is trusted, and before the first judged round (the rest of the rules are about the rounds)
PRE_CALIB = ("gates_", "ram_budget_", "dryrun", "link_gate", "segments_covering", "trace_hidden", "anchors_", "fit_band_")
PRE_FREEZE = PRE_CALIB + ("prediction_judged", "freeze", "span_both_nodes")


def check_partition(plan_toml: str, n_layer: int, nodes: tuple[str, ...] = ("a", "b")) -> list[str]:
    """The stages cut the layers 0..n_layer in order, one stage per node, the first node first (the leader holds the first layers)."""
    sp = plan_spans(plan_toml)
    why = []
    if [n for n, _, _ in sp] != list(nodes):
        why.append(f"the stages sit on {[n for n, _, _ in sp]}, not one per node in order {list(nodes)}")
    pos = 0
    for n, lo, hi in sp:
        if lo != pos or hi <= lo:
            why.append(f"stage on {n} covers {lo}..{hi} after layer {pos}")
        pos = hi
    if pos != n_layer:
        why.append(f"the stages end at layer {pos}, the model has {n_layer}")
    return why


def check_span(plan_toml: str, nodes: tuple[str, ...] | None = None) -> list[str]:
    try:
        got, why = plan_nodes(plan_toml)
    except (OSError, ValueError) as e:
        return [f"cannot read the plan: {e}"]
    want = sorted(nodes) if nodes else got
    if len(got) < 2 or got != want:
        why.append(f"the plan computes on {got}, a judged candidate must compute on every node {want or ['a', 'b']}")
    return why


def pick(candidates_json: str, plans_dir: str, nodes: tuple[str, ...] = ("a", "b"), n_expert: int | None = None) -> tuple[dict, list[str]]:
    """The analytic pick: lowest predicted request time among the candidates whose plan computes on both nodes. A plan on one
    node is never considered (ADR 0091 Case S). With `n_expert` a plan that is not covering is not considered either."""
    doc = load(candidates_json)
    spanning, dropped = [], []
    for c in doc.get("candidates", []):
        plan = c.get("plan")
        p = os.path.join(plans_dir, plan) if plan else ""
        if plan and os.path.isfile(p) and not check_span(p, nodes) and not (n_expert and check_covering(p, n_expert)):
            spanning.append((c["prediction"]["request_s"], c["id"], plan))
        else:
            dropped.append(c.get("id"))
    if not spanning:
        what = "computes on both nodes" + (f" and covers all {n_expert} experts" if n_expert else "")
        return {"error": f"no candidate {what}", "dropped": dropped}, [f"no candidate {what}"]
    t, cid, plan = min(spanning)
    return {"id": cid, "plan": plan, "request_s": t, "spanning": len(spanning), "dropped_single_node": dropped}, []


# ---- the link ----------------------------------------------------------------------------------------------------------------


def curve_time(points: list, nbytes: float) -> float:
    """profiler::curve_time: linear in bytes between points, flat below the first, extended with the last slope above."""
    pts = sorted((float(a), float(b)) for a, b in points)
    if nbytes <= pts[0][0]:
        return pts[0][1]
    if nbytes >= pts[-1][0]:
        if len(pts) == 1:
            return pts[-1][1] * nbytes / pts[-1][0]
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        return y1 + max((y1 - y0) / (x1 - x0), 0.0) * (nbytes - x1)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if nbytes <= x1:
            return y0 + (y1 - y0) * (nbytes - x0) / (x1 - x0)
    raise AssertionError("unreachable")


def link_readings(probe: dict) -> dict[str, float]:
    out = {}
    for c in probe.get("curves", []):
        if c.get("op") == "copy" and len(c.get("ranks", [])) == 2:
            d = "->".join(r.split("/", 1)[0] for r in c["ranks"])
            for b in PROBE_POINTS:
                out[f"{d} {b}"] = curve_time(c["points"], b)
    return out


def check_link(start: dict, end: dict, band: float = BAND) -> tuple[list[str], list[dict]]:
    s, e = link_readings(start), link_readings(end)
    rows, why = [], []
    if len(s) < 2 * len(PROBE_POINTS) or set(s) != set(e):
        return ["a probe lacks a direction or a size (both directions are needed at both ends)"], rows
    for k in sorted(s):
        diff = e[k] / s[k] - 1.0
        rows.append({"point": k, "start_s": s[k], "end_s": e[k], "diff": diff, "ok": abs(diff) <= band})
        if abs(diff) > band:
            why.append(f"{k} B: {s[k] * 1e3:.2f} -> {e[k] * 1e3:.2f} ms ({diff:+.3f} against {band})")
    return why, rows


# ---- rounds ------------------------------------------------------------------------------------------------------------------


def seg_files(rd: str) -> list[str]:
    return sorted(f for f in glob.glob(os.path.join(rd, "seg-*.json")) if not f.endswith(".gpu.json"))


def vram_peak(seg_file: str) -> list[str]:
    """ADR 0091 History 2026-11-26: on a card over 24 GiB the peak GPU memory of the round (the warm request included, so the timed one
    after it cannot exceed it) stays under 24,576 MiB, so the card stands in for a 24 GiB one. `gpu_mem_total_mib` is the card's size and
    `gpu_mem_peak_mib` the most `nvidia-smi` showed in use over the round (phase6_box.sh gpumem, twice a second), both written by the driver
    into the segment file; a card of 24 GiB or less needs no peak."""
    try:
        d = load(seg_file)
    except (OSError, ValueError):
        return []  # ss.unclean reports an unreadable file
    total = d.get("gpu_mem_total_mib")
    if total is None:
        return ["no card memory reading (gpu_mem_total_mib): the 24 GiB peak rule cannot be checked"]
    if total <= CARD_MIB:
        return []
    peak = d.get("gpu_mem_peak_mib")
    if peak is None:
        return [f"a {total} MiB card without a peak GPU memory reading (gpu_mem_peak_mib)"]
    if peak >= CARD_MIB:
        return [f"peak GPU memory {peak:.0f} MiB on a {total} MiB card, not under {CARD_MIB} MiB"]
    return []


def round_unclean(rd: str, prompt_hash: str | None, frozen_sha: str | None = None, t_frozen: float | None = None) -> list[str]:
    """Why a judged round is not clean: the per-segment rules of sc_summary on each segment's JSON, plus the prompt and the freeze."""
    why: list[str] = []
    try:
        cl = ss.run_of(load(os.path.join(rd, "cluster.json")))
    except (OSError, ValueError):
        return ["no cluster.json"]
    if not cl.get("prefill_tok_s") or not cl.get("decode_tok_s"):
        why.append("no rates")
    segs = seg_files(rd)
    if len(segs) < 2:
        why.append(f"{len(segs)} segment file(s), one per node needed")
    for f in segs:
        why += [f"{os.path.basename(f)[4:-5]}: {w}" for w in ss.unclean(f) + vram_peak(f)]
    p = (cl.get("prompt") or {})
    if prompt_hash and p.get("fnv1a64") != prompt_hash:
        why.append(f"prompt hash {p.get('fnv1a64')} is not the frozen {prompt_hash}")
    try:
        meta = load(os.path.join(rd, "meta.json"))
    except (OSError, ValueError):
        meta = {}
    if frozen_sha is not None and meta.get("frozen_sha256") != frozen_sha:
        why.append(f"ran under {meta.get('frozen_sha256')}, not the frozen {frozen_sha}")
    if t_frozen is not None and not (meta.get("t_start") or 0) > t_frozen:
        why.append("started before the prediction was frozen")
    return why


def round_dirs(res: str) -> list[str]:
    ds = [d for d in glob.glob(os.path.join(res, "judged", "r[0-9]*")) if os.path.isdir(d)]
    return sorted(ds, key=lambda d: int(re.search(r"r(\d+)$", d).group(1)))


def frozen_facts(res: str) -> tuple[str | None, float | None]:
    try:
        sha = open(os.path.join(res, "predictions", "FROZEN.sha256"), encoding="utf-8").read().split()[0]
    except (OSError, IndexError):
        sha = None
    try:
        t = load(os.path.join(res, "predictions", "frozen.json")).get("t_frozen")
    except (OSError, ValueError):
        t = None
    return sha, t


# ---- the session ------------------------------------------------------------------------------------------------------------


def node_rules(res: str, sess: dict, rule) -> None:
    ph = (sess.get("prompt") or {}).get("fnv1a64")
    for seg in sess.get("segments", []):
        n = seg["node"]
        cd = os.path.join(res, "calib", n)
        files = sorted(glob.glob(os.path.join(cd, "anchors", "a[12]-r[12].json")))
        names = {os.path.basename(f) for f in files}
        full = names == {"a1-r1.json", "a1-r2.json", "a2-r1.json", "a2-r2.json"}
        if seg["span"][0] == 0:  # a span from layer 0 takes tokens: no hidden rows, so the prompt hash is the evidence
            bad = []
            for f in files:
                j = load(f)
                got = (ss.run_of(j).get("prompt") or j.get("prompt") or {}).get("fnv1a64")  # exit_regime writes it at the top of the file
                if not ph or got != ph:
                    bad.append(f"{os.path.basename(f)} prompt {got}")
            why = bad
        else:
            why = sg.check_anchors_real(files)
        if not full:
            why = why + [f"anchors {sorted(names)}: two truncations of two rounds each are needed"]
        rule(f"anchors_{n}", not why, "; ".join(why) or "real input, two truncations, two rounds")
        # the anchors run the batch the judged plan runs (ADR 0091 History 2026-11-26): the pick's micro-batch of its first step, not the
        # whole prompt as one batch, so the fit calibrates the regime that is judged
        try:
            want = pick_chunk(os.path.join(res, "plans", "judged-plan.toml"), int((sess.get("prompt") or {}).get("tokens") or 2048))
            got = sorted({load(f).get("exit_ubatch") for f in files}, key=str)
            ok = bool(files) and got == [want]
            rule(f"anchors_step_{n}", ok, f"anchors ran batches of {got} tokens, the pick's micro-batch is {want}" if not ok
                 else f"anchors ran batches of {want} tokens, the pick's")
        except (OSError, ValueError, TypeError) as e:
            rule(f"anchors_step_{n}", False, f"no plan or anchors to compare ({e})")
        why = sg.check_fit_band(os.path.join(cd, "pred.json"), sg.FIT_BAND_LIMIT)
        rule(f"fit_band_{n}", not why, "; ".join(why) or "within the limit")


def summarize(res: str) -> dict:
    sess = load(os.path.join(res, "session.json"))
    segs = sess.get("segments", [])
    nodes = tuple(sorted(s["node"] for s in segs))
    out: dict = {"case": "S", "line": LINE, "band": BAND, "band_source": sess.get("band_source", "bench/spec/phase0.toml")}
    rules: list[dict] = []

    def rule(name: str, ok: bool, detail: str) -> None:
        rules.append({"name": name, "ok": bool(ok), "detail": detail})

    # gates and node budgets, before anything is judged
    for n in nodes:
        g = os.path.join(res, "gates", n)
        why: list[str] = []
        try:
            why += sg.check_numa(open(os.path.join(g, "lscpu.txt"), encoding="utf-8").read())
        except OSError:
            why.append("no lscpu.txt")
        why += sg.check_cotenant(os.path.join(g, "cotenant.json"))
        try:
            gj = load(os.path.join(g, "gates.json"))
            why += [f"{k} is {gj.get(k)!r}" for k in ("power_ok", "idle_ok") if gj.get(k) is not True]
        except (OSError, ValueError):
            why.append("no gates.json")
        rule(f"gates_{n}", not why, "; ".join(why) or "one NUMA node, co-tenant under the limit, power and idle gates passed")
        try:
            rb = load(os.path.join(res, "env", n, "env.json"))["ram_budget"]["ram_budget_bytes"]
            rule(f"ram_budget_{n}", 0 < rb <= EMULATED_RAM_B, f"{rb} B against {EMULATED_RAM_B} emulated")
        except (OSError, ValueError, KeyError, TypeError):
            rule(f"ram_budget_{n}", False, "no measured RAM budget in env.json")
    # the hidden rows of the trace run reached both boxes unchanged
    try:
        want = open(os.path.join(res, "trace", "hidden.sha256"), encoding="utf-8").read().split()[0]
        got = {n: open(os.path.join(res, "hidden", f"{n}.sha256"), encoding="utf-8").read().split()[0] for n in nodes}
        rule("trace_hidden", all(v == want for v in got.values()), f"trace {want[:12]}, boxes {({k: v[:12] for k, v in got.items()})}")
    except (OSError, IndexError):
        rule("trace_hidden", False, "missing hash of the hidden rows")
    node_rules(res, sess, rule)
    why = dryrun_check(res, sess)
    rule("dryrun", not why, "; ".join(why) or "the dry run passed on both boxes before the download")
    why = budget_check(res, sess)
    rule("ram_budget_remeasured", not why, "; ".join(why) or "RAM budget measured again after the model install, the pick still the covering pick")

    # the prediction: judged, from the trace, frozen before the first judged round
    frozen_sha, t_frozen = frozen_facts(res)
    pc_path = os.path.join(res, "predictions", "predict-cluster.json")
    pc: dict = {}
    try:
        pc = load(pc_path)
        tsha = open(os.path.join(res, "trace", "routing.sha256"), encoding="utf-8").read().split()[0]
        why = []
        if pc.get("routing") != "trace" or pc.get("judged") is not True:
            why.append(f"routing {pc.get('routing')!r}, judged {pc.get('judged')!r}")
        if (pc.get("routing_trace") or {}).get("sha256") != tsha:
            why.append("priced from another trace than the one on disk")
        if (pc.get("routing_trace") or {}).get("prompt_hash") != (sess.get("prompt") or {}).get("fnv1a64"):
            why.append("priced for another prompt")
        if (pc.get("workload") or {}) != {"prompt": sess["prompt"]["tokens"], "gen": sess["prompt"]["gen"]}:
            why.append(f"workload {pc.get('workload')}")
        rule("prediction_judged", not why, "; ".join(why) or "routing trace, judged, the exit prompt")
    except (OSError, ValueError, KeyError, IndexError) as e:
        rule("prediction_judged", False, f"no usable predict-cluster.json ({e})")
    why = sg.check_freeze(res)
    rds = round_dirs(res)
    for rd in rds + [os.path.join(res, "judged", "aa-last")]:
        try:
            m = load(os.path.join(rd, "meta.json"))
        except (OSError, ValueError):
            continue  # a missing aa-last has its own rule
        if m.get("frozen_sha256") != frozen_sha:
            why.append(f"{os.path.relpath(rd, res)} ran under {m.get('frozen_sha256')}")
        if t_frozen is None or not (m.get("t_start") or 0) > t_frozen:
            why.append(f"{os.path.relpath(rd, res)} started before the freeze")
    rule("freeze", not why, "; ".join(why) or f"{(frozen_sha or '')[:16]} frozen before every judged round")

    # span and pick
    plan = os.path.join(res, "plans", "judged-plan.toml")
    try:
        n_expert = int(load(os.path.join(res, "layout.json"))["n_expert"])
    except (OSError, ValueError, KeyError, TypeError):
        n_expert = None
    # ADR 0091 History 2026-11-25: judged segments are covering and run the pick's own slots, the anchors' included
    try:
        why = [] if n_expert else ["no layout.json with n_expert"]
        if n_expert:
            why += check_covering(plan, n_expert)
        got = {n: (v, r) for n, v, r in stage_slots(plan)}
        for seg in segs:
            want = tuple(seg.get("slots") or ())
            if want != got.get(seg["node"]):
                why.append(f"{seg['node']}: the anchors ran slots {list(want)}, the pick has {list(got.get(seg['node']) or ())}")
        rule("segments_covering", not why, "; ".join(why) or "every stage holds all experts, the anchors ran the pick's slots: "
             + ", ".join(f"{n} {v}+{r}" for n, (v, r) in sorted(got.items())))
    except (OSError, ValueError, KeyError) as e:
        rule("segments_covering", False, f"no plan ({e})")
    try:
        why = check_span(plan, nodes)
        info, w2 = pick(os.path.join(res, "plans", "candidates.json"), os.path.join(res, "plans", "candidates"), nodes, n_expert)
        why += w2
        if not w2:
            if sha_file(plan) != sha_file(os.path.join(res, "plans", "candidates", info["plan"])):
                why.append(f"the judged plan is not the pick {info['id']}")
            if pc and (pc.get("plan_sha256") != sha_file(plan)):
                why.append("predict-cluster.json priced another plan")
        rule("span_both_nodes", not why, "; ".join(why) or f"pick {info.get('id')} of {info.get('spanning')} spanning plans")
    except (OSError, ValueError, KeyError) as e:
        rule("span_both_nodes", False, f"no plan or candidates ({e})")

    gate, why = gate_check(res)
    out["link_gate"] = gate
    rule("link_gate", not why, "; ".join(why) or (
        f"probes {gate['gap_s']:.0f} s apart (at least {gate['min_gap_s']:.0f}), the frozen candidate re-priced: pp {gate['pp_move']:+.3f}, "
        f"tg {gate['tg_move']:+.3f} against {BAND}"))

    # the link: raw probe points per probe are reported; the rule is how far the end probe moves the frozen plan's prediction
    out["link"] = []
    try:
        start = load(os.path.join(res, "link", "probe-start.json"))
        for f in sorted(glob.glob(os.path.join(res, "link", "probe-*.json"))):
            if os.path.basename(f) != "probe-start.json":
                _, rows = check_link(start, load(f), BAND)
                out["link"].append({"probe": os.path.basename(f), "points": rows})
    except (OSError, ValueError):
        pass
    pe: dict = {}
    try:
        pe = load(os.path.join(res, "predictions", "predict-cluster-end.json")).get("prediction") or {}
        pf = (pc.get("prediction") or {}) if pc else {}
        mv = {k: pe[k] / pf[k] - 1.0 for k in ("pp_tps", "tg_tps")}
        out["link_reprice"] = {"pp_move": mv["pp_tps"], "tg_move": mv["tg_tps"], "band": BAND}
        rule("link_reprice", all(abs(v) <= BAND + 1e-9 for v in mv.values()),
             f"end-priced against frozen: pp {mv['pp_tps']:+.3f}, tg {mv['tg_tps']:+.3f} against {BAND}")
    except (OSError, ValueError, KeyError, ZeroDivisionError) as e:
        rule("link_reprice", False, f"no end-priced prediction ({e})")

    # rounds
    pr = (pc.get("prediction") or {}) if pc else {}
    per, clean = [], []
    ph = (sess.get("prompt") or {}).get("fnv1a64")
    for rd in rds:
        why = round_unclean(rd, ph, frozen_sha, t_frozen)
        cl = ss.run_of(load(os.path.join(rd, "cluster.json"))) if os.path.exists(os.path.join(rd, "cluster.json")) else {}
        row = {"round": os.path.relpath(rd, res).replace(os.sep, "/"), "pp": cl.get("prefill_tok_s"), "tg": cl.get("decode_tok_s"),
               "e_pp": ss.e(pr.get("pp_tps", 0.0), cl.get("prefill_tok_s")) if pr else None,
               "e_tg": ss.e(pr.get("tg_tps", 0.0), cl.get("decode_tok_s")) if pr else None,
               "clean": not why, "unclean_reasons": why}
        for f in seg_files(rd):
            r = ss.run_of(load(f))
            row[os.path.basename(f)[4:-5]] = ss.layer_stats(r)
        per.append(row)
        if not why:
            clean.append(row)
    out["rounds"] = per
    rule("rounds_clean", len(clean) >= CLEAN_NEEDED and len(per) <= MAX_ROUNDS,
         f"{len(clean)} clean of {len(per)} rounds (need {CLEAN_NEEDED} within {MAX_ROUNDS})")
    rule("prompt_hash", bool(per) and all(not any("prompt hash" in w for w in r["unclean_reasons"]) for r in per),
         "every round ran the frozen prompt" if per else "no rounds")
    if clean and pr:
        m = {"pp": med([c["pp"] for c in clean]), "tg": med([c["tg"] for c in clean])}
        out["median"] = m
        out["e_pp"], out["e_tg"] = ss.e(pr["pp_tps"], m["pp"]), ss.e(pr["tg_tps"], m["tg"])
        gate, held = {}, True
        for n in nodes:
            try:
                rp = load(os.path.join(res, "calib", n, "pred.json"))["prediction"]
            except (OSError, ValueError, KeyError):
                held = False
                gate[n] = None
                continue
            ms = {k: med([c[n][k] for c in clean if n in c]) for k in ("vram_pair_share", "full_hit_rate_mean", "ram_hit_rate_mean")}
            g = {"vram_share": (rp["replay_vram_share"], ms["vram_pair_share"], GATE_SHARE),
                 "full_hit_share": (rp["replay_full_hit_share"], ms["full_hit_rate_mean"], GATE_SHARE),
                 "ram_hit_rate": (rp["replay_ram_hit_rate"], ms["ram_hit_rate_mean"], GATE_RAM)}
            gate[n] = {k: {"replay": a, "measured": b, "line": ln, "diff": None if b is None else b - a,
                           "ok": b is not None and abs(b - a) <= ln} for k, (a, b, ln) in g.items()}
            held = held and all(x["ok"] for x in gate[n].values())
        out["gate"] = gate
        rule("routing_gate", held, "; ".join(f"{n} {k} {x['diff']:+.4f} against {x['line']}" for n, g in gate.items() if g
                                           for k, x in g.items() if x["diff"] is not None) or "no readings")
    else:
        rule("routing_gate", False, "no clean round or no prediction")
    j = [os.path.join(rd, "cluster.json") for rd in rds]

    def aa_rule(name: str, a: str | None, b: str | None) -> None:
        if not a or not b or not os.path.exists(a) or not os.path.exists(b):
            rule(name, False, "no such run")
            return
        r = sg.aa(load(a), load(b), BAND)
        out[name] = r
        rule(name, r["ok"], f"T {r['t_first']:.2f} -> {r['t_second']:.2f} s, {r['diff']:+.3f} against {BAND}")

    aa_rule("aa_early", j[0] if j else None, j[1] if len(j) > 1 else None)
    aa_rule("aa", j[0] if j else None, os.path.join(res, "judged", "aa-last", "cluster.json"))

    out["rules"] = rules
    out["prediction"] = {"pp_tps": pr.get("pp_tps"), "tg_tps": pr.get("tg_tps"), "frozen_sha256": frozen_sha}
    bad = [r for r in rules if not r["ok"]]
    out["clean"] = not bad
    out["verdict"] = "no verdict"
    if bad:
        out["status"] = "no verdict: not clean (" + "; ".join(f"{r['name']}: {r['detail']}" for r in bad) + ")"
        if out.get("median"):
            out["status"] += f"; on the medians e_pp {out['e_pp']:+.3f}, e_tg {out['e_tg']:+.3f} against {LINE}, not a verdict"
    else:
        m = out["median"]
        out["e_pp_end"], out["e_tg_end"] = ss.e(pe["pp_tps"], m["pp"]), ss.e(pe["tg_tps"], m["tg"])
        ok_f = abs(out["e_pp"]) <= LINE and abs(out["e_tg"]) <= LINE
        ok_e = abs(out["e_pp_end"]) <= LINE and abs(out["e_tg_end"]) <= LINE
        detail = (f"frozen e_pp {out['e_pp']:+.3f}, e_tg {out['e_tg']:+.3f}; end-priced e_pp {out['e_pp_end']:+.3f}, "
                  f"e_tg {out['e_tg_end']:+.3f}; against {LINE}")
        if ok_f and ok_e:
            out["verdict"], out["status"] = "pass", f"pass: {detail}"
        elif not ok_f and not ok_e:
            out["verdict"], out["status"] = "miss", f"miss: {detail}"
        else:
            out["status"] = f"no verdict: holds against only one of the frozen and the end-priced prediction ({detail})"
    # report only: errors against the prediction priced from a second prompt's routing trace
    try:
        p2 = load(os.path.join(res, "predictions", "predict-cluster-p2.json"))["prediction"]
        if out.get("median"):
            out["second_prompt_report"] = {"pp_tps": p2["pp_tps"], "tg_tps": p2["tg_tps"],
                                           "e_pp": ss.e(p2["pp_tps"], out["median"]["pp"]), "e_tg": ss.e(p2["tg_tps"], out["median"]["tg"])}
    except (OSError, ValueError, KeyError):
        pass
    # report only: the single-node reference of each round
    out["reference"] = [{"file": os.path.relpath(f, res).replace(os.sep, "/"), "pp": ss.run_of(load(f)).get("prefill_tok_s"),
                         "tg": ss.run_of(load(f)).get("decode_tok_s")} for f in sorted(glob.glob(os.path.join(res, "reference", "r[0-9]*.json")))]
    if out.get("median"):
        t_two = sg.PROMPT / out["median"]["pp"] + sg.GEN / out["median"]["tg"]
        ref = [sg.t_of(ss.run_of(load(f))) for f in sorted(glob.glob(os.path.join(res, "reference", "r[0-9]*.json")))
               if ss.run_of(load(f)).get("prefill_tok_s")]
        out["paired_ratio_T_single_over_two"] = (med(ref) / t_two) if ref else None  # for docs/features.md, not gated
    return out


def print_tables(o: dict) -> None:
    print("status:", o["status"])
    if o.get("second_prompt_report"):
        q = o["second_prompt_report"]
        print(f"  report only, second prompt's routing: e_pp {q['e_pp']:+.3f}, e_tg {q['e_tg']:+.3f}")
    for r in o["rounds"]:
        print(f"  {r['round']:14} pp {r['pp'] or 0:8.2f} tg {r['tg'] or 0:7.3f} e_pp {r['e_pp'] or 0:+.3f} e_tg {r['e_tg'] or 0:+.3f} "
              f"{'clean' if r['clean'] else 'UNCLEAN ' + '; '.join(r['unclean_reasons'])}")
    for r in o["rules"]:
        print(f"  rule {r['name']:20} {'ok  ' if r['ok'] else 'FAIL'} {r['detail']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary").add_argument("res")
    p = sub.add_parser("pick")
    p.add_argument("candidates")
    p.add_argument("plans_dir")
    p.add_argument("out")
    p.add_argument("--n-expert", type=int, help="drop plans that do not hold all experts in VRAM plus RAM slots on every stage")
    ls = sub.add_parser("layout-same")
    ls.add_argument("committed")
    ls.add_argument("model")
    sub.add_parser("span").add_argument("plan")
    sl = sub.add_parser("slots")
    sl.add_argument("plan")
    sl.add_argument("--n-expert", type=int, help="also check that every stage covers all experts")
    lk = sub.add_parser("link")
    lk.add_argument("start")
    lk.add_argument("end")
    lk.add_argument("--band", type=float, default=BAND)
    sub.add_parser("gate").add_argument("res")
    sub.add_parser("dryrun").add_argument("res")
    ck = sub.add_parser("chunk")
    ck.add_argument("plan")
    ck.add_argument("--prompt", type=int, default=2048)
    r = sub.add_parser("round")
    r.add_argument("dir")
    r.add_argument("--prompt-hash")
    r.add_argument("--frozen-sha")
    sp_ = sub.add_parser("spans")
    sp_.add_argument("plan")
    sp_.add_argument("--n-layer", type=int, help="also check that the stages partition the layers, one per node in order")
    pc = sub.add_parser("precheck")
    pc.add_argument("res")
    pc.add_argument("stage", choices=["calib", "freeze"])
    nr = sub.add_parser("next-round")
    nr.add_argument("res")
    a = ap.parse_args()
    if a.cmd == "summary":
        o = summarize(a.res)
        with open(os.path.join(a.res, "verdict-s.json"), "w", encoding="utf-8") as f:
            json.dump(o, f, indent=2)
        print_tables(o)
        return 0
    if a.cmd == "pick":
        info, why = pick(a.candidates, a.plans_dir, n_expert=a.n_expert)
        print(json.dumps(info))
        if why:
            return 1
        with open(a.out, "wb") as f, open(os.path.join(a.plans_dir, info["plan"]), "rb") as src:
            f.write(src.read())
        return 0
    if a.cmd == "layout-same":
        why = layout_same(a.committed, a.model)
        for w in why:
            print(f"GATE FAILED layout: {w}")
        return 1 if why else 0
    if a.cmd == "span":
        why = check_span(a.plan)
        for w in why:
            print(f"GATE FAILED span: {w}")
        return 1 if why else 0
    if a.cmd == "slots":
        why = check_covering(a.plan, a.n_expert) if a.n_expert else []
        for n, v, r in stage_slots(a.plan):
            print(f"{n} {v} {r}")
        for w in why:
            print(f"GATE FAILED slots: {w}", file=sys.stderr)
        return 1 if why else 0
    if a.cmd == "link":
        why, rows = check_link(load(a.start), load(a.end), a.band)
        for rw in rows:
            print(f"  {rw['point']:24} {rw['start_s'] * 1e3:9.3f} -> {rw['end_s'] * 1e3:9.3f} ms {rw['diff']:+.3f} {'ok' if rw['ok'] else 'OUTSIDE THE BAND'}")
        for w in why:
            print(f"raw points outside {a.band} (reported, not gated; the rule is the end-priced prediction): {w}")
        return 0
    if a.cmd == "gate":
        info, why = gate_check(a.res)
        info["pass"] = not why
        info["reasons"] = why
        os.makedirs(os.path.join(a.res, "link"), exist_ok=True)
        with open(os.path.join(a.res, "link", "gate.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print("link gate " + ("passes" if not why else "FAILS") + f" (gap {info.get('gap_s')} s, pp move {info.get('pp_move')}, tg move {info.get('tg_move')})")
        for w in why:
            print(f"GATE FAILED link_gate: {w}")
        return 1 if why else 0
    if a.cmd == "dryrun":
        why = dryrun_check(a.res, load(os.path.join(a.res, "session.json")))
        for w in why:
            print(f"GATE FAILED dryrun: {w}")
        return 1 if why else 0
    if a.cmd == "chunk":
        print(pick_chunk(a.plan, a.prompt))
        return 0
    if a.cmd == "spans":
        for n, lo, hi in plan_spans(a.plan):
            print(f"{n} {lo}..{hi}")
        why = check_partition(a.plan, a.n_layer) if a.n_layer else []
        for w in why:
            print(f"GATE FAILED spans: {w}", file=sys.stderr)
        return 1 if why else 0
    if a.cmd == "precheck":
        want = PRE_CALIB if a.stage == "calib" else PRE_FREEZE
        bad = [r for r in summarize(a.res)["rules"] if not r["ok"] and r["name"].startswith(want)]
        for r in bad:
            print(f"GATE FAILED {r['name']}: {r['detail']}")
        if not bad:
            print(f"precheck {a.stage} ok")
        return 1 if bad else 0
    if a.cmd == "round":
        why = round_unclean(a.dir, a.prompt_hash, a.frozen_sha)
        for w in why:
            print(f"UNCLEAN {os.path.basename(a.dir)}: {w}")
        return 1 if why else 0
    # next-round: another round is due while fewer than three are clean and fewer than eight have run
    sess = load(os.path.join(a.res, "session.json"))
    ph = (sess.get("prompt") or {}).get("fnv1a64")
    frozen_sha, t_frozen = frozen_facts(a.res)
    rds = round_dirs(a.res)
    clean = sum(not round_unclean(d, ph, frozen_sha, t_frozen) for d in rds)
    print(f"{clean} clean of {len(rds)} rounds")
    return 0 if clean < CLEAN_NEEDED and len(rds) < MAX_ROUNDS else 1


if __name__ == "__main__":
    sys.exit(main())
