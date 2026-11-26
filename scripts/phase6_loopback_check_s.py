#!/usr/bin/env python3
"""Asserts what the real-binary Case S loopback left (CI step "phase 6 Case S loopback"): `LOOPBACK=1 bash scripts/phase6_exit_s.sh` with
STAGES="gates bulk setup budget plan linkgate fetch calib predict judged aalast linkend summary" ran every stage of the dry run of the exit session on
the tiny sharded MoE with the real `tetraeder`, `tetraeder-plan` and `tetraeder-bench` binaries: the bulk screen, the model cut into shards and
installed under one id on both nodes, the pinned head-led two-node plan, the link gate with a short gap, the RAM budget measured again and the pick
checked again, the per-node calibration (real exit_regime runs on the CPU, the fit), `cluster merge` of the fitted graphs, `predict --cluster --judged`
with the freeze, at least three judged rounds of the real `tetraeder cluster judged-round` with the early A/A, the A/A last, the end probe, and the
real summary on the session marked `tiny`.

    phase6_loopback_check_s.py RESULTS_DIR LB_DIR

What it does not assert: any timing rule (the bands are set wide, the numbers are a CPU runner's) or a verdict (the summary says "no verdict" on a tiny
session by design). The stand-ins of this run are three things a CI runner cannot make: the RAM budget (read by the real balloon_guard.py from a
made-up cgroup directory), the GPU throttle record (a made-up line) and the cluster graph (no GPU on a runner: the committed two-3090 graph carries
the real probe's link, plans/hw-standin.json). Prints one stable line per fact (no colon-space inside, so a tree node can quote it).
"""

from __future__ import annotations

import filecmp
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = os.path.join(HERE, "..", "tests", "fixtures", "phase6", "tiny-case-s-plan.toml")


def main() -> int:
    res, lb = sys.argv[1:3]
    bad: list[str] = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            bad.append(what)

    def load(*p: str):
        with open(os.path.join(res, *p), encoding="utf-8") as f:
            return json.load(f)

    # the real binary, not the stand-in
    check(not os.path.exists(os.path.join(lb, "tetraeder")), "the stand-in shim is present: not the real binary")
    # the model: three shards, one id on both nodes, installed with its manifest
    sess = load("session.json")
    mid = sess.get("model_sha256", "")
    check(bool(re.fullmatch(r"[0-9a-f]{64}", mid)), f"model_sha256 {mid!r}")
    for n in ("a", "b"):
        d = os.path.join(lb, n, "models")
        names = sorted(os.listdir(d)) if os.path.isdir(d) else []
        check(any(x.startswith(mid) for x in names), f"node {n} has no model installed under {mid[:12]}: {names}")
        man = os.path.join(d, mid, "manifest.txt")
        check(os.path.isfile(man) and len(open(man, encoding="utf-8").read().split("\n")) >= 3, f"node {n}: no manifest of three shards")
        check(open(os.path.join(lb, n, "model.sha256"), encoding="utf-8").read().strip() == mid, f"node {n}: model.sha256 is not the id")
    # the plan is the pinned one, its slots are the pick's and cover the experts
    check(filecmp.cmp(os.path.join(res, "plans", "judged-plan.toml"), PLAN, shallow=False), "plans/judged-plan.toml is not the pinned plan")
    slots = open(os.path.join(res, "plans", "slots.txt"), encoding="utf-8").read().split()
    check(slots == ["a", "4", "8", "b", "4", "8"], f"slots {slots}")
    check([s["slots"] for s in sess["segments"]] == [[4, 8], [4, 8]], f"session segments {sess.get('segments')}")
    check(load("layout.json")["n_expert"] == 8, "layout n_expert")
    # calibration on both nodes: anchors, the fit and the replay, from real exit_regime runs
    for n in ("a", "b"):
        for f in ("a1-r1", "a1-r2", "a2-r1", "a2-r2"):
            r = load("calib", n, "anchors", f"{f}.json")
            run = (r.get("runs") or [r])[-1]
            check(bool(run.get("prefill_tok_s")) and bool(run.get("decode_tok_s")), f"{n} {f} has no rates")
            # the anchors ran the pinned plan's micro-batch (ubatch 16, two micro-batches, a 64-token prompt: batches of 16), stamped by sc_box.sh
            check(r.get("exit_ubatch") == 16, f"{n} {f} ran batches of {r.get('exit_ubatch')} tokens, the pinned plan's micro-batch is 16")
        check(os.path.getsize(os.path.join(res, "calib", n, "fitted.toml")) > 0, f"{n} fitted.toml is empty")
        check("band" in load("calib", n, "pred.json")["prediction"], f"{n} pred.json has no fit band")
    # the merge of the fitted graphs into the cluster graph (a real `tetraeder cluster merge`)
    hw = load("predictions", "hw-cluster.json")
    ids = {d["id"] for d in hw["devices"]}
    check({"a/cpu0", "b/cpu0"} <= ids, f"hw-cluster.json devices {sorted(ids)}")
    # the judged round: a real `tetraeder cluster judged-round`, accepted by the summary's own check
    rd = os.path.join(res, "judged", "r1")
    cl = load("judged", "r1", "cluster.json")
    run = cl["runs"][-1]
    check(run.get("prefill_tok_s", 0) > 0 and run.get("decode_tok_s", 0) > 0, "judged round has no rates")
    for n in ("a", "b"):
        check(os.path.isfile(os.path.join(rd, f"seg-{n}.json")), f"no seg-{n}.json")
    ph = sess["prompt"]["fnv1a64"]
    r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "phase6_exit_s_summary.py"), "round", rd, "--prompt-hash", ph], capture_output=True, text=True)
    check(r.returncode == 0, f"phase6_exit_s_summary.py round rejects the round: {r.stdout.strip()}")
    # the bulk screen: raw TCP both ways, interleaved, and its verdict
    runs = load("link", "bulk-runs.json")
    check(len(runs["a->b"]) >= 3 and len(runs["b->a"]) >= 3, f"bulk repeats {runs}")
    check(load("link", "bulk.json").get("pass") is True, "link/bulk.json does not pass")
    # the stand-in graph went in, the real probe's link with it
    hw = load("plans", "hw-plan.json")
    check({"a/gpu0", "b/gpu0"} <= {d["id"] for d in hw["devices"]}, "plans/hw-plan.json has no GPU: the stand-in graph was not used")
    # the link gate: two full probes a short gap apart, the judged pick priced with each on the fitted rates
    gate = load("link", "gate.json")
    check(gate.get("pass") is True and gate.get("gap_s", 0) >= 1, f"link gate {gate}")
    # the RAM budget measured again after the install and the pick checked again
    rp = load("plans", "repick.json")
    check(rp.get("same") is True, f"repick {rp}")
    check({n: sorted(v) for n, v in sess["budget"].items()} == {"a": ["installed", "plan"], "b": ["installed", "plan"]}, f"session budget {sess.get('budget')}")
    # the freeze: the prediction is judged, priced from the trace on disk for the session's prompt, and the rounds ran under its hash
    pc = load("predictions", "predict-cluster.json")
    check(pc.get("judged") is True and pc.get("routing") == "trace" and pc["prediction"]["pp_tps"] > 0, "predict-cluster.json is not a judged trace prediction")
    check(pc["workload"] == {"prompt": 64, "gen": 16} == {"prompt": sess["prompt"]["tokens"], "gen": sess["prompt"]["gen"]}, f"workload {pc['workload']}")
    frozen = open(os.path.join(res, "predictions", "FROZEN.sha256"), encoding="utf-8").read().split()[0]
    rounds = sorted(d for d in os.listdir(os.path.join(res, "judged")) if re.fullmatch(r"r\d+", d))
    check(len(rounds) >= 3, f"rounds {rounds}")
    for d in rounds:
        check(load("judged", d, "meta.json").get("frozen_sha256") == frozen, f"round {d} ran under another hash")
    check(os.path.isfile(os.path.join(res, "judged", "aa-early.txt")) and os.path.isfile(os.path.join(res, "judged", "aa-last", "cluster.json")), "no early A/A or A/A last")
    check(os.path.isfile(os.path.join(res, "predictions", "predict-cluster-end.json")) and os.path.isfile(os.path.join(res, "link", "probe-end.json")), "no end probe or end price")
    # the summary: the real one, on a session marked tiny
    v = load("verdict-s.json")
    check(v.get("tiny") is True and v.get("clean") is True and v.get("verdict") == "no verdict", f"summary {v.get('status')}")
    if bad:
        for b in bad:
            print("P6_LOOPBACK_S FAIL " + b)
        return 1
    print(f"P6_LOOPBACK_S real binaries, tiny MoE in 3 shards under one id {mid[:12]} on 2 nodes, pinned head-led plan with covering slots 4+8")
    print("P6_LOOPBACK_S per-node calibration (anchors in the pick's micro-batch of 16 tokens, fit, replay) and cluster merge of the fitted graphs done")
    print(f"P6_LOOPBACK_S bulk screen, link gate, re-measured budget and re-pick, predict --judged with the freeze, {len(rounds)} judged rounds with the early A/A and the A/A last, end probe and summary")
    print(f"P6_LOOPBACK_S one real judged-round accepted by the summary round check, pp {run['prefill_tok_s']:.1f} tg {run['decode_tok_s']:.2f} tok/s (CPU runner, wiring only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
