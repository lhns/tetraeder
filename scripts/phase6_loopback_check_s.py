#!/usr/bin/env python3
"""Asserts what the real-binary Case S loopback left (CI step "phase 6 Case S loopback"): `LOOPBACK=1 bash scripts/phase6_exit_s.sh` with
STAGES="gates setup budget plan fetch calib merge judged" ran the dry run of the exit session on the tiny sharded MoE with the real
`tetraeder` binary: the model cut into shards and installed under one id on both nodes, the pinned head-led two-node plan, the per-node
calibration (real exit_regime runs on the CPU, the fit), `cluster merge` of the fitted graphs and one real `tetraeder cluster judged-round`,
which `phase6_exit_s_summary.py round` accepts.

    phase6_loopback_check_s.py RESULTS_DIR LB_DIR

What it does not assert: any timing rule (the fit band is set loose, the numbers are a CPU runner's) and the frozen prediction (nothing is
priced or frozen in the dry run). The stand-ins of this run are two files that a CI runner cannot make: the RAM budget (read by the real
balloon_guard.py from a made-up cgroup directory) and the GPU throttle record (a made-up line). Prints one stable line per fact (no colon-space
inside, so a tree node can quote it).
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
    if bad:
        for b in bad:
            print("P6_LOOPBACK_S FAIL " + b)
        return 1
    print(f"P6_LOOPBACK_S real binaries, tiny MoE in 3 shards under one id {mid[:12]} on 2 nodes, pinned head-led plan with covering slots 4+8")
    print("P6_LOOPBACK_S per-node calibration (anchors, fit, replay) and cluster merge of the fitted graphs done")
    print(f"P6_LOOPBACK_S one real judged-round accepted by the summary round check, pp {run['prefill_tok_s']:.1f} tg {run['decode_tok_s']:.2f} tok/s (CPU runner, wiring only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
