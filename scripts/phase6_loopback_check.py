#!/usr/bin/env python3
"""Asserts the verdict of the real-binary Case F loopback (CI step "phase 6 loopback"): `LOOPBACK=1 bash scripts/phase6_exit_f.sh` ran a real
`tetraeder serve --cluster` head and two real `tetraeder node` daemons on the tiny MoE fixture, with real kill -9 and SIGSTOP faults fired by the
trigger. The verdict is `fail` there by design (20 s of load is not 180 s), so what is asserted is every rule except the load shape.

    phase6_loopback_check.py RESULTS_DIR LB_DIR

Prints one stable line per fact (no colon-space inside, so a tree node can quote it).
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    res, lb = sys.argv[1:3]
    v = json.load(open(os.path.join(res, "verdict-f.json"), encoding="utf-8"))
    bad: list[str] = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            bad.append(what)

    modes = [t["mode"] for t in v["trials"]]
    check(modes == ["worker"] * 3 + ["daemon"] * 3 + ["stop"] * 3, f"judged trials {modes}")
    for t in v["trials"]:
        failed = {k: r["detail"] for k, r in t["rules"].items() if not r["ok"]}
        check(not failed, f"{t['mode']} trial {t['trial']} rules {failed}")
        s = t["stats"]
        check((s.get("live_b_at_fault") or 0) >= 1, f"{t['mode']} {t['trial']} no request in flight on B at the fault")
        check((s.get("in_flight_errors") or 0) >= 1, f"{t['mode']} {t['trial']} no in-flight error")
        check(s.get("rejoin_s") is not None, f"{t['mode']} {t['trial']} B did not rejoin")
    session_failed = [r["name"] for r in v["session_rules"] if not r["ok"]]
    check(session_failed == ["load_shape"], f"session rules failing {session_failed}")
    check(v["reading"] == "every rule holds in every mode", f"reading {v['reading']}")
    # the real binaries, not the stand-in: no shim, the head registered its prefix listener in the daemon's node-state
    check(not os.path.exists(os.path.join(lb, "tetraeder")), "the stand-in shim is present: not the real binary")
    prefix_head = os.path.join(lb, "a", "node-state", "prefix-head")
    check(os.path.isfile(prefix_head), f"no {prefix_head}: the head's in-process replica did not register in the daemon's node-state")
    check(not os.path.exists("node-state"), "a ./node-state exists: a process used the working directory's default")
    if bad:
        for b in bad:
            print("P6_LOOPBACK_REAL FAIL " + b)
        return 1
    att = v.get("attempts", [])
    unc = max(t["stats"]["fault_uncertainty_s"] for t in v["trials"])
    ck = next(r["detail"] for r in v["session_rules"] if r["name"] == "clock_offset")
    print(f"P6_LOOPBACK_REAL real head and 2 real node daemons, 9 judged trials of 3 modes, all per-trial rules ok, {len(att)} attempts of which "
          f"{sum(not a['judged'] for a in att)} repeated")
    print(f"P6_LOOPBACK_REAL fault time known within {unc} s at most, clock rule ok ({ck})")
    print("P6_LOOPBACK_REAL serve --node-state wired, prefix-head registered in the daemon's node-state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
