#!/usr/bin/env python3
"""Reads a Case F load file (scripts/phase6_load.py) and the fault time and applies the client-side pass rules
of ADR 0091 Case F, rehearsal only (no verdict):

  inflight_errors_in_bound   every request in flight at the fault that failed ended with one clean error within
                             `bound` seconds (node_down_after + 1 s = 3 s)
  one_clean_error_each       an error is an HTTP 5xx before headers, or one error event followed by nothing
  none_after_bound_failed    every request submitted from `bound` after the fault on succeeded
  other_failures             failures that are neither in flight at the fault nor submitted in its first `bound`
                             seconds (a request on the other replica can only fail here, or in flight)
  exactly_once               each request has exactly one outcome (a [DONE] with a finish reason, or an error)

Not observable from the client: which replica an in-flight request ran on (the head's log and the node logs
carry it), requests waiting at the router, and the rejoin time. The driver reads those from the logs.

    python3 scripts/phase6_casef.py load.jsonl --fault 1795575000.5 [--restore 1795575060.5] --out result.json
"""

import argparse
import json


def classify(r):
    """ok, error or ambiguous."""
    err = r.get("error")
    if err is None and r.get("done") and r.get("finish") in ("stop", "length", "tool_calls"):
        return "ok"
    if err is not None and not (r.get("done") and r.get("finish") in ("stop", "length")):
        return "error"
    return "ambiguous"


def clean_error(r):
    """One error, nothing after it: an HTTP error before the stream, or an error event that ended it."""
    if r.get("status") not in (None, 200):
        return True
    return r.get("error") is not None and r.get("after_error", 0) == 0 and not r.get("finish")


def analyse(rows, t_fault, bound, t_restore=None):
    for r in rows:
        r["class"] = classify(r)
    infl = [r for r in rows if r["t_submit"] < t_fault <= (r["t_end"] or 1e18)]
    infl_err = [r for r in infl if r["class"] != "ok"]
    after = [r for r in rows if r["t_submit"] >= t_fault + bound]
    early = [r for r in rows if t_fault <= r["t_submit"] < t_fault + bound]
    before_fail = [r for r in rows if r["t_end"] and r["t_end"] < t_fault and r["class"] != "ok"]
    other = [r for r in rows if r["class"] != "ok" and r not in infl and r not in early]
    delays = sorted(round(r["t_end"] - t_fault, 3) for r in infl_err)
    res = {
        "t_fault": t_fault,
        "bound_s": bound,
        "requests": len(rows),
        "ok": sum(r["class"] == "ok" for r in rows),
        "error": sum(r["class"] == "error" for r in rows),
        "ambiguous": sum(r["class"] == "ambiguous" for r in rows),
        "in_flight_at_fault": len(infl),
        "in_flight_ok": len(infl) - len(infl_err),
        "in_flight_errors": len(infl_err),
        "in_flight_error_delay_s": delays,
        "in_flight_error_messages": sorted({(r.get("error") or "")[:120] for r in infl_err}),
        "submitted_in_bound": len(early),
        "submitted_in_bound_failed": sum(r["class"] != "ok" for r in early),
        "submitted_after_bound": len(after),
        "submitted_after_bound_failed": sum(r["class"] != "ok" for r in after),
        "failed_before_fault": len(before_fail),
        "other_failures": len(other),
        "tokens_after_error": sum(r.get("after_error", 0) for r in rows),
    }
    res["rules"] = {
        "inflight_errors_in_bound": all(d <= bound for d in delays),
        "one_clean_error_each": all(clean_error(r) for r in infl_err),
        "none_after_bound_failed": res["submitted_after_bound_failed"] == 0,
        "other_failures_zero": res["other_failures"] == 0,
        "exactly_once": res["ambiguous"] == 0 and res["tokens_after_error"] == 0,
    }
    if t_restore is not None:
        res["t_restore"] = t_restore
        ok_after = sorted(r["t_end"] for r in rows if r["class"] == "ok" and r["t_submit"] >= t_restore)
        res["first_ok_submitted_after_restore_ended_s"] = round(ok_after[0] - t_restore, 2) if ok_after else None
    # per 10 s bucket relative to the fault: ok and failed ends
    buckets = {}
    for r in rows:
        b = int((r["t_end"] - t_fault) // 10) * 10
        k = buckets.setdefault(b, [0, 0])
        k[0 if r["class"] == "ok" else 1] += 1
    res["ends_per_10s_ok_failed"] = {str(b): v for b, v in sorted(buckets.items())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("load")
    ap.add_argument("--fault", type=float, required=True, help="epoch seconds of the fault")
    ap.add_argument("--restore", type=float)
    ap.add_argument("--bound", type=float, default=3.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    with open(a.load) as f:
        rows = [json.loads(x) for x in f if x.strip()]
    res = analyse(rows, a.fault, a.bound, a.restore)
    text = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
