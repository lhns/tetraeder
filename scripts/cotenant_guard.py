"""Flag a stage that ran beside a co-tenant on the shared host (ADR 0045, 20261112g: a co-tenant saturated the host
and the stage decoded 3.2x slower, with nothing in the result to say so).

  cotenant_guard.py merge --json F    reads host.decode of the stage JSON F, sets cotenant_ok / cotenant_reason in it
                                      and prints one summary line; never fails the run
Thresholds, on the decode phase of the sampler's host block:
  outside_cores > max(4, 15% of the host's physical cores (host_cpus / 2))
  smt_pairs_busy > half the cores of the engine's threads (ceil(threads / 2)), only with outside_cores >= 2, because
  the engine's own threads fill sibling pairs too
Without the inputs the flag is null (not judged), never false.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

MIN_OUTSIDE = 4.0
OUTSIDE_FRAC = 0.15
SMT_MIN_OUTSIDE = 2.0


def evaluate(decode: dict | None, threads: int | None) -> tuple[bool | None, str]:
    """(cotenant_ok, reason) from a host decode block; ok is None when the inputs are missing."""
    d = decode or {}
    outside, cpus, pairs = d.get("outside_cores"), d.get("host_cpus"), d.get("smt_pairs_busy")
    if outside is None or not cpus:
        return None, "no host CPU counters for the decode phase"
    reasons = []
    limit = max(MIN_OUTSIDE, OUTSIDE_FRAC * cpus / 2)
    if outside > limit:
        reasons.append(f"{outside:.1f} cores busy outside the container (limit {limit:.1f} of {cpus // 2} physical)")
    if pairs is not None and threads and outside >= SMT_MIN_OUTSIDE:
        cores = math.ceil(threads / 2)
        if pairs > cores / 2:
            reasons.append(f"{pairs} SMT pairs busy, more than half of the engine's {cores} cores")
    return (not reasons), "; ".join(reasons)


def merge(path: str) -> str:
    """Add cotenant_ok (and cotenant_reason) to the JSON object at path (a sidecar F.cotenant.json when it is no
    object); returns the summary line."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = None
    host = data.get("host") if isinstance(data, dict) else None
    decode = host.get("decode") if isinstance(host, dict) else None
    threads = ((data or {}).get("host_windows") or {}).get("threads") if isinstance(data, dict) else None
    ok, why = evaluate(decode, threads)
    if not isinstance(data, dict):
        path += ".cotenant.json"
        data = {}
    data["cotenant_ok"] = ok
    if why and not ok:
        data["cotenant_reason"] = why
    else:
        data.pop("cotenant_reason", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    d = decode or {}
    state = "unknown" if ok is None else ("ok" if ok else "CO-TENANT")
    line = (f"cotenant {state}: host {d.get('host_busy_cores')} cores busy, container {d.get('container_cores')}, "
            f"engine {d.get('engine_cores')}, outside {d.get('outside_cores')}, smt pairs {d.get('smt_pairs_busy')}")
    return line + (f" ({why})" if why else "")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["merge"])
    p.add_argument("--json", required=True)
    a = p.parse_args()
    print(merge(a.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
