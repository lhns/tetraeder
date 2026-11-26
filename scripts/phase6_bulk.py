#!/usr/bin/env python3
"""The bulk screen of the Case S pair (ADR 0091 History 2026-11-26): 64 MiB each way over raw TCP between the two boxes, a few times,
right after the gates, before the build. A slow or drifting link costs minutes here, not the build and the setup.

    phase6_bulk.py serve --port P --count N [--timeout S]     receive N transfers (a length, the bytes) and ack each with one byte
    phase6_bulk.py send --addr HOST:PORT [--mib 64]           send one transfer (connect retried, not timed), print {"seconds": ..} (ack included)
    phase6_bulk.py eval RUNS.json [--out bulk.json]           the screen's rule on {"a->b": [seconds..], "b->a": [..], "mib": 64}; exit 1 on a fail
    phase6_bulk.py derive                                     the constants of the rule, recomputed from the committed sessions

The rule. The Case S pick is priced on fitted rates (ADR 0091 History 2026-11-26), where its prefill time is a constant plus the two
32 MiB hops of one micro-batch of 2048 tokens (the a to b frame and the b to a return):

    T = C + t_ab(32 MiB) + t_ba(32 MiB),   pp = 2048 / T

C is fixed by two committed prices of the covering pick (crates/planner/tests/cluster_discovered_ram_experts.rs): pp 196.0 on rehearsal 1's
link (hops 0.663 s and 0.576 s, `phase6-rehearsal-20261125/a/probe-auto.json`) and pp 81.4 on rehearsal 2's first probe (hops 15.315 s and
0.631 s, `crates/planner/fixtures/phase6-rehearsal/link-probes/probe-start.json`). Both give C = 9.21 s, with a slope of 1.000 on the hops
(`derive` recomputes it). A transfer of 64 MiB takes twice a hop, so a hop is half the measured time.

The screen fails a pair whose measured spread would move the frozen prediction by more than the gate's 0.08: with the fastest and the
slowest repeat of each direction, T_best = C + t_ab,min + t_ba,min and T_worst = C + t_ab,max + t_ba,max, the move is T_worst / T_best - 1.
Interleaved a to b and b to a repeats make the spread cover time, not one direction. At rehearsal 1's link the two hops are 1.24 s, so the
prediction moves 0.08 only when the hops of the slowest repeats exceed the fastest by about 0.84 s (a spread of 1.7 times); pair 1 of
20261126 (64 MiB in 5.0 and 6.9 s a to b, 1.9 to 8.5 s b to a) would have failed with a move of 0.34.
It is necessary, not sufficient: raw TCP at 64 MiB is not the daemon's transport and a 32 MiB frame is priced from the probe's curve; the
link gate (two full probes 10 minutes apart, priced on the judged pick) still runs later, unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import time

MIB = 1 << 20
LIMIT = 0.08  # the link gate's band on the predicted pp (ADR 0091 answer 10)
PROMPT = 2048
# C and its inputs (see the header): recomputed by `derive`, pinned by tests/test_phase6_bulk.py
R1_PP, R1_HOPS = 196.0, (0.663, 0.576)
R2_PP, R2_HOPS = 81.4, (15.315, 0.631)
C_S = 9.21


def derive(root: str | None = None) -> dict:
    """C from the two committed prices and the hops of the two committed probes (32 MiB, interpolated as the cost model does)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import phase6_exit_s_summary as s  # noqa: PLC0415

    root = root or s.ROOT

    def hops(path: str) -> tuple[float, float]:
        r = s.link_readings(s.load(os.path.join(root, path)))
        return r["a->b 33554432"], r["b->a 33554432"]

    h1 = hops("bench/results/phase6-rehearsal-20261125/a/probe-auto.json")
    h2 = hops("crates/planner/fixtures/phase6-rehearsal/link-probes/probe-start.json")
    t1, t2 = PROMPT / R1_PP, PROMPT / R2_PP
    slope = (t2 - t1) / (sum(h2) - sum(h1))
    return {"c_r1_s": t1 - sum(h1), "c_r2_s": t2 - sum(h2), "slope": slope, "hops_r1": h1, "hops_r2": h2}


def evaluate(runs: dict, limit: float = LIMIT, c_s: float = C_S) -> dict:
    """The rule on the repeats of both directions (seconds of one 64 MiB transfer each)."""
    mib = float(runs.get("mib", 64))
    scale = 32.0 / mib  # a 32 MiB hop from a transfer of `mib`, linear in bytes at this size
    ab, ba = runs.get("a->b") or [], runs.get("b->a") or []
    why: list[str] = []
    if len(ab) < 2 or len(ba) < 2:
        return {"pass": False, "why": [f"need at least 2 repeats each way, got {len(ab)} and {len(ba)}"], "limit": limit, "c_s": c_s}
    best = c_s + (min(ab) + min(ba)) * scale
    worst = c_s + (max(ab) + max(ba)) * scale
    move = worst / best - 1.0
    if move > limit:
        why.append(f"the prediction would move {move:+.3f} between the fastest and the slowest repeat, against {limit}")
    return {"pass": not why, "why": why, "limit": limit, "c_s": c_s, "mib": mib, "move": move, "t_best_s": best, "t_worst_s": worst,
            "pp_best": PROMPT / best, "pp_worst": PROMPT / worst,
            "a_to_b_s": {"min": min(ab), "max": max(ab), "n": len(ab)}, "b_to_a_s": {"min": min(ba), "max": max(ba), "n": len(ba)},
            "rate_mb_s": {"a_to_b": mib * 1.048576 / (sorted(ab)[len(ab) // 2]), "b_to_a": mib * 1.048576 / (sorted(ba)[len(ba) // 2])}}


def recv_all(c: socket.socket, n: int) -> None:
    left = n
    while left:
        b = c.recv(min(left, 1 << 22))
        if not b:
            raise ConnectionError("closed before the end of the transfer")
        left -= len(b)


def serve(port: int, count: int, timeout: float) -> int:
    ls = socket.socket()
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind(("0.0.0.0", port))
    ls.listen(4)
    ls.settimeout(timeout)
    for _ in range(count):
        c, _ = ls.accept()
        with c:
            c.settimeout(timeout)
            hdr = b""
            while len(hdr) < 8:
                b = c.recv(8 - len(hdr))
                if not b:
                    raise ConnectionError("no header")
                hdr += b
            (n,) = struct.unpack(">Q", hdr)
            recv_all(c, n)
            c.sendall(b"k")
    ls.close()
    return 0


def connect(addr: str, wait_s: float = 45.0) -> socket.socket:
    """The server is started a moment before the first send: retry the connection for `wait_s`."""
    host, _, port = addr.rpartition(":")
    t_end = time.monotonic() + wait_s
    while True:
        try:
            return socket.create_connection((host, int(port)), timeout=30)
        except OSError:
            if time.monotonic() > t_end:
                raise
            time.sleep(0.5)


def send(addr: str, mib: int) -> float:
    buf = os.urandom(1 << 20)
    with connect(addr) as c:
        c.settimeout(600)
        t0 = time.perf_counter()
        c.sendall(struct.pack(">Q", mib * MIB))
        for _ in range(mib):
            c.sendall(buf)
        if c.recv(1) != b"k":
            raise ConnectionError("no ack")
        return time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--count", type=int, default=4)
    s.add_argument("--timeout", type=float, default=180.0)
    d = sub.add_parser("send")
    d.add_argument("--addr", required=True)
    d.add_argument("--mib", type=int, default=64)
    e = sub.add_parser("eval")
    e.add_argument("runs")
    e.add_argument("--out")
    sub.add_parser("derive")
    a = ap.parse_args()
    if a.cmd == "serve":
        return serve(a.port, a.count, a.timeout)
    if a.cmd == "send":
        print(json.dumps({"seconds": send(a.addr, a.mib), "mib": a.mib}))
        return 0
    if a.cmd == "derive":
        print(json.dumps(derive(), indent=2))
        return 0
    with open(a.runs, encoding="utf-8") as f:
        r = evaluate(json.load(f))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
    print("bulk screen " + ("passes" if r["pass"] else "FAILS") + (f": the prediction would move {r['move']:+.3f} (limit {r['limit']})" if "move" in r else ""))
    for w in r["why"]:
        print(f"GATE FAILED bulk: {w}")
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
