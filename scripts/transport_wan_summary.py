#!/usr/bin/env python3
"""Markdown tables from the transport-bench JSON files of one result dir (ADR 0023).

usage: transport_wan_summary.py <scenario dir>...   (each holds <transport>-<cipher>-r<rep>.json)
Per (transport, cipher, size): median over the repetitions of p50, p90, p99 and throughput, plus the range of p50.
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

NAMES = {"tcp": "TCP", "tls": "TCP+TLS", "quic": "QUIC"}
SIZES = {8192: "8 KB", 1048576: "1 MB"}


def load(d: Path):
    cells = defaultdict(list)
    for f in sorted(d.glob("*-r*.json")):
        t, cipher, _ = f.stem.split("-")
        for r in json.loads(f.read_text()):
            cells[(r["size"], cipher, t)].append(r)
    return cells


def ms(us: float) -> str:
    return f"{us / 1000:.1f}" if us >= 10000 else f"{us / 1000:.2f}"


def table(d: Path) -> str:
    cells = load(d)
    out = [
        f"### {d.parent.name} / {d.name}",
        "",
        "| Size | Cipher | Transport | reps | p50 ms | p50 range ms | p90 ms | p99 ms | MiB/s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for size in sorted({k[0] for k in cells}):
        for cipher in ("aes", "chacha"):
            for t in ("tcp", "tls", "quic"):
                rows = cells.get((size, cipher, t))
                if not rows:
                    continue
                p50 = [r["p50_us"] for r in rows]
                out.append(
                    f"| {SIZES.get(size, size)} | {cipher} | {NAMES[t]} | {len(rows)} | {ms(st.median(p50))} "
                    f"| {ms(min(p50))} to {ms(max(p50))} | {ms(st.median(r['p90_us'] for r in rows))} "
                    f"| {ms(st.median(r['p99_us'] for r in rows))} "
                    f"| {st.median(r['throughput_mib_s'] for r in rows):.0f} |"
                )
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(table(Path(arg)))
