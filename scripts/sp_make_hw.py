#!/usr/bin/env python3
"""The hardware graph of one stream-predict arm: the RTX 3090 spec-sheet prior with this box's inputs.

Only box facts change, not the cost model: the container's RAM budget of the arm, the GPU's total memory, the CPU thread
quota and the disk's measured queue curve (`tetraeder-bench disk-curve`). Everything else stays the prior of
`crates/planner/fixtures/rtx3090_prior.toml`. `--keep-disk-prior` leaves the NVMe curve at the prior (the prediction
with no measurement of this box's disk).

    python scripts/sp_make_hw.py --prior P.toml --ram-bytes N --gpu-bytes N --threads T --disk-curve C.json --out hw.toml
"""

from __future__ import annotations

import argparse
import json
import re
import sys


def section(text: str, dev_id: str) -> tuple[int, int]:
    """Span of the [[devices]] block with this id."""
    m = re.search(r'\[\[devices\]\]\s*\nid = "%s"' % re.escape(dev_id), text)
    if not m:
        sys.exit(f"no device {dev_id} in the prior")
    nxt = re.search(r"\n\[\[", text[m.end():])
    end = m.end() + nxt.start() if nxt else len(text)
    return m.start(), end


def sub_in(text: str, dev_id: str, pattern: str, repl: str) -> str:
    a, b = section(text, dev_id)
    block, n = re.subn(pattern, repl, text[a:b], count=1, flags=re.S)
    if n != 1:
        sys.exit(f"{dev_id}: pattern {pattern!r} not found")
    return text[:a] + block + text[b:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--ram-bytes", type=int, required=True)
    ap.add_argument("--gpu-bytes", type=int, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--disk-curve")
    ap.add_argument("--keep-disk-prior", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    text = open(a.prior, encoding="utf-8").read()
    text = sub_in(text, "cpu0", r"capacity_bytes = \d+", f"capacity_bytes = {a.ram_bytes}")
    text = sub_in(text, "cpu0", r"threads = \d+", f"threads = {a.threads}")
    text = sub_in(text, "gpu0", r"capacity_bytes = \d+", f"capacity_bytes = {a.gpu_bytes}")
    if not a.keep_disk_prior:
        if not a.disk_curve:
            sys.exit("--disk-curve or --keep-disk-prior")
        pts = json.load(open(a.disk_curve, encoding="utf-8"))["queue_curve"]
        pts = sorted(pts, key=lambda p: p["depth"])
        curve = ",\n".join(f'  {{ depth = {p["depth"]}, bytes_per_s = {p["bytes_per_s"]:.6e} }}' for p in pts)
        at16 = next((p["bytes_per_s"] for p in pts if p["depth"] >= 16), pts[-1]["bytes_per_s"])
        text = sub_in(text, "nvme0", r"queue_curve = \[.*?\n\]", f"queue_curve = [\n{curve},\n]")
        text = sub_in(
            text, "nvme0", r'bandwidth_bytes_per_s = \{[^}]*\}',
            f'bandwidth_bytes_per_s = {{ value = {at16:.6e}, source = "measured" }}',
        )
    open(a.out, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main()
