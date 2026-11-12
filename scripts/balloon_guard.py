"""Check and record the RAM balloon of scripts/exit_run.sh (a dead balloon means a run without memory pressure).

  balloon_guard.py check --pid P --target BYTES [--min-frac 0.9]   prints "ok" or the reason; exit 1 when not ok
  balloon_guard.py merge --json F --ok true|false [--reason R]     sets balloon_ok / balloon_reason in the stage JSON
                                                                    (a sidecar F.balloon.json when F is no object)
"""

from __future__ import annotations

import argparse
import json
import sys


def parse_status(text: str) -> tuple[str, int]:
    """(state letter, VmRSS bytes) from /proc/<pid>/status text; ("", 0) when a field is missing."""
    state, rss = "", 0
    for line in text.splitlines():
        if line.startswith("State:"):
            state = line.split()[1]
        elif line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
    return state, rss


def evaluate(status: str | None, target: int, min_frac: float = 0.9) -> tuple[bool, str]:
    """status is the text of /proc/<pid>/status, None when the process is gone."""
    if status is None:
        return False, "balloon process gone (OOM-killed or exited)"
    state, rss = parse_status(status)
    if state in ("Z", "X"):
        return False, f"balloon process is {state} (dead)"
    if rss < min_frac * target:
        return False, f"balloon RSS {rss} B is below {min_frac:.0%} of target {target} B"
    return True, "ok"


def read_status(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def merge(path: str, ok: bool, reason: str) -> str:
    """Add balloon_ok (and balloon_reason when not ok) to the JSON object at path; returns the file written."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = None
    if not isinstance(data, dict):
        path += ".balloon.json"
        data = {}
    data["balloon_ok"] = ok
    if reason and not ok:
        data["balloon_reason"] = reason
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--pid", type=int, required=True)
    c.add_argument("--target", type=int, required=True)
    c.add_argument("--min-frac", type=float, default=0.9)
    m = sub.add_parser("merge")
    m.add_argument("--json", required=True)
    m.add_argument("--ok", required=True, choices=["true", "false"])
    m.add_argument("--reason", default="")
    a = p.parse_args()
    if a.cmd == "check":
        ok, why = evaluate(read_status(a.pid), a.target, a.min_frac)
        print(why)
        return 0 if ok else 1
    merge(a.json, a.ok == "true", a.reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
