"""Check and record the RAM balloon of scripts/exit_run.sh (a dead balloon means a run without memory pressure).

  balloon_guard.py check --pid P --target BYTES [--min-frac 0.9]   prints "ok" or the reason; exit 1 when not ok
  balloon_guard.py merge --json F --ok true|false [--reason R]     sets balloon_ok / balloon_reason in the stage JSON
                                                                    (a sidecar F.balloon.json when F is no object)
  balloon_guard.py wait --pid P --target BYTES [--timeout S]       waits until the balloon holds its FULL target
  balloon_guard.py budget --cgroup DIR [--env-json F] [--hw TOML]  the planner's RAM budget under the balloon (ADR 0092)

The budget is `memory.max - memory.current + inactive_file - 1 GiB`, read after the balloon is full: the
container's own overhead (the balloon's page tables, other processes) is already in `memory.current`, the page
cache the kernel can drop is `inactive_file`, and the GiB covers the engine's page tables, slab, allocator growth
and the reclaim floor.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

GIB = 1 << 30


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


def read_cgroup(cg: str) -> dict[str, int]:
    """memory.max, memory.current and inactive_file (memory.stat) of a cgroup v2 directory, bytes."""

    def one(name: str) -> int | None:
        text = open(os.path.join(cg, name), encoding="utf-8").read().strip()
        return None if text == "max" else int(text)

    stat: dict[str, int] = {}
    with open(os.path.join(cg, "memory.stat"), encoding="utf-8") as f:
        for line in f:
            key, _, val = line.partition(" ")
            if val.strip().isdigit():
                stat[key] = int(val)
    mx, cur = one("memory.max"), one("memory.current")
    if mx is None or cur is None:
        raise ValueError(f"{cg}: no memory limit, so no budget under a balloon")
    return {"memory_max": mx, "memory_current": cur, "inactive_file": stat.get("inactive_file", 0)}


def ram_budget(c: dict[str, int], reserve: int = GIB) -> int:
    """What the planner may plan RAM against: the cgroup's headroom with the page cache it can drop."""
    return c["memory_max"] - c["memory_current"] + c["inactive_file"] - reserve


def record_budget(cg: str, env_json: str | None, hw: str | None, reserve: int = GIB) -> dict[str, int]:
    """Reads the cgroup, adds `ram_budget` and its inputs to env.json and sets cpu0 capacity in the graph."""
    inputs = read_cgroup(cg)
    out = {**inputs, "reserve_bytes": reserve, "ram_budget_bytes": ram_budget(inputs, reserve)}
    if env_json:
        try:
            with open(env_json, encoding="utf-8") as f:
                env = json.load(f)
        except (OSError, ValueError):
            env = {}
        if not isinstance(env, dict):
            env = {}
        env["ram_budget"] = out
        with open(env_json, "w", encoding="utf-8") as f:
            json.dump(env, f, indent=2)
    if hw:
        with open(hw, encoding="utf-8") as f:
            text = f.read()
        with open(hw, "w", encoding="utf-8") as f:
            f.write(set_cpu_capacity(text, out["ram_budget_bytes"]))
    return out


def set_cpu_capacity(text: str, budget: int) -> str:
    """The graph TOML with `capacity_bytes` of the `cpu0` device set to `budget`."""
    m = re.search(r'\[\[devices\]\]\s*\nid = "cpu0"', text)
    if not m:
        raise ValueError("no device cpu0 in the graph")
    nxt = re.search(r"\n\[\[", text[m.end():])
    end = m.end() + nxt.start() if nxt else len(text)
    block, n = re.subn(r"capacity_bytes = \d+", f"capacity_bytes = {budget}", text[m.start():end], count=1)
    if n != 1:
        raise ValueError("cpu0 has no capacity_bytes")
    return text[: m.start()] + block + text[end:]


def wait_full(check, timeout: float, interval: float = 5.0) -> tuple[bool, str]:
    """Polls `check() -> (ok, why)` until it is ok or `timeout` seconds pass."""
    deadline = time.monotonic() + timeout
    while True:
        ok, why = check()
        if ok or time.monotonic() >= deadline:
            return ok, why
        time.sleep(interval)


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wait")
    w.add_argument("--pid", type=int, required=True)
    w.add_argument("--target", type=int, required=True)
    w.add_argument("--timeout", type=float, default=1800.0)
    b = sub.add_parser("budget")
    b.add_argument("--cgroup", default="/sys/fs/cgroup")
    b.add_argument("--env-json")
    b.add_argument("--hw")
    b.add_argument("--reserve", type=int, default=GIB)
    c = sub.add_parser("check")
    c.add_argument("--pid", type=int, required=True)
    c.add_argument("--target", type=int, required=True)
    c.add_argument("--min-frac", type=float, default=0.9)
    m = sub.add_parser("merge")
    m.add_argument("--json", required=True)
    m.add_argument("--ok", required=True, choices=["true", "false"])
    m.add_argument("--reason", default="")
    a = p.parse_args()
    if a.cmd == "wait":
        ok, why = wait_full(lambda: evaluate(read_status(a.pid), a.target, 1.0), a.timeout)
        print(why)
        return 0 if ok else 1
    if a.cmd == "budget":
        try:
            print(json.dumps(record_budget(a.cgroup, a.env_json, a.hw, a.reserve)))
        except (OSError, ValueError) as e:
            print(f"ram budget not measured: {e}", file=sys.stderr)
            return 2
        return 0
    if a.cmd == "check":
        ok, why = evaluate(read_status(a.pid), a.target, a.min_frac)
        print(why)
        return 0 if ok else 1
    merge(a.json, a.ok == "true", a.reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
