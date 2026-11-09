#!/usr/bin/env python3
"""Scores a Claude Code benchmark directory (scripts/claude_code_proxy.py output) into summary.json.

    python scripts/claude_code_proxy_summary.py bench/results/<date>/claude-code-proxy

Reads session-<engine>-r<rep>-<task>.json and proxy-*.jsonl; the engine, rep and task come from the file contents.
"""

import glob
import json
import math
import os
import statistics
import sys


def percentile(values, p):
    """Nearest rank; None for an empty list."""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def median(values):
    return statistics.median(values) if values else None


def _r(x, nd=3):
    return None if x is None else round(x, nd)


def load_dir(path):
    sessions, requests = [], []
    for f in sorted(glob.glob(os.path.join(path, "session-*.json"))):
        with open(f, encoding="utf-8") as fh:
            sessions.append(json.load(fh))
    for f in sorted(glob.glob(os.path.join(path, "proxy-*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            requests += [json.loads(line) for line in fh if line.strip()]
    return sessions, requests


def _turns(session):
    c = session.get("claude") or {}
    return c.get("num_turns")


def engine_stats(sessions, requests):
    """Numbers for the sessions and proxy lines of one engine."""
    successes = sum(1 for s in sessions if s.get("success"))
    per_task = {}
    for s in sessions:
        t = per_task.setdefault(s["task"], {"runs": 0, "successes": 0})
        t["runs"] += 1
        t["successes"] += 1 if s.get("success") else 0
    for t in per_task.values():
        t["rate"] = _r(t["successes"] / t["runs"])
    non200 = [r for r in requests if r.get("status") != 200 or r.get("error_type")]
    is_error = sum(1 for s in sessions if (s.get("claude") or {}).get("is_error"))
    ttft = [r["ttft_s"] for r in requests if r.get("ttft_s") is not None]
    decode = [r["decode_tok_s"] for r in requests if r.get("decode_tok_s") is not None]
    wall = [s["wall_s"] for s in sessions if s.get("wall_s") is not None]

    def tok(key):
        return sum(r.get(key) or 0 for r in requests)

    errors_by_class = {}
    for r in requests:
        for cls, n in (r.get("tool_errors") or {}).items():
            errors_by_class[cls] = errors_by_class.get(cls, 0) + n
    tool_uses = tok("tool_uses")
    tool_errors = sum(errors_by_class.values())
    return {
        "runs": len(sessions),
        "successes": successes,
        "success_rate": _r(successes / len(sessions)) if sessions else None,
        "per_task": dict(sorted(per_task.items())),
        "turns_median": median([t for t in map(_turns, sessions) if t is not None]),
        "timeouts": sum(1 for s in sessions if s.get("timed_out")),
        "is_error_sessions": is_error,
        "non_200_requests": len(non200),
        "protocol_errors": len(non200) + is_error,
        "requests": len(requests),
        "ttft_median_s": _r(median(ttft)),
        "ttft_p90_s": _r(percentile(ttft, 90)),
        "decode_tok_s_median": _r(median(decode)),
        "input_tokens_total": tok("input_tokens"),
        "output_tokens_total": tok("output_tokens"),
        "cache_read_tokens_total": tok("cache_read_tokens"),
        "wall_s_median": _r(median(wall), 1),
        "wall_s_total": _r(sum(wall), 1),
        "tool_uses_total": tool_uses,
        "tool_errors_by_class": dict(sorted(errors_by_class.items())),
        "tool_names_unknown_total": tok("tool_names_unknown"),
        "leaked_markers_total": tok("leaked_markers"),
        "tool_error_rate": _r(tool_errors / tool_uses) if tool_uses else None,
    }


def _task_row(sessions, requests):
    wall = [s["wall_s"] for s in sessions if s.get("wall_s") is not None]
    ttft = [r["ttft_s"] for r in requests if r.get("ttft_s") is not None]
    decode = [r["decode_tok_s"] for r in requests if r.get("decode_tok_s") is not None]
    return {
        "runs": len(sessions),
        "successes": sum(1 for s in sessions if s.get("success")),
        "turns_median": median([t for t in map(_turns, sessions) if t is not None]),
        "wall_s_median": _r(median(wall), 1),
        "requests": len(requests),
        "ttft_median_s": _r(median(ttft)),
        "decode_tok_s_median": _r(median(decode)),
    }


def summarize(sessions, requests):
    engines = sorted({s["engine"] for s in sessions} | {r["engine"] for r in requests})
    out = {"engines": {}, "comparison": {}}
    for e in engines:
        out["engines"][e] = engine_stats(
            [s for s in sessions if s["engine"] == e], [r for r in requests if r["engine"] == e]
        )
    tasks = sorted({s["task"] for s in sessions})
    for task in tasks:
        row = {}
        for e in engines:
            ss = [s for s in sessions if s["engine"] == e and s["task"] == task]
            rr = [r for r in requests if r["engine"] == e and r["task"] == task]
            if ss:
                row[e] = _task_row(ss, rr)
        a, b = row.get("tetraeder"), row.get("llama-server")
        if a and b and a["wall_s_median"] and b["wall_s_median"]:
            row["wall_ratio_tetraeder_over_llama"] = _r(a["wall_s_median"] / b["wall_s_median"])
        if a and b and a["decode_tok_s_median"] and b["decode_tok_s_median"]:
            row["decode_ratio_tetraeder_over_llama"] = _r(a["decode_tok_s_median"] / b["decode_tok_s_median"])
        out["comparison"][task] = row
    return out


def format_table(summary):
    lines = []
    head = f"{'engine':<14}{'ok/runs':>9}{'turns':>7}{'perr':>6}{'terr':>6}{'reqs':>7}{'ttft50':>9}{'ttft90':>9}{'tok/s':>8}{'wall50':>9}"
    lines.append(head)
    for e, s in summary["engines"].items():
        lines.append(
            f"{e:<14}{str(s['successes']) + '/' + str(s['runs']):>9}{str(s['turns_median']):>7}"
            f"{s['protocol_errors']:>6}{sum(s['tool_errors_by_class'].values()):>6}{s['requests']:>7}{str(s['ttft_median_s']):>9}{str(s['ttft_p90_s']):>9}"
            f"{str(s['decode_tok_s_median']):>8}{str(s['wall_s_median']):>9}"
        )
    return "\n".join(lines)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    path = argv[0]
    sessions, requests = load_dir(path)
    if not sessions:
        print(f"no session-*.json in {path}", file=sys.stderr)
        return 1
    summary = summarize(sessions, requests)
    with open(os.path.join(path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(format_table(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
