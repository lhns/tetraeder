#!/usr/bin/env python3
"""The clock of a Case S session (ADR 0091 History 2026-11-26): per-stage time estimates against the 3 h cap per box (AGENTS.md `max_box_hours`).

    phase6_clock.py table                                 the stages, minutes (measured, assumed, padded) and their basis
    phase6_clock.py check --created A,B [--now T] [--from STAGE]
                                                          exit 1 when the OLDER box (smaller creation epoch) cannot finish STAGE..end under the cap

The driver runs `check --from fetch` before the 142 GB download, the point of no return in time and money: a pair that cannot finish the stages
after it is destroyed first. It runs `--from second_trace` and `--from report` before the report-only runs, which are skipped when the clock is short.

The estimate. Each row is `measured` minutes (taken from a committed session, named in the basis) plus `assumed` minutes (a guess, with its basis);
the check pays the measured minutes as they are and the assumed ones times PAD (1.25), so a stage that has never run on a pair counts a quarter
more. The pair works in parallel (setup, dryrun, budget, fetch, install and calibration run on both boxes at once), so a row is the longer box.
Replace an assumed number with the first measured one of a full session. The committed README of pair 1 (20261126) put the whole session at "about
2 h 45 min from a ready pair at best"; the rows below add up to 2 h 50 min padded from create (10 min of it is `ready`), 2 h 29 min at the nominal
rows. The difference to the README's figure is the padding of the assumed rows, the bulk screen and the heavier dry run (+20 min), against the
steps now run on both boxes at once (gates, the tiny model's split, builds, installs, calibration: about -10 min) and a leaner `commit`. There
is no extra margin on top of the padding: a pair that fits only with the padded rows has about 10 minutes to spare, and one that does not is
destroyed before the download. The report-only runs (+2.6 and +7 min padded) need that spare, so on a padded session they are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tomllib
from pathlib import Path

def _box_hours() -> float:
    """P6_BOX_HOURS (tests), else taxameter.toml [policy] max_box_hours (3 h; a one-off in AGENTS.md may raise it), 3 when unreadable."""
    if os.environ.get("P6_BOX_HOURS"):
        return float(os.environ["P6_BOX_HOURS"])
    try:
        with open(Path(__file__).resolve().parent.parent / "taxameter.toml", "rb") as f:
            return float(tomllib.load(f)["policy"]["max_box_hours"])
    except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError):
        return 3.0


CAP_MIN = _box_hours() * 60  # AGENTS.md: the per-box cap (max_box_hours), the one the taxameter watchdog stops a box at
PAD = 1.25  # multiplier of the assumed minutes

# (stage, measured min, assumed min, optional, basis). The session's order.
STAGES: list[tuple[str, float, float, bool, str]] = [
    ("ready", 0.0, 8.0, False, "create to ssh and `cuda: ok` (pair 1 of 20261126: about 6 to 10 min)"),
    ("gates", 0.0, 3.0, False, "power, idle, a 20 s co-tenant sample, both boxes at once"),
    ("bulk", 0.0, 4.0, False, "6 transfers of 64 MiB at pair 1's 2 to 8 s each, plus ssh calls"),
    ("setup", 17.0, 2.0, False, "build 17 min on 6 cores (pair 1 node b), 13 min on 11 (segment-confirm-20261125b), both boxes at once; certificates and daemons 2"),
    ("dryrun", 0.0, 14.0, False, "every stage on the tiny model: 3 probe repeats, a 20 s gate gap, 8 sc_box runs of about 40 s; no full run exists yet"),
    ("budget", 2.5, 0.0, False, "balloon 1.6 min (segment-confirm-20261125b) and the env stage, both boxes at once"),
    ("plan", 0.0, 7.0, False, "20 probe repeats (pair 1: 64 MiB took 2 to 8 s, so about 5 min), profile, candidates"),
    ("linkgate", 10.0, 4.0, False, "the 10 min between the probes is the rule; two probes of 10 repeats and two prices assumed"),
    ("fetch", 28.0, 5.0, False, "download 28 min at 85 MB/s (segment-confirm-20261125b); the hashing of 142 GB (about 5 min) and ssh waits assumed"),
    ("rebudget", 1.6, 1.4, False, "balloon 1.6 min (segment-confirm-20261125b), then merge and the covering pick"),
    ("calib", 4.1, 5.4, False, "disk 0.4, trace 2.0, anchors 1.7 (segment-confirm-20261125b); the trace to node b, launches and the start probe assumed"),
    ("predict", 0.0, 4.0, False, "merge, price, freeze, the checks"),
    ("second_trace", 2.0, 0.5, True, "report only: the whole-model run (trace-whole took 2 min on segment-confirm-20261125b)"),
    ("commit", 0.0, 5.0, False, "commit and push between the two invocations"),
    ("judged", 0.0, 12.0, False, "4 rounds (3 clean and one rerun) of about 3 min with the link probe before each (judged-r1..r3 took 1.5 to 2 min on segment-confirm-20261125b per segment)"),
    ("aalast", 0.0, 3.0, False, "one round"),
    ("linkend", 0.0, 2.5, False, "the end probe and the re-pricing"),
    ("summary", 0.5, 0.0, False, "seconds"),
    ("report", 2.0, 4.0, True, "report only: daemons stopped, the second prompt priced, the single-node reference (2 min of engine, launch and copy assumed)"),
    ("wrapup", 0.0, 5.0, False, "fetch of the results, verify, destroy both"),
]


def minutes(row: tuple[str, float, float, bool, str], pad: float = PAD) -> float:
    return row[1] + pad * row[2]


def names() -> list[str]:
    return [r[0] for r in STAGES]


def remaining(frm: str, pad: float = PAD) -> tuple[float, list[tuple[str, float]]]:
    """Minutes of the stages from `frm` to the end: every required stage, and `frm` itself when it is an optional one."""
    if frm not in names():
        raise SystemExit(f"unknown stage {frm}: {names()}")
    rows = [(r[0], minutes(r, pad)) for r in STAGES[names().index(frm):] if not r[3] or r[0] == frm]
    return sum(m for _, m in rows), rows


def check(created: list[float], now: float, frm: str, cap: float = CAP_MIN) -> tuple[bool, str]:
    left, rows = remaining(frm)
    age = (now - min(created)) / 60.0
    need = age + left
    msg = (f"older box is {age:.0f} min old; {frm} to the end needs {left:.0f} min padded ({', '.join(f'{n} {m:.1f}' for n, m in rows)}): "
           f"{need:.0f} min against the {cap:.0f} min cap")
    return need <= cap, msg


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table")
    c = sub.add_parser("check")
    c.add_argument("--created", required=True, help="creation epochs of the boxes, comma separated")
    c.add_argument("--now", type=float)
    c.add_argument("--from", dest="frm", default="fetch")
    a = ap.parse_args()
    if a.cmd == "table":
        nom = pad = 0.0
        for r in STAGES:
            print(f"{r[0]:12} measured {r[1]:5.1f} assumed {r[2]:5.1f} padded {minutes(r):5.1f} min{' (optional)' if r[3] else ''}  {r[4]}")
            if not r[3]:
                nom += r[1] + r[2]
                pad += minutes(r)
        print(f"{'total':12} nominal {nom:.0f} min, padded {pad:.0f} min of {CAP_MIN:.0f} (optional stages not counted)")
        return 0
    ok, msg = check([float(x) for x in a.created.split(",")], a.now if a.now is not None else time.time(), a.frm)
    print(("clock ok: " if ok else "CLOCK SHORT: ") + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
