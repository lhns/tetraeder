#!/usr/bin/env python3
"""Gates of the segment confirmation as code (ADR 0091 owner answers 7 and 8, ADR 0092 section 6).

The session script (`scripts/sc_box.sh`) and the summary (`scripts/sc_summary.py`) call these; nothing here
is a judgement call. Each check returns a list of reasons, empty when it passes; the CLI prints them and exits 1.

    sc_gates.py numa LSCPU_FILE                  exactly one NUMA node (checked before the model download)
    sc_gates.py cotenant FILE                    FILE is the output of `engine_sample.py cotenant` and is under the limit
    sc_gates.py fitband PRED_JSON [--limit L]    the anchors' fit band is within the limit (refuses the freeze)
    sc_gates.py anchors-real FILE...             every anchor ran on real hidden rows (a dump file), not fixed-seed input
    sc_gates.py freeze --results DIR [--frozen-sha S]
                                                 FROZEN.sha256 exists, equals S, and every frozen file matches SHA256SUMS
    sc_gates.py judged-start --results DIR --frozen-sha S
                                                 all of the above, as the judged stages need them
    sc_gates.py aa A.json B.json [--band B]      A/A on T = prompt / pp + gen / tg of two runs of the same plan
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cotenant_guard  # noqa: E402

# Mirrors FIT_BAND_LIMIT of crates/planner/src/segment.rs (tests/test_sc_gates.py keeps them equal). The band is the
# largest |ln(pred / meas)| over the anchors after the fit: 0.07 measured on real hidden rows, 1.23 on fixed-seed input.
FIT_BAND_LIMIT = 0.15
AA_BAND = 0.08  # bench/spec/phase0.toml, ADR 0068 and ADR 0091 Case S: band on T
PROMPT, GEN = 2048, 128
COTENANT_SOURCE = "engine_sample.py cotenant"


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_of(j: dict) -> dict:
    runs = j.get("runs")
    return runs[-1] if isinstance(runs, list) and runs else j


# ---- host gates (before the model download) -----------------------------------------------------------------------


def numa_nodes(lscpu: str) -> int | None:
    m = re.search(r"^NUMA node\(s\):\s*(\d+)\s*$", lscpu, re.M)
    return int(m.group(1)) if m else None


def check_numa(lscpu: str) -> list[str]:
    n = numa_nodes(lscpu)
    if n is None:
        return ["lscpu does not say how many NUMA nodes the host has"]
    return [] if n == 1 else [f"the host has {n} NUMA nodes, a judged session needs exactly one"]


def check_cotenant(path: str) -> list[str]:
    """The SSH-time reading must come from `engine_sample.py cotenant` and be under the limit. The limit is
    recomputed here from host_cpus, so a file that states a looser one does not pass."""
    try:
        r = load(path)
    except (OSError, ValueError):
        return [f"no co-tenant reading at {path}"]
    if r.get("source") != COTENANT_SOURCE:
        return [f"the co-tenant reading is not from `{COTENANT_SOURCE}` (source {r.get('source')!r}): no ad-hoc sampler"]
    outside, cpus = r.get("outside_cores"), r.get("host_cpus")
    if outside is None or not cpus:
        return ["the co-tenant reading has no CPU counters"]
    limit = cotenant_guard.limit_of(cpus)
    if outside > limit:
        return [f"{outside:.1f} cores busy outside the container, over the limit {limit:.1f} of {cpus // 2} physical"]
    return []


# ---- the freeze ---------------------------------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def check_fit_band(pred_path: str, limit: float = FIT_BAND_LIMIT) -> list[str]:
    try:
        band = load(pred_path)["prediction"]["band"]
    except (OSError, ValueError, KeyError):
        return [f"no fit band in {pred_path}"]
    if not isinstance(band, (int, float)) or band != band or band > limit:
        return [f"the anchors' fit band {band} is above the limit {limit}: the prediction is not frozen"]
    return []


def check_anchors_real(paths: list[str]) -> list[str]:
    """Anchors take the hidden rows dumped from the frozen prompt (`hidden_in.file`), never a seed."""
    if not paths:
        return ["no anchor results"]
    bad = []
    for p in paths:
        try:
            seg = run_of(load(p)).get("segment") or {}
        except (OSError, ValueError):
            bad.append(f"{p}: unreadable")
            continue
        h = seg.get("hidden_in") or {}
        if not h.get("file") or "seed" in h:
            bad.append(f"{os.path.basename(p)}: input is {h or 'none'}, not real hidden rows")
    return bad


def check_freeze(results: str, frozen_sha: str | None = None) -> list[str]:
    pdir = os.path.join(results, "predictions")
    fz = os.path.join(pdir, "FROZEN.sha256")
    sums = os.path.join(pdir, "SHA256SUMS")
    if not os.path.isfile(fz) or not os.path.getsize(fz):
        return [f"no {fz}: the prediction is not written"]
    if not os.path.isfile(sums):
        return [f"no {sums}"]
    want = open(fz, encoding="utf-8").read().split()[0]
    why = []
    if frozen_sha is not None and frozen_sha != want:
        why.append(f"FROZEN_SHA {frozen_sha!r} is not the frozen hash {want}: commit the prediction first, then pass it")
    if sha256_file(sums) != want:
        why.append("SHA256SUMS does not hash to FROZEN.sha256")
    for line in open(sums, encoding="utf-8").read().splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        path = os.path.join(results, name.strip())
        if not os.path.isfile(path):
            why.append(f"frozen file missing: {name}")
        elif sha256_file(path) != digest:
            why.append(f"frozen file changed: {name}")
    return why


def anchor_files(results: str) -> list[str]:
    d = os.path.join(results, "anchors")
    return sorted(os.path.join(d, f) for f in os.listdir(d) if re.fullmatch(r"a\d+-r\d+\.json", f)) if os.path.isdir(d) else []


def check_judged_start(results: str, frozen_sha: str | None) -> list[str]:
    """What must hold before any judged round runs (ADR 0091 answers 7 and 8)."""
    why = []
    env = os.path.join(results, "env")
    lscpu = os.path.join(env, "lscpu.txt")
    try:
        why += [f"numa: {w}" for w in check_numa(open(lscpu, encoding="utf-8").read())]
    except OSError:
        why.append(f"numa: no {lscpu}")
    why += [f"co-tenant: {w}" for w in check_cotenant(os.path.join(env, "cotenant.json"))]
    why += [f"freeze: {w}" for w in check_freeze(results, frozen_sha)]
    why += [f"fit band: {w}" for w in check_fit_band(os.path.join(results, "predictions", "pred.json"))]
    why += [f"anchors: {w}" for w in check_anchors_real(anchor_files(results))]
    return why


# ---- A/A ----------------------------------------------------------------------------------------------------------


def t_of(run: dict, prompt: int = PROMPT, gen: int = GEN) -> float:
    return prompt / run["prefill_tok_s"] + gen / run["decode_tok_s"]


def aa(first: dict, second: dict, band: float = AA_BAND) -> dict:
    """T of the second run against the first, both result JSONs of the same plan."""
    t0, t1 = t_of(run_of(first)), t_of(run_of(second))
    d = t1 / t0 - 1.0
    return {"t_first": t0, "t_second": t1, "diff": d, "band": band, "ok": abs(d) <= band}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("numa").add_argument("lscpu")
    sub.add_parser("cotenant").add_argument("file")
    f = sub.add_parser("fitband")
    f.add_argument("pred")
    f.add_argument("--limit", type=float, default=FIT_BAND_LIMIT)
    sub.add_parser("anchors-real").add_argument("files", nargs="+")
    z = sub.add_parser("freeze")
    z.add_argument("--results", required=True)
    z.add_argument("--frozen-sha")
    j = sub.add_parser("judged-start")
    j.add_argument("--results", required=True)
    j.add_argument("--frozen-sha", required=True)
    a = sub.add_parser("aa")
    a.add_argument("first")
    a.add_argument("second")
    a.add_argument("--band", type=float, default=AA_BAND)
    args = ap.parse_args()
    if args.cmd == "aa":
        r = aa(load(args.first), load(args.second), args.band)
        print(f"A/A on T: {r['t_first']:.2f} s -> {r['t_second']:.2f} s, {r['diff']:+.3f} against {r['band']} "
              f"{'ok' if r['ok'] else 'OUTSIDE THE BAND'}")
        return 0 if r["ok"] else 1
    why = {
        "numa": lambda: check_numa(open(args.lscpu, encoding="utf-8").read()),
        "cotenant": lambda: check_cotenant(args.file),
        "fitband": lambda: check_fit_band(args.pred, args.limit),
        "anchors-real": lambda: check_anchors_real(args.files),
        "freeze": lambda: check_freeze(args.results, args.frozen_sha),
        "judged-start": lambda: check_judged_start(args.results, args.frozen_sha),
    }[args.cmd]()
    for w in why:
        print(f"GATE FAILED {args.cmd}: {w}")
    if not why:
        print(f"gate {args.cmd} ok")
    return 1 if why else 0


if __name__ == "__main__":
    sys.exit(main())
