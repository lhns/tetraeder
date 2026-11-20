#!/usr/bin/env python3
"""decode-clocks.json for cost model 5 (ADR 0082, Brief V5): the SM clock a GPU runs at during timed decode.

    python -I scripts/decode_clocks.py DIR [--out decode-clocks.json] [--style anchors|diag]

DIR holds, per run, a clock log and a window file written by `scripts/phase5_box.sh` (`run_one`):

    clocks-<run>.csv     nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,
                         clocks_event_reasons.active --format=csv -lms 20 (the box's UTC clock)
    window-<run>.txt     start=<epoch s> end=<epoch s> (`date +%s.%N` before and after the process)

`<run>.clocks.csv` and `<run>.window.txt` (as `s3e_diag_box.sh` part C names them) are read too. The run's rates come
from `anchor-parts/<model>-<plan>-L<n>-r<k>.json` (style `anchors`, run `<model>-anchor-<plan>-L<n>-r<k>`) or
`<run>.json` (style `diag`, runs `<model>-single[-L<n>]-tg128` and `<model>-layer-<g>-tg128`).

A run is cut to its timed decode tail: the last `reps x n_gen / tg_tok_s` seconds before the window end, `reps` being
the samples of the tg row. Medians are per GPU over that tail. Whole-window and 1 Hz medians are not used: they
measure model loading (ADR 0082, V2). A run is withheld, with the reason, when a GPU that computes in it has under
25 tail samples, or shows 0x8, 0x20, 0x40 or 0x80 (slowdown, thermal, power brake) in its tail.

`busy_mhz[model]`: median over the model's `single` runs of the tail clock of GPU 0 (both depths, both rounds). The
key is `layout.model` of `DIR/layouts/<run model>.json` (what `features_multi` sets, e.g. "Qwen3 30B Gptq Fp16"), the
run's own model name when DIR has no layout.
`low_mhz`: median over the layer anchors `layer-<g>@L<n>` (style `diag`: the full-depth `layer-<g>` runs) of the tail
clocks of the GPUs without the head, GPUs 0 to g - 2. Without a valid layer run `low_mhz` is null, and
`tetraeder-plan calibrate --clocks` leaves the graph without clocks (cost model 4). Standard library only.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import statistics
import sys
from pathlib import Path

DUTY_THRESHOLD = 0.32
MIN_SAMPLES = 25
# clocks_event_reasons.active bits: hw slowdown, sw thermal, hw thermal, hw power brake
BAD_REASONS = 0x8 | 0x20 | 0x40 | 0x80

ANCHOR_RUN = re.compile(r"(?P<model>.+)-anchor-(?P<plan>single|layer-\d+@L\d+|tp-\d+)-L(?P<lim>\d+)-r(?P<k>\d+)")
DIAG_RUN = re.compile(r"(?P<model>qwen3-[0-9a-z-]+?)-(?P<plan>single|layer-\d+)(?:-L(?P<lim>\d+))?-tg128")


def read_window(path: Path) -> tuple[float, float]:
    kv = dict(p.split("=", 1) for p in path.read_text().split())
    return float(kv["start"]), float(kv["end"])


def read_clocks(path: Path) -> list[tuple[float, int, float, int | None]]:
    """`(epoch s, gpu index, sm MHz, event reasons or None)` of every row of an nvidia-smi csv."""
    lines = path.read_text().splitlines()
    head = [c.strip() for c in lines[0].split(",")]
    col = {name: i for i, name in enumerate(head)}
    find = lambda *prefixes: next((i for n, i in col.items() if n.startswith(prefixes)), None)  # noqa: E731
    t_i, idx_i = find("timestamp"), find("index")
    sm_i, why_i = find("clocks.current.sm", "clocks.sm"), find("clocks_event_reasons", "clocks_throttle_reasons")
    if t_i is None or idx_i is None or sm_i is None:
        raise ValueError(f"{path}: no timestamp, index and sm clock columns in {head}")
    out = []
    for ln in lines[1:]:
        f = [c.strip() for c in ln.split(",")]
        if len(f) < len(head):
            continue
        t = datetime.datetime.strptime(f[t_i], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc)
        why = None
        if why_i is not None:
            try:
                why = int(f[why_i], 16)
            except ValueError:
                why = 0 if f[why_i].lower().startswith("not active") else None
        out.append((t.timestamp(), int(f[idx_i]), float(f[sm_i].split()[0]), why))
    return out


def tail_seconds(rows: list) -> float | None:
    """`reps x n_gen / tg_tok_s` of the tg row of llama-bench JSON rows."""
    for r in rows:
        if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0 and r.get("avg_ts"):
            reps = len(r.get("samples_ts") or []) or 1
            return reps * r["n_gen"] / r["avg_ts"]
    return None


def gpu_tail(samples: list, window: tuple[float, float], tail_s: float, gpu: int) -> dict:
    """Median SM clock of one GPU over the last `tail_s` seconds before the window end."""
    lo, hi = window[1] - tail_s, window[1]
    rows = [s for s in samples if s[1] == gpu and lo <= s[0] <= hi]
    why = 0
    for s in rows:
        why |= s[3] or 0
    out = {"samples": len(rows), "median": statistics.median(s[2] for s in rows) if rows else None,
           "reasons": f"0x{why:x}" if any(s[3] is not None for s in rows) else None}
    bad = []
    if len(rows) < MIN_SAMPLES:
        bad.append(f"gpu{gpu}: {len(rows)} tail samples, under {MIN_SAMPLES}")
    if why & BAD_REASONS:
        bad.append(f"gpu{gpu}: event reasons 0x{why & BAD_REASONS:x} in the tail")
    out["withheld"] = bad
    return out


def parse_run(run: str, style: str) -> dict | None:
    m = (ANCHOR_RUN if style == "anchors" else DIAG_RUN).fullmatch(run)
    if not m:
        return None
    d = m.groupdict()
    plan = d["plan"]
    g = re.match(r"layer-(\d+)", plan)
    d["ranks"] = int(g.group(1)) if g else 1
    if style == "anchors":
        d["kind"] = "single" if plan == "single" else "layer" if "@L" in plan else None
    else:
        # s3e part C: single at every depth, layer-g only at full depth (a truncated layer run drops stages)
        d["kind"] = "single" if plan == "single" else "layer" if g and d["lim"] is None else None
    return d


def run_files(d: Path, run: str) -> tuple[Path, Path] | None:
    for c, w in ((f"clocks-{run}.csv", f"window-{run}.txt"), (f"{run}.clocks.csv", f"{run}.window.txt")):
        if (d / c).is_file() and (d / w).is_file():
            return d / c, d / w
    return None


def runs_in(d: Path) -> list[str]:
    names = {p.name[len("clocks-"):-len(".csv")] for p in d.glob("clocks-*.csv")}
    names |= {p.name[:-len(".clocks.csv")] for p in d.glob("*.clocks.csv")}
    return sorted(names)


def rates_file(d: Path, run: str, info: dict, style: str) -> Path:
    if style == "diag":
        return d / f"{run}.json"
    return d / "anchor-parts" / f"{info['model']}-{info['plan']}-L{info['lim']}-r{info['k']}.json"


def layout_model(d: Path, name: str) -> str:
    """`layout.model` of `layouts/<name>.json`, the key `RankFeatures::model` carries; `name` without the file."""
    f = d / "layouts" / f"{name}.json"
    return json.loads(f.read_text())["model"] if f.is_file() else name


def summarize(d: Path, style: str = "anchors") -> dict:
    runs: dict = {}
    for run in runs_in(d):
        info = parse_run(run, style)
        if info is None:
            continue
        files, rates = run_files(d, run), rates_file(d, run, info, style)
        entry = {"model": layout_model(d, info["model"]), "plan": info["plan"], "limit": int(info["lim"]) if info["lim"] else None,
                 "kind": info["kind"], "withheld": []}
        runs[run] = entry
        if files is None or not rates.is_file():
            entry["withheld"].append("no clock log, window or rates file")
            continue
        tail_s = tail_seconds(json.loads(rates.read_text()))
        if tail_s is None:
            entry["withheld"].append("no tg row in the rates file")
            continue
        samples, window = read_clocks(files[0]), read_window(files[1])
        entry["tail_s"] = tail_s
        # the GPUs that compute: GPU 0 of a single run, the GPUs without the head (0 to g - 2) of a layer run
        gpus = range(info["ranks"]) if info["kind"] == "layer" else [0]
        entry["gpus"] = {str(g): gpu_tail(samples, window, tail_s, g) for g in range(max(info["ranks"], 1))}
        for g in gpus:
            entry["withheld"] += entry["gpus"][str(g)]["withheld"]
        entry["use"] = [g for g in gpus if info["kind"] != "layer" or g < info["ranks"] - 1]
    busy: dict = {}
    low: list = []
    withheld = []
    for run, e in runs.items():
        if e["kind"] is None:
            continue
        if e["withheld"]:
            withheld.append(f"{run}: " + "; ".join(e["withheld"]))
            continue
        vals = [e["gpus"][str(g)]["median"] for g in e["use"]]
        if e["kind"] == "single":
            busy.setdefault(e["model"], []).extend(vals)
        else:
            low.extend(vals)
    return {"low_mhz": statistics.median(low) if low else None,
            "busy_mhz": {m: statistics.median(v) for m, v in sorted(busy.items())},
            "duty_threshold": DUTY_THRESHOLD, "withheld": withheld, "runs": runs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="default: DIR/decode-clocks.json")
    ap.add_argument("--style", choices=["anchors", "diag"], default="anchors")
    a = ap.parse_args()
    res = summarize(a.dir, a.style)
    out = a.out or a.dir / "decode-clocks.json"
    out.write_text(json.dumps(res, indent=1) + "\n", newline="\n")
    print(f"low {res['low_mhz']} MHz, busy {res['busy_mhz']} -> {out}")
    for w in res["withheld"]:
        print(f"withheld {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
