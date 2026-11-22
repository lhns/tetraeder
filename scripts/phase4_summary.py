#!/usr/bin/env python3
"""Phase 4 verdict (ADR 0065, "Verdict") from the results of scripts/phase4_box.sh.

    python scripts/phase4_summary.py bench/results/<session>/phase4          writes verdict.json there
    python scripts/phase4_summary.py DIR --pick-variant q8b                  prints kvu or nokvu (pre-pass winner)
    python scripts/phase4_summary.py DIR --probe --probe-run q8b:tetraeder-dense [--parity q8b:failed]
                                                                             probe.json, exit 3 when every judged model is stopped
    python scripts/phase4_summary.py DIR --stopped                           prints the models probe.json stopped

Input files in DIR (names fixed by the ADR): bench-<model>-<engine>-<c1|c4>-r<rep>.json, prepass-<model>-<kvu|nokvu>.json,
replay-<model>-<on|off>-r<rep>.json, gpu-<run>.txt (lines `before: <hex>` / `after: <hex>`, the clocks_throttle_reasons
bitmask; `power-before: <limit>, <draw>` / `power-after: ...` are copied to verdict.json as is), server-*.log, env.json. A run is `<model>-<engine>-r<rep>`; replay runs are `replay-<model>-<on|off>-r<rep>`.
gpu-samples.csv (scripts/gpu_sampler.sh, optional): with it a pair is judged on the share of window samples carrying
0x200 or 0x400 (ADR 0065 Verdict, "Throttle record") instead of the single before sample; verdict.json records the rule per pair.
Reps 1 and 2 are the judged pairs: rep 1 runs (baseline, Tetraeder), rep 2 (Tetraeder, baseline). Rep 3 is the one
rerun pair of phase4_box.sh when rep 1 or 2 is unclean (errors, throttle, co-tenant, a missing result); the judged pairs
are then the first two CLEAN pairs. Anything missing makes the pair unclean (a reason is recorded) and the verdict
`null`, never true.

    python scripts/phase4_summary.py DIR --rerun-reps q8b                    prints the unclean rep(s) of 1 and 2 that one
                                                                             rep-3 pair can replace (empty: no rerun)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EPS = 1e-9
REPS = (1, 2)
RERUN = 3
ALL_REPS = (1, 2, 3, 4)
BASELINE = "llama-server"
# model -> (judged engine, engine used when no judged-engine files exist, counts then)
JUDGED = {"q8b": ("tetraeder-dense", None), "q30b": ("tetraeder-moe", "tetraeder-llama")}
# clocks_throttle_reasons bits that are fine: GPU idle (0x1) and applications clocks setting (0x2) before a run; after a
# run also the SW power cap (0x4), which the card's own load sets (ADR 0065 Verdict, "Throttle record"). Everything else is unclean.
BENIGN_THROTTLE = {"before": 0x3, "after": 0x7}
# With sampler data (gpu-samples.csv, ADR 0065 addendum "BoardLimit and Reliability bits judged under load"): 0x200
# (BoardLimit) and 0x400 (Reliability) no longer disqualify the single "before" sample; they are judged by their share
# of the samples in the measured window instead.
BOARD_BITS = 0x200 | 0x400
SAMPLER_BEFORE = BENIGN_THROTTLE["before"] | BOARD_BITS
# bits that are fine in a window sample; anything else (0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal, 0x80 power
# brake, unknown) makes the run unclean
WINDOW_OK = 0x1 | 0x2 | 0x4 | 0x100 | BOARD_BITS
SHARE_TOL = 0.10  # max difference of the two sides' shares of samples with a board bit


def _load_bench():
    spec = importlib.util.spec_from_file_location("gateway_bench", HERE / "gateway_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


percentile = _load_bench().percentile


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---- band ---------------------------------------------------------------------------------------------------


def class_slug(gpu_name: str | None) -> str:
    """Same as `class_slug` in crates/bench/src/noise.rs, CUDA case."""
    words = [w for w in re.split(r"[^a-z0-9]+", (gpu_name or "").lower()) if w and w not in ("nvidia", "geforce")]
    return "-".join(words) if words else "cuda-unknown"


def choose_band(d: Path, noise_dir: Path, spec_path: Path) -> tuple[float, str]:
    """(band, band_source): the measured band of the GPU class if its n is at least 3, else the spec's cuda-default."""
    env = read_json(d / "env.json") or {}
    cls = class_slug(env.get("gpu"))
    measured = read_json(noise_dir / f"{cls}.json")
    if isinstance(measured, dict) and measured.get("n", 0) >= 3 and 0 < measured.get("band", 0) < 1:
        return float(measured["band"]), f"measured {cls} n={measured['n']}"
    spec = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    band = next(float(b["band"]) for b in spec["noise_band"] if b["class"] == "cuda-default")
    why = ""
    if isinstance(measured, dict):
        why = f" (measured n={measured.get('n', 0)} < 3)"
    return band, f"provisional cuda-default{why}"


# ---- runs ---------------------------------------------------------------------------------------------------


def throttle_record(d: Path, run: str, relaxed: bool = False) -> tuple[list[str], dict]:
    """(reasons the throttle record of a run is unclean, the raw record). The record keeps the hex values per GPU
    (`before`, `after`) and the raw `power-before` / `power-after` lines (`power.limit, power.draw`) as written.
    `relaxed`: the pair has sampler data, so 0x200 and 0x400 do not disqualify the before sample."""
    p = d / f"gpu-{run}.txt"
    rec: dict = {"before": [], "after": [], "power_before": [], "power_after": []}
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return [f"no {p.name}"], rec
    seen: dict[str, list[int]] = {"before": [], "after": []}
    bad = []
    for m in re.finditer(r"^(before|after):[ \t]*(\S+)", text, re.M):
        rec[m.group(1)].append(m.group(2))
        try:
            seen[m.group(1)].append(int(m.group(2), 16))
        except ValueError:
            bad.append(f"{m.group(1)}: unreadable throttle value {m.group(2)!r}")
    for m in re.finditer(r"^power-(before|after):[ \t]*(.*\S)", text, re.M):
        rec[f"power_{m.group(1)}"].append(m.group(2))
    out = bad
    for when, vals in seen.items():
        if not vals:
            if not any(r.startswith(when) for r in bad):
                out.append(f"no {when} throttle record")
        for v in vals:
            ok = SAMPLER_BEFORE if relaxed and when == "before" else BENIGN_THROTTLE[when]
            if v & ~ok:
                out.append(f"throttle reasons {when} the run: {v:#x}")
    return out, rec


def throttle_reasons(d: Path, run: str, relaxed: bool = False) -> list[str]:
    """Reasons the throttle record of a run is unclean; empty when before shows only idle / app clocks and after
    also at most the SW power cap."""
    return throttle_record(d, run, relaxed)[0]


# ---- sampler ------------------------------------------------------------------------------------------------


def load_samples(d: Path) -> list[tuple[datetime, int]] | None:
    """(time, reasons mask) per line of gpu-samples.csv (`epoch,timestamp,reasons,...`, scripts/gpu_sampler.sh); the
    time is nvidia-smi's naive box-local timestamp, the same clock as the result files' `finished`. None when there is
    no file or no readable line."""
    try:
        text = (d / "gpu-samples.csv").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    out = []
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 3:
            continue
        try:
            t = datetime.strptime(f[1].split(".")[0], "%Y/%m/%d %H:%M:%S")
            out.append((t, int(f[2], 16)))
        except ValueError:
            continue
    return out or None


def result_window(j: dict | None) -> tuple[datetime, datetime] | None:
    """Measured window of a bench or replay result: it ends at `finished` (replay and new bench files) or `started`
    (older bench files stamp it after the run) and lasts wall_s. One second of slack each side for the stamp's
    truncation."""
    if not j:
        return None
    wall = (j.get("summary") or {}).get("wall_s", j.get("wall_s"))
    stamp = j.get("finished") or j.get("started")
    if not isinstance(wall, (int, float)) or not stamp:
        return None
    try:
        end = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    return end - timedelta(seconds=wall + 1), end + timedelta(seconds=1)


def window_stats(samples: list[tuple[datetime, int]] | None, windows: list) -> dict | None:
    """{samples, share, disqualifying, bad_bits} for the samples inside any of the windows (share: fraction carrying
    0x200 or 0x400; disqualifying: samples with a bit outside WINDOW_OK). None without sampler data for the run."""
    ws = [w for w in windows if w]
    if samples is None or not ws:
        return None
    inside = [m for t, m in samples if any(a <= t <= b for a, b in ws)]
    if not inside:
        return None
    bad, bits = 0, 0
    for m in inside:
        if m & ~WINDOW_OK:
            bad += 1
            bits |= m & ~WINDOW_OK
    return {
        "samples": len(inside),
        "share": sum(1 for m in inside if m & BOARD_BITS) / len(inside),
        "disqualifying": bad,
        "bad_bits": f"{bits:#x}",
    }


def pair_sampler(a: dict | None, b: dict | None) -> tuple[bool, list[str]]:
    """(the sampler rule applies, its reasons). It needs data for both sides; otherwise the old rule."""
    if a is None or b is None:
        return False, []
    out = []
    if abs(a["share"] - b["share"]) > SHARE_TOL + EPS:
        out.append(f"board-limit share differs: {a['share']:.2f} vs {b['share']:.2f} (> {SHARE_TOL})")
    for who, x in (("first", a), ("second", b)):
        if x["disqualifying"]:
            out.append(f"{who} side: {x['disqualifying']} window sample(s) with throttle bits {x['bad_bits']}")
    return True, out


def run_reasons(d: Path, model: str, engine: str, rep: int, relaxed: bool = False) -> tuple[list[str], dict]:
    """(reasons the run is unclean, {test: result json}) for one server start."""
    run = f"{model}-{engine}-r{rep}"
    res, reasons = {}, []
    for test in ("c1", "c4"):
        j = read_json(d / f"bench-{model}-{engine}-{test}-r{rep}.json")
        if j is None:
            reasons.append(f"{run}: no {test} result")
            continue
        res[test] = j
        errs = (j.get("summary") or {}).get("errors")
        if errs != 0:
            reasons.append(f"{run} {test}: errors={errs}")
        if j.get("cotenant_ok") is False:
            reasons.append(f"{run} {test}: co-tenant ({j.get('cotenant_reason', '?')})")
    reasons += [f"{run}: {r}" for r in throttle_reasons(d, run, relaxed)]
    return reasons, res


def run_sampler(d: Path, model: str, engine: str, rep: int, samples) -> dict | None:
    """Sampler stats over the c1 and c4 windows of one server start."""
    ws = [result_window(read_json(d / f"bench-{model}-{engine}-{t}-r{rep}.json")) for t in ("c1", "c4")]
    return window_stats(samples, ws)


def pair_check(d: Path, model: str, engine: str, rep: int) -> tuple[list[str], dict, dict, dict]:
    """(reasons, base results, own results, sampler record) of the pair of rep (baseline vs engine)."""
    samples = load_samples(d)
    sb = run_sampler(d, model, BASELINE, rep, samples)
    sj = run_sampler(d, model, engine, rep, samples)
    used, sreasons = pair_sampler(sb, sj)
    rb, base = run_reasons(d, model, BASELINE, rep, used)
    rj, own = run_reasons(d, model, engine, rep, used)
    reasons = rb + rj + [f"r{rep} {r}" for r in sreasons] + [f"r{rep} {r}" for r in tokens_differ(base, own, BASELINE, engine)]
    return reasons, base, own, {"rule": "sampler" if used else "single-sample", "baseline": sb, "judged": sj}


def field(res: dict, test: str, key: str) -> float | None:
    v = ((res.get(test) or {}).get("summary") or {}).get(key)
    return float(v) if isinstance(v, (int, float)) and v > 0 else None


def ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def pick_variant(d: Path, model: str) -> str | None:
    """Pre-pass winner: the variant with the higher c4 throughput_tps (a tie goes to kvu); None if either is missing."""
    t = {}
    for v in ("kvu", "nokvu"):
        j = read_json(d / f"prepass-{model}-{v}.json")
        t[v] = field({"c4": j}, "c4", "throughput_tps") if j else None
    if t["kvu"] is None or t["nokvu"] is None:
        return None
    return "kvu" if t["kvu"] >= t["nokvu"] else "nokvu"


def kv_unified(d: Path, model: str, variant: str | None) -> str | None:
    """Value of the `kv_unified = '...'` line of the baseline's server log (the chosen pre-pass log first)."""
    logs = ([d / f"server-prepass-{model}-{variant}.log"] if variant else []) + [
        d / f"server-{model}-{BASELINE}-r{r}.log" for r in ALL_REPS
    ]
    for p in logs:
        try:
            m = re.search(r"kv_unified\s*=\s*'?([A-Za-z0-9_]+)'?", p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if m:
            return m.group(1)
    return None


def rerun_reps(d: Path, model: str) -> list[int]:
    """Unclean reps among 1 and 2, each replaced by one more pair (rep 3 for the first, rep 4 for the second; the
    order of the unclean rep, so both unclean give 3 then 4 in the order of 1 then 2). Still the same after the
    reruns ran, so a resumed session finishes them."""
    engine, _ = judged_engine(d, model)
    return [r for r in REPS if pair_reasons(d, model, engine, r)]


def judged_engine(d: Path, model: str) -> tuple[str, bool]:
    engine, fallback = JUDGED[model]
    if fallback and not any(d.glob(f"bench-{model}-{engine}-*.json")):
        return fallback, False
    return engine, True


def pair_reasons(d: Path, model: str, engine: str, rep: int) -> list[str]:
    """Why the pair of rep is unclean: either run's own reasons, or unequal completion-token totals."""
    return pair_check(d, model, engine, rep)[0]


def stopped_verdict(model: str, probe_model: dict) -> dict:
    """A model the stop rule skipped (failed gpu-parity or probe c4 below the line): no pairs, `pass` null.
    It counts when its probe ran the judged engine, so a skipped model never lets the others pass alone."""
    engine = probe_model.get("engine") or JUDGED[model][0]
    return {
        "engine": engine,
        "counts": engine == JUDGED[model][0],
        "stopped": True,
        "baseline_variant": None,
        "baseline_kv_unified": None,
        "aa": {"c4": None, "c1": None},
        "pairs": [],
        "worst": {"c4": None, "c1": None},
        "pass": None,
        "reasons": list(probe_model.get("reasons") or ["stopped"]),
    }


def tokens_differ(base: dict, own: dict, run_b: str, run_o: str) -> list[str]:
    """Reasons when the two engines' completion-token totals differ in a test: not the same work."""
    out = []
    for test in ("c1", "c4"):
        a = ((base.get(test) or {}).get("summary") or {}).get("completion_tokens")
        b = ((own.get(test) or {}).get("summary") or {}).get("completion_tokens")
        if test in base and test in own and a != b:
            out.append(f"{test}: completion tokens differ ({run_b} {a}, {run_o} {b})")
    return out


def model_verdict(d: Path, model: str, band: float) -> dict:
    engine, counts = judged_engine(d, model)
    variant = pick_variant(d, model)
    out = {
        "engine": engine,
        "counts": counts,
        "baseline_variant": variant,
        "baseline_kv_unified": kv_unified(d, model, variant),
        "aa": {"c4": None, "c1": None},
        "pairs": [],
        "worst": {"c4": None, "c1": None},
        "pass": None,
        "reasons": [],
    }
    base = {r: run_reasons(d, model, BASELINE, r) for r in ALL_REPS}
    judged = {r: run_reasons(d, model, engine, r) for r in ALL_REPS}
    cand = [r for r in ALL_REPS if r in REPS or any(d.glob(f"bench-{model}-*-r{r}.json"))]
    clean = [r for r in cand if not pair_reasons(d, model, engine, r)]
    reps = clean[:2] if len(clean) >= 2 else list(REPS)  # fewer than two clean pairs: the unclean first two, no verdict
    out["reps_used"] = reps
    out["dropped_pairs"] = [{"rep": r, "reasons": pair_reasons(d, model, engine, r)} for r in cand if r not in reps]

    # clean session: the two baseline runs agree within the band on both fields
    session_reasons = []
    for key, test, fld in (("c4", "c4", "throughput_tps"), ("c1", "c1", "decode_tps_median")):
        r = ratio(field(base[reps[1]][1], test, fld), field(base[reps[0]][1], test, fld))
        if r is None:
            session_reasons.append(f"A/A {key}: baseline results missing")
            continue
        out["aa"][key] = r - 1
        if abs(r - 1) > band + EPS:
            session_reasons.append(f"A/A {key}: |B2/B1 - 1| = {abs(r - 1):.4f} > band {band}")

    for rep in reps:
        reasons = pair_reasons(d, model, engine, rep) + session_reasons
        pair = {
            "reps": [rep],
            "c4_throughput_ratio": ratio(field(judged[rep][1], "c4", "throughput_tps"), field(base[rep][1], "c4", "throughput_tps")),
            "c1_decode_ratio": ratio(field(judged[rep][1], "c1", "decode_tps_median"), field(base[rep][1], "c1", "decode_tps_median")),
            "clean": not reasons,
            "reasons": reasons,
            "throttle": {
                "baseline": throttle_record(d, f"{model}-{BASELINE}-r{rep}")[1],
                "judged": throttle_record(d, f"{model}-{engine}-r{rep}")[1],
            },
            "sampler": pair_check(d, model, engine, rep)[3],
        }
        out["pairs"].append(pair)
    for key, pk in (("c4", "c4_throughput_ratio"), ("c1", "c1_decode_ratio")):
        rs = [p[pk] for p in out["pairs"] if p[pk] is not None]
        out["worst"][key] = min(rs) if rs else None

    out["reasons"] = session_reasons + [r for p in out["pairs"] for r in p["reasons"] if r not in session_reasons]
    if not all(p["clean"] for p in out["pairs"]):
        return out  # an unclean pair is rerun, never dropped: no verdict
    if any(out["worst"][k] is None for k in ("c4", "c1")):
        out["reasons"].append("ratio missing")
        return out
    out["pass"] = all(out["worst"][k] >= 1 - band - EPS for k in ("c4", "c1"))
    return out


# ---- TTFT ---------------------------------------------------------------------------------------------------


def median(v: list[float]) -> float | None:
    s = sorted(v)
    return s[len(s) // 2] if s else None


def replay_run(d: Path, mode: str, rep: int, relaxed: bool = False) -> tuple[list[str], dict | None]:
    run = f"replay-q8b-{mode}-r{rep}"
    j = read_json(d / f"{run}.json")
    if j is None:
        return [f"{run}: no result"], None
    reasons = [f"{run}: {r}" for r in throttle_reasons(d, run, relaxed)]
    if j.get("cotenant_ok") is False:
        reasons.append(f"{run}: co-tenant ({j.get('cotenant_reason', '?')})")
    return reasons, j


def ttft_verdict(d: Path, band: float) -> dict:
    out = {"log_sha256": None, "requests": None, "pairs": [], "pass": None}
    for rep in REPS:
        samples = load_samples(d)
        so, sf = (window_stats(samples, [result_window(read_json(d / f"replay-q8b-{m}-r{rep}.json"))]) for m in ("on", "off"))
        used, sreasons = pair_sampler(so, sf)
        r_on, on = replay_run(d, "on", rep, used)
        r_off, off = replay_run(d, "off", rep, used)
        reasons = r_on + r_off + [f"r{rep} {r}" for r in sreasons]
        pair = {
            "median_ratio": None, "p90_ratio": None, "clean": False, "reasons": reasons,
            "throttle": {m: throttle_record(d, f"replay-q8b-{m}-r{rep}")[1] for m in ("on", "off")},
            "sampler": {"rule": "sampler" if used else "single-sample", "on": so, "off": sf},
        }
        if on and off:
            for j in (on, off):
                p = j.get("params") or {}
                out["log_sha256"] = out["log_sha256"] or p.get("log_sha256")
                out["requests"] = out["requests"] or p.get("requests")
            if (on["summary"].get("errors")) != (off["summary"].get("errors")):
                reasons.append(f"errors differ: on {on['summary'].get('errors')}, off {off['summary'].get('errors')}")
            ok_on = {r["index"]: r.get("ttft_s") for r in on["results"] if r.get("status") == 200}
            ok_off = {r["index"]: r.get("ttft_s") for r in off["results"] if r.get("status") == 200}
            both = sorted(i for i in ok_on if i in ok_off and ok_on[i] is not None and ok_off[i] is not None)
            if not both:
                reasons.append("no request with status 200 in both runs")
            else:
                a, b = [ok_on[i] for i in both], [ok_off[i] for i in both]
                pair["median_ratio"] = ratio(median(a), median(b))
                pair["p90_ratio"] = ratio(percentile(a, 90), percentile(b, 90))
        pair["clean"] = not reasons
        out["pairs"].append(pair)
    if not all(p["clean"] for p in out["pairs"]):
        return out
    out["pass"] = all(p[k] is not None and p[k] <= 1 - band + EPS for p in out["pairs"] for k in ("median_ratio", "p90_ratio"))
    return out


# ---- probe --------------------------------------------------------------------------------------------------

# Fixed in the box plan: the pass line is about 0.925 (band 0.075), below 0.85 the pairs cannot pass.
PROBE_MIN_C4 = 0.85


def probe(d: Path, runs: dict[str, str], parity: str | dict = "ok", min_c4: float = PROBE_MIN_C4) -> dict:
    """The per-model stop rule of scripts/phase4_box.sh from probe-<model>-<engine>-<c1|c4>.json (judged engine and
    llama-server): a model is stopped when its gpu-parity failed (its probe did not run) or its judged engine's c4
    throughput ratio is missing or below PROBE_MIN_C4. A stopped model's pairs are skipped; `stop` (the session ends)
    only when every model is stopped. `runs` maps model to its judged engine name; `parity` is one status for all
    models or {model: status}."""
    par = parity if isinstance(parity, dict) else {m: parity for m in runs}
    out: dict = {
        "threshold_c4": min_c4,
        "gpu_parity": "ok" if all(par.get(m, "ok") == "ok" for m in runs) else "failed",
        "models": {},
        "reasons": [],
        "stopped": [],
    }
    for m, engine in runs.items():
        e: dict = {"engine": engine, "gpu_parity": par.get(m, "ok"), "reasons": []}
        if e["gpu_parity"] != "ok":
            e["reasons"].append(f"gpu-parity {e['gpu_parity']}")
        for test in ("c1", "c4"):
            tps = {}
            for name in (BASELINE, engine):
                j = read_json(d / f"probe-{m}-{name}-{test}.json")
                tps[name] = ((j or {}).get("summary") or {}).get("throughput_tps")
                e[f"{test}_{name}_errors"] = ((j or {}).get("summary") or {}).get("errors")
            base, own = tps[BASELINE], tps[engine]
            e[f"{test}_ratio"] = own / base if base and own is not None else None
        c4 = e["c4_ratio"]
        if not e["reasons"]:
            if c4 is None:
                e["reasons"].append("no probe c4 ratio")
            elif c4 < min_c4:
                e["reasons"].append(f"c4 ratio {c4:.3f} < {min_c4}")
        e["stop"] = bool(e["reasons"])
        if e["stop"]:
            out["stopped"].append(m)
            out["reasons"] += [f"{m} {engine}: {r}" for r in e["reasons"]]
        out["models"][m] = e
    out["stop"] = bool(runs) and len(out["stopped"]) == len(runs)
    return out


# ---- verdict ------------------------------------------------------------------------------------------------


def verdict(d: Path, noise_dir: Path = ROOT / "bench" / "noise", spec: Path = ROOT / "bench" / "spec" / "phase0.toml") -> dict:
    band, source = choose_band(d, noise_dir, spec)
    pr = (read_json(d / "probe.json") or {}).get("models") or {}
    models = {
        m: stopped_verdict(m, pr[m]) if (pr.get(m) or {}).get("stop") else model_verdict(d, m, band) for m in JUDGED
    }
    ttft = ttft_verdict(d, band)
    judged = [m["pass"] for m in models.values() if m["counts"]] + [ttft["pass"]]
    final = False if any(p is False for p in judged) else (True if all(p is True for p in judged) else None)
    return {"band": band, "band_source": source, "models": models, "ttft": ttft, "phase4_pass": final}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--pick-variant", metavar="MODEL", help="print the pre-pass winner (kvu|nokvu) and exit")
    ap.add_argument("--rerun-reps", metavar="MODEL", help="print the unclean rep (1 or 2) a rep-3 pair replaces and exit")
    ap.add_argument("--probe", action="store_true", help="write probe.json; exit 3 when the stop rule fires")
    ap.add_argument("--probe-run", action="append", default=[], metavar="MODEL:ENGINE")
    ap.add_argument("--probe-min", type=float, default=PROBE_MIN_C4, help="with --probe: the c4 stop line (the CPU rehearsal lowers it)")
    ap.add_argument("--parity", action="append", default=[], metavar="[MODEL:]failed",
                    help="with --probe: a model whose gpu-parity failed (bare `failed`: all models)")
    ap.add_argument("--stopped", action="store_true", help="print the models probe.json stopped and exit")
    ap.add_argument("--noise-dir", type=Path, default=ROOT / "bench" / "noise")
    ap.add_argument("--spec", type=Path, default=ROOT / "bench" / "spec" / "phase0.toml")
    a = ap.parse_args(argv)
    if a.stopped:
        print(" ".join((read_json(a.dir / "probe.json") or {}).get("stopped") or []))
        return 0
    if a.probe:
        runs = dict(r.split(":", 1) for r in a.probe_run)
        par: dict = {}
        for x in a.parity:
            m, _, st = x.rpartition(":")
            for k in [m] if m else runs:
                par[k] = st
        pr = probe(a.dir, runs, par, a.probe_min)
        (a.dir / "probe.json").write_text(json.dumps(pr, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"probe_stop": pr["stop"], "stopped": pr["stopped"], "reasons": pr["reasons"]}))
        return 3 if pr["stop"] else 0
    if a.rerun_reps:
        print(" ".join(str(r) for r in rerun_reps(a.dir, a.rerun_reps)))
        return 0
    if a.pick_variant:
        v = pick_variant(a.dir, a.pick_variant)
        print(v or "")
        return 0 if v else 1
    v = verdict(a.dir, a.noise_dir, a.spec)
    (a.dir / "verdict.json").write_text(json.dumps(v, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"phase4_pass": v["phase4_pass"], "band": v["band"], "band_source": v["band_source"],
                      "models": {m: x["pass"] for m, x in v["models"].items()}, "ttft": v["ttft"]["pass"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
