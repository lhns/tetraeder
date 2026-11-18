#!/usr/bin/env python3
"""Phase 5 verdict (ADR 0068) from the results of scripts/phase5_box.sh.

    python scripts/phase5_summary.py bench/results/<session>/phase5      writes verdict.json and verdict-notes.json there
    python scripts/phase5_summary.py DIR --round-order MODEL K           configurations of round K, one per line
    python scripts/phase5_summary.py DIR --need-rerun MODEL              configurations with fewer than two clean rounds
    python scripts/phase5_summary.py DIR --only "tp-2 llama-bench:tensor"  writes step0.json (own tp-2 against -sm tensor)
    python scripts/phase5_summary.py DIR --first MODEL                   the first candidate (the A/A run repeats it)
    python scripts/phase5_summary.py --merge V.json [V.json ...] --out F  the four cases of several sessions

Input files in DIR (names fixed by ADR 0068 unless marked box):
  env.json                                 gpu_names, gpu_count, p2p, comm_impl (the box script writes them)
  candidates-<model>.json                  `tetraeder-plan candidates --multi`: candidates (order = case order) and pruned
  predict-<model>-<n>x.json                `tetraeder-plan predict`, committed before the first candidate runs
  bench-<model>-<id>-r<k>.json             one candidate in round k (llama-bench rows); bench-<model>-<id>-aa.json is the A/A run
  baseline-<model>-<engine>-<sm>-r<k>.json llama-bench / ik-llama-bench rows, or a gateway_bench result for vllm
  prepass-<model>-<engine>-<sm>.json       box: {"ok", "rc", "comm"}: which split modes the baseline build accepts
  run-<run>.rc                             box: exit code of a run; gpu-<run>.txt throttle reasons before and after;
                                           cotenant-<run>.json the host check (cotenant_ok)
  gates.json                               box: {"gpu_parity", "comm_probe"} = ok | failed | skipped
  gpu-samples.csv                          box: scripts/gpu_sampler.sh, 1 Hz, one line per GPU per second (ADR 0068 addendum
                                           2026-11-17); window-<run>.json {"start", "end"} box-local stamps of each run
  timeline.jsonl                           box: {"t", "event": predict|run, "model", "kind", "sha256"} in order
A run is `<model>-<id>-r<k>`, `<model>-<id>-aa` or `<model>-<engine>-<sm>-r<k>`. Anything missing makes a run unclean
or a result null, never true.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import statistics
import sys
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EPS = 1e-9
PROMPT, GEN = 2048, 128
TOL = 0.15  # predicted against measured rate, both rates, every candidate
MIN_CLEAN = 2  # clean rounds a configuration needs
GATED = "llama-bench"  # baseline engine whose best split mode gates the pick
ENGINES = ("llama-bench", "ik-llama-bench", "vllm")
CASES = ("D2", "M2", "D4", "M4")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_p4 = _load("phase4_summary")  # throttle record format, GPU class slug: one source
throttle_reasons = _p4.throttle_reasons
class_slug = _p4.class_slug
read_json = _p4.read_json
# Throttle by the session sampler (ADR 0068 addenda 2026-11-17, 2026-11-18): bits that are fine in a window sample; the
# shares of samples with 0x200/0x400 and with 0x4 are compared between the rounds of one configuration, never across.
BOARD_BITS = 0x200 | 0x400
CAP_BIT = 0x4
WINDOW_OK = 0x1 | 0x2 | 0x4 | 0x100 | BOARD_BITS
SHARE_TOL = 0.15
SAMPLE_FMT = "%Y/%m/%d %H:%M:%S"


def p2p_class(text: str) -> str | None:
    """`nvidia-smi topo -p2p r` matrix (rows may be indented; X is the diagonal): all | none | partial | None."""
    cells = [t for ln in text.splitlines() if ln.strip().startswith("GPU") for t in ln.split()[1:]]
    cells = [c for c in cells if c in ("OK", "NS", "CNS", "GNS", "TNS", "NA", "U", "CNS")]
    if not cells:
        return None
    ok = sum(c == "OK" for c in cells)
    return "all" if ok == len(cells) else ("none" if ok == 0 else "partial")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---- band ---------------------------------------------------------------------------------------------------


def band_class(env: dict) -> str:
    names = env.get("gpu_names") or []
    n = env.get("gpu_count") or len(names)
    return f"{n}x-{class_slug(names[0] if names else None)}"


def choose_band(env: dict, noise_dir: Path, spec_path: Path) -> tuple[float, str]:
    """(band, band_source): bench/noise/<n>x-<gpu-class>.json if its n is at least 3, else the spec's cuda-default."""
    cls = band_class(env)
    measured = read_json(noise_dir / f"{cls}.json")
    if isinstance(measured, dict) and measured.get("n", 0) >= 3 and 0 < measured.get("band", 0) < 1:
        return float(measured["band"]), f"measured {cls} n={measured['n']}"
    spec = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    band = next(float(b["band"]) for b in spec["noise_band"] if b["class"] == "cuda-default")
    why = f" (measured n={measured.get('n', 0)} < 3)" if isinstance(measured, dict) else ""
    return band, f"provisional cuda-default{why}"


# ---- throttle by the session sampler ------------------------------------------------------------------------


def load_samples(d: Path) -> list[tuple[datetime, int, int]] | None:
    """(time, reasons mask, GPU index) per line of gpu-samples.csv (`epoch,timestamp,reasons,...,index`); a file
    without the index column (older sampler) counts as GPU 0. None when there is no file or no readable line."""
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
            t = datetime.strptime(f[1].split(".")[0], SAMPLE_FMT)
            out.append((t, int(f[2], 16), int(f[7]) if len(f) > 7 else 0))
        except ValueError:
            continue
    return out or None


def run_window(d: Path, run: str, engine: str = "") -> tuple[datetime, datetime] | None:
    """Measured window of a run. Source by run type:
    - vllm: the gateway_bench result (`finished` minus summary.wall_s), as Phase 4;
    - every other run: window-<run>.json, the box-local start and end of the process (phase5_box.sh run_one), narrowed to
      the last 2 * (sum of the rows' avg_ns) + 10 s when the result rows carry avg_ns, which cuts the model load
      (llama-bench warms up each test once, hence the factor 2).
    One second of slack each side for the stamps' truncation. None when there is no usable stamp."""
    if engine == "vllm":
        return _p4.result_window(read_json(d / f"baseline-{run}.json"))
    w = read_json(d / f"window-{run}.json")
    try:
        a = datetime.strptime(w["start"], "%Y-%m-%dT%H:%M:%S")
        b = datetime.strptime(w["end"], "%Y-%m-%dT%H:%M:%S")
    except (TypeError, KeyError, ValueError):
        return None
    for kind in ("bench", "baseline"):
        j = read_json(d / f"{kind}-{run}.json")
        ns = [r["avg_ns"] for r in (j if isinstance(j, list) else []) if isinstance(r, dict)
              and isinstance(r.get("avg_ns"), (int, float)) and r["avg_ns"] > 0]
        if ns:
            a = max(a, b - timedelta(seconds=2 * sum(ns) / 1e9 + 10))
            break
    return a - timedelta(seconds=1), b + timedelta(seconds=1)


def gpu_stats(samples, win) -> dict[int, dict] | None:
    """{gpu: {samples, share_board, share_cap, disqualifying, bad_bits}} over the samples inside the window; shares are
    fractions of the window's samples with 0x200/0x400 and with 0x4. None without samples inside."""
    if samples is None or win is None:
        return None
    per: dict[int, list[int]] = {}
    for t, m, g in samples:
        if win[0] <= t <= win[1]:
            per.setdefault(g, []).append(m)
    if not per:
        return None
    out = {}
    for g, ms in sorted(per.items()):
        bad = [m for m in ms if m & ~WINDOW_OK]
        bits = 0
        for m in bad:
            bits |= m & ~WINDOW_OK
        out[g] = {"samples": len(ms), "share_board": sum(1 for m in ms if m & BOARD_BITS) / len(ms),
                  "share_cap": sum(1 for m in ms if m & CAP_BIT) / len(ms), "disqualifying": len(bad),
                  "bad_bits": f"{bits:#x}"}
    return out


_SESSION_CACHE: dict = {}


def _stamp(d: Path):
    """What changes when the box adds a run or a sample: the cache key of session_sampler."""
    parts = [d / "gpu-samples.csv", d]
    key = [(p.stat().st_mtime_ns, p.stat().st_size) if p.exists() else None for p in parts]
    return (str(d), tuple(key), len(list(d.glob("window-*.json"))), len(list(d.glob("run-*.rc"))))


def config_of(run: str) -> str:
    """The configuration a run belongs to: the run name without its round (`-r<k>`) or `-aa`."""
    return re.sub(r"-(r\d+|aa)$", "", run)


def _round_rank(run: str) -> int:
    m = re.search(r"-r(\d+)$", run)
    return int(m.group(1)) if m else 10**6  # the A/A run is the last one of its configuration


def session_sampler(d: Path) -> dict | None:
    """{runs: {run: {gpu: stats}}, config: {config: {gpu: {share_board, share_cap}}}, outliers: {run: [reason]}} over the
    session, or None without sampler data. `config` holds the median share over the rounds of one configuration; a
    configuration whose share on a GPU spreads by more than SHARE_TOL between its rounds has the round furthest from the
    median as an outlier (a tie goes to the later round). Shares are never compared across configurations (ADR 0068
    addendum 2026-11-18)."""
    key = _stamp(d)
    if _SESSION_CACHE.get("key") == key:
        return _SESSION_CACHE["val"]
    samples = load_samples(d)
    val = None
    if samples is not None:
        runs = {}
        for rc in sorted(d.glob("run-*.rc")):
            run = rc.stem[len("run-") :]
            if "-anchor-" in run:
                continue
            st = gpu_stats(samples, run_window(d, run, "vllm" if "-vllm-" in run else ""))
            if st:
                runs[run] = st
        by_cfg: dict[str, list[str]] = {}
        for run in runs:
            by_cfg.setdefault(config_of(run), []).append(run)
        cfg: dict[str, dict] = {}
        outliers: dict[str, list[str]] = {}
        for c, members in by_cfg.items():
            cfg[c] = {}
            for g in sorted({g for r in members for g in runs[r]}):
                have = [r for r in members if g in runs[r]]
                cfg[c][g] = {}
                for name, k in (("0x200/0x400", "share_board"), ("0x4", "share_cap")):
                    vals = {r: runs[r][g][k] for r in have}
                    med = statistics.median(vals.values())
                    cfg[c][g][k] = med
                    if len(have) > 1 and max(vals.values()) - min(vals.values()) > SHARE_TOL + EPS:
                        far = max(have, key=lambda r: (abs(vals[r] - med), _round_rank(r)))
                        outliers.setdefault(far, []).append(
                            f"{far}: GPU {g}: share of {name} samples {vals[far]:.2f} is furthest from the median "
                            f"{med:.2f} of {c}, whose rounds differ by {max(vals.values()) - min(vals.values()):.2f} "
                            f"(more than {SHARE_TOL})")
        val = {"runs": runs, "config": cfg, "outliers": outliers}
    _SESSION_CACHE.update(key=key, val=val)
    return val


def sampler_judgement(d: Path, run: str) -> tuple[list[str], dict]:
    """(reasons the run is unclean by the sampler rule, the record for verdict.json). The rule `sampler` applies when
    the run has window samples; otherwise `single-sample` (the before and after records, ADR 0065)."""
    sess = session_sampler(d)
    st = sess["runs"].get(run) if sess else None
    if st is None:
        return [f"{run}: {x}" for x in throttle_reasons(d, run)], {"rule": "single-sample"}
    reasons, gpus = [], {}
    cfg = sess["config"][config_of(run)]
    for g, x in st.items():
        gpus[str(g)] = {**x, "config_median_share_board": cfg[g]["share_board"], "config_median_share_cap": cfg[g]["share_cap"]}
        if x["disqualifying"]:
            reasons.append(f"{run}: GPU {g}: {x['disqualifying']} window sample(s) with throttle bits {x['bad_bits']}")
    reasons += sess["outliers"].get(run, [])
    rec = {"rule": "sampler", "gpus": gpus, "disqualifying": sum(x["disqualifying"] for x in st.values()),
           "samples": sum(x["samples"] for x in st.values())}
    # the single before and after samples are recorded, not judged
    rec["before_after"] = _p4.throttle_record(d, run)[1]
    return reasons, rec


# ---- runs ---------------------------------------------------------------------------------------------------


def rates(path: Path) -> tuple[float, float] | None:
    """(pp_tps, tg_tps) from llama-bench rows (n_prompt only / n_gen only, avg_ts); None if either is missing."""
    j = read_json(path)
    rows = j if isinstance(j, list) else (j.get("rows") or j.get("results")) if isinstance(j, dict) else None
    pp = tg = None
    for r in rows or []:
        if not isinstance(r, dict) or not isinstance(r.get("avg_ts"), (int, float)) or r["avg_ts"] <= 0:
            continue
        np_, ng = r.get("n_prompt") or 0, r.get("n_gen") or 0
        if np_ > 0 and ng == 0:
            pp = float(r["avg_ts"])
        elif ng > 0 and np_ == 0:
            tg = float(r["avg_ts"])
    return (pp, tg) if pp and tg else None


def request_time(pp: float, tg: float) -> float:
    return PROMPT / pp + GEN / tg


def vllm_time(path: Path) -> float | None:
    """vLLM is measured through gateway_bench (report only): ttft plus the decode time of GEN tokens."""
    s = (read_json(path) or {}).get("summary") or {}
    ttft, rate = s.get("ttft_median_s"), s.get("decode_tps_median")
    return ttft + GEN / rate if ttft is not None and rate else None


def run_status(d: Path, kind: str, run: str, engine: str = "") -> dict:
    """One run: {clean, failed, reasons, pp, tg, t}. failed = the run exited non-zero (it did not produce a result)."""
    reasons: list[str] = []
    rc_p = d / f"run-{run}.rc"
    rc = None
    try:
        rc = int(rc_p.read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        reasons.append(f"{run}: no exit code record")
    if rc not in (0, None):
        reasons.append(f"{run}: exit code {rc}")
    f = d / f"{kind}-{run}.json"
    pp = tg = t = None
    if engine == "vllm":
        t = vllm_time(f)
        if t is None:
            reasons.append(f"{run}: no vllm result")
    else:
        r = rates(f)
        if r is None:
            reasons.append(f"{run}: no pp and tg rows")
        else:
            pp, tg = r
            t = request_time(pp, tg)
    thr, rec = sampler_judgement(d, run)
    reasons += thr
    cot = read_json(d / f"cotenant-{run}.json")
    if isinstance(cot, dict) and cot.get("cotenant_ok") is False:
        reasons.append(f"{run}: co-tenant ({cot.get('cotenant_reason', '?')})")
    return {"clean": not reasons, "failed": rc not in (0, None), "reasons": reasons, "pp": pp, "tg": tg, "t": t,
            "throttle": rec}


def rounds_of(d: Path, kind: str, prefix: str) -> list[int]:
    """Round numbers k that have a result or an exit code for `<prefix>-r<k>`."""
    pat = re.compile(re.escape(prefix) + r"-r(\d+)")
    ks = set()
    for p in list(d.glob(f"run-{prefix}-r*.rc")) + list(d.glob(f"{kind}-{prefix}-r*.json")):
        m = pat.fullmatch(p.stem[len("run-") :] if p.suffix == ".rc" else p.stem[len(kind) + 1 :])
        if m:
            ks.add(int(m.group(1)))
    return sorted(ks)


def measure(d: Path, kind: str, prefix: str, engine: str = "") -> dict:
    """All rounds of one configuration and the median of its clean ones."""
    runs = {k: run_status(d, kind, f"{prefix}-r{k}", engine) for k in rounds_of(d, kind, prefix)}
    clean = [r for r in runs.values() if r["clean"]]
    out = {"runs": runs, "clean": len(clean), "pp": None, "tg": None, "t": None,
           "failed_on_box": sum(r["failed"] for r in runs.values()) >= 2 and not any(r["t"] and not r["failed"] for r in runs.values())}
    if clean:
        if engine == "vllm":
            out["t"] = statistics.median(r["t"] for r in clean)
        else:
            out["pp"] = statistics.median(r["pp"] for r in clean)
            out["tg"] = statistics.median(r["tg"] for r in clean)
            out["t"] = request_time(out["pp"], out["tg"])
    return out


# ---- configurations -----------------------------------------------------------------------------------------


def models_in(d: Path) -> list[str]:
    return sorted(p.stem[len("candidates-") :] for p in d.glob("candidates-*.json"))


def shape_of(cid: str) -> str:
    if cid.startswith("single"):
        return "single"
    if cid.startswith("layer-"):
        return "layer"
    if cid.startswith("tp2-pp2"):
        return "tp-pp"
    if cid.startswith("tp-"):
        return "tp"
    if cid.startswith("ep-"):
        return "ep"
    return cid


def candidate_ids(d: Path, model: str) -> list[str]:
    """Run candidates in the case order of `candidates --multi` (pruned shapes are not run)."""
    c = read_json(d / f"candidates-{model}.json") or {}
    return [x["id"] for x in c.get("candidates", [])]


def baseline_configs(d: Path, model: str) -> list[tuple[str, str]]:
    """(engine, sm) the baselines run: accepted in the pre-pass, or with files. llama-bench first, sm in a fixed order."""
    found: set[tuple[str, str]] = set()
    for p in d.glob(f"prepass-{model}-*.json"):
        rest = p.stem[len(f"prepass-{model}-") :]
        for e in ENGINES:
            if rest.startswith(e + "-") and (read_json(p) or {}).get("ok") is True:
                found.add((e, rest[len(e) + 1 :]))
    for p in d.glob(f"baseline-{model}-*-r*.json"):
        m = re.fullmatch(rf"baseline-{re.escape(model)}-({'|'.join(ENGINES)})-(.+)-r\d+", p.stem)
        if m:
            found.add((m.group(1), m.group(2)))
    order = {"layer": 0, "row": 1, "tensor": 2}
    return sorted(found, key=lambda x: (ENGINES.index(x[0]), order.get(x[1], 9), x[1]))


def baseline_prefix(model: str, engine: str, sm: str) -> str:
    return f"{model}-{engine}-{sm}"


def configs(d: Path, model: str, with_report_only: bool = True) -> list[tuple[str, str, str]]:
    """('tet', id, '') for candidates, then ('base', engine, sm)."""
    out = [("tet", c, "") for c in candidate_ids(d, model)]
    out += [("base", e, sm) for e, sm in baseline_configs(d, model) if with_report_only or e == GATED]
    return out


def measure_config(d: Path, model: str, cfg: tuple[str, str, str]) -> dict:
    kind, a, b = cfg
    if kind == "tet":
        return measure(d, "bench", f"{model}-{a}")
    return measure(d, "baseline", baseline_prefix(model, a, b), a)


def round_order(d: Path, model: str, k: int, only: list[tuple[str, str, str]] | None = None,
                alternate: bool = False) -> list[tuple[str, str, str]]:
    """Case order rotated by k; round 2 reversed. Candidates that failed on the box twice are not run again.
    `alternate` (the ONLY knob, step-0 pairs): the given order in odd rounds, reversed in even ones, so two
    configurations swap places every round (the rotation would repeat the order for two)."""
    cfgs = [c for c in (only if only is not None else configs(d, model))
            if not (c[0] == "tet" and measure_config(d, model, c)["failed_on_box"])]
    if not cfgs:
        return []
    if alternate:
        return cfgs if k % 2 else cfgs[::-1]
    if k == 2:
        return cfgs[::-1]
    r = k % len(cfgs)
    return cfgs[r:] + cfgs[:r]


def need_rerun(d: Path, model: str) -> list[tuple[str, str, str]]:
    """Gated configurations (candidates, llama-bench modes) with fewer than two clean rounds."""
    return [c for c in configs(d, model, with_report_only=False)
            if (m := measure_config(d, model, c))["clean"] < MIN_CLEAN and not m["failed_on_box"]]


# ---- prediction validity ------------------------------------------------------------------------------------


def prediction_problem(d: Path, model: str, pred: Path) -> str | None:
    """A prediction file written after (or changed since) the first candidate run makes the case invalid."""
    events = []
    try:
        events = [json.loads(ln) for ln in (d / "timeline.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    except (OSError, ValueError):
        pass
    events = [e for e in events if isinstance(e, dict) and e.get("model") == model]
    first_run = min((e["t"] for e in events if e.get("event") == "run" and e.get("kind") in ("candidate", "aa")), default=None)
    preds = [e for e in events if e.get("event") == "predict"]
    if preds:
        # the first event that names this very file is when it was committed (a resume may log it again)
        same = [e for e in preds if e.get("sha256") == sha256_file(pred)]
        if not same:
            return "prediction file changed after it was committed"
        if first_run is not None and same[0]["t"] > first_run:
            return "prediction written after the first candidate run"
        return None
    runs = [p for c in candidate_ids(d, model) for p in list(d.glob(f"bench-{model}-{c}-*.json")) + list(d.glob(f"run-{model}-{c}-*.rc"))]
    if runs and pred.stat().st_mtime > min(p.stat().st_mtime for p in runs) + EPS:
        return "prediction file is newer than a candidate run"
    return None


# ---- case ---------------------------------------------------------------------------------------------------


def case_verdict(d: Path, model: str, env: dict, band: float, gates: dict) -> tuple[dict, dict | None, list]:
    """(case dict of the ADR, predict sha entry, unclean runs)."""
    n = env.get("gpu_count") or 0
    box = f"{n}x"
    cand_file = read_json(d / f"candidates-{model}.json") or {}
    pred_p = d / f"predict-{model}-{box}.json"
    if not pred_p.exists():
        alt = sorted(d.glob(f"predict-{model}-*.json"))
        pred_p = alt[0] if alt else pred_p
    pred = read_json(pred_p) or {}
    prow = {r["id"]: r for r in pred.get("candidates", [])}
    reasons: list[str] = []  # why the session is not clean: no verdict
    notes: list[str] = []  # failures that count against the case
    case = {
        "case": ("M" if "a3b" in model or "moe" in model else "D") + str(n),
        "model": model, "box_gpus": n, "gpu_class": band_class(env), "p2p": env.get("p2p"),
        "comm_impl": env.get("comm_impl"),
        "candidates": [], "pick": None, "winner_set": [], "pick_ok": None, "prediction_ok": None,
        "baselines": [], "best_baseline": None, "baseline_ratio": None, "baseline_ok": None,
        "aa": {"t_start": None, "t_end": None}, "clean": False, "reasons": reasons, "pass": None,
    }
    sha = sha256_file(pred_p) if pred_p.exists() else None
    unclean: list[dict] = []
    invalid = None
    if not pred_p.exists():
        invalid = "no predict file"
    else:
        invalid = prediction_problem(d, model, pred_p)
    if invalid:
        reasons.append(f"invalid: {invalid}")
    for g in ("gpu_parity", "comm_probe", "baseline_modes"):
        if gates.get(g) != "ok" and (g != "baseline_modes" or g in gates):  # the older box scripts had no such gate
            reasons.append(f"gate {g}: {gates.get(g, 'missing')}")

    # candidates
    cids = candidate_ids(d, model)
    meas = {c: measure(d, "bench", f"{model}-{c}") for c in cids}
    for c in cand_file.get("pruned", []):
        case["candidates"].append({"id": c["id"], "shape": shape_of(c["id"]), "pruned": True, "reason": c["reason"],
                                   "predicted": None, "measured": None, "err": None, "within": None})
    failed = []
    for c in cids:
        m, p = meas[c], prow.get(c)
        entry = {"id": c, "shape": shape_of(c), "pruned": False, "reason": None,
                 "predicted": {"pp_tps": p["pp_tps"], "tg_tps": p["tg_tps"], "t_s": p["t_s"]} if p else None,
                 "measured": {"pp_tps": m["pp"], "tg_tps": m["tg"], "t_s": m["t"], "rounds_clean": m["clean"]},
                 "err": None, "within": None}
        if p is None:
            reasons.append(f"{c}: no prediction")
        if m["failed_on_box"]:
            failed.append(c)
            entry["within"] = False
            entry["reason"] = "failed on the box"
            notes.append(f"{c}: failed on the box (exit code of its runs not 0)")
        elif p is not None and m["t"] is not None:
            entry["err"] = {"pp": p["pp_tps"] / m["pp"] - 1, "tg": p["tg_tps"] / m["tg"] - 1}
            entry["within"] = abs(entry["err"]["pp"]) <= TOL + EPS and abs(entry["err"]["tg"]) <= TOL + EPS
        case["candidates"].append(entry)
        for k, r in m["runs"].items():
            if not r["clean"]:
                unclean.append({"run": f"{model}-{c}-r{k}", "reasons": r["reasons"]})

    # enough clean rounds
    thin = [f"{c}: {meas[c]['clean']} clean round(s), need {MIN_CLEAN}" for c in cids
            if meas[c]["clean"] < MIN_CLEAN and not meas[c]["failed_on_box"]]
    reasons += thin

    # baselines
    bl = {}
    for e, sm in baseline_configs(d, model):
        m = measure(d, "baseline", baseline_prefix(model, e, sm), e)
        bl[(e, sm)] = m
        pp = read_json(d / f"prepass-{model}-{e}-{sm}.json") or {}
        case["baselines"].append({"engine": e, "sm": sm, "t_s": m["t"], "comm": pp.get("comm")})
        for k, r in m["runs"].items():
            if not r["clean"]:
                unclean.append({"run": f"{model}-{e}-{sm}-r{k}", "reasons": r["reasons"]})
        if e == GATED and m["clean"] < MIN_CLEAN:
            reasons.append(f"{e} -sm {sm}: {m['clean']} clean round(s), need {MIN_CLEAN}")

    # pick and winner set
    ranked = [c for c in cids if c in prow]
    pick = min(ranked, key=lambda c: prow[c]["t_s"]) if ranked else None
    case["pick"] = pick
    done = [c for c in cids if meas[c]["t"] is not None and meas[c]["clean"] >= MIN_CLEAN]
    undone = [c for c in cids if c not in done and c not in failed]
    judged = [e for e in case["candidates"] if not e["pruned"]]
    if failed:
        case["prediction_ok"] = False  # predicted feasible, failed on the box
    if not undone:
        if done:
            best = min(meas[c]["t"] for c in done)
            case["winner_set"] = [c for c in done if meas[c]["t"] <= best * (1 + band) + EPS]
            case["pick_ok"] = pick in case["winner_set"]
        if not failed and judged and all(e["within"] is not None for e in judged):
            case["prediction_ok"] = all(e["within"] for e in judged)

    # baseline gate
    gated = {k: m for k, m in bl.items() if k[0] == GATED}
    if gated and all(m["t"] is not None and m["clean"] >= MIN_CLEAN for m in gated.values()):
        (be, bs), bm = min(gated.items(), key=lambda kv: kv[1]["t"])
        case["best_baseline"] = {"engine": be, "sm": bs, "t_s": bm["t"]}
        t_pick = meas[pick]["t"] if pick in meas and meas[pick]["clean"] >= MIN_CLEAN else None
        if t_pick:
            case["baseline_ratio"] = bm["t"] / t_pick
            case["baseline_ok"] = case["baseline_ratio"] >= 1 - band - EPS
    elif not gated:
        reasons.append(f"no {GATED} baseline result")

    # A/A: the first candidate again as the last run of the case
    first = cids[0] if cids else None
    if first:
        start = next((r for k, r in sorted(meas[first]["runs"].items()) if r["clean"]), None)
        end = run_status(d, "bench", f"{model}-{first}-aa")
        case["aa"]["t_start"] = start["t"] if start else None
        case["aa"]["t_end"] = end["t"] if end["clean"] else None
        if not end["clean"]:
            reasons += [f"A/A: {x}" for x in (end["reasons"] or ["missing"])]
            unclean.append({"run": f"{model}-{first}-aa", "reasons": end["reasons"]})
        elif start is None:
            reasons.append("A/A: no clean start run")
        elif abs(end["t"] / start["t"] - 1) > band + EPS:
            reasons.append(f"A/A: |T_end/T_start - 1| = {abs(end['t'] / start['t'] - 1):.4f} > band {band}")
    else:
        reasons.append("no candidates")

    case["clean"] = not reasons
    reasons += notes
    flags = [case["pick_ok"], case["prediction_ok"], case["baseline_ok"]]
    if not case["clean"]:
        case["pass"] = None
    elif False in flags:
        case["pass"] = False
    elif all(f is True for f in flags):
        case["pass"] = True
    return case, ({"model": model, "sha256": sha} if sha else None), unclean


# ---- verdict ------------------------------------------------------------------------------------------------


def verdict(d: Path, noise_dir: Path = ROOT / "bench" / "noise", spec: Path = ROOT / "bench" / "spec" / "phase5.toml") -> tuple[dict, dict]:
    env = read_json(d / "env.json") or {}
    gates = read_json(d / "gates.json") or {}
    band, source = choose_band(env, noise_dir, spec)
    cases, shas, notes = [], {}, {"unclean_runs": []}
    for model in models_in(d):
        c, s, unclean = case_verdict(d, model, env, band, gates)
        cases.append(c)
        if s:
            shas[model] = s["sha256"]
        notes["unclean_runs"] += unclean
    return {"band": band, "band_source": source, "predict_sha256": shas, "cases": cases, "phase5_pass": final_pass(cases),
            "throttle": throttle_record_of_session(d)}, notes


def throttle_record_of_session(d: Path) -> dict:
    """Per run (anchors excluded): the rule used, the shares per GPU against the median of its configuration, disqualifying
    and sample counts (ADR 0068 addenda 2026-11-17, 2026-11-18)."""
    runs = {}
    for rc in sorted(d.glob("run-*.rc")):
        run = rc.stem[len("run-") :]
        if "-anchor-" not in run:
            runs[run] = sampler_judgement(d, run)[1]
    sess = session_sampler(d)
    rules = {r["rule"] for r in runs.values()}
    rule = "sampler" if rules == {"sampler"} else "single-sample" if rules <= {"single-sample"} else "mixed"
    return {"rule": rule, "compare": "per-configuration", "share_tolerance": SHARE_TOL,
            "config_median_share": {c: {str(g): m for g, m in gs.items()} for c, gs in (sess["config"] if sess else {}).items()},
            "runs": runs}


# ---- step-0 check (ADR 0069) ----------------------------------------------------------------------------------


def parse_only(spec: str) -> list[tuple[str, str, str]]:
    """The ONLY knob of phase5_box.sh: space-separated configuration ids. `tp-2` is a candidate (plan id), `engine:sm`
    (for example `llama-bench:tensor`) a baseline."""
    out = []
    for tok in spec.split():
        if ":" in tok:
            e, sm = tok.split(":", 1)
            out.append(("base", e, sm))
        else:
            out.append(("tet", tok, ""))
    return out


def step0(d: Path, spec: str, noise_dir: Path = ROOT / "bench" / "noise",
          spec_path: Path = ROOT / "bench" / "spec" / "phase5.toml") -> dict:
    """Own executor against the baseline of ONLY, per model and round (ADR 0069 "Step-0 check"): the ratios own / baseline
    of pp and tg, from the pairs of rounds in which both runs are clean. Passes when every model has at least two clean
    pairs and all their ratios are at least 1 - band; false when a clean ratio is below; null when pairs are missing."""
    cfgs = parse_only(spec)
    own = next((c for c in cfgs if c[0] == "tet"), None)
    base = next((c for c in cfgs if c[0] == "base"), None)
    env = read_json(d / "env.json") or {}
    band, source = choose_band(env, noise_dir, spec_path)
    models = {}
    for m in models_in(d):
        rounds = []
        if own and base:
            po, pb = f"{m}-{own[1]}", baseline_prefix(m, base[1], base[2])
            ks = sorted(set(rounds_of(d, "bench", po)) & set(rounds_of(d, "baseline", pb)))
            for k in ks:
                ro = run_status(d, "bench", f"{po}-r{k}")
                rb = run_status(d, "baseline", f"{pb}-r{k}", base[1])
                clean = ro["clean"] and rb["clean"]
                rounds.append({
                    "round": k, "clean": clean,
                    "own": {"pp_tps": ro["pp"], "tg_tps": ro["tg"]}, "baseline": {"pp_tps": rb["pp"], "tg_tps": rb["tg"]},
                    "ratio_pp": ro["pp"] / rb["pp"] if clean else None, "ratio_tg": ro["tg"] / rb["tg"] if clean else None,
                    "reasons": ro["reasons"] + rb["reasons"]})
        good = [r for r in rounds if r["clean"]]
        low = [r for r in good if min(r["ratio_pp"], r["ratio_tg"]) < 1 - band - EPS]
        ok = False if low else (True if len(good) >= 2 else None)
        models[m] = {"own": own[1] if own else None, "baseline": f"{base[1]}:{base[2]}" if base else None,
                     "rounds": rounds, "clean_pairs": len(good),
                     "ratio_pp_min": min((r["ratio_pp"] for r in good), default=None),
                     "ratio_tg_min": min((r["ratio_tg"] for r in good), default=None), "ok": ok}
    oks = [v["ok"] for v in models.values()]
    passed = None if not oks else False if any(o is False for o in oks) else True if all(o is True for o in oks) else None
    return {"band": band, "band_source": source, "threshold": 1 - band, "only": spec, "models": models,
            "step0_pass": passed, "throttle": throttle_record_of_session(d)}


def final_pass(cases: list[dict]) -> bool | None:
    """True only when all four cases (D2, M2, D4, M4) are present and pass: one session never has more than two."""
    if {c["case"] for c in cases} != set(CASES):
        return False
    ps = [c["pass"] for c in cases]
    return False if any(p is False for p in ps) else (True if all(p is True for p in ps) else None)


def merge(files: list[Path]) -> dict:
    """The cases of several session verdicts (V2, V4a, V4b); a later file replaces a case of the same name."""
    by_case, shas, sessions = {}, {}, []
    for f in files:
        v = json.loads(f.read_text(encoding="utf-8"))
        sessions.append({"file": f.name, "band": v["band"], "band_source": v["band_source"]})
        for c in v["cases"]:
            by_case[c["case"]] = c
            if c["model"] in v["predict_sha256"]:
                shas[f"{c['model']}-{c['box_gpus']}x"] = v["predict_sha256"][c["model"]]
    cases = [by_case[k] for k in CASES if k in by_case]
    first = sessions[0] if sessions else {"band": None, "band_source": None}
    return {"band": first["band"], "band_source": first["band_source"], "sessions": sessions, "predict_sha256": shas,
            "cases": cases, "phase5_pass": final_pass(cases)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, nargs="?")
    ap.add_argument("--round-order", nargs=2, metavar=("MODEL", "K"), help="print the configurations of round K and exit")
    ap.add_argument("--need-rerun", metavar="MODEL", help="print the gated configurations with fewer than two clean rounds")
    ap.add_argument("--only-need", action="store_true", help="with --round-order: only the configurations --need-rerun lists")
    ap.add_argument("--only", metavar="IDS", help="with --round-order: only these configurations (tet ids, engine:sm); "
                    "alone: write step0.json for them (ADR 0069 step-0 check)")
    ap.add_argument("--first", metavar="MODEL", help="print the first candidate of the case (the A/A run repeats it)")
    ap.add_argument("--p2p", metavar="FILE", help="print all|none|partial for a `nvidia-smi topo -p2p r` file and exit")
    ap.add_argument("--merge", nargs="+", type=Path, metavar="VERDICT")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--noise-dir", type=Path, default=ROOT / "bench" / "noise")
    ap.add_argument("--spec", type=Path, default=ROOT / "bench" / "spec" / "phase5.toml")
    a = ap.parse_args(argv)
    if a.merge:
        v = merge(a.merge)
        (a.out or Path("verdict-all.json")).write_text(json.dumps(v, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"phase5_pass": v["phase5_pass"], "cases": {c["case"]: c["pass"] for c in v["cases"]}}))
        return 0
    if a.p2p:
        try:
            print(p2p_class(Path(a.p2p).read_text(encoding="utf-8", errors="replace")) or "")
        except OSError:
            print("")
        return 0
    if a.dir is None:
        ap.error("dir is required")
    fmt = lambda cs: "\n".join(f"tet {c[1]}" if c[0] == "tet" else f"base {c[1]} {c[2]}" for c in cs)  # noqa: E731
    if a.first:
        print((candidate_ids(a.dir, a.first) or [""])[0])
        return 0
    if a.need_rerun:
        print(fmt(need_rerun(a.dir, a.need_rerun)))
        return 0
    if a.round_order:
        model, k = a.round_order[0], int(a.round_order[1])
        if a.only:  # a shape the enumeration pruned, or a split mode the baseline refuses, is not run
            have = configs(a.dir, model)
            only = [c for c in parse_only(a.only) if c in have]
        else:
            only = need_rerun(a.dir, model) if a.only_need else None
        print(fmt(round_order(a.dir, model, k, only, alternate=bool(a.only))))
        return 0
    if a.only:
        v = step0(a.dir, a.only, a.noise_dir, a.spec)
        (a.dir / "step0.json").write_text(json.dumps(v, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"step0_pass": v["step0_pass"], "band": v["band"],
                          "models": {m: x["ok"] for m, x in v["models"].items()}}))
        return 0
    v, notes = verdict(a.dir, a.noise_dir, a.spec)
    (a.dir / "verdict.json").write_text(json.dumps(v, indent=2) + "\n", encoding="utf-8")
    (a.dir / "verdict-notes.json").write_text(json.dumps(notes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"phase5_pass": v["phase5_pass"], "band": v["band"], "band_source": v["band_source"],
                      "cases": {c["case"]: c["pass"] for c in v["cases"]}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
