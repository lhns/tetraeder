#!/usr/bin/env python3
"""Verdict of the segment confirmation (tree node bench-235b-segment-confirm, ADR 0092 section 6).

    sc_summary.py bench/results/<name>        writes <name>/verdict.json and prints the tables

Clean, then judged (ADR 0091 answers 7 and 8; every rule is a line of `rules` in verdict.json, all must hold):
  rounds_clean     per judged round: warm block says covering and not unclean (timed prefill read 0 bytes), balloon_ok not
                   false, cotenant_ok not false, no disqualifying GPU throttle bit before or after the run; at least
                   ROUNDS (3) clean rounds
  routing_gate     medians of the clean rounds against the replay of the frozen prediction: decode vram share within 0.02
                   of vram_pair_share, full-hit share within 0.02, RAM hit within 0.005
  aa_early         judged r1 against r2 (right after warm-up) within 0.08 on T = 2048 / pp + 128 / tg
  aa               judged r1 against judged/aa-last.json, the judged plan repeated as the last run, within 0.08 on T
  one_numa_node    env/lscpu.txt shows exactly one NUMA node
  cotenant_ssh     env/cotenant.json written by `engine_sample.py cotenant`, under the limit
  freeze           predictions/FROZEN.sha256 and every file of SHA256SUMS match
  fit_band         the anchors' fit band within FIT_BAND_LIMIT (scripts/sc_gates.py, crates/planner/src/segment.rs)
  anchors_real_hidden  every anchor ran on the dumped hidden rows, not fixed-seed input
There is no prefill noise band (owner answer 8). An unclean session gives no verdict; the errors on the medians are still printed.
Then, only when clean: e = pred / meas - 1 on the medians of the clean rounds, for pp (prefill_tok_s) and tg
(decode_tok_s); pass when both are within 0.15. The line is not changed after a miss.
Everything else (anchors, fixed-seed segment, whole-model run, refit on real-hidden anchors) is report only.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc_gates as sg  # noqa: E402

LINE = 0.15
GATE_SHARE = 0.02
GATE_RAM = 0.005
AA_BAND = sg.AA_BAND
ROUNDS_NEEDED = 3
OK_BITS = 0x1 | 0x2 | 0x4 | 0x100 | 0x200 | 0x400  # idle, app clocks, power cap software, and the idle gate's others


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_of(j: dict) -> dict:
    runs = j.get("runs")
    return runs[-1] if isinstance(runs, list) and runs else j


def throttle(path: str) -> list[str]:
    """Reasons from <json>.gpu.txt (lines `before: 0x..,power,limit,default` and `after: ...`)."""
    bad = []
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        return ["no gpu throttle record"]
    seen = set()
    for line in lines:
        when, _, rest = line.partition(":")
        parts = [x.strip() for x in rest.split(",")]
        if not parts or not parts[0].startswith("0x"):
            continue
        seen.add(when)
        bits = int(parts[0], 16)
        if bits & ~OK_BITS:
            bad.append(f"throttle bits {parts[0]} {when} the run")
        if len(parts) >= 4:
            try:
                if abs(float(parts[2].split()[0]) - float(parts[3].split()[0])) > 0.5:
                    bad.append(f"power limit {parts[2]} below default {parts[3]}")
            except ValueError:
                pass
    if not {"before", "after"} <= seen:
        bad.append("incomplete gpu throttle record")
    return bad


def unclean(path: str) -> list[str]:
    j = load(path)
    r = run_of(j)
    why = []
    w = r.get("warm") or {}
    if w.get("covering") is not True:
        why.append("plan does not cover every expert")
    if w.get("unclean"):
        why.append(f"timed prefill read {w.get('timed_prefill_disk_bytes')} bytes from disk")
    if (r.get("prefill_io") or {}).get("bytes"):
        why.append(f"prefill_io.bytes {(r.get('prefill_io') or {}).get('bytes')}")
    if j.get("balloon_ok") is False:
        why.append(f"balloon: {j.get('balloon_reason', '')}")
    if j.get("cotenant_ok") is False:
        why.append(f"co-tenant: {j.get('cotenant_reason', '')}")
    if not r.get("prefill_tok_s") or not r.get("decode_tok_s"):
        why.append("no rates")
    why += throttle(path[: -len(".json")] + ".gpu.txt")
    return why


def med(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def numa(j: dict) -> dict:
    """Share of the engine's thread samples on its busiest NUMA node, prefill and decode window (1 = one node)."""
    h = j.get("host") or {}
    return {"numa_top_prefill": (h.get("prefill") or {}).get("top_node_share"),
            "numa_top_decode": (h.get("decode") or {}).get("top_node_share")}


def layer_stats(r: dict) -> dict:
    d = r.get("decode_layers") or {}
    return {
        "vram_pair_share": d.get("vram_pair_share"),
        "full_hit_rate_mean": d.get("full_hit_rate_mean"),
        "ram_hit_rate_mean": d.get("ram_hit_rate_mean"),
    }


def e(pred: float, meas: float | None) -> float | None:
    return None if not meas else pred / meas - 1.0


def rounds(res: str, pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(res, pattern))):
        if f.endswith(".gpu.json"):
            continue
        r = run_of(load(f))
        out.append({"file": os.path.relpath(f, res).replace(os.sep, "/"), "pp": r.get("prefill_tok_s"), "tg": r.get("decode_tok_s"),
                    "bytes": (r.get("prefill_io") or {}).get("bytes"), **numa(load(f)), **layer_stats(r)})
    return out


def main() -> int:
    res = sys.argv[1]
    pred = load(os.path.join(res, "predictions", "pred.json"))
    pr = pred["prediction"]
    out: dict = {"prediction": {"pp_tps": pr["pp_tps"], "tg_tps": pr["tg_tps"], "routing": pred["routing"],
                                "routing_trace": pred["routing_trace"], "hw_sha256": pred["hw_sha256"],
                                "uncalibrated_pp_tps": pr["uncalibrated_pp_tps"],
                                "uncalibrated_tg_tps": pr["uncalibrated_tg_tps"], "band": pr["band"]}}
    try:
        frozen = open(os.path.join(res, "predictions", "FROZEN.sha256"), encoding="utf-8").read().split()[0]
    except (OSError, IndexError):
        frozen = ""
    out["frozen_sha256"] = frozen

    files = sorted(glob.glob(os.path.join(res, "judged", "r[0-9]*.json")))
    files = [f for f in files if not f.endswith(".gpu.json")]
    per, clean = [], []
    for f in files:
        why = unclean(f)
        r = run_of(load(f))
        row = {"file": os.path.relpath(f, res).replace(os.sep, "/"), "pp": r.get("prefill_tok_s"), "tg": r.get("decode_tok_s"),
               "e_pp": e(pr["pp_tps"], r.get("prefill_tok_s")), "e_tg": e(pr["tg_tps"], r.get("decode_tok_s")),
               "clean": not why, "unclean_reasons": why, **numa(load(f)), **layer_stats(r)}
        per.append(row)
        if not why:
            clean.append(row)
    out["rounds"] = per
    rules: list[dict] = []

    def rule(name: str, ok: bool, detail: str) -> None:
        rules.append({"name": name, "ok": bool(ok), "detail": detail})

    rule("rounds_clean", len(clean) >= ROUNDS_NEEDED, f"{len(clean)} clean rounds of {len(per)} (need {ROUNDS_NEEDED})")
    if clean:
        m = {k: med([c[k] for c in clean]) for k in ("pp", "tg", "vram_pair_share", "full_hit_rate_mean", "ram_hit_rate_mean")}
        out["median"] = m
        out["e_pp"], out["e_tg"] = e(pr["pp_tps"], m["pp"]), e(pr["tg_tps"], m["tg"])
        gate = {
            "vram_share": {"replay": pr["replay_vram_share"], "measured": m["vram_pair_share"], "line": GATE_SHARE},
            "full_hit_share": {"replay": pr["replay_full_hit_share"], "measured": m["full_hit_rate_mean"], "line": GATE_SHARE},
            "ram_hit_rate": {"replay": pr["replay_ram_hit_rate"], "measured": m["ram_hit_rate_mean"], "line": GATE_RAM},
        }
        for g in gate.values():
            g["diff"] = None if g["measured"] is None else g["measured"] - g["replay"]
            g["ok"] = g["diff"] is not None and abs(g["diff"]) <= g["line"]
        out["gate"] = gate
        out["gate_held"] = all(g["ok"] for g in gate.values())
        rule("routing_gate", out["gate_held"], "; ".join(f"{k} {g['diff']:+.4f} against {g['line']}" for k, g in gate.items() if g["diff"] is not None))

    # A/A (ADR 0068, ADR 0091 case S, owner answer 8): the first judged round against the second right after warm-up
    # (the early abort) and against a repeat of the judged plan as the last run of the session (judged/aa-last.json)
    def aa_rule(name: str, a_file: str | None, b_file: str | None) -> None:
        if not a_file or not b_file or not os.path.exists(a_file) or not os.path.exists(b_file):
            out[name] = None
            rule(name, False, "no such run")
            return
        r = sg.aa(load(a_file), load(b_file), AA_BAND)
        r.update(first=os.path.relpath(a_file, res).replace(os.sep, "/"), second=os.path.relpath(b_file, res).replace(os.sep, "/"))
        out[name] = r
        rule(name, r["ok"], f"T {r['t_first']:.2f} -> {r['t_second']:.2f} s, {r['diff']:+.3f} against {AA_BAND}")

    aa_rule("aa_early", files[0] if files else None, files[1] if len(files) > 1 else None)
    aa_rule("aa", files[0] if files else None, os.path.join(res, "judged", "aa-last.json"))
    out["t_rounds"] = [sg.t_of(run_of(load(f))) for f in files if run_of(load(f)).get("prefill_tok_s")]

    # host, freeze and calibration gates, re-checked from the committed files
    env = os.path.join(res, "env")
    try:
        why = sg.check_numa(open(os.path.join(env, "lscpu.txt"), encoding="utf-8").read())
    except OSError:
        why = ["no env/lscpu.txt"]
    rule("one_numa_node", not why, "; ".join(why) or "lscpu shows one NUMA node")
    why = sg.check_cotenant(os.path.join(env, "cotenant.json"))
    rule("cotenant_ssh", not why, "; ".join(why) or "under the limit, read by engine_sample.py")
    why = sg.check_freeze(res)
    rule("freeze", not why, "; ".join(why) or "FROZEN.sha256 and every frozen file match")
    why = sg.check_fit_band(os.path.join(res, "predictions", "pred.json"))
    rule("fit_band", not why, "; ".join(why) or f"{pr['band']:.3f} within {sg.FIT_BAND_LIMIT}")
    why = sg.check_anchors_real(sg.anchor_files(res))
    rule("anchors_real_hidden", not why, "; ".join(why) or "every anchor ran on the dumped hidden rows")

    out["rules"] = rules
    out["clean"] = all(r["ok"] for r in rules)
    bad = [r for r in rules if not r["ok"]]
    out["verdict"] = "no verdict"
    status = ""
    if bad:
        status = "no verdict: the session is not clean (" + "; ".join(f"{r['name']}: {r['detail']}" for r in bad) + ")"
        if out.get("median"):
            status += f"; on the medians e_pp {out['e_pp']:+.3f}, e_tg {out['e_tg']:+.3f} against {LINE}, not a verdict"
    else:
        ok = abs(out["e_pp"]) <= LINE and abs(out["e_tg"]) <= LINE
        out["verdict"] = "pass" if ok else "miss"
        status = ("pass" if ok else "miss") + f": e_pp {out['e_pp']:+.3f}, e_tg {out['e_tg']:+.3f} against {LINE}"
    out["status"] = status

    # report only
    out["anchors"] = {"fit": pr["anchors"], "residuals": pr["residuals"]}
    out["whole_model_run"] = None
    wm = os.path.join(res, "trace", "whole-model.json")
    if os.path.exists(wm):
        r = run_of(load(wm))
        out["whole_model_run"] = {"pp_tok_s": r.get("prefill_tok_s"), "tg_tok_s": r.get("decode_tok_s"),
                                  "peak_rss_bytes": r.get("peak_rss_bytes")}
    seed = rounds(res, "report/seed-r[0-9]*.json")
    out["fixed_seed_segment"] = seed
    if seed and out.get("median"):
        out["fixed_seed_median"] = {k: med([s[k] for s in seed]) for k in ("pp", "tg", "vram_pair_share", "full_hit_rate_mean", "ram_hit_rate_mean")}
    extra = rounds(res, "report/extra-r[0-9]*.json")
    out["extra_rounds"] = extra
    if extra and out.get("median"):
        every = [{"pp": c["pp"], "tg": c["tg"]} for c in clean] + [{"pp": x["pp"], "tg": x["tg"]} for x in extra]
        mp, mt = med([x["pp"] for x in every]), med([x["tg"] for x in every])
        out["all_rounds_report"] = {
            "n": len(every), "pp": sorted(x["pp"] for x in every), "tg": sorted(x["tg"] for x in every),
            "median_pp": mp, "median_tg": mt, "e_pp": e(pr["pp_tps"], mp), "e_tg": e(pr["tg_tps"], mt),
            "pp_max_over_min": max(x["pp"] for x in every) / min(x["pp"] for x in every),
            "tg_max_over_min": max(x["tg"] for x in every) / min(x["tg"] for x in every),
        }
    out["anchor_repeat"] = rounds(res, "report/anchor-repeat-a[0-9]-r[0-9]*.json")
    real = os.path.join(res, "report", "pred-real-anchors.json")
    if os.path.exists(real) and out.get("median"):
        rp = load(real)["prediction"]
        out["refit_real_anchors"] = {"pp_tps": rp["pp_tps"], "tg_tps": rp["tg_tps"], "band": rp["band"],
                                     "e_pp": e(rp["pp_tps"], out["median"]["pp"]), "e_tg": e(rp["tg_tps"], out["median"]["tg"])}
    out["anchor_rounds"] = rounds(res, "anchors/a[0-9]-r[0-9]*.json")
    # GPU throttle samples over the whole session (1 Hz), report only
    csvp = os.path.join(res, "gpu", "gpu-samples.csv")
    if os.path.exists(csvp):
        bits: dict[str, int] = {}
        with open(csvp, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    bits[row[2]] = bits.get(row[2], 0) + 1
        out["gpu_reason_samples"] = bits
    with open(os.path.join(res, "verdict.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("status:", status)
    print(f"prediction (frozen {frozen[:16]}): pp {pr['pp_tps']:.2f} tg {pr['tg_tps']:.3f} tok/s (uncalibrated {pr['uncalibrated_pp_tps']:.2f}, {pr['uncalibrated_tg_tps']:.3f})")
    for r in per:
        print(f"  {r['file']:22} pp {r['pp']:8.2f} tg {r['tg']:7.3f} e_pp {r['e_pp']:+.3f} e_tg {r['e_tg']:+.3f} "
              f"numa top {r['numa_top_prefill'] or 0:.2f}/{r['numa_top_decode'] or 0:.2f} "
              f"{'clean' if r['clean'] else 'UNCLEAN ' + '; '.join(r['unclean_reasons'])}")
    if out.get("median"):
        m = out["median"]
        print(f"  median of {len(clean)} clean: pp {m['pp']:.2f} tg {m['tg']:.3f}  e_pp {out['e_pp']:+.3f} e_tg {out['e_tg']:+.3f}")
        for k, g in out["gate"].items():
            print(f"  gate {k:15} replay {g['replay']:.4f} measured {g['measured']:.4f} diff {g['diff']:+.4f} line {g['line']} {'ok' if g['ok'] else 'MISS'}")
    print(f"  T of the judged rounds {', '.join(f'{t:.1f}' for t in out['t_rounds'])} s")
    for r in rules:
        print(f"  rule {r['name']:20} {'ok  ' if r['ok'] else 'FAIL'} {r['detail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
