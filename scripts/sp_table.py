#!/usr/bin/env python3
"""Tables of the stream-predict session (tree node bench-235b-stream-predict).

    sp_table.py predictions <results>/predictions   the frozen picks and predictions
    sp_table.py results <results>                   e = pred / meas - 1 per arm and prediction, medians of the rounds

e_pp and e_tg are judged against 0.15; there is no verdict on M3 here.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys

LINE = 0.15


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def predictions(d: str) -> None:
    for f in sorted(glob.glob(os.path.join(d, "pred-*.json"))):
        j = load(f)
        k = j["pick"]
        print(
            f"{os.path.basename(f):38} routing={j['routing'][:28]:28} pick v{k['vram_slots']} r{k['ram_slots']} ub{k['ubatch']} "
            f"pp {k['pp_tps']:8.2f} tg {k['tg_tps']:7.3f} tok/s  T {k['request_s']:7.1f} s  binding {k['binding']}"
        )


def measured(res: str, arm: str) -> dict | None:
    pp, tg, cold = [], [], []
    sub = "arms-d2" if arm == "D2" else "arms"
    files = sorted(glob.glob(os.path.join(res, sub, f"exit-{arm}-r*.json")))
    for f in files:
        r = load(f)["runs"][0]
        pp.append(r["prefill_tok_s"])
        tg.append(r["decode_tok_s"])
        w = r.get("warm_prefill")
        if w:
            cold.append(w["prefill_tok_s"])
    if not pp:
        return None
    return {
        "rounds": len(pp),
        "pp": statistics.median(pp),
        "tg": statistics.median(tg),
        "pp_all": pp,
        "tg_all": tg,
        "pp_cold": statistics.median(cold) if cold else None,
    }


def results(res: str) -> None:
    out = {}
    for arm in ("D", "D2", "R"):
        m = measured(res, arm)
        if m is None:
            continue
        print(f"arm {arm}: {m['rounds']} rounds, pp median {m['pp']:.2f} tok/s {['%.2f' % x for x in m['pp_all']]}, "
              f"tg median {m['tg']:.3f} tok/s {['%.3f' % x for x in m['tg_all']]}"
              + (f", cold first prefill median {m['pp_cold']:.2f}" if m["pp_cold"] else ""))
        pdir = os.path.join(res, "predictions", "d2") if arm == "D2" else os.path.join(res, "predictions")
        for f in sorted(glob.glob(os.path.join(pdir, f"pred-{arm}-*.json"))):
            k = load(f)["pick"]
            e_pp = k["pp_tps"] / m["pp"] - 1
            e_tg = k["tg_tps"] / m["tg"] - 1
            ok = abs(e_pp) <= LINE and abs(e_tg) <= LINE
            tag = os.path.basename(f)[len("pred-"):-len(".json")]
            print(f"  {tag:24} pred pp {k['pp_tps']:8.2f} tg {k['tg_tps']:7.3f}   e_pp {e_pp:+.3f} e_tg {e_tg:+.3f}   "
                  f"{'both within 0.15' if ok else 'above 0.15'}")
            out[f"{arm}/{tag}"] = {"pp_pred": k["pp_tps"], "tg_pred": k["tg_tps"], "pp_meas": m["pp"], "tg_meas": m["tg"],
                                   "e_pp": e_pp, "e_tg": e_tg, "within": ok}
    with open(os.path.join(res, "errors.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("predictions", "results"):
        sys.exit(__doc__)
    (predictions if sys.argv[1] == "predictions" else results)(sys.argv[2])
