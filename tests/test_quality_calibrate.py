"""Calibration job additions (ADR 0048 update): knapsack-shaped checks and the per-format scale.

Pure functions of a profile; no llama.cpp or GPU. Runs on the committed Qwen3-30B-A3B profile.
"""

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "quality"))
calibrate = importlib.import_module("calibrate")
compare_mix = importlib.import_module("compare_mix")
qlib = importlib.import_module("qlib")

PROFILE = ROOT / "bench" / "quality" / "qwen3-30b-bf16.quality.json"


def load():
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def group_bytes(prof, mix):
    base = prof["base"]
    in_base = sum(next(e["bytes"] for e in g["entries"] if e["format"] == base["format"]) for g in prof["groups"])
    chosen = sum(next(e["bytes"] for e in g["entries"] if e["format"] == mix[g["id"]]) for g in prof["groups"])
    return base["bytes"] - in_base + chosen


def test_knapsack_mixes_are_mixed_distinct_and_ordered_by_budget():
    prof = load()
    mixes = calibrate.knapsack_check_mixes(prof)
    assert list(mixes) == ["knapsack-20", "knapsack-35", "knapsack-50"]
    assert len({json.dumps(m, sort_keys=True) for m in mixes.values()}) == 3
    ids = {g["id"] for g in prof["groups"]}
    sizes = []
    for mix in mixes.values():
        assert set(mix) == ids
        # Unlike a uniform check, a knapsack mix spreads over several formats.
        assert len(set(mix.values())) >= 3
        sizes.append(group_bytes(prof, mix))
    assert sizes == sorted(sizes) and len(set(sizes)) == 3
    assert sizes[-1] <= prof["base"]["bytes"]
    # More bytes, less estimated KL.
    est = [qlib.estimate(prof, m) for m in mixes.values()]
    assert est == sorted(est, reverse=True)


def test_knapsack_mixes_are_the_reference_greedy_at_their_budget():
    prof = load()
    base = prof["base"]
    lo = group_bytes(prof, {g["id"]: min(g["entries"], key=lambda e: e["bytes"])["format"] for g in prof["groups"]})
    mixes = calibrate.knapsack_check_mixes(prof, (0.35,))
    want, _ = compare_mix.greedy_mix(prof, int(lo + 0.35 * (base["bytes"] - lo)))
    assert mixes == {"knapsack-35": want}


def test_knapsack_mixes_drop_repeats():
    prof = load()
    assert list(calibrate.knapsack_check_mixes(prof, (0.35, 0.35))) == ["knapsack-35"]
    assert calibrate.knapsack_check_mixes(prof, ()) == {}


def test_format_scale_caps_only_the_underestimated_format():
    prof = load()
    scale = calibrate.fit_format_scale(prof)
    assert list(scale) == ["Q2_K"]
    assert 1.3 < scale["Q2_K"] < 1.4
    # The committed profile carries what the job would write.
    assert prof["format_scale"] == scale


def test_format_scale_closes_the_uniform_check():
    prof = load()
    scale = calibrate.fit_format_scale(prof)
    chk = next(c for c in prof["checks"] if c["name"] == "uniform-Q2_K")
    base_kl = prof["base"]["kl"]
    est = qlib.estimate(prof, chk["formats"])
    assert (chk["measured"] - est) / chk["measured"] > 0.2
    scaled = base_kl + scale["Q2_K"] * (est - base_kl)
    assert abs(scaled - chk["measured"]) / chk["measured"] < 0.001


def test_format_scale_ignores_non_uniform_and_overestimated_checks():
    prof = load()
    prof["checks"] = [c for c in prof["checks"] if c["name"] != "uniform-Q2_K"]
    assert calibrate.fit_format_scale(prof) == {}
    # A random mix that is underestimated is not a uniform check, so it sets no scale.
    prof["checks"].append({"name": "random-0", "formats": prof["checks"][0]["formats"], "estimate": 0.1,
                           "measured": 10.0})
    assert calibrate.fit_format_scale(prof) == {}
