"""scripts/costmodel_lines.py: per-shape step-time lines from truncated anchors (ADR 0076)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("costmodel_lines", ROOT / "scripts" / "costmodel_lines.py")
cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cl)

S2F = ROOT / "bench" / "results" / "s2f-20261118" / "phase5"


def bench_rows(pp, tg, **extra):
    return [{"n_prompt": 2048, "n_gen": 0, "avg_ts": pp, **extra}, {"n_prompt": 0, "n_gen": 128, "avg_ts": tg, **extra}]


def line_rates(lim, step_pp=0.01, layer_pp=0.008, step_tg=0.06, layer_tg=0.0135):
    return 2048 / (step_pp + lim * layer_pp), 128 / (step_tg + lim * layer_tg)


def write_session(d: Path, lims=(12, 24, 36)):
    (d / "layouts").mkdir(parents=True)
    (d / "layouts" / "m.json").write_text(json.dumps({"n_layer": 48}))
    for k in (1, 2):
        rows = []
        for lim in lims:
            rows += bench_rows(*line_rates(lim), plan_id="single", layer_limit=lim)
        (d / f"anchors-m-r{k}.json").write_text(json.dumps(rows))
    (d / "bench-m-single-r1.json").write_text(json.dumps(bench_rows(*line_rates(48))))
    (d / "bench-m-single-aa.json").write_text(json.dumps(bench_rows(1.0, 1.0)))  # ignored
    (d / "bench-m-layer-2-r1.json").write_text(json.dumps(bench_rows(5000.0, 200.0)))
    (d / "predict-m-2x.json").write_text(json.dumps({"candidates": [{"id": "single", "pp_tps": 1.0, "tg_tps": 2.0}]}))


def test_fit_line_is_exact_on_a_line_and_needs_two_layer_counts():
    a, b = cl.fit_line([(12, 0.1 + 12 * 0.01), (24, 0.1 + 24 * 0.01), (36, 0.1 + 36 * 0.01)])
    assert a == pytest.approx(0.1) and b == pytest.approx(0.01)
    with pytest.raises(ValueError):
        cl.fit_line([(12, 1.0), (12, 2.0)])


def test_full_model_prediction_and_unanchored_shapes(tmp_path):
    write_session(tmp_path)
    rows = {(r["shape"], r["rate"]): r for r in cl.evaluate(tmp_path, tmp_path)}
    pp, tg = line_rates(48)
    assert rows[("single", "pp")]["predicted"] == pytest.approx(pp)
    assert rows[("single", "tg")]["predicted"] == pytest.approx(tg)
    assert rows[("single", "pp")]["within"] and rows[("single", "pp")]["points"] == 3
    assert rows[("single", "tg")]["committed"] == 2.0  # the session's own prediction, for comparison
    # the A/A run is not a round
    assert rows[("single", "pp")]["measured"] == pytest.approx(pp)
    # no anchors of its own shape: never predicted from another shape
    assert rows[("layer-2", "pp")]["predicted"] is None and rows[("layer-2", "pp")]["reason"] == "unanchored"


def test_holdout_predicts_the_largest_truncation(tmp_path):
    write_session(tmp_path)
    rows = {(r["shape"], r["rate"]): r for r in cl.evaluate(tmp_path, tmp_path, holdout=True)}
    assert rows[("single", "pp")]["at"] == "L36"
    assert rows[("single", "pp")]["predicted"] == pytest.approx(line_rates(36)[0])
    assert rows[("single", "pp")]["points"] == 2


def test_holdout_needs_three_truncations(tmp_path):
    write_session(tmp_path, lims=(12, 24))
    rows = cl.evaluate(tmp_path, tmp_path, holdout=True)
    assert all(r["predicted"] is None and r["reason"] == "needs 3 truncations" for r in rows)


@pytest.mark.skipif(not S2F.is_dir(), reason="committed S2f results not present")
def test_s2f_lines_hold_for_anchored_shapes():
    rows = cl.evaluate(S2F, S2F)
    anchored = [r for r in rows if r["predicted"] is not None]
    assert {(r["model"], r["shape"]) for r in anchored} == {
        ("qwen3-30b-a3b", "single"), ("qwen3-30b-a3b", "tp-2"), ("qwen3-32b", "single"), ("qwen3-32b", "tp-2")}
    # S2f: every anchored rate within 3% (the committed analytic prediction missed by up to 61%)
    assert max(abs(r["err"]) for r in anchored) < 0.03
    assert all(r["reason"] == "unanchored" for r in rows if r["shape"] in ("layer-2", "ep-2"))
