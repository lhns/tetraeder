"""scripts/costmodel_replica.py: the Phase 5 cost model offline (ADR 0070)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("costmodel_replica", ROOT / "scripts" / "costmodel_replica.py")
cr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cr)

S2F = ROOT / "bench" / "results" / "s2f-20261118" / "phase5"
V2 = ROOT / "bench" / "results" / "s2f-20261118" / "costmodel-v2"

needs_s2f = pytest.mark.skipif(not (S2F / "calib.json").is_file(), reason="S2f session files not committed")


@needs_s2f
def test_version_1_matches_the_committed_predictions():
    """MODEL_VERSION 1 on S2f's own calib.json gives its predict-*.json to the last bits."""
    hw = cr.load_json(S2F / "calib.json")["hw"]
    rows = cr.predict_session(hw, S2F, 1)
    assert len(rows) == 7
    for r in rows:
        committed = cr.load_json(S2F / f"predict-{r['model']}-2x.json")
        c = next(c for c in committed["candidates"] if c["id"] == r["id"])
        assert r["pp"] == pytest.approx(c["pp_tps"], rel=1e-12), r
        assert r["tg"] == pytest.approx(c["tg_tps"], rel=1e-12), r
        assert r["t"] == pytest.approx(c["t_s"], rel=1e-12), r


@needs_s2f
@pytest.mark.parametrize("name", ["fit-s2f.json", "fit-s2e-eval-s2f.json"])
def test_version_2_rows_follow_from_the_committed_graph(name):
    """The committed version-2 rows are the replica's predictions on the graph stored beside them."""
    v = json.loads((V2 / name).read_text())
    assert v["version"] == 2 and v["ar_form"] == "offset"
    rows = cr.predict_session(v["hw"], ROOT / v["eval"], 2)
    for r, c in zip(rows, v["rows"]):
        assert (r["model"], r["id"]) == (c["model"], c["id"])
        assert r["pp"] == c["pp"] and r["tg"] == c["tg"]


def test_the_allreduce_pays_its_latency_share():
    hw = {"devices": [], "links": [], "range_costs": {"allreduce_latency_share": 0.25},
          "collectives": [{"op": "all_reduce", "ranks": ["gpu0", "gpu1"], "points": [[4096.0, 3e-5], [1e6, 1e-4]]}]}
    c = {"ar": ["gpu0", "gpu1"], "bpt": 8192.0}
    # 512 * 8192 bytes on the curve, minus 0.75 of the zero-byte time
    t = cr.curve_time(hw["collectives"][0]["points"], 512 * 8192.0)
    assert cr.coll_time(hw, c, 512.0, 2)[1] == pytest.approx(t - 0.75 * 3e-5)
    assert cr.coll_time(hw, c, 512.0, 1)[1] == pytest.approx(t)


def test_expected_max_local_matches_the_rust_bounds():
    a = cr.expected_max_local([1.0] * 128, 8, 2, cr.EP_STEPS, cr.EP_SEED)
    assert 4.0 <= a < 8.0
    skew = [1.0] * 64 + [0.0] * 64
    assert cr.expected_max_local(skew, 8, 2, 50, cr.EP_SEED) == 8.0
