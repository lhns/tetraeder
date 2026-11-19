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


@pytest.mark.parametrize("session,box", [("s2g-20261118", "2x"), ("s3-20261119", "4x"), ("s3b-20261119", "4x")])
def test_version_2_on_the_session_graph_gives_its_predictions(session, box):
    """Version 2, pipelined prefill (ADR 0077) included, on the session's calib.json gives predict-*.json."""
    d = ROOT / "bench" / "results" / session / "phase5"
    if not (d / "calib.json").is_file():
        pytest.skip(f"{session} not committed")
    hw = cr.load_json(d / "calib.json")["hw"]
    for r in cr.predict_session(hw, d, 2):
        c = next(c for c in cr.load_json(d / f"predict-{r['model']}-{box}.json")["candidates"] if c["id"] == r["id"])
        assert r["pp"] == pytest.approx(c["pp_tps"], rel=1e-12), r
        assert r["tg"] == pytest.approx(c["tg_tps"], rel=1e-12), r


def test_the_allreduce_pays_its_in_step_latency_in_version_3():
    """ADR 0078: byte part of the curve plus a + 2 (p - 1) b; the share is not used; without b every group pays a."""
    pts = {2: [[4096.0, 4e-5], [1e6, 2e-4]], 4: [[4096.0, 3e-5], [1e6, 3e-4]]}
    ids = lambda n: [f"gpu{i}" for i in range(n)]
    for b in (None, 1e-6):
        rc = {"allreduce_latency_share": 0.25, "allreduce_latency_s": 1.2e-5}
        if b:
            rc["allreduce_step_s"] = b
        hw = {"devices": [], "links": [], "range_costs": rc,
              "collectives": [{"op": "all_reduce", "ranks": ids(n), "points": p} for n, p in pts.items()]}
        for n in (2, 4):
            c = {"ar": ids(n), "bpt": 1000.0}
            lat = 1.2e-5 + 2 * (n - 1) * (b or 0.0)
            assert cr.coll_time(hw, c, 1.0, 3)[1] == pytest.approx(lat)
            t = cr.curve_time(pts[n], 512e3)
            assert cr.coll_time(hw, c, 512.0, 3)[1] == pytest.approx(t - pts[n][0][1] + lat)


def test_routing_reaches_expert_parallel_only():
    """ADR 0070's routing input: uniform tables (and --skew 0.25) are today's uniform prices exactly; skew moves ep-4
    and nothing else; the anchors on a 4-GPU box find tp-4 in the session's own plans."""
    d = ROOT / "bench" / "results" / "s3b-20261119" / "phase5"
    if not (d / "calib.json").is_file():
        pytest.skip("s3b not committed")
    hw = cr.load_json(d / "calib.json")["hw"]
    m = "qwen3-30b-a3b"
    layout = cr.load_json(d / "layouts" / f"{m}.json")
    pl = cr.plans(d, m)
    e, n = layout["n_expert"], layout["n_layer"]
    uniform = cr.routing_tables({"prefill": [[16] * e] * n, "decode": [[1] * e] * n})
    assert all(x == 1.0 for row in uniform["prefill"] for x in row)
    assert cr.synthetic_routing(layout, 0.25) == uniform

    def rates(routing):
        return {cid: cr.predict(hw, cr.features(layout, plan, None, 3, routing), cr.PROMPT, cr.GEN, 3)
                for cid, plan in pl.items()}

    base, same, skew = rates(None), rates(uniform), rates(cr.synthetic_routing(layout, 0.68))
    for cid in pl:
        assert (same[cid]["pp"], same[cid]["tg"]) == (base[cid]["pp"], base[cid]["tg"]), cid
        if cid.startswith("ep-"):
            assert skew[cid]["pp"] < base[cid]["pp"] and skew[cid]["tg"] < base[cid]["tg"]
        else:
            assert (skew[cid]["pp"], skew[cid]["tg"]) == (base[cid]["pp"], base[cid]["tg"]), cid
    with pytest.raises(ValueError):
        cr.routing_tables({"prefill": [[0] * e], "decode": [[1] * e]})


def test_anchor_plans_fall_back_to_the_session_plans(tmp_path):
    """plans-anchor/ holds single and tp-2 (2-GPU graph); a tp-4 anchor uses the session's plan."""
    cands = {"model": "x", "gpus": 4, "pruned": [],
             "candidates": [{"id": "single", "plan": {"tag": "main"}}, {"id": "tp-4", "plan": {"tag": "main"}}]}
    anchor = {"model": "x", "gpus": 2, "pruned": [],
              "candidates": [{"id": "single", "plan": {"tag": "anchor"}}, {"id": "tp-2", "plan": {"tag": "anchor"}}]}
    (tmp_path / "plans-anchor").mkdir()
    (tmp_path / "candidates-x.json").write_text(json.dumps(cands))
    (tmp_path / "plans-anchor" / "candidates-x.json").write_text(json.dumps(anchor))
    pl = cr.anchor_plans(tmp_path, "x")
    assert {k: v["tag"] for k, v in pl.items()} == {"single": "anchor", "tp-4": "main", "tp-2": "anchor"}


def test_routed_slot_bytes_of_the_30b():
    """ADR 0080: k = 8, n_ff_exp 768, n_embd 2048 give 690944 bytes per token and MoE layer on a whole-expert rank;
    a tp-2 rank halves only the n_ff_exp part; versions before 4 price none."""
    layout = {"n_expert_used": 8, "n_ff_exp": 768, "n_embd": 2048}
    assert cr.slot_bytes(layout, 1.0, 4) == 690944.0
    assert cr.slot_bytes(layout, 0.5, 4) == 690944.0 - 8 * 25.125 * 384
    assert cr.slot_bytes(layout, 1.0, 3) == 0.0


@pytest.mark.parametrize("path", ["s2f-20261118/costmodel-v4/fit-s2f.json",
                                  "s2f-20261118/costmodel-v4/fit-s2e-eval-s2f.json",
                                  "s3c-20261119/costmodel-v4/fit-s3c.json"])
def test_version_4_rows_follow_from_the_committed_graph(path):
    """The committed version-4 rows (s3c with its routing record) are the replica's predictions on the graph beside
    them, which carries expert_slot_traffic for Rust."""
    p = ROOT / "bench" / "results" / path
    if not p.is_file():
        pytest.skip(f"{path} not committed")
    v = json.loads(p.read_text())
    assert v["version"] == 4 and v["hw"]["range_costs"]["expert_slot_traffic"] is True
    ev = ROOT / v["eval"]
    saved = dict(cr.ROUTING)
    try:
        for m in cr.models(ev):
            f = ev / f"routing-{m}.json"
            if f.exists():
                cr.ROUTING[m] = cr.routing_tables(cr.load_json(f))
        rows = cr.predict_session(v["hw"], ev, 4)
    finally:
        cr.ROUTING.clear()
        cr.ROUTING.update(saved)
    for r, c in zip(rows, v["rows"]):
        assert (r["model"], r["id"]) == (c["model"], c["id"])
        assert r["pp"] == c["pp"] and r["tg"] == c["tg"]
