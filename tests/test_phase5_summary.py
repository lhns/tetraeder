"""scripts/phase5_summary.py over synthetic result directories: the rules of ADR 0068 ("Verdict", "Settled")."""

from __future__ import annotations

import importlib.util
import json
import os
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("phase5_summary", ROOT / "scripts" / "phase5_summary.py")
p5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p5)

MODEL = "qwen3-32b"
# measured (pp_tps, tg_tps) -> T: single 6.83 s, layer-2 7.04 s, tp-2 4.76 s (best; W = {tp-2} at band 0.08)
MEAS = {"single": (800.0, 30.0), "layer-2": (780.0, 29.0), "tp-2": (1200.0, 42.0)}
BASE = {"layer": (700.0, 28.0), "row": (600.0, 25.0), "tensor": (1000.0, 36.0)}  # best T 5.6 s
OK_THROTTLE = "before: 0x1\nbefore: 0x1\nafter: 0x1\nafter: 0x1\n"


def rows(pp, tg):
    return [{"n_prompt": 2048, "n_gen": 0, "avg_ts": pp, "plan_id": "x"},
            {"n_prompt": 0, "n_gen": 128, "avg_ts": tg, "plan_id": "x"}]


def write_run(d, kind, run, pp, tg, rc=0, throttle=OK_THROTTLE, cotenant=None, result=True):
    if result:
        (d / f"{kind}-{run}.json").write_text(json.dumps(rows(pp, tg)))
    (d / f"run-{run}.rc").write_text(f"{rc}\n")
    (d / f"gpu-{run}.txt").write_text(throttle)
    if cotenant is not None:
        (d / f"cotenant-{run}.json").write_text(json.dumps({"cotenant_ok": cotenant}))


def pred_of(meas, pred_err=None):
    """Prediction rows: the measured rates, scaled per candidate by (1 + err) on pp, tg unchanged."""
    pred_err = pred_err or {}
    rows_ = []
    for cid, (pp, tg) in meas.items():
        pp_p, tg_p = pp * (1 + pred_err.get(cid, 0.0)), tg
        rows_.append({"id": cid, "pp_tps": pp_p, "tg_tps": tg_p, "t_s": 2048 / pp_p + 128 / tg_p, "binding": "x"})
    return rows_


def make(d, meas=MEAS, base=BASE, pred=None, pruned=(("ep-2", "dense model has no experts"),), rounds=(1, 2, 3),
         aa_scale=1.0, gates=None, sm_fail=()):
    """A complete session: enumeration, prediction (older than every run), rounds, baselines, A/A."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "env.json").write_text(json.dumps({"gpu_count": 2, "gpu_names": ["NVIDIA GeForce RTX 3090"] * 2,
                                            "p2p": "none", "comm_impl": "internal"}))
    (d / "gates.json").write_text(json.dumps(gates or {"gpu_parity": "ok", "comm_probe": "ok"}))
    ids = list(meas)
    (d / f"candidates-{MODEL}.json").write_text(json.dumps({
        "model": MODEL, "gpus": 2, "candidates": [{"id": i} for i in ids],
        "pruned": [{"id": i, "reason": r} for i, r in pruned]}))
    pf = d / f"predict-{MODEL}-2x.json"
    pf.write_text(json.dumps({"model": MODEL, "candidates": pred if pred is not None else pred_of(meas)}))
    os.utime(pf, (1_000_000, 1_000_000))
    for sm, (pp, tg) in base.items():
        (d / f"prepass-{MODEL}-llama-bench-{sm}.json").write_text(json.dumps({"ok": True, "rc": 0, "comm": "internal" if sm == "tensor" else None}))
    for k in rounds:
        for cid, (pp, tg) in meas.items():
            write_run(d, "bench", f"{MODEL}-{cid}-r{k}", pp, tg)
        for sm, (pp, tg) in base.items():
            write_run(d, "baseline", f"{MODEL}-llama-bench-{sm}-r{k}", pp, tg)
    first = ids[0]
    pp, tg = meas[first]
    write_run(d, "bench", f"{MODEL}-{first}-aa", pp * aa_scale, tg * aa_scale)
    for p in d.glob("*.json"):
        if p != pf:
            os.utime(p, (2_000_000, 2_000_000))
    for p in list(d.glob("*.rc")) + list(d.glob("gpu-*.txt")):
        os.utime(p, (2_000_000, 2_000_000))
    return d


def verdict(d, tmp_path):
    noise = tmp_path / "noise"
    noise.mkdir(exist_ok=True)
    v, notes = p5.verdict(d, noise, ROOT / "bench" / "spec" / "phase5.toml")
    return v, notes, v["cases"][0]


def test_all_pass(tmp_path):
    v, _, c = verdict(make(tmp_path / "d"), tmp_path)
    assert c["pass"] is True and c["clean"] is True and c["reasons"] == []
    assert c["case"] == "D2" and c["gpu_class"] == "2x-rtx-3090" and c["box_gpus"] == 2
    assert c["pick"] == "tp-2" and c["winner_set"] == ["tp-2"] and c["pick_ok"] is True
    assert c["prediction_ok"] is True and c["baseline_ok"] is True
    assert c["best_baseline"]["sm"] == "tensor" and c["baseline_ratio"] == pytest.approx(5.6 / 4.76, rel=1e-2)
    assert c["aa"]["t_start"] == pytest.approx(c["aa"]["t_end"])
    assert v["band"] == 0.08 and v["band_source"].startswith("provisional cuda-default")
    assert v["predict_sha256"][MODEL] == p5.sha256_file(tmp_path / "d" / f"predict-{MODEL}-2x.json")
    # one session is never the whole exit test
    assert v["phase5_pass"] is False
    assert {b["sm"] for b in c["baselines"]} == {"layer", "row", "tensor"}
    cand = {x["id"]: x for x in c["candidates"]}
    assert cand["tp-2"]["measured"]["rounds_clean"] == 3 and cand["tp-2"]["shape"] == "tp"
    assert cand["tp-2"]["err"] == {"pp": pytest.approx(0.0), "tg": pytest.approx(0.0)} and cand["tp-2"]["within"] is True


def test_schema_fields_of_the_adr(tmp_path):
    v, _, c = verdict(make(tmp_path / "d"), tmp_path)
    assert set(v) == {"band", "band_source", "predict_sha256", "cases", "phase5_pass", "throttle"}
    assert set(c) == {"case", "model", "box_gpus", "gpu_class", "p2p", "comm_impl", "candidates", "pick", "winner_set",
                      "pick_ok", "prediction_ok", "baselines", "best_baseline", "baseline_ratio", "baseline_ok", "aa",
                      "clean", "reasons", "pass"}
    assert set(c["candidates"][0]) == {"id", "shape", "pruned", "reason", "predicted", "measured", "err", "within"}
    assert set(c["candidates"][-1]["measured"]) == {"pp_tps", "tg_tps", "t_s", "rounds_clean"}
    assert set(c["baselines"][0]) == {"engine", "sm", "t_s", "comm"} and set(c["aa"]) == {"t_start", "t_end"}


def test_pick_outside_the_winner_set_fails(tmp_path):
    # the planner thinks layer-2 is fastest; tp-2 measures 1.5x faster
    pred = pred_of(MEAS)
    for r in pred:
        if r["id"] == "layer-2":
            r["t_s"] = 3.0
    _, _, c = verdict(make(tmp_path / "d", pred=pred), tmp_path)
    assert c["pick"] == "layer-2" and c["pick_ok"] is False and c["pass"] is False


def test_pick_inside_the_winner_set_when_two_are_within_the_band(tmp_path):
    meas = {**MEAS, "layer-2": (1170.0, 41.0)}  # T 4.88 s, within 8% of tp-2's 4.76
    pred = pred_of(meas)
    pred[1]["t_s"] = 1.0
    _, _, c = verdict(make(tmp_path / "d", meas=meas, pred=pred), tmp_path)
    assert set(c["winner_set"]) == {"layer-2", "tp-2"} and c["pick"] == "layer-2" and c["pick_ok"] is True


def test_one_candidate_at_16_percent_fails_the_case(tmp_path):
    _, _, c = verdict(make(tmp_path / "d", pred=pred_of(MEAS, {"single": 0.16})), tmp_path)
    cand = {x["id"]: x for x in c["candidates"]}
    assert cand["single"]["within"] is False and cand["tp-2"]["within"] is True
    assert c["prediction_ok"] is False and c["pick_ok"] is True and c["pass"] is False


def test_14_percent_passes(tmp_path):
    _, _, c = verdict(make(tmp_path / "d", pred=pred_of(MEAS, {"single": 0.14, "layer-2": -0.14})), tmp_path)
    assert c["prediction_ok"] is True and c["pass"] is True


def test_baseline_faster_beyond_the_band_fails(tmp_path):
    _, _, c = verdict(make(tmp_path / "d", base={**BASE, "tensor": (1500.0, 50.0)}), tmp_path)  # 3.93 s against 4.76 s
    assert c["baseline_ok"] is False and c["baseline_ratio"] < 0.92 and c["pass"] is False
    assert c["pick_ok"] is True and c["prediction_ok"] is True


def test_baseline_within_the_band_passes(tmp_path):
    # 4.52 s: the pick is 5% slower than the best baseline, inside the 8% band
    _, _, c = verdict(make(tmp_path / "d", base={**BASE, "tensor": (1275.0, 44.0)}), tmp_path)
    assert 0.92 < c["baseline_ratio"] < 1 and c["baseline_ok"] is True and c["pass"] is True


def test_ik_and_vllm_are_reported_not_gated(tmp_path):
    d = make(tmp_path / "d")
    (d / f"prepass-{MODEL}-ik-llama-bench-layer.json").write_text(json.dumps({"ok": True}))
    for k in (1, 2, 3):
        write_run(d, "baseline", f"{MODEL}-ik-llama-bench-layer-r{k}", 5000.0, 200.0)  # far faster than the pick
    (d / f"baseline-{MODEL}-vllm-tp-r1.json").write_text(json.dumps({"summary": {"ttft_median_s": 0.5, "decode_tps_median": 100.0}}))
    write_run(d, "baseline", f"{MODEL}-vllm-tp-r1", 0, 0, result=False)
    _, _, c = verdict(d, tmp_path)
    by = {(b["engine"], b["sm"]): b for b in c["baselines"]}
    assert by[("ik-llama-bench", "layer")]["t_s"] < 1.1 and by[("vllm", "tp")]["t_s"] == pytest.approx(1.78)
    assert c["best_baseline"]["engine"] == "llama-bench" and c["baseline_ok"] is True and c["pass"] is True


def test_unclean_run_rerun_in_a_later_round_passes(tmp_path):
    d = make(tmp_path / "d", rounds=(1, 2, 3))
    pp, tg = MEAS["tp-2"]
    write_run(d, "bench", f"{MODEL}-tp-2-r2", pp, tg, throttle="before: 0x1\nafter: 0x8\n")  # hw slowdown
    assert [c for c in p5.need_rerun(d, MODEL)] == []  # tp-2 still has rounds 1 and 3
    write_run(d, "bench", f"{MODEL}-tp-2-r3", pp, tg, cotenant=False)  # now only round 1 is clean
    assert p5.need_rerun(d, MODEL) == [("tet", "tp-2", "")]
    write_run(d, "bench", f"{MODEL}-tp-2-r4", pp, tg)
    v, notes, c = verdict(d, tmp_path)
    assert p5.need_rerun(d, MODEL) == []
    assert c["pass"] is True and c["clean"] is True
    assert {x["id"]: x for x in c["candidates"]}["tp-2"]["measured"]["rounds_clean"] == 2
    runs = {u["run"] for u in notes["unclean_runs"]}
    assert {f"{MODEL}-tp-2-r2", f"{MODEL}-tp-2-r3"} <= runs  # recorded, not dropped silently


def test_candidate_with_one_clean_round_has_no_verdict(tmp_path):
    d = make(tmp_path / "d")
    pp, tg = MEAS["tp-2"]
    write_run(d, "bench", f"{MODEL}-tp-2-r2", pp, tg, throttle="before: 0x1\nafter: 0x8\n")
    write_run(d, "bench", f"{MODEL}-tp-2-r3", pp, tg, rc=137)
    _, _, c = verdict(d, tmp_path)
    assert c["clean"] is False and c["pass"] is None
    assert any("tp-2" in r and "clean round" in r for r in c["reasons"])


def test_gated_baseline_with_one_clean_round_has_no_verdict(tmp_path):
    d = make(tmp_path / "d")
    for k in (2, 3):
        write_run(d, "baseline", f"{MODEL}-llama-bench-tensor-r{k}", 1000.0, 36.0, throttle="before: 0x1\nafter: 0x20\n")
    _, _, c = verdict(d, tmp_path)
    assert c["pass"] is None and c["baseline_ok"] is None and any("tensor" in r for r in c["reasons"])


def test_aa_outside_the_band_gives_no_verdict(tmp_path):
    _, _, c = verdict(make(tmp_path / "d", aa_scale=0.85), tmp_path)  # 15% slower at the end of the session
    assert c["clean"] is False and c["pass"] is None
    assert any(r.startswith("A/A") and "band" in r for r in c["reasons"])
    assert c["aa"]["t_end"] > c["aa"]["t_start"]


def test_aa_just_inside_the_band_passes(tmp_path):
    _, _, c = verdict(make(tmp_path / "d", aa_scale=0.95), tmp_path)
    assert c["clean"] is True and c["pass"] is True


def test_missing_aa_gives_no_verdict(tmp_path):
    d = make(tmp_path / "d")
    for p in d.glob(f"*{MODEL}-single-aa*"):
        p.unlink()
    _, _, c = verdict(d, tmp_path)
    assert c["pass"] is None and c["aa"]["t_end"] is None


def test_prediction_newer_than_a_candidate_run_is_invalid(tmp_path):
    d = make(tmp_path / "d")
    os.utime(d / f"predict-{MODEL}-2x.json", (3_000_000, 3_000_000))
    _, _, c = verdict(d, tmp_path)
    assert c["pass"] is None and any(r.startswith("invalid") for r in c["reasons"])


def test_timeline_order_and_prediction_hash(tmp_path):
    d = make(tmp_path / "d")
    pf = d / f"predict-{MODEL}-2x.json"
    sha = p5.sha256_file(pf)
    ev = lambda **k: json.dumps({"model": MODEL, **k})  # noqa: E731
    (d / "timeline.jsonl").write_text(
        ev(t=100, event="predict", sha256=sha) + "\n" + ev(t=200, event="run", kind="candidate", run="x") + "\n")
    os.utime(pf, (3_000_000, 3_000_000))  # mtime would say invalid; the timeline is what counts after a fetch
    assert verdict(d, tmp_path)[2]["pass"] is True
    (d / "timeline.jsonl").write_text(
        ev(t=300, event="predict", sha256=sha) + "\n" + ev(t=200, event="run", kind="candidate", run="x") + "\n")
    c = verdict(d, tmp_path)[2]
    assert c["pass"] is None and "after the first candidate run" in " ".join(c["reasons"])
    (d / "timeline.jsonl").write_text(ev(t=100, event="predict", sha256="0" * 64) + "\n")
    assert "changed after" in " ".join(verdict(d, tmp_path)[2]["reasons"])


def test_pruned_candidates_are_listed_and_not_counted(tmp_path):
    _, _, c = verdict(make(tmp_path / "d", pruned=(("ep-2", "dense model has no experts"), ("tp2-pp2", "needs 4 GPUs"))), tmp_path)
    pr = [x for x in c["candidates"] if x["pruned"]]
    assert [(x["id"], x["shape"], x["reason"]) for x in pr] == [
        ("ep-2", "ep", "dense model has no experts"), ("tp2-pp2", "tp-pp", "needs 4 GPUs")]
    assert all(x["within"] is None and x["measured"] is None for x in pr)
    assert c["pass"] is True and "ep-2" not in c["winner_set"]


def test_missing_data_is_null_not_true(tmp_path):
    d = make(tmp_path / "d")
    for p in d.glob(f"*{MODEL}-tp-2-*"):
        p.unlink()
    _, _, c = verdict(d, tmp_path)
    cand = {x["id"]: x for x in c["candidates"]}
    assert cand["tp-2"]["measured"]["t_s"] is None and cand["tp-2"]["within"] is None
    assert c["pick_ok"] is None and c["prediction_ok"] is None and c["pass"] is None
    # no baselines at all: the gate is null too
    d2 = make(tmp_path / "d2")
    for p in list(d2.glob("baseline-*")) + list(d2.glob("prepass-*")):
        p.unlink()
    c2 = verdict(d2, tmp_path)[2]
    assert c2["baseline_ok"] is None and c2["pass"] is None and c2["best_baseline"] is None


def test_no_prediction_file_is_null(tmp_path):
    d = make(tmp_path / "d")
    (d / f"predict-{MODEL}-2x.json").unlink()
    c = verdict(d, tmp_path)[2]
    assert c["pass"] is None and any("predict" in r for r in c["reasons"])


def test_candidate_that_fails_on_the_box_fails_the_case(tmp_path):
    d = make(tmp_path / "d")
    for k in (1, 2, 3):
        for p in d.glob(f"bench-{MODEL}-layer-2-r{k}.json"):
            p.unlink()
        (d / f"run-{MODEL}-layer-2-r{k}.rc").write_text("1\n")  # out of memory
    _, _, c = verdict(d, tmp_path)
    cand = {x["id"]: x for x in c["candidates"]}
    assert cand["layer-2"]["within"] is False and c["prediction_ok"] is False and c["pass"] is False
    assert c["pick_ok"] is True


def test_failed_gate_gives_no_verdict(tmp_path):
    _, _, c = verdict(make(tmp_path / "d", gates={"gpu_parity": "failed", "comm_probe": "ok"}), tmp_path)
    assert c["pass"] is None and "gate gpu_parity: failed" in c["reasons"]
    _, _, c = verdict(make(tmp_path / "e", gates={"gpu_parity": "skipped", "comm_probe": "skipped"}), tmp_path)
    assert c["pass"] is None  # a CPU rehearsal is never a verdict


def test_band_from_noise_file_with_three_sessions(tmp_path):
    d = make(tmp_path / "d")
    noise = tmp_path / "noise"
    noise.mkdir()
    (noise / "2x-rtx-3090.json").write_text(json.dumps({"n": 3, "band": 0.03}))
    v = p5.verdict(d, noise, ROOT / "bench" / "spec" / "phase5.toml")[0]
    assert v["band"] == 0.03 and v["band_source"] == "measured 2x-rtx-3090 n=3"
    (noise / "2x-rtx-3090.json").write_text(json.dumps({"n": 2, "band": 0.03}))
    v = p5.verdict(d, noise, ROOT / "bench" / "spec" / "phase5.toml")[0]
    assert v["band"] == 0.08 and "n=2 < 3" in v["band_source"]


# ---- rounds -------------------------------------------------------------------------------------------------


def configs_of(lines):
    return [ln.replace("base llama-bench ", "b:") .replace("tet ", "t:") for ln in lines]


def test_round_order_rotates_and_round_2_reverses(tmp_path):
    d = make(tmp_path / "d")
    base = [("tet", "single", ""), ("tet", "layer-2", ""), ("tet", "tp-2", ""), ("base", "llama-bench", "layer"),
            ("base", "llama-bench", "row"), ("base", "llama-bench", "tensor")]
    assert p5.configs(d, MODEL) == base
    assert p5.round_order(d, MODEL, 1) == base[1:] + base[:1]
    assert p5.round_order(d, MODEL, 2) == base[::-1]
    assert p5.round_order(d, MODEL, 3) == base[3:] + base[:3]
    assert p5.round_order(d, MODEL, 7) == base[1:] + base[:1]
    # two rounds never start on the same configuration
    assert p5.round_order(d, MODEL, 1)[0] != p5.round_order(d, MODEL, 2)[0]


def test_round_order_skips_a_candidate_that_failed_twice(tmp_path):
    d = make(tmp_path / "d", rounds=())
    for k in (1, 2):
        (d / f"run-{MODEL}-single-r{k}.rc").write_text("1\n")
    assert ("tet", "single", "") not in p5.round_order(d, MODEL, 3)


def test_cli_round_order_and_summary(tmp_path, capsys):
    d = make(tmp_path / "d")
    assert p5.main([str(d), "--round-order", MODEL, "1"]) == 0
    out = capsys.readouterr().out.split("\n")
    assert out[0] == "tet layer-2" and out[3] == "base llama-bench row" and out[-2] == "tet single"
    assert p5.main([str(d), "--need-rerun", MODEL]) == 0 and capsys.readouterr().out.strip() == ""
    assert p5.main([str(d), "--noise-dir", str(tmp_path / "none")]) == 0
    v = json.loads((d / "verdict.json").read_text())
    assert v["cases"][0]["pass"] is True and (d / "verdict-notes.json").exists()


# ---- all four cases -----------------------------------------------------------------------------------------


def case(name, pass_):
    return {"case": name, "model": "m", "box_gpus": 2, "pass": pass_}


def test_phase5_pass_needs_all_four_cases():
    assert p5.final_pass([case(c, True) for c in p5.CASES]) is True
    assert p5.final_pass([case(c, True) for c in p5.CASES[:3]]) is False  # partial stays false
    assert p5.final_pass([case("D2", True), case("M2", False), case("D4", True), case("M4", True)]) is False
    assert p5.final_pass([case("D2", True), case("M2", None), case("D4", True), case("M4", True)]) is None


def test_merge_sessions(tmp_path):
    v2, _, _ = verdict(make(tmp_path / "a"), tmp_path)
    v2["cases"].append({**v2["cases"][0], "case": "M2", "model": "qwen3-30b-a3b"})
    f = tmp_path / "v2.json"
    f.write_text(json.dumps(v2))
    g = tmp_path / "v4.json"
    g.write_text(json.dumps({"band": 0.08, "band_source": "x", "predict_sha256": {},
                             "cases": [{**v2["cases"][0], "case": "D4", "box_gpus": 4}, {**v2["cases"][0], "case": "M4", "box_gpus": 4}]}))
    m = p5.merge([f, g])
    assert [c["case"] for c in m["cases"]] == ["D2", "M2", "D4", "M4"] and m["phase5_pass"] is True
    assert m["predict_sha256"] == {f"{MODEL}-2x": v2["predict_sha256"][MODEL]}
    assert p5.merge([f])["phase5_pass"] is False


# ---- spec ---------------------------------------------------------------------------------------------------


def test_spec_matches_adr_and_phase0_band():
    s = tomllib.loads((ROOT / "bench" / "spec" / "phase5.toml").read_text(encoding="utf-8"))
    p0 = tomllib.loads((ROOT / "bench" / "spec" / "phase0.toml").read_text(encoding="utf-8"))
    band = lambda t: next(b["band"] for b in t["noise_band"] if b["class"] == "cuda-default")  # noqa: E731
    assert band(s) == band(p0)
    assert {e["id"] for e in s["engine"]} == {"tetraeder-exec", "llama-bench", "ik-llama-bench"}
    assert s["workload"]["prompt_len"] == 2048 and s["workload"]["gen_len"] == 128 and s["workload"]["ubatch"] == 512
    assert set(s["model"]) == {"qwen3-32b", "qwen3-30b-a3b"}
    pin = {k: v for k, v in (ln.split("=", 1) for ln in (ROOT / "vendor" / "PIN").read_text().split() if "=" in ln)}
    assert s["pins"]["llama_cpp"] == pin["baseline_commit"] and s["pins"]["ik_llama_cpp"] == pin["ik_commit"]


def test_resumed_prediction_logged_twice_stays_valid(tmp_path):
    d = make(tmp_path / "d")
    sha = p5.sha256_file(d / f"predict-{MODEL}-2x.json")
    ev = lambda **k: json.dumps({"model": MODEL, **k})  # noqa: E731
    (d / "timeline.jsonl").write_text("\n".join([
        ev(t=100, event="predict", sha256=sha), ev(t=200, event="run", kind="candidate", run="x"),
        ev(t=300, event="predict", sha256=sha)]) + "\n")  # an older box script logged it again on resume
    assert verdict(d, tmp_path)[2]["pass"] is True


def test_p2p_matrix_with_indented_rows():
    both = "        GPU0    GPU1\n GPU0   X       OK\n GPU1   OK      X\n"
    assert p5.p2p_class(both) == "all"
    assert p5.p2p_class(both.replace("OK", "NS")) == "none"
    four = "\tGPU0\tGPU1\tGPU2\tGPU3\n GPU0\tX\tOK\tNS\tNS\n GPU1\tOK\tX\tNS\tNS\n GPU2\tNS\tNS\tX\tOK\n GPU3\tNS\tNS\tOK\tX\n"
    assert p5.p2p_class(four) == "partial" and p5.p2p_class("") is None


def test_baseline_modes_gate(tmp_path):
    d = make(tmp_path / "d", gates={"gpu_parity": "ok", "comm_probe": "ok", "baseline_modes": "failed"})
    c = verdict(d, tmp_path)[2]
    assert c["pass"] is None and "gate baseline_modes: failed" in c["reasons"]
    ok = make(tmp_path / "e", gates={"gpu_parity": "ok", "comm_probe": "ok", "baseline_modes": "ok"})
    assert verdict(ok, tmp_path)[2]["pass"] is True


# ---- throttle by the session sampler (ADR 0068 addenda 2026-11-17, 2026-11-18) ----------------------------------------------

from datetime import datetime, timedelta  # noqa: E402

T0 = datetime(2026, 11, 17, 10, 0, 0)
BOARD, CAP, IDLE = 0x600, 0x4, 0x1


def sampler(d, shares=None, extra=None, gpus=(0, 1), windows=True, index_column=True):
    """gpu-samples.csv for the runs of d (anchors excluded): one 100 s slot per run, one line per GPU per second.
    shares(run, gpu) -> (share of 0x600, share of 0x4); extra(run, gpu) -> masks that replace the first samples."""
    shares = shares or (lambda run, g: (0.5, 0.2))
    lines = []
    runs = sorted(p.stem[len("run-"):] for p in d.glob("run-*.rc"))
    for i, run in enumerate(runs):
        t = T0 + timedelta(seconds=200 * i)
        if windows:
            (d / f"window-{run}.json").write_text(json.dumps({
                "start": t.strftime("%Y-%m-%dT%H:%M:%S"), "end": (t + timedelta(seconds=99)).strftime("%Y-%m-%dT%H:%M:%S")}))
        for g in gpus:
            board, cap = shares(run, g)
            masks = [BOARD] * round(board * 100) + [CAP] * round(cap * 100)
            masks += [IDLE] * (100 - len(masks))
            ex = extra(run, g) if extra else []
            masks[:len(ex)] = ex
            for s, m in enumerate(masks):
                ts = (t + timedelta(seconds=s)).strftime("%Y/%m/%d %H:%M:%S")
                lines.append(f"{1e9 + s},{ts}.123,{m:#018x},100.0,1400,9000,60" + (f",{g}" if index_column else ""))
    (d / "gpu-samples.csv").write_text("\n".join(lines) + "\n")


def test_equal_shares_on_two_gpus_are_clean(tmp_path):
    d = make(tmp_path / "d")
    sampler(d)
    v, notes, c = verdict(d, tmp_path)
    assert c["pass"] is True and not notes["unclean_runs"]
    t = v["throttle"]
    assert t["rule"] == "sampler" and t["compare"] == "per-configuration"
    assert t["config_median_share"][f"{MODEL}-tp-2"]["1"] == {"share_board": 0.5, "share_cap": 0.2}
    r = t["runs"][f"{MODEL}-tp-2-r1"]
    assert r["rule"] == "sampler" and set(r["gpus"]) == {"0", "1"} and r["disqualifying"] == 0 and r["samples"] == 200
    assert r["gpus"]["1"]["samples"] == 100 and r["gpus"]["1"]["share_board"] == 0.5 and r["gpus"]["1"]["share_cap"] == 0.2


def test_shares_differing_across_configurations_do_not_make_runs_unclean(tmp_path):
    d = make(tmp_path / "d")
    shape = {"single": (0.1, 0.8), "layer-2": (0.9, 0.0), "tp-2": (0.5, 0.2)}
    sampler(d, lambda run, g: next(v for k, v in shape.items() if f"{MODEL}-{k}-" in run + "-") if "llama" not in run else (0.3, 0.5))
    v, notes, c = verdict(d, tmp_path)
    assert not notes["unclean_runs"] and c["pass"] is True


def test_the_round_furthest_from_the_configuration_median_is_unclean(tmp_path):
    d = make(tmp_path / "d")
    bad = f"{MODEL}-tp-2-r1"
    sampler(d, lambda run, g: (0.7, 0.2) if (run, g) == (bad, 1) else (0.5, 0.2))  # 0.2 apart on GPU 1 only
    v, notes, c = verdict(d, tmp_path)
    assert [u["run"] for u in notes["unclean_runs"]] == [bad]
    assert "GPU 1" in notes["unclean_runs"][0]["reasons"][0] and "0x200/0x400" in notes["unclean_runs"][0]["reasons"][0]
    assert c["pass"] is True  # two clean rounds of tp-2 are left
    assert next(x for x in c["candidates"] if x["id"] == "tp-2")["measured"]["rounds_clean"] == 2
    assert v["throttle"]["runs"][bad]["gpus"]["1"]["config_median_share_board"] == 0.5


def test_two_rounds_tie_goes_to_the_later_round(tmp_path):
    d = make(tmp_path / "d", rounds=(1, 2))
    sampler(d, lambda run, g: (0.7, 0.2) if run == f"{MODEL}-tp-2-r2" else (0.5, 0.2))
    assert [u["run"] for u in verdict(d, tmp_path)[1]["unclean_runs"]] == [f"{MODEL}-tp-2-r2"]


def test_the_aa_run_belongs_to_its_configuration(tmp_path):
    d = make(tmp_path / "d")
    aa = f"{MODEL}-single-aa"
    sampler(d, lambda run, g: (0.8, 0.2) if run == aa else (0.5, 0.2))
    assert [u["run"] for u in verdict(d, tmp_path)[1]["unclean_runs"]] == [aa]


def test_cap_share_is_judged_separately_and_exactly_at_tolerance_passes(tmp_path):
    d = make(tmp_path / "d")
    bad = f"{MODEL}-single-r2"
    sampler(d, lambda run, g: (0.5, 0.45) if (run, g) == (bad, 0) else (0.5, 0.2))
    assert [u["run"] for u in verdict(d, tmp_path)[1]["unclean_runs"]] == [bad]
    ok = make(tmp_path / "e")
    sampler(ok, lambda run, g: (0.5, 0.3) if (run, g) == (bad, 0) else (0.5, 0.2))  # 0.10 apart
    assert not verdict(ok, tmp_path)[1]["unclean_runs"]
    edge = make(tmp_path / "f")
    sampler(edge, lambda run, g: (0.65, 0.2) if (run, g) == (bad, 0) else (0.5, 0.2))  # exactly 0.15 apart
    assert not verdict(edge, tmp_path)[1]["unclean_runs"]


def test_one_round_only_checks_the_bits(tmp_path):
    d = make(tmp_path / "d", rounds=(1,))
    one = f"{MODEL}-tp-2-r1"
    sampler(d, lambda run, g: (0.9, 0.0) if run == one else (0.5, 0.2))  # alone in its configuration: no spread
    assert not verdict(d, tmp_path)[1]["unclean_runs"]
    sampler(d, lambda run, g: (0.9, 0.0) if run == one else (0.5, 0.2),
            extra=lambda run, g: [0x8] if (run, g) == (one, 0) else [])
    assert [u["run"] for u in verdict(d, tmp_path)[1]["unclean_runs"]] == [one]


def test_a_disqualifying_sample_makes_the_run_unclean(tmp_path):
    d = make(tmp_path / "d")
    bad = f"{MODEL}-layer-2-r3"
    sampler(d, extra=lambda run, g: [0x8 | BOARD] if (run, g) == (bad, 0) else [])
    v, notes, c = verdict(d, tmp_path)
    assert [u["run"] for u in notes["unclean_runs"]] == [bad]
    assert "1 window sample(s) with throttle bits 0x8" in notes["unclean_runs"][0]["reasons"][0]
    assert v["throttle"]["runs"][bad]["disqualifying"] == 1


def test_without_a_csv_the_old_rule_applies(tmp_path):
    d = make(tmp_path / "d")
    write_run(d, "bench", f"{MODEL}-tp-2-r1", *MEAS["tp-2"], throttle="before: 0x4\nafter: 0x4\n")
    v, notes, c = verdict(d, tmp_path)
    assert v["throttle"]["rule"] == "single-sample" and v["throttle"]["runs"][f"{MODEL}-tp-2-r1"] == {"rule": "single-sample"}
    assert [u["run"] for u in notes["unclean_runs"]] == [f"{MODEL}-tp-2-r1"]
    assert "throttle reasons before" in notes["unclean_runs"][0]["reasons"][0]


def test_with_sampler_data_a_cap_bit_before_does_not_disqualify(tmp_path):
    d = make(tmp_path / "d")
    write_run(d, "bench", f"{MODEL}-tp-2-r1", *MEAS["tp-2"], throttle="before: 0x4\nafter: 0x600\n")
    sampler(d)
    v, notes, _ = verdict(d, tmp_path)
    assert not notes["unclean_runs"]
    assert v["throttle"]["runs"][f"{MODEL}-tp-2-r1"]["before_after"]["before"] == ["0x4"]  # recorded, not judged


def test_a_run_without_a_window_falls_back_to_its_single_samples(tmp_path):
    d = make(tmp_path / "d")
    sampler(d)
    (d / f"window-{MODEL}-single-r1.json").unlink()
    write_run(d, "bench", f"{MODEL}-single-r1", *MEAS["single"], throttle="before: 0x4\n")
    v, notes, _ = verdict(d, tmp_path)
    assert v["throttle"]["rule"] == "mixed" and v["throttle"]["runs"][f"{MODEL}-single-r1"]["rule"] == "single-sample"
    assert [u["run"] for u in notes["unclean_runs"]] == [f"{MODEL}-single-r1"]


def test_csv_without_index_column_counts_as_gpu_0(tmp_path):
    d = make(tmp_path / "d")
    sampler(d, gpus=(0,), index_column=False)
    v, _, c = verdict(d, tmp_path)
    assert c["pass"] is True and set(v["throttle"]["runs"][f"{MODEL}-tp-2-r1"]["gpus"]) == {"0"}


def test_window_is_narrowed_by_avg_ns(tmp_path):
    d = make(tmp_path / "d")
    run = f"{MODEL}-tp-2-r1"
    (d / f"window-{run}.json").write_text(json.dumps({"start": "2026-11-17T10:00:00", "end": "2026-11-17T10:01:39"}))
    (d / f"bench-{run}.json").write_text(json.dumps([{"n_prompt": 2048, "n_gen": 0, "avg_ts": 1200.0, "avg_ns": 1_000_000_000},
                                                      {"n_prompt": 0, "n_gen": 128, "avg_ts": 42.0, "avg_ns": 2_000_000_000}]))
    a, b = p5.run_window(d, run)
    assert (b - a).total_seconds() == 2 * 3 + 10 + 2  # 2 * sum(avg_ns) + 10 s, one second of slack each side
    assert b == datetime(2026, 11, 17, 10, 1, 40)


# ---- step-0 check (ADR 0069) ---------------------------------------------------------------------------------------


def step0_of(d, tmp_path, only="tp-2 llama-bench:tensor"):
    noise = tmp_path / "noise"
    noise.mkdir(exist_ok=True)
    return p5.step0(d, only, noise, ROOT / "bench" / "spec" / "phase5.toml")


def test_step0_ratios_per_model_and_round(tmp_path):
    d = make(tmp_path / "d", rounds=(1, 2))
    v = step0_of(d, tmp_path)
    m = v["models"][MODEL]
    assert v["step0_pass"] is True and v["threshold"] == pytest.approx(0.92) and m["clean_pairs"] == 2 and m["ok"] is True
    r = m["rounds"][0]
    assert r["round"] == 1 and r["ratio_pp"] == pytest.approx(1.2) and r["ratio_tg"] == pytest.approx(42 / 36)
    assert r["own"] == {"pp_tps": 1200.0, "tg_tps": 42.0} and m["baseline"] == "llama-bench:tensor" and m["own"] == "tp-2"


def test_step0_one_rate_below_1_minus_band_fails(tmp_path):
    d = make(tmp_path / "d", rounds=(1, 2), meas={**MEAS, "tp-2": (1200.0, 32.0)})  # tg 32 / 36 = 0.89 < 0.92
    v = step0_of(d, tmp_path)
    assert v["step0_pass"] is False and v["models"][MODEL]["ratio_tg_min"] == pytest.approx(32 / 36)
    ok = make(tmp_path / "e", rounds=(1, 2), meas={**MEAS, "tp-2": (1200.0, 33.5)})  # 0.93: inside the band
    assert step0_of(ok, tmp_path)["step0_pass"] is True


def test_step0_unclean_pair_is_not_judged_and_missing_pairs_give_null(tmp_path):
    d = make(tmp_path / "d", rounds=(1, 2))
    write_run(d, "bench", f"{MODEL}-tp-2-r2", 1.0, 1.0, throttle="before: 0x8\n")  # unclean, awful rates
    v = step0_of(d, tmp_path)
    m = v["models"][MODEL]
    assert m["clean_pairs"] == 1 and m["ok"] is None and v["step0_pass"] is None
    assert m["rounds"][1]["clean"] is False and m["rounds"][1]["ratio_pp"] is None


def test_step0_cli_writes_step0_json_and_not_verdict(tmp_path):
    d = make(tmp_path / "d", rounds=(1, 2))
    assert p5.main([str(d), "--only", "tp-2 llama-bench:tensor", "--spec", str(ROOT / "bench" / "spec" / "phase5.toml"),
                    "--noise-dir", str(tmp_path / "none")]) == 0
    assert json.loads((d / "step0.json").read_text())["step0_pass"] is True and not (d / "verdict.json").exists()


def test_only_filters_the_round_order_and_alternates(tmp_path, capsys):
    d = make(tmp_path / "d")

    def order(k):
        p5.main([str(d), "--round-order", MODEL, str(k), "--only", "tp-2 llama-bench:tensor ep-2 llama-bench:nonexistent"])
        return capsys.readouterr().out.splitlines()

    assert order(1) == ["tet tp-2", "base llama-bench tensor"]  # ep-2 is pruned, nonexistent is not accepted
    assert order(2) == ["base llama-bench tensor", "tet tp-2"]
    assert order(3) == ["tet tp-2", "base llama-bench tensor"]
    p5.main([str(d), "--round-order", MODEL, "1"])  # ONLY unset: the full rotation
    assert len(capsys.readouterr().out.splitlines()) == 6
