"""Case S summary of the Phase 6 exit (scripts/phase6_exit_s_summary.py): the freeze refusal, the link band, the span rule, the
A/A aborts, the unclean-round rule and the 15% line, on synthetic two-node sessions. Hardware label: none."""

import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import phase6_exit_s_summary as s6  # noqa: E402
from test_sc_gates import LSCPU_1, LSCPU_2, gpu_txt, reading, run_json, write  # noqa: E402

PH = "9f8e7d6c5b4a3920"
PLAN = """[[replicas]]
[[replicas.stages]]
layers = [0, 47]
devices = ["a/gpu0"]
[[replicas.stages]]
layers = [47, 94]
devices = ["b/gpu0"]
"""
SINGLE = """[[replicas]]
[[replicas.stages]]
layers = [0, 94]
devices = ["a/gpu0"]
"""
POINTS = [[4096, 0.032], [16384, 0.046], [65536, 0.054], [262144, 0.055], [1048576, 0.056], [4194304, 0.099], [16777216, 0.324], [67108864, 1.342]]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def probe(scale=1.0, scale32=1.0):
    pts = [[x, y * scale] for x, y in POINTS]
    pts2 = [[x, y * scale * (scale32 if x >= 16777216 else 1.0)] for x, y in POINTS]
    return {"curves": [{"op": "copy", "ranks": ["a/cpu0", "b/cpu0"], "comm": "tls", "points": pts2},
                       {"op": "copy", "ranks": ["b/cpu0", "a/cpu0"], "comm": "tls", "points": pts}]}


def cluster_json(pp=295.0, tg=5.1, prompt=PH):
    return {"runs": [{"prefill_tok_s": pp, "decode_tok_s": tg, "prompt": {"tokens": 2048, "fnv1a64": prompt}}]}


def anchor(node, prompt=PH, seed=False):
    seg = {} if node == "a" else {"hidden_in": {"seed": 1} if seed else {"file": "/root/hidden.bin"}}
    r = run_json()["runs"][0]
    r.update(prompt={"tokens": 2048, "fnv1a64": prompt}, segment=seg)
    return {"runs": [r]}


def make_round(d, *, t_start, frozen, pp=295.0, tg=5.1, share=0.39, prompt=PH, **seg_kw):
    write(d / "cluster.json", cluster_json(pp, tg, prompt))
    for n in ("a", "b"):
        kw = dict(seg_kw.get(n, {}))
        kw.setdefault("share", share)
        write(d / f"seg-{n}.json", run_json(**kw))
        write(d / f"seg-{n}.gpu.txt", gpu_txt())
    write(d / "meta.json", {"t_start": t_start, "frozen_sha256": frozen})


def build(tmp_path, *, rounds=None, aa_last=True, fit=None, lscpu_b=LSCPU_1, plan=PLAN, probe_end=None, hidden_b=None,
          anchor_b_seed=False, anchor_a_prompt=PH, cand_single_faster=True, mutate=None, frozen_t=1000.0, freeze_pp=300.0, freeze_tg=5.0, end_move=(1.0, 1.0),
          second=None, gate=(700.0, 1.0, 1.0)):
    res = tmp_path / "res"
    write(res / "session.json", {"model_sha256": "ab" * 32, "prompt": {"tokens": 2048, "gen": 128, "fnv1a64": PH},
                                 "segments": [{"node": "a", "span": [0, 47], "embed": True, "head": False},
                                              {"node": "b", "span": [47, 94], "embed": False, "head": False}]})
    for n in ("a", "b"):
        write(res / "gates" / n / "lscpu.txt", lscpu_b if n == "b" else LSCPU_1)
        write(res / "gates" / n / "cotenant.json", reading(1.0, 256))
        write(res / "gates" / n / "gates.json", {"power_ok": True, "idle_ok": True})
        write(res / "env" / n / "env.json", {"ram_budget": {"ram_budget_bytes": 59_450_000_000}})
        write(res / "hidden" / f"{n}.sha256", (hidden_b if (n == "b" and hidden_b) else "h" * 64) + "  hidden\n")
        for a in ("a1-r1", "a1-r2", "a2-r1", "a2-r2"):
            write(res / "calib" / n / "anchors" / f"{a}.json", anchor(n, anchor_a_prompt if n == "a" else PH, anchor_b_seed and n == "b"))
        band = (fit or {}).get(n, 0.05)
        write(res / "calib" / n / "pred.json", {"prediction": {"band": band, "replay_vram_share": 0.39, "replay_full_hit_share": 0.005,
                                                               "replay_ram_hit_rate": 1.0}})
    write(res / "trace" / "routing.sha256", "t" * 64 + "  routing\n")
    write(res / "trace" / "hidden.sha256", "h" * 64 + "  hidden\n")
    if gate is not None:  # the link gate before the download: (seconds between the probes, pp and tg of the second price / the first)
        write(res / "link" / "probe-gate1.json", probe())
        write(res / "link" / "probe-gate2.json", probe(1.02))
        write(res / "link" / "gate-times.json", {"gate1": 100.0, "gate2": 100.0 + gate[0]})
        write(res / "link" / "gate" / "predict-gate1.json", {"prediction": {"pp_tps": 280.0, "tg_tps": 4.8, "request_s": 30.0}})
        write(res / "link" / "gate" / "predict-gate2.json", {"prediction": {"pp_tps": 280.0 * gate[1], "tg_tps": 4.8 * gate[2], "request_s": 30.0}})
    write(res / "link" / "probe-start.json", probe())
    write(res / "link" / "probe-end.json", probe_end or probe(1.02))
    (res / "plans" / "candidates").mkdir(parents=True)
    write(res / "plans" / "judged-plan.toml", plan)
    write(res / "plans" / "candidates" / "m-c1.toml", PLAN)
    write(res / "plans" / "candidates" / "m-c2.toml", SINGLE)
    write(res / "plans" / "candidates.json", {"candidates": [
        {"id": "c1", "plan": "m-c1.toml", "prediction": {"request_s": 40.0}},
        {"id": "c2", "plan": "m-c2.toml", "prediction": {"request_s": 20.0 if cand_single_faster else 90.0}}]})
    write(res / "predictions" / "predict-cluster.json", {
        "kind": "predict-cluster", "routing": "trace", "judged": True, "plan_sha256": sha((res / "plans" / "judged-plan.toml").read_bytes()),
        "routing_trace": {"sha256": "t" * 64, "prompt_hash": PH}, "workload": {"prompt": 2048, "gen": 128},
        "prediction": {"pp_tps": freeze_pp, "tg_tps": freeze_tg, "request_s": 30.0}})
    names = ["predictions/predict-cluster.json", "plans/judged-plan.toml", "calib/a/pred.json", "calib/b/pred.json"]
    if end_move is not None:  # the frozen plan priced again with the end probe's link: only the link differs
        write(res / "predictions" / "predict-cluster-end.json", {"prediction": {"pp_tps": freeze_pp * end_move[0], "tg_tps": freeze_tg * end_move[1]}})
    if second:
        write(res / "predictions" / "predict-cluster-p2.json", {"prediction": {"pp_tps": second[0], "tg_tps": second[1]}})
    write(res / "predictions" / "SHA256SUMS", "".join(f"{sha((res / n).read_bytes())}  {n}\n" for n in names))
    frozen = sha((res / "predictions" / "SHA256SUMS").read_bytes())
    write(res / "predictions" / "FROZEN.sha256", f"{frozen}  SHA256SUMS\n")
    write(res / "predictions" / "frozen.json", {"t_frozen": frozen_t, "sha256": frozen})
    for i, kw in enumerate(rounds if rounds is not None else [{}, {}, {}], 1):
        make_round(res / "judged" / f"r{i}", t_start=2000.0 + i, frozen=frozen, **kw)
    if aa_last:
        make_round(res / "judged" / "aa-last", t_start=3000.0, frozen=frozen, **(aa_last if isinstance(aa_last, dict) else {}))
    if mutate:
        mutate(res, frozen)
    return res, frozen


def summary(res):
    return s6.summarize(str(res))


def failed(o):
    return sorted(r["name"] for r in o["rules"] if not r["ok"])


def test_a_clean_session_passes(tmp_path):
    res, _ = build(tmp_path)
    o = summary(res)
    assert failed(o) == [] and o["verdict"] == "pass", o["status"]
    assert abs(o["e_pp"] - (300.0 / 295.0 - 1)) < 1e-9 and abs(o["e_tg"] - (5.0 / 5.1 - 1)) < 1e-9
    assert any(r["point"].endswith("33554432") for x in o["link"] for r in x["points"])  # 32 MiB is read, interpolated


def test_a_clean_session_that_misses_the_line_is_a_miss_not_a_pass(tmp_path):
    res, _ = build(tmp_path, freeze_tg=6.5)  # tg_pred / tg_meas - 1 = +0.27
    o = summary(res)
    assert failed(o) == [] and o["verdict"] == "miss" and o["e_tg"] > 0.15


# ---- the freeze ---------------------------------------------------------------------------------------------------------------


def test_a_frozen_file_changed_after_the_freeze_refuses(tmp_path):
    res, frozen = build(tmp_path)
    write(res / "plans" / "judged-plan.toml", PLAN + "\n# edited\n")
    o = summary(res)
    assert "freeze" in failed(o) and o["verdict"] == "no verdict"
    assert "frozen file changed" in next(r["detail"] for r in o["rules"] if r["name"] == "freeze")
    assert sg_freeze(res, frozen) != []


def sg_freeze(res, frozen_sha):
    sys.path.insert(0, str(ROOT / "scripts"))
    import sc_gates

    return sc_gates.check_freeze(str(res), frozen_sha)


def test_the_driver_refuses_a_frozen_sha_that_does_not_match(tmp_path):
    res, frozen = build(tmp_path)
    assert sg_freeze(res, frozen) == []
    assert "is not the frozen hash" in sg_freeze(res, "0" * 64)[0]
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "sc_gates.py"), "freeze", "--results", str(res), "--frozen-sha", "0" * 64],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "GATE FAILED freeze" in r.stdout


def test_a_round_that_started_before_the_freeze_or_ran_under_another_hash_is_no_verdict(tmp_path):
    res, _ = build(tmp_path, frozen_t=2001.5)  # r1 started at 2001, before the freeze
    o = summary(res)
    assert "freeze" in failed(o) and "started before the freeze" in next(r["detail"] for r in o["rules"] if r["name"] == "freeze")

    def other(res, frozen):
        write(res / "judged" / "r2" / "meta.json", {"t_start": 2002.0, "frozen_sha256": "e" * 64})

    res, _ = build(tmp_path / "x", mutate=other)
    assert "freeze" in failed(summary(res))


def test_the_prediction_must_be_judged_trace_priced_for_the_exit_prompt(tmp_path):
    def m(res, frozen):
        pc = json.loads((res / "predictions" / "predict-cluster.json").read_text())
        pc["routing"] = "uniform"
        write(res / "predictions" / "predict-cluster.json", pc)

    res, _ = build(tmp_path, mutate=m)
    assert "prediction_judged" in failed(summary(res))

    def m2(res, frozen):
        pc = json.loads((res / "predictions" / "predict-cluster.json").read_text())
        pc["routing_trace"]["sha256"] = "x" * 64
        write(res / "predictions" / "predict-cluster.json", pc)

    res, _ = build(tmp_path / "y", mutate=m2)
    assert "prediction_judged" in failed(summary(res))


# ---- the fit band and the anchors -----------------------------------------------------------------------------------------------


def test_the_fit_band_limit_is_0_10_per_node(tmp_path):
    res, _ = build(tmp_path, fit={"a": 0.05, "b": 0.12})
    o = summary(res)
    assert failed(o) == ["fit_band_b"] and o["verdict"] == "no verdict"
    res, _ = build(tmp_path / "ok", fit={"a": 0.10, "b": 0.099})
    assert failed(summary(res)) == []


def test_anchors_on_fixed_seed_input_or_another_prompt_are_refused(tmp_path):
    res, _ = build(tmp_path, anchor_b_seed=True)
    assert failed(summary(res)) == ["anchors_b"]
    res, _ = build(tmp_path / "p", anchor_a_prompt="deadbeef")  # layer 0 takes tokens: the prompt hash is the evidence
    assert failed(summary(res)) == ["anchors_a"]


def test_the_hidden_rows_must_be_the_same_on_both_boxes(tmp_path):
    res, _ = build(tmp_path, hidden_b="f" * 64)
    assert failed(summary(res)) == ["trace_hidden"]


# ---- the link ---------------------------------------------------------------------------------------------------------------------


def test_raw_small_message_latency_drifting_15_percent_is_clean_when_the_prediction_barely_moves(tmp_path):
    # the end probe's 16 KiB point is 15% up, as the rehearsal's WAN pair moved; the end-priced prediction moves 2%
    p = probe()
    for c in p["curves"]:
        c["points"] = [[x, y * (1.15 if x < 1024 * 1024 else 1.0)] for x, y in c["points"]]
    res, _ = build(tmp_path, probe_end=p, end_move=(0.98, 0.98))
    o = summary(res)
    assert failed(o) == [] and o["verdict"] == "pass", o["status"]
    rows = next(x for x in o["link"] if x["probe"] == "probe-end.json")["points"]
    assert any(not r["ok"] for r in rows)  # the raw point is reported out of band, and not gated
    assert abs(o["link_reprice"]["pp_move"] + 0.02) < 1e-9


def test_a_link_that_moves_the_frozen_prediction_past_0_08_is_not_clean(tmp_path):
    res, _ = build(tmp_path, end_move=(0.91, 1.0))
    o = summary(res)
    assert failed(o) == ["link_reprice"] and o["verdict"] == "no verdict"
    res, _ = build(tmp_path / "tg", end_move=(1.0, 1.09))
    assert failed(summary(res)) == ["link_reprice"]
    res, _ = build(tmp_path / "edge", end_move=(0.92, 1.08))  # exactly on the band: inside
    assert failed(summary(res)) == []


def test_frozen_passes_but_end_priced_fails_is_no_verdict(tmp_path):
    # measured pp 295, tg 5.1: frozen pp 330 is +11.9% (inside), the end-priced 330 * 1.07 = 353 is +19.7% (outside), the link moved 7%
    res, _ = build(tmp_path, freeze_pp=330.0, end_move=(1.07, 1.0))
    o = summary(res)
    assert failed(o) == [] and o["verdict"] == "no verdict" and "only one of the frozen and the end-priced" in o["status"]
    assert abs(o["e_pp"]) <= 0.15 < abs(o["e_pp_end"])


def test_outside_against_both_is_a_miss_and_inside_against_both_a_pass(tmp_path):
    res, _ = build(tmp_path, freeze_pp=400.0, end_move=(1.0, 1.0))
    assert summary(res)["verdict"] == "miss"
    res, _ = build(tmp_path / "ok", end_move=(1.05, 0.95))
    o = summary(res)
    assert o["verdict"] == "pass" and abs(o["e_pp_end"]) < 0.15


def test_no_end_priced_prediction_is_no_verdict(tmp_path):
    res, _ = build(tmp_path, end_move=None)
    assert failed(summary(res)) == ["link_reprice"]


def test_the_second_prompt_prediction_is_reported_beside_the_judged_errors(tmp_path):
    res, _ = build(tmp_path, second=(280.0, 4.8))
    o = summary(res)
    assert o["verdict"] == "pass" and abs(o["second_prompt_report"]["e_pp"] - (280.0 / 295.0 - 1)) < 1e-9
    assert "second_prompt_report" not in summary(build(tmp_path / "none")[0])


def test_the_link_cli_reports_raw_points_and_never_fails(tmp_path):
    res, _ = build(tmp_path)
    a, b = res / "link" / "probe-start.json", res / "link" / "probe-end.json"
    write(b, probe(1.2))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_s_summary.py"), "link", str(a), str(b)], capture_output=True, text=True)
    assert r.returncode == 0 and "OUTSIDE THE BAND" in r.stdout and "not gated" in r.stdout


def test_the_interpolation_is_the_cost_models_curve_time():
    assert s6.curve_time(POINTS, 16 * 1024 * 1024) == pytest.approx(0.324)
    mid = s6.curve_time(POINTS, 32 * 1024 * 1024)
    assert mid == pytest.approx(0.324 + (1.342 - 0.324) * (32 - 16) / (64 - 16))
    assert s6.curve_time(POINTS, 100 * 2**20) == pytest.approx(1.342 + (1.342 - 0.324) / (48 * 2**20) * (36 * 2**20))


# ---- spanning ------------------------------------------------------------------------------------------------------------------------


def test_a_judged_plan_on_one_node_is_refused_and_the_pick_skips_single_node_candidates(tmp_path):
    res, _ = build(tmp_path, plan=SINGLE)
    o = summary(res)
    assert "span_both_nodes" in failed(o)
    # the single-node candidate is the fastest on paper and must not be the pick
    info, why = s6.pick(str(res / "plans" / "candidates.json"), str(res / "plans" / "candidates"))
    assert not why and info["id"] == "c1" and info["dropped_single_node"] == ["c2"]
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_s_summary.py"), "span", str(res / "plans" / "judged-plan.toml")],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "GATE FAILED span" in r.stdout
    out = tmp_path / "pick.toml"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_s_summary.py"), "pick", str(res / "plans" / "candidates.json"),
                        str(res / "plans" / "candidates"), str(out)], capture_output=True, text=True)
    assert r.returncode == 0 and out.read_text() == PLAN


def test_a_judged_plan_that_is_not_the_pick_is_refused(tmp_path):
    res, _ = build(tmp_path, plan=PLAN + "# another plan\n")
    assert "span_both_nodes" in failed(summary(res))


def test_no_spanning_candidate_is_an_error(tmp_path):
    res, _ = build(tmp_path)
    write(res / "plans" / "candidates" / "m-c1.toml", SINGLE)
    info, why = s6.pick(str(res / "plans" / "candidates.json"), str(res / "plans" / "candidates"))
    assert why and "error" in info


# ---- A/A -----------------------------------------------------------------------------------------------------------------------------------


def test_the_early_aa_outside_the_band_is_no_verdict(tmp_path):
    res, _ = build(tmp_path, rounds=[{}, {"pp": 150.0}, {}])
    o = summary(res)
    assert "aa_early" in failed(o) and o["verdict"] == "no verdict"
    assert o["aa_early"]["diff"] < -0.08 or o["aa_early"]["diff"] > 0.08


def test_the_aa_last_outside_the_band_or_missing_is_no_verdict(tmp_path):
    res, _ = build(tmp_path, aa_last={"pp": 150.0})
    assert failed(summary(res)) == ["aa"]
    res, _ = build(tmp_path / "m", aa_last=False)
    o = summary(res)
    assert failed(o) == ["aa", "freeze"][:1] or "aa" in failed(o)
    assert o["verdict"] == "no verdict"


# ---- the unclean-round rule -----------------------------------------------------------------------------------------------------------


def unclean_balloon():
    return {"a": {"balloon_ok": False, "balloon_reason": "dead"}}


def test_unclean_rounds_are_rerun_up_to_eight_and_every_round_is_reported(tmp_path):
    rounds = [{"a": {"cotenant_ok": False}}, {}, {"b": {"prefill_bytes": 4096}}, {}, {}]  # r1 and r3 unclean, 3 clean of 5
    res, _ = build(tmp_path, rounds=rounds)
    o = summary(res)
    assert failed(o) == [] and o["verdict"] == "pass"
    assert [r["clean"] for r in o["rounds"]] == [False, True, False, True, True]
    assert "co-tenant" in o["rounds"][0]["unclean_reasons"][0] and "prefill" in o["rounds"][2]["unclean_reasons"][0]
    assert len(o["rounds"]) == 5  # all reported
    # the medians are those of the clean rounds only
    res, _ = build(tmp_path / "m", rounds=[{"pp": 10.0, "a": {"balloon_ok": False}}, {}, {}, {}])
    assert abs(summary(res)["median"]["pp"] - 295.0) < 1e-9


def test_fewer_than_three_clean_rounds_in_eight_is_no_verdict_and_a_ninth_round_is_invalid(tmp_path):
    bad = {"a": {"cotenant_ok": False}}
    res, _ = build(tmp_path, rounds=[{}, {}] + [bad] * 6)
    o = summary(res)
    assert "rounds_clean" in failed(o) and "2 clean of 8" in next(r["detail"] for r in o["rules"] if r["name"] == "rounds_clean")
    res, _ = build(tmp_path / "n", rounds=[{}, {}, bad] * 3)  # nine rounds run: more than the rule allows
    assert "rounds_clean" in failed(summary(res))


def test_next_round_runs_until_three_clean_or_eight(tmp_path):
    cmd = [sys.executable, str(ROOT / "scripts" / "phase6_exit_s_summary.py"), "next-round"]
    res, _ = build(tmp_path, rounds=[{}, {}])
    assert subprocess.run(cmd + [str(res)], capture_output=True).returncode == 0  # two clean: one more
    res, _ = build(tmp_path / "a", rounds=[{}, {}, {}])
    assert subprocess.run(cmd + [str(res)], capture_output=True).returncode == 1
    bad = {"a": {"cotenant_ok": False}}
    res, _ = build(tmp_path / "b", rounds=[bad, {}, bad, {}, bad])
    assert subprocess.run(cmd + [str(res)], capture_output=True).returncode == 0
    res, _ = build(tmp_path / "c", rounds=[{}, {}] + [bad] * 6)
    assert subprocess.run(cmd + [str(res)], capture_output=True).returncode == 1


def test_a_round_on_another_prompt_or_with_a_throttle_is_unclean(tmp_path):
    res, _ = build(tmp_path, rounds=[{}, {}, {}, {"prompt": "0badf00d"}])
    o = summary(res)
    assert not o["rounds"][3]["clean"] and "prompt_hash" in failed(o)
    res, _ = build(tmp_path / "t", rounds=[{}, {}, {}])
    write(res / "judged" / "r2" / "seg-b.gpu.txt", gpu_txt("0x20"))
    assert [r["clean"] for r in summary(res)["rounds"]] == [True, False, True]


# ---- the routing gate --------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("share,ok", [(0.39 + 0.01, True), (0.39 + 0.03, False), (0.39 - 0.019, True)])
def test_the_routing_gate_is_per_segment(tmp_path, share, ok):
    res, _ = build(tmp_path, rounds=[{"b": {"share": share}}] * 3)
    o = summary(res)
    assert ("routing_gate" not in failed(o)) is ok
    if not ok:
        assert o["verdict"] == "no verdict"


# ---- budgets and gates ---------------------------------------------------------------------------------------------------------------------


def test_gates_and_budgets_on_both_boxes(tmp_path):
    res, _ = build(tmp_path, lscpu_b=LSCPU_2)
    assert failed(summary(res)) == ["gates_b"]
    res, _ = build(tmp_path / "r")
    (res / "env" / "a" / "env.json").unlink()
    assert failed(summary(res)) == ["ram_budget_a"]
    res, _ = build(tmp_path / "s")
    write(res / "gates" / "a" / "gates.json", {"power_ok": True, "idle_ok": False})
    assert failed(summary(res)) == ["gates_a"]


def test_the_cli_writes_the_verdict(tmp_path):
    res, _ = build(tmp_path)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_s_summary.py"), "summary", str(res)], capture_output=True, text=True)
    assert r.returncode == 0 and "status: pass" in r.stdout
    assert json.loads((res / "verdict-s.json").read_text())["verdict"] == "pass"


def test_the_plans_slots_must_be_the_anchors(tmp_path):
    plan = tmp_path / "p.toml"
    plan.write_text(
        '[[replicas]]\n[[replicas.stages]]\nlayers = [0, 47]\ndevices = ["a/gpu0"]\n'
        '[[replicas.stages.expert_cache]]\ndevice = "a/gpu0"\nslots = 32\n[[replicas.stages.expert_cache]]\ndevice = "a/cpu0"\nslots = 96\n'
    )
    assert s6.check_slots(str(plan), 32, 96) == []
    why = s6.check_slots(str(plan), 32, 64)
    assert len(why) == 1 and "cpu0 has 96 slots" in why[0]
    plan.write_text('[[replicas]]\n[[replicas.stages]]\nlayers = [0, 47]\ndevices = ["a/gpu0"]\n')
    assert s6.check_slots(str(plan), 1, 1) == []  # no cache, nothing to compare

# ---- the link gate before the download -----------------------------------------------------------------------------------------


def test_the_link_gate_passes_when_two_full_probes_ten_minutes_apart_move_the_prediction_within_the_band(tmp_path):
    res, _ = build(tmp_path, gate=(600.0, 1.07, 0.93))
    o = summary(res)
    assert failed(o) == [] and o["link_gate"]["pp_move"] == pytest.approx(0.07) and o["link_gate"]["tg_move"] == pytest.approx(-0.07)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_s_summary.py"), "gate", str(res)], capture_output=True, text=True)
    assert r.returncode == 0 and "passes" in r.stdout
    g = json.loads((res / "link" / "gate.json").read_text())
    assert g["pass"] is True and g["gap_s"] == 600.0 and g["min_gap_s"] == 600.0


@pytest.mark.parametrize("gate,why", [
    ((599.0, 1.0, 1.0), "599 s apart"),
    ((900.0, 1.09, 1.0), "pp moves +0.090"),
    ((900.0, 1.0, 0.90), "tg moves -0.100"),
    ((900.0, 0.5, 1.0), "pp moves -0.500"),
])
def test_the_link_gate_aborts_a_pair_whose_probes_are_too_close_or_whose_prediction_moves(tmp_path, gate, why):
    res, _ = build(tmp_path, gate=gate)
    o = summary(res)
    assert "link_gate" in failed(o) and why in next(r["detail"] for r in o["rules"] if r["name"] == "link_gate")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_s_summary.py"), "gate", str(res)], capture_output=True, text=True)
    assert r.returncode == 1 and "GATE FAILED link_gate" in r.stdout and json.loads((res / "link" / "gate.json").read_text())["pass"] is False


def test_the_link_gate_needs_full_probes_and_all_its_evidence(tmp_path):
    def truncated(res, frozen):
        pts = [[x, y] for x, y in POINTS if x <= 4194304]  # --max-bytes 4 MiB, as the rehearsal fell back to
        write(res / "link" / "probe-gate2.json", {"curves": [{"op": "copy", "ranks": ["a/cpu0", "b/cpu0"], "comm": "tls", "points": pts}]})

    o = summary(build(tmp_path, mutate=truncated)[0])
    assert "link_gate" in failed(o) and "a full probe reaches" in next(r["detail"] for r in o["rules"] if r["name"] == "link_gate")
    res, _ = build(tmp_path / "none", gate=None)  # nothing recorded: the rule fails, the session is not a verdict
    o = summary(res)
    assert "link_gate" in failed(o) and o["verdict"] != "pass"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_s_summary.py"), "precheck", str(res), "calib"], capture_output=True, text=True)
    assert r.returncode == 1 and "link_gate" in r.stdout  # the calibration refuses without the gate


def test_the_gate_files_are_not_frozen(tmp_path):
    # the freeze covers the prediction and what it was priced from; the gate predicts another thing (uniform routing, a committed layout)
    res, _ = build(tmp_path)
    sums = (res / "predictions" / "SHA256SUMS").read_text()
    assert "gate" not in sums
