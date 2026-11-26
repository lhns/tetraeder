"""Case S summary of the Phase 6 exit (scripts/phase6_exit_s_summary.py): the freeze refusal, the link band, the span rule, the
A/A aborts, the unclean-round rule and the 15% line, on synthetic two-node sessions. Hardware label: none."""

import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import phase6_exit_s_summary as s6  # noqa: E402
from test_sc_gates import LSCPU_1, LSCPU_2, gpu_txt, reading, run_json, write  # noqa: E402

PH = "9f8e7d6c5b4a3920"
def plan_of(v=32, r=96):
    stage = '[[replicas.stages]]\nlayers = [{lo}, {hi}]\ndevices = ["{n}/gpu0"]\n' \
            '[[replicas.stages.expert_cache]]\ndevice = "{n}/gpu0"\nslots = {v}\n' \
            '[[replicas.stages.expert_cache]]\ndevice = "{n}/cpu0"\nslots = {r}\n'
    return "[[replicas]]\n" + stage.format(lo=0, hi=47, n="a", v=v, r=r) + stage.format(lo=47, hi=94, n="b", v=v, r=r)


PLAN = plan_of()
THIN = plan_of(32, 64)  # 96 of 128 experts: not covering
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


def anchor(node, prompt=PH, seed=False, ubatch=2048):
    seg = {} if node == "a" else {"hidden_in": {"seed": 1} if seed else {"file": "/root/hidden.bin"}}
    r = run_json()["runs"][0]
    r.update(prompt={"tokens": 2048, "fnv1a64": prompt}, segment=seg)
    return {"runs": [r], "exit_ubatch": ubatch}  # sc_box.sh stamps the batch size of the run


def make_round(d, *, t_start, frozen, pp=295.0, tg=5.1, share=0.39, prompt=PH, **seg_kw):
    write(d / "cluster.json", cluster_json(pp, tg, prompt))
    for n in ("a", "b"):
        kw = dict(seg_kw.get(n, {}))
        kw.setdefault("share", share)
        kw.setdefault("gpu_mem_total_mib", 24576)
        kw.setdefault("gpu_mem_peak_mib", 21000.0)
        write(d / f"seg-{n}.json", run_json(**kw))
        write(d / f"seg-{n}.gpu.txt", gpu_txt())
    write(d / "meta.json", {"t_start": t_start, "frozen_sha256": frozen})


def build(tmp_path, *, rounds=None, aa_last=True, fit=None, lscpu_b=LSCPU_1, plan=PLAN, probe_end=None, hidden_b=None,
          anchor_b_seed=False, anchor_a_prompt=PH, cand_single_faster=True, mutate=None, frozen_t=1000.0, freeze_pp=300.0, freeze_tg=5.0, end_move=(1.0, 1.0),
          second=None, gate=(700.0, 1.0, 1.0), dryrun=True, repick="same", budgets=(59_450_000_000, 59_430_000_000), anchor_ubatch=2048,
          fitted_rates_in_gate=True):
    res = tmp_path / "res"
    write(res / "session.json", {"model_sha256": "ab" * 32, "prompt": {"tokens": 2048, "gen": 128, "fnv1a64": PH},
                                 "segments": [{"node": "a", "span": [0, 47], "embed": True, "head": False, "slots": [32, 96]},
                                              {"node": "b", "span": [47, 94], "embed": False, "head": False, "slots": [32, 96]}],
                                 "boxes": {"a": {"id": "11"}, "b": {"id": "22"}},
                                 "budget": {n: {"plan": budgets[0], "installed": budgets[1]} for n in ("a", "b")}})
    if dryrun:  # the dry run on the tiny MoE passed on both boxes of this session before the download
        write(res / "dryrun.json", {"passed": True, "name": "s-dryrun", "nodes": ["a", "b"], "boxes": {"a": "11", "b": "22"}})
    write(res / "layout.json", {"n_expert": 128, "n_layer": 94})
    for n in ("a", "b"):
        write(res / "gates" / n / "lscpu.txt", lscpu_b if n == "b" else LSCPU_1)
        write(res / "gates" / n / "cotenant.json", reading(1.0, 256))
        write(res / "gates" / n / "gates.json", {"power_ok": True, "idle_ok": True})
        write(res / "env" / n / "env.json", {"ram_budget": {"ram_budget_bytes": budgets[1]}})  # measured again after the install
        write(res / "env" / n / "env-plan.json", {"ram_budget": {"ram_budget_bytes": budgets[0]}})  # the reading the plan was made on
        write(res / "hidden" / f"{n}.sha256", (hidden_b if (n == "b" and hidden_b) else "h" * 64) + "  hidden\n")
        for a in ("a1-r1", "a1-r2", "a2-r1", "a2-r2"):
            write(res / "calib" / n / "anchors" / f"{a}.json", anchor(n, anchor_a_prompt if n == "a" else PH, anchor_b_seed and n == "b", anchor_ubatch))
        band = (fit or {}).get(n, 0.05)
        write(res / "calib" / n / "pred.json", {"prediction": {"band": band, "replay_vram_share": 0.39, "replay_full_hit_share": 0.005,
                                                               "replay_ram_hit_rate": 1.0}})
    write(res / "trace" / "routing.sha256", "t" * 64 + "  routing\n")
    write(res / "trace" / "hidden.sha256", "h" * 64 + "  hidden\n")
    write(res / "plans" / "judged-plan.toml", plan)  # the gate prices it
    plan_sha = sha((res / "plans" / "judged-plan.toml").read_bytes())
    if repick == "same":  # the covering pick made again on the budgets measured after the install is the judged plan
        write(res / "plans" / "repick.json", {"plan_sha256": plan_sha, "judged_sha256": plan_sha, "same": True})
    elif repick == "other":
        write(res / "plans" / "repick.json", {"plan_sha256": "0" * 64, "judged_sha256": plan_sha, "same": False})
    if gate is not None:  # the link gate before the download: (seconds between the probes, pp and tg of the second price / the first)
        (res / "link" / "gate").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(s6.gate_fitted_path(), res / "link" / "gate" / "fitted.toml")
        rates = s6.fitted_rates(s6.gate_fitted_path())
        for k in ("gate1", "gate2"):  # the graph each price was made on: both nodes with the fitted rates (or the discovered priors)
            devs = [{"id": f"{n}/{d}", "compute": {"flops_per_s": r if fitted_rates_in_gate else 1.0}} for n in ("a", "b") for d, r in rates.items()]
            write(res / "link" / "gate" / f"hw-{k}.json", {"devices": devs})
        write(res / "link" / "probe-gate1.json", probe())
        write(res / "link" / "probe-gate2.json", probe(1.02))
        write(res / "link" / "gate-times.json", {"gate1": 100.0, "gate2": 100.0 + gate[0]})
        write(res / "link" / "gate" / "predict-gate1.json", {"plan_sha256": plan_sha, "hw_sha256": sha((res / "link" / "gate" / "hw-gate1.json").read_bytes()),
                                                              "prediction": {"pp_tps": 280.0, "tg_tps": 4.8, "request_s": 30.0}})
        write(res / "link" / "gate" / "predict-gate2.json", {"plan_sha256": plan_sha, "hw_sha256": sha((res / "link" / "gate" / "hw-gate2.json").read_bytes()),
                                                              "prediction": {"pp_tps": 280.0 * gate[1], "tg_tps": 4.8 * gate[2], "request_s": 30.0}})
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
    assert failed(summary(res)) == ["ram_budget_a", "ram_budget_remeasured"]  # no reading after the install either
    res, _ = build(tmp_path / "s")
    write(res / "gates" / "a" / "gates.json", {"power_ok": True, "idle_ok": False})
    assert failed(summary(res)) == ["gates_a"]


def test_the_cli_writes_the_verdict(tmp_path):
    res, _ = build(tmp_path)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_s_summary.py"), "summary", str(res)], capture_output=True, text=True)
    assert r.returncode == 0 and "status: pass" in r.stdout
    assert json.loads((res / "verdict-s.json").read_text())["verdict"] == "pass"


def test_the_pick_must_cover_every_expert_and_the_anchors_run_its_slots(tmp_path):
    res, _ = build(tmp_path)
    assert "segments_covering" not in failed(summary(res))
    # a plan holding 96 of 128 experts per layer is no judged segment (ADR 0091 History 2026-11-25)
    res, _ = build(tmp_path / "thin", plan=THIN)
    o = summary(res)
    assert "segments_covering" in failed(o)
    assert "96 of 128 experts" in next(r["detail"] for r in o["rules"] if r["name"] == "segments_covering")
    # anchors that ran other slots than the pick's
    res, _ = build(tmp_path / "other", plan=plan_of(40, 88))
    why = next(r["detail"] for r in summary(res)["rules"] if r["name"] == "segments_covering")
    assert "the anchors ran slots [32, 96], the pick has [40, 88]" in why


def test_the_slots_cli_prints_the_picks_slots_per_node_and_refuses_a_thin_plan(tmp_path):
    cli = [sys.executable, str(ROOT / "scripts" / "phase6_exit_s_summary.py"), "slots"]
    good, thin = tmp_path / "good.toml", tmp_path / "thin.toml"
    good.write_text(plan_of(41, 87))
    thin.write_text(THIN)
    r = subprocess.run([*cli, str(good), "--n-expert", "128"], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.split("\n")[:2] == ["a 41 87", "b 41 87"]
    r = subprocess.run([*cli, str(thin), "--n-expert", "128"], capture_output=True, text=True)
    assert r.returncode == 1 and "GATE FAILED slots" in r.stderr and "not a covering segment" in r.stderr


def test_the_pick_drops_candidates_that_do_not_cover(tmp_path):
    res, _ = build(tmp_path)
    write(res / "plans" / "candidates" / "m-c3.toml", THIN)
    doc = json.loads((res / "plans" / "candidates.json").read_text())
    doc["candidates"].append({"id": "c3", "plan": "m-c3.toml", "prediction": {"request_s": 10.0}})  # the fastest, but thin
    write(res / "plans" / "candidates.json", doc)
    args = (str(res / "plans" / "candidates.json"), str(res / "plans" / "candidates"))
    assert s6.pick(*args)[0]["id"] == "c3"
    info, why = s6.pick(*args, n_expert=128)
    assert not why and info["id"] == "c1" and set(info["dropped_single_node"]) == {"c2", "c3"}
    write(res / "plans" / "candidates" / "m-c1.toml", THIN)
    info, why = s6.pick(*args, n_expert=128)
    assert why and "covers all 128 experts" in why[0]


def test_the_link_gate_must_have_priced_the_judged_pick(tmp_path):
    res, _ = build(tmp_path)
    write(res / "link" / "gate" / "predict-gate2.json", {"plan_sha256": "0" * 64, "hw_sha256": sha((res / "link" / "gate" / "hw-gate2.json").read_bytes()),
                                                          "prediction": {"pp_tps": 280.0, "tg_tps": 4.8, "request_s": 30.0}})
    o = summary(res)
    assert "link_gate" in failed(o)
    assert "not the judged pick" in next(r["detail"] for r in o["rules"] if r["name"] == "link_gate")


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


# ---- ADR 0091 History 2026-11-26: fitted link gate, step size, dry run, memory, card peak ------------------------------------------


def rule_detail(o, name):
    return next(r["detail"] for r in o["rules"] if r["name"] == name)


def test_the_link_gate_prices_on_the_fitted_rates_of_both_nodes_not_on_priors(tmp_path):
    res, _ = build(tmp_path, fitted_rates_in_gate=False)  # the graphs the prices were made on carry the discovered rates
    o = summary(res)
    assert "link_gate" in failed(o) and "is not the fitted" in rule_detail(o, "link_gate")
    res, _ = build(tmp_path / "ok")
    assert failed(summary(res)) == []


def test_the_link_gate_refuses_a_fit_that_is_not_the_committed_one_and_a_price_made_on_another_graph(tmp_path):
    def other_fit(res, frozen):
        (res / "link" / "gate" / "fitted.toml").write_text("# another fit\n")

    assert "not the committed fitted graph" in rule_detail(summary(build(tmp_path, mutate=other_fit)[0]), "link_gate")

    def other_graph(res, frozen):
        write(res / "link" / "gate" / "hw-gate1.json", {"devices": []})

    o = summary(build(tmp_path / "g", mutate=other_graph)[0])
    assert "link_gate" in failed(o) and "not hw-gate1.json" in rule_detail(o, "link_gate")


KNOBS = "[knobs]\nubatch = 2048\nprefill_micro_batches = 4\n\n"


def test_the_picks_micro_batch_is_the_step_cut_as_the_pipeline_cuts_it(tmp_path):
    f = tmp_path / "p.toml"
    for knobs, prompt, want in [("ubatch = 2048\nprefill_micro_batches = 4", 2048, 512), ("ubatch = 128\nprefill_micro_batches = 4", 2048, 128),
                                ("ubatch = 2048\nprefill_micro_batches = 1", 2048, 2048), ("ubatch = 2048\nprefill_micro_batches = 3", 2048, 683),
                                ("ubatch = 16\nprefill_micro_batches = 2", 64, 16), ("ubatch = 1024\nprefill_micro_batches = 8", 100, 13)]:
        f.write_text(f"[knobs]\n{knobs}\n\n" + PLAN)
        assert s6.pick_chunk(str(f), prompt) == want, (knobs, prompt)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_s_summary.py"), "chunk", str(f), "--prompt", str(prompt)], capture_output=True, text=True)
        assert r.stdout.split() == [str(want)]
    f.write_text(PLAN)  # no knobs: one batch of the prompt
    assert s6.pick_chunk(str(f), 2048) == 2048


def pick_has_knobs(res, frozen):  # the planner's pick carries its knobs: the judged plan is a copy of it
    write(res / "plans" / "candidates" / "m-c1.toml", KNOBS + PLAN)


def test_the_anchors_must_run_the_picks_micro_batch(tmp_path):
    res, _ = build(tmp_path, plan=KNOBS + PLAN, anchor_ubatch=512, mutate=pick_has_knobs)
    assert failed(summary(res)) == []
    res, _ = build(tmp_path / "whole", plan=KNOBS + PLAN, anchor_ubatch=2048, mutate=pick_has_knobs)  # the whole prompt as one batch, as segment-confirm ran them
    o = summary(res)
    assert failed(o) == ["anchors_step_a", "anchors_step_b"] and "micro-batch is 512" in rule_detail(o, "anchors_step_a")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_s_summary.py"), "precheck", str(res), "calib"], capture_output=True, text=True)
    assert r.returncode == 1 and "anchors_step_a" in r.stdout  # the calibration is refused

    def unstamped(res, frozen):  # an anchor that does not say its batch
        d = json.loads((res / "calib" / "a" / "anchors" / "a1-r1.json").read_text())
        d.pop("exit_ubatch")
        write(res / "calib" / "a" / "anchors" / "a1-r1.json", d)

    assert "anchors_step_a" in failed(summary(build(tmp_path / "u", mutate=unstamped)[0]))


def test_a_download_needs_a_dry_run_that_passed_on_both_boxes_of_this_session(tmp_path):
    res, _ = build(tmp_path, dryrun=False)
    o = summary(res)
    assert failed(o) == ["dryrun"]
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_s_summary.py"), "dryrun", str(res)], capture_output=True, text=True)
    assert r.returncode == 1 and "no dryrun.json" in r.stdout

    def one_node(res, frozen):
        write(res / "dryrun.json", {"passed": True, "nodes": ["a"], "boxes": {"a": "11", "b": "22"}})

    assert "covers nodes ['a']" in rule_detail(summary(build(tmp_path / "one", mutate=one_node)[0]), "dryrun")

    def other_boxes(res, frozen):  # a dry run of earlier boxes does not count
        write(res / "dryrun.json", {"passed": True, "nodes": ["a", "b"], "boxes": {"a": "11", "b": "99"}})

    assert "ran on boxes" in rule_detail(summary(build(tmp_path / "boxes", mutate=other_boxes)[0]), "dryrun")

    def failed_dry(res, frozen):
        write(res / "dryrun.json", {"passed": False, "nodes": ["a", "b"], "boxes": {"a": "11", "b": "22"}})

    assert "did not pass" in rule_detail(summary(build(tmp_path / "failed", mutate=failed_dry)[0]), "dryrun")
    res, _ = build(tmp_path / "ok")
    assert subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_s_summary.py"), "dryrun", str(res)], capture_output=True).returncode == 0


def test_the_ram_budget_is_measured_again_after_the_install_and_the_pick_must_be_the_judged_plan(tmp_path):
    res, _ = build(tmp_path, repick=None)
    o = summary(res)
    assert failed(o) == ["ram_budget_remeasured"] and "no plans/repick.json" in rule_detail(o, "ram_budget_remeasured")
    o = summary(build(tmp_path / "other", repick="other")[0])
    assert failed(o) == ["ram_budget_remeasured"] and "not the judged plan" in rule_detail(o, "ram_budget_remeasured")

    def no_second_reading(res, frozen):
        (res / "env" / "b" / "env-plan.json").unlink()

    o = summary(build(tmp_path / "once", mutate=no_second_reading)[0])
    assert "b: no budget before and after" in rule_detail(o, "ram_budget_remeasured")
    # the calibration is refused on it
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_s_summary.py"), "precheck", str(build(tmp_path / "pc", repick=None)[0]), "calib"],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "ram_budget_remeasured" in r.stdout
    # the session file must say what the files say
    o = summary(build(tmp_path / "lie", budgets=(59_450_000_000, 59_430_000_000), mutate=lambda res, f: write(
        res / "env" / "a" / "env.json", {"ram_budget": {"ram_budget_bytes": 60_000_000_000}}))[0])
    assert "session.json budget" in rule_detail(o, "ram_budget_remeasured")


CARD = {"a": 32768, "b": 32768}


def peak_round(total, peak):
    return {n: {"gpu_mem_total_mib": total, "gpu_mem_peak_mib": peak} for n in ("a", "b")}


@pytest.mark.parametrize("total,peak,clean", [
    (24576, 24576.0, True),   # a 24 GiB card is the card: no extra rule
    (32768, 24000.0, True),   # a bigger card that stays under 24,576 MiB
    (32768, 24575.0, True),
    (32768, 24576.0, False),  # at the line is not under it
    (32768, 26000.0, False),
    (49152, 30000.0, False),
])
def test_on_a_card_over_24_gib_the_peak_stays_under_24576_mib(tmp_path, total, peak, clean):
    res, _ = build(tmp_path, rounds=[peak_round(total, peak)] * 3)
    o = summary(res)
    assert all(r["clean"] for r in o["rounds"]) == clean, [r["unclean_reasons"] for r in o["rounds"]]
    if not clean:
        assert "peak GPU memory" in o["rounds"][0]["unclean_reasons"][0] and "24576" in o["rounds"][0]["unclean_reasons"][0]
        assert "rounds_clean" in failed(o)


def test_a_card_over_24_gib_without_a_peak_reading_or_a_round_without_the_card_size_is_unclean(tmp_path):
    res, _ = build(tmp_path, rounds=[{n: {"gpu_mem_total_mib": 32768, "gpu_mem_peak_mib": None} for n in ("a", "b")}] * 3)
    assert not any(r["clean"] for r in summary(res)["rounds"])
    res, _ = build(tmp_path / "nocard", rounds=[{"a": {"gpu_mem_total_mib": None}}] * 3)
    why = summary(res)["rounds"][0]["unclean_reasons"]
    assert any("gpu_mem_total_mib" in w for w in why) or any("card" in w for w in why), why


def test_a_cpu_only_segment_has_no_card_and_the_peak_rule_is_n_a(tmp_path):
    res, _ = build(tmp_path, rounds=[{n: {"gpu_mem_total_mib": 0, "gpu_mem_peak_mib": None} for n in ("a", "b")}] * 3)
    assert all(r["clean"] for r in summary(res)["rounds"])


def test_a_gpu_segment_without_a_card_reading_stays_unclean_next_to_a_cpu_one(tmp_path):
    res, _ = build(tmp_path, rounds=[{"a": {"gpu_mem_total_mib": 0, "gpu_mem_peak_mib": None}, "b": {"gpu_mem_total_mib": None}}] * 3)
    why = summary(res)["rounds"][0]["unclean_reasons"]
    assert not any(r["clean"] for r in summary(res)["rounds"])
    assert any(w.startswith("b:") and "gpu_mem_total_mib" in w for w in why) and not any(w.startswith("a:") for w in why), why
