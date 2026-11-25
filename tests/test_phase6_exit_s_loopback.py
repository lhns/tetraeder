"""The loopback dry run of Case S: scripts/phase6_exit_s.sh end to end with the stand-in tools of scripts/phase6_loopback.py (two "boxes" as
directories and processes on this machine, fixed-number files in the layouts the real tools write). It runs the driver's own flow: the
gates, the pick, the calibration gates, the freeze and its refusals, the early A/A, the unclean-round rule, the A/A last, the link probes
and the verdict, with LB_SCENARIO breaking one rule at a time. It proves the driver, not the engine. Hardware label: none (needs a POSIX
userland with pgrep, setsid and flock; skips cleanly without)."""

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

needs_posix = pytest.mark.skipif(
    os.name == "nt" or any(shutil.which(t) is None for t in ("bash", "pgrep", "setsid", "flock")),
    reason="needs bash, pgrep, setsid and flock",
)


def kill_all(tmp_path):
    for pidfile in (tmp_path / "lb").glob("*/pids/*"):
        try:
            os.kill(int(pidfile.read_text().split()[0]), 9)
        except (ValueError, OSError, IndexError):
            pass


def run_s(tmp_path, stages, scenario="", **env):
    e = {**os.environ, "LOOPBACK": "1", "RESULTS_NAME": "s", "RESULTS_ROOT": str(tmp_path / "res"), "LB_DIR": str(tmp_path / "lb"),
         "STAGES": stages, "LB_SCENARIO": scenario, **env}
    r = subprocess.run(["bash", str(SCRIPTS / "phase6_exit_s.sh")], cwd=ROOT, env=e, capture_output=True, text=True, timeout=900)
    return r, tmp_path / "res" / "s"


def frozen_sha(out):
    return (out / "predictions" / "FROZEN.sha256").read_text().split()[0]


def text(r):
    return r.stdout + r.stderr


def session_s(tmp_path, scenario="", second_scenario=None):
    """The two invocations of the driver; returns (first result, second result or None, results dir)."""
    try:
        r1, out = run_s(tmp_path, "gates setup budget plan calib predict", scenario)
        if r1.returncode != 0 or not (out / "predictions" / "FROZEN.sha256").exists():
            return r1, None, out
        r2, out = run_s(tmp_path, "judged aalast linkend summary", second_scenario if second_scenario is not None else scenario,
                        FROZEN_SHA=frozen_sha(out))
        return r1, r2, out
    finally:
        kill_all(tmp_path)


def verdict_s(out):
    return json.loads((out / "verdict-s.json").read_text())


def failed_rules(v):
    return sorted(r["name"] for r in v["rules"] if not r["ok"])


@needs_posix
def test_case_s_runs_end_to_end_on_loopback_and_passes(tmp_path):
    r1, r2, out = session_s(tmp_path)
    log = text(r1)[-2500:] + (text(r2)[-2500:] if r2 else "")
    assert r2 is not None and r2.returncode == 0, log
    v = verdict_s(out)
    assert v["verdict"] == "pass" and failed_rules(v) == [], v["status"]
    assert len(v["rounds"]) == 3 and all(r["clean"] for r in v["rounds"])
    assert (out / "judged" / "aa-last" / "cluster.json").exists() and (out / "judged" / "aa-early.txt").exists()
    # the prediction was frozen before the judged rounds and the judged plan spans both nodes
    t_frozen = json.loads((out / "predictions" / "frozen.json").read_text())["t_frozen"]
    assert t_frozen < json.loads((out / "judged" / "r1" / "meta.json").read_text())["t_start"]
    spans = subprocess.run([sys.executable, str(SCRIPTS / "phase6_exit_s_summary.py"), "spans", str(out / "plans" / "judged-plan.toml")],
                           capture_output=True, text=True).stdout
    assert spans.split() == ["a", "0..47", "b", "47..94"]


@needs_posix
@pytest.mark.parametrize("scenario", ["fit_band_miss", "seed_anchors"])
def test_case_s_a_fit_band_over_the_limit_or_fixed_seed_anchors_freeze_nothing(tmp_path, scenario):
    r1, r2, out = session_s(tmp_path, scenario)
    assert r2 is None and r1.returncode != 0, text(r1)[-2000:]
    assert "nothing frozen" in text(r1)
    assert not (out / "predictions" / "FROZEN.sha256").exists() and not list((out / "judged").glob("r*"))


@needs_posix
def test_case_s_no_plan_on_both_nodes_stops_at_the_pick(tmp_path):
    r1, r2, out = session_s(tmp_path, "single_node")
    assert r2 is None and r1.returncode != 0
    assert "no plan that computes on both nodes" in text(r1) and not (out / "calib" / "a" / "pred.json").exists()


@needs_posix
def test_case_s_refuses_to_judge_on_a_frozen_hash_that_does_not_match_or_a_changed_file(tmp_path):
    try:
        r1, out = run_s(tmp_path, "gates setup budget plan calib predict")
        assert r1.returncode == 0, text(r1)[-2500:]
        r2, _ = run_s(tmp_path, "judged", FROZEN_SHA="0" * 64)
        assert r2.returncode != 0 and "is not the frozen hash" in text(r2)
        assert not list((out / "judged").glob("r*"))
        # a frozen file edited after the freeze
        plan = out / "plans" / "judged-plan.toml"
        plan.write_text(plan.read_text() + "# edited\n")
        r3, _ = run_s(tmp_path, "judged", FROZEN_SHA=frozen_sha(out))
        assert r3.returncode != 0 and "frozen file changed" in text(r3)
        assert not list((out / "judged").glob("r*"))
    finally:
        kill_all(tmp_path)


@needs_posix
def test_case_s_early_aa_outside_the_band_aborts_after_round_2(tmp_path):
    r1, r2, out = session_s(tmp_path, "", "aa_early_miss")
    assert r2 is not None and r2.returncode != 0
    assert "early A/A outside the band" in text(r2)
    assert sorted(p.name for p in (out / "judged").glob("r*")) == ["r1", "r2"] and not (out / "judged" / "aa-last").exists()


@needs_posix
def test_case_s_aa_last_outside_the_band_gives_no_verdict(tmp_path):
    r1, r2, out = session_s(tmp_path, "", "aa_last_miss")
    v = verdict_s(out)
    assert v["verdict"] == "no verdict" and failed_rules(v) == ["aa"]


@needs_posix
def test_case_s_small_message_drift_is_reported_not_gated(tmp_path):
    r1, r2, out = session_s(tmp_path, "small_drift")
    v = verdict_s(out)
    assert v["verdict"] == "pass" and failed_rules(v) == [], v["status"]
    assert abs(v["link_reprice"]["pp_move"]) < 0.08
    raw = [r for x in v["link"] for r in x["points"]]
    assert any(not r["ok"] for r in raw)  # the raw 16 KiB points are out of the old band and reported
    assert (out / "link" / "probe-r1.json").exists() and (out / "link" / "probe-aalast.json").exists()  # a probe before every round


@needs_posix
def test_case_s_a_link_that_moves_the_prediction_past_0_08_is_not_clean(tmp_path):
    r1, r2, out = session_s(tmp_path, "link_move")
    v = verdict_s(out)
    assert v["verdict"] == "no verdict" and failed_rules(v) == ["link_reprice"]


@needs_posix
def test_case_s_frozen_passes_but_end_priced_fails_is_no_verdict(tmp_path):
    r1, r2, out = session_s(tmp_path, "end_fails")
    v = verdict_s(out)
    assert failed_rules(v) == [] and v["verdict"] == "no verdict" and abs(v["e_pp"]) <= 0.15 < abs(v["e_pp_end"])


@needs_posix
def test_case_s_unclean_rounds_are_rerun_and_all_reported(tmp_path):
    r1, r2, out = session_s(tmp_path, "unclean_rounds")
    v = verdict_s(out)
    assert [r["clean"] for r in v["rounds"]] == [False, True, False, True, True], v["status"]
    assert v["verdict"] == "pass" and r2.returncode == 0


@needs_posix
def test_case_s_a_wrong_prompt_runs_eight_rounds_and_gives_no_verdict(tmp_path):
    r1, r2, out = session_s(tmp_path, "wrong_prompt")
    v = verdict_s(out)
    assert len(v["rounds"]) == 8 and not any(r["clean"] for r in v["rounds"])
    assert v["verdict"] == "no verdict" and "rounds_clean" in failed_rules(v) and "prompt_hash" in failed_rules(v)


@pytest.mark.skipif(os.name == "nt", reason="bash -n runs on Linux; the Windows runner has only the WSL stub")
def test_case_s_scripts_parse_and_run_the_judged_round_and_the_merge_through_the_binary():
    for f in ("phase6_exit_s.sh", "phase6_exit_lib.sh", "sc_box.sh"):
        assert subprocess.run(["bash", "-n", str(SCRIPTS / f)]).returncode == 0
    body = (SCRIPTS / "phase6_exit_s.sh").read_text()
    # the judged round and the graph merge are `tetraeder` subcommands, checked in Rust (crates/tetraeder)
    assert "cluster judged-round" in body and "cluster merge" in body and "--ram-budget" in body
    assert "JUDGED_ROUND_CMD" not in body and "phase6_exit_s_merge" not in body and "--builder-disk" not in body
    for name in ("sc_box.sh", "sc_gates.py", "fetch_exit_model.sh", "phase6_exit_s_summary.py", "phase6_loopback.py",
                 "phase6_box.sh"):
        assert name in body and (SCRIPTS / name).exists(), name


def test_the_pick_must_partition_the_layers_one_stage_per_node_leader_first(tmp_path):
    sys.path.insert(0, str(SCRIPTS))
    import phase6_exit_s_summary as s6

    def plan(*stages):
        body = "[[replicas]]\n" + "".join(f'[[replicas.stages]]\nlayers = [{lo}, {hi}]\ndevices = ["{n}/gpu0"]\n' for n, lo, hi in stages)
        f = tmp_path / "p.toml"
        f.write_text(body)
        return str(f)

    assert s6.check_partition(plan(("a", 0, 40), ("b", 40, 94)), 94) == []  # another cut than the confirmed 47 is a valid pick
    assert s6.check_partition(plan(("b", 0, 47), ("a", 47, 94)), 94)  # the leader holds the first layers
    assert s6.check_partition(plan(("a", 0, 47), ("b", 48, 94)), 94)  # a hole
    assert s6.check_partition(plan(("a", 0, 47), ("b", 47, 90)), 94)  # not the whole model
    assert s6.check_partition(plan(("a", 0, 94)), 94)  # one node
