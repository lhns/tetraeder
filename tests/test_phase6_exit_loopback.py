"""The loopback dry run of the Phase 6 exit drivers with the fake head (LOOPBACK=fake, the summary-rule fixtures): scripts/phase6_exit_f.sh end
to end with two "boxes" as two sets of processes on this machine (scripts/phase6_loopback.py in place of tetraeder; the real binaries run in
the CI step "phase 6 loopback", scripts/phase6_loopback_check.py), real fault injection through the trigger (kill -9 of the engine, kill -9 of the daemon and its
children, SIGSTOP and SIGCONT), the load, the epochs and the verdict. Hardware label: none (needs a POSIX userland with pgrep, setsid and
flock; skips cleanly without)."""

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


def run_f(tmp_path, **env):
    e = {**os.environ, "LOOPBACK": "fake", "RESULTS_NAME": "f", "RESULTS_ROOT": str(tmp_path / "res"), "LB_DIR": str(tmp_path / "lb"),
         "TRIALS": "1", **env}
    r = subprocess.run(["bash", str(SCRIPTS / "phase6_exit_f.sh")], cwd=ROOT, env=e, capture_output=True, text=True, timeout=900)
    return r, tmp_path / "res" / "f"


def kill_all(tmp_path):
    for pidfile in (tmp_path / "lb").glob("*/pids/*"):
        try:
            pid = int(pidfile.read_text().split()[0])
            os.kill(pid, 9)
        except (ValueError, OSError, IndexError):
            pass


@needs_posix
def test_case_f_runs_end_to_end_on_loopback(tmp_path):
    try:
        r, out = run_f(tmp_path)
    finally:
        kill_all(tmp_path)
    log = r.stdout + r.stderr
    v = json.loads((out / "verdict-f.json").read_text()) if (out / "verdict-f.json").exists() else None
    assert v is not None, log[-3000:]
    # three modes, one trial each, every per-trial rule holds on the stand-in head
    assert [t["mode"] for t in v["trials"]] == ["worker", "daemon", "stop"], log[-3000:]
    bad = {f"{t['mode']}.{k}": x["detail"] for t in v["trials"] for k, x in t["rules"].items() if not x["ok"]}
    assert not bad, (bad, log[-3000:])
    for t in v["trials"]:
        assert t["stats"]["live_b_at_fault"] >= 1 and t["stats"]["fault_uncertainty_s"] < 0.5  # the trigger's evidence, B's own reading
        assert t["stats"]["in_flight_errors"] >= 1 and max(t["stats"]["in_flight_error_delay_s"]) <= v["bound_s"]
        assert t["stats"]["rejoin_s"] is not None and t["stats"]["rejoin_s"] <= v["rejoin_limit_s"]
    # the dry run is not the full spec: only the load shape (14 s against 180 s) may fail among the session rules
    failed = [x["name"] for x in v["session_rules"] if not x["ok"]]
    assert failed == ["load_shape"], (failed, [x for x in v["session_rules"] if not x["ok"]])
    assert v["verdict"] == "fail" and v["reading"] == "every rule holds in every mode"
    for mode in ("worker", "daemon", "stop"):
        d = out / "trials" / f"{mode}-1"
        meta = json.loads((d / "meta.json").read_text())
        assert meta["t_restore"] is not None and meta["t_fault"] < meta["t_restore"]
        assert (d / "load.jsonl").stat().st_size > 0 and (d / "b-admits.txt").exists() and (d / "fault.log").exists()


@needs_posix
def test_a_trial_whose_b_does_not_come_back_stops_and_asks_for_a_hand(tmp_path):
    # B cannot rejoin when the engine takes longer to load than the whole wait: the driver must stop, not restart anything
    try:
        r, out = run_f(tmp_path, MODES="worker", LB_LOAD_S="60", B_UP_WAIT_S="6")
    finally:
        kill_all(tmp_path)
    assert r.returncode != 0
    assert "B is not in service" in r.stdout + r.stderr and "Nothing was restarted" in r.stdout + r.stderr
    assert not (out / "trials" / "worker-1" / "meta.json").exists()


@needs_posix
def test_note_records_a_manual_action(tmp_path):
    e = {**os.environ, "RESULTS_NAME": "n", "RESULTS_ROOT": str(tmp_path)}
    r = subprocess.run(["bash", str(SCRIPTS / "phase6_exit_f.sh"), "note", "restarted the head"], cwd=ROOT, env=e, capture_output=True, text=True)
    assert r.returncode == 0
    line = json.loads((tmp_path / "n" / "manual.jsonl").read_text().splitlines()[0])
    assert line["what"] == "restarted the head" and line["t"] > 1e9


@pytest.mark.skipif(os.name == "nt", reason="bash -n runs on Linux; the Windows runner has only the WSL stub")
def test_the_driver_scripts_parse_and_reference_existing_files():
    for f in ("phase6_exit_f.sh", "phase6_box.sh", "phase6_fault.sh"):
        assert subprocess.run(["bash", "-n", str(SCRIPTS / f)]).returncode == 0
    text = (SCRIPTS / "phase6_exit_f.sh").read_text()
    for name in ("phase6_box.sh", "phase6_fault.sh", "phase6_load.py", "phase6_cluster_toml.py", "phase6_exit_f_summary.py", "phase6_loopback.py", "sc_gates.py"):
        assert name in text and (SCRIPTS / name).exists(), name
