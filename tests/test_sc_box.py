"""Fixes after segment-confirm-20261125b: the driver's disk, the fetch patterns for the frozen prediction, the gates stage
of scripts/sc_box.sh and its early A/A abort with the DONE file (ADR 0092). Hardware label: none (fake nvidia-smi, fixture JSON)."""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from test_sc_gates import failed, make_session, summary  # noqa: E402

posix = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs a POSIX userland")


def test_the_fetch_brings_back_the_frozen_prediction_files():
    inc = tomllib.loads((ROOT / "taxameter.toml").read_text(encoding="utf-8"))["results"]["include"]
    for pat in ("*.toml", "*.sha256", "SHA256SUMS", "DONE"):  # hw.toml, fitted.toml, FROZEN.sha256, trace/*.sha256
        assert pat in inc, pat


@posix
def test_the_driver_sets_the_box_disk_for_the_235b(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "uv").write_text('#!/usr/bin/env bash\necho "uv $*" > "$UV_LOG"\n')
    (fake / "uv").chmod(0o755)
    work = tmp_path / "work"
    (work / ".vast").mkdir(parents=True)
    (work / ".vast" / "config.json").write_text(json.dumps({"keep": 1, "disk_gb": 80}))
    env = {**os.environ, "PATH": f"{fake}{os.pathsep}{os.environ['PATH']}", "UV_LOG": str(tmp_path / "uv.log")}
    for cmd in ("offers", "create"):
        r = subprocess.run(["bash", str(SCRIPTS / "sc_offers.sh"), cmd], cwd=work, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert json.loads((work / ".vast" / "config.json").read_text()) == {"keep": 1, "disk_gb": 230}
        assert f"taxameter {cmd} --download-gb 150" in (tmp_path / "uv.log").read_text()
    r = subprocess.run(["bash", str(SCRIPTS / "sc_offers.sh"), "create"], cwd=work, env={**env, "DISK_GB": "80"},
                       capture_output=True, text=True)
    assert r.returncode == 2


def box(tmp_path, stages, extra_env=None):
    """Run sc_box.sh in a temp results root with the engine runs, the freeze gate and the sampler replaced."""
    ov = tmp_path / "overrides.sh"
    ov.write_text('gpu_sampler_start() { :; }\nfreeze_gate() { :; }\njudged_env() { :; }\nrun_one() { echo "$1" >> "$CALLS"; }\n')
    tax = tmp_path / "tax"
    tax.mkdir(exist_ok=True)
    (tax / "cuda_env.sh").write_text("")
    env = {**os.environ, "STAGES": stages, "RESULTS_NAME": "s", "RESULTS_ROOT": str(tmp_path / "res"),
           "SC_LOG": str(tmp_path / "sc.log"), "SC_LOCK": str(tmp_path / "lock"), "SC_BOX_OVERRIDES": str(ov),
           "TAXAMETER_HOME": str(tax), "PROMPT_FILE": "", "CALLS": str(tmp_path / "calls.txt"), **(extra_env or {})}
    r = subprocess.run(["bash", str(SCRIPTS / "sc_box.sh")], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    calls = (tmp_path / "calls.txt").read_text().split() if (tmp_path / "calls.txt").exists() else []
    return r, calls, tmp_path / "res" / "s", (tmp_path / "sc.log").read_text()


def judged_session(tmp_path, r2_pp):
    """A frozen session with judged r1 and r2 in place (the replaced run_one keeps them) and no r3 or A/A last."""
    src, _ = make_session(tmp_path / "src", rounds=[{}, {"pp": r2_pp}], aa_last=None)
    shutil.copytree(src, tmp_path / "res" / "s")


@posix
def test_the_early_aa_outside_the_band_stops_the_session_after_round_2(tmp_path):
    judged_session(tmp_path, r2_pp=100.0)  # r1 190 tok/s prefill: T differs far beyond 0.08
    r, calls, out, log = box(tmp_path, "judged aalast")
    assert r.returncode == 1, r.stdout + r.stderr
    assert calls == ["judged-r1", "judged-r2"]
    assert "early A/A outside the band" in log and "OUTSIDE THE BAND" in (out / "judged" / "aa-early.txt").read_text()
    assert "SC_BOX_DONE rc=1" in log and "SC_BOX_DONE rc=0" not in log
    assert (out / "DONE").read_text().startswith("SC_BOX_DONE rc=1")
    v = summary(out)
    assert v["verdict"] == "no verdict" and "aa_early" in failed(v)


@posix
def test_a_pass_through_the_early_aa_runs_every_round_and_writes_done(tmp_path):
    judged_session(tmp_path, r2_pp=190.0)
    r, calls, out, log = box(tmp_path, "judged")
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls == ["judged-r1", "judged-r2", "judged-r3"]
    assert "SC_BOX_DONE rc=0" in log
    assert (out / "DONE").read_text().startswith("SC_BOX_DONE rc=0 stages='judged'")


FAKE_SMI = """#!/usr/bin/env bash
case "$*" in
  *power.limit*) echo "0, $POWER_LIMIT, 350.00, 400.00" ;;
  *clocks_event_reasons*) echo "$REASONS" ;;
esac
"""


@posix
@pytest.mark.parametrize("limit,reasons,why", [("250.00", "0x0", "default power limit"), ("350.00", "0x20", "idle gate")])
def test_the_gates_stage_runs_the_power_and_idle_gates_and_fails_the_session(tmp_path, limit, reasons, why):
    smi = tmp_path / "bin" / "nvidia-smi"
    smi.parent.mkdir()
    smi.write_text(FAKE_SMI)
    smi.chmod(0o755)
    r, calls, out, log = box(tmp_path, "gates", {
        "PATH": f"{smi.parent}{os.pathsep}{os.environ['PATH']}", "NVIDIA_SMI": str(smi), "POWER_LIMIT": limit,
        "REASONS": reasons, "SAMPLES": "2", "INTERVAL": "0"})
    assert r.returncode == 1, r.stdout + r.stderr
    assert "SC_BOX_DONE rc=1" in log and why in log
    assert (out / "DONE").read_text().startswith("SC_BOX_DONE rc=1")
    assert not (out / "env" / "cotenant.json").exists()  # stopped before the NUMA and co-tenant checks


def test_the_gates_stage_has_both_gates_before_the_other_checks_and_done_is_written():
    s = (SCRIPTS / "sc_box.sh").read_text(encoding="utf-8")
    g = s[s.index("st_gates() {"):s.index("st_env() {")]
    assert g.index("power_gate.sh") < g.index("idle_gate.sh") < g.index("lscpu")
    assert 'rc=0 stages=' in s and '> "$out/DONE"' in s
