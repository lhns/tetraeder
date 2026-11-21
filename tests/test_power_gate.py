"""scripts/power_gate.sh with a fake nvidia-smi, and its place in phase5_box.sh (ADR 0068, "Power limit")."""

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GATE = ROOT / "scripts" / "power_gate.sh"
BOX = ROOT / "scripts" / "phase5_box.sh"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="needs bash on Linux (runs on the ubuntu CI job)"
)


def gate(tmp_path, output, rc=0):
    """Run the gate against a fake nvidia-smi that prints `output` and exits with `rc`."""
    smi = tmp_path / "nvidia-smi"
    smi.write_text(f"#!/usr/bin/env bash\ncat <<'EOF'\n{output}\nEOF\nexit {rc}\n")
    smi.chmod(0o755)
    r = subprocess.run(["bash", str(GATE)], capture_output=True, text=True, cwd=ROOT, timeout=60,
                       env={**os.environ, "NVIDIA_SMI": str(smi)})
    return r.returncode, r.stdout


def test_default_limits_pass(tmp_path):
    rc, out = gate(tmp_path, "0, 350.00, 350.00, 400.00\n1, 450.00, 450.00, 450.00")
    assert rc == 0, out
    assert "limit 350.00 W, default 350.00 W, max 400.00 W ok" in out


def test_within_one_watt_passes(tmp_path):
    assert gate(tmp_path, "0, 349.50, 350.00, 400.00")[0] == 0


def test_lowered_limit_fails(tmp_path):
    rc, out = gate(tmp_path, "0, 250.00, 350.00, 400.00\n1, 450.00, 450.00, 450.00")
    assert rc == 1
    assert "LIMITED" in out and "1 of 2 GPUs" in out


def test_empty_output_fails(tmp_path):
    assert gate(tmp_path, "")[0] == 1
    assert gate(tmp_path, "", rc=9)[0] == 1


def test_unparsable_values_fail(tmp_path):
    assert gate(tmp_path, "0, [N/A], [N/A], [N/A]")[0] == 1


def test_phase5_box_stops_at_gpus_on_lowered_power(tmp_path):
    e = {**os.environ, "DRY_RUN": "1", "DRY_POWER": "fail", "P5_OUT": str(tmp_path / "out"), "THREADS": "8",
         "PY": sys.executable}
    r = subprocess.run(["bash", str(BOX), "s1", "2"], capture_output=True, text=True, env=e, cwd=ROOT, timeout=300)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "STAGE env" not in r.stdout and "power limit below default" in r.stdout
    assert "PHASE5_BOX_DONE rc=1" in r.stdout


def test_phase5_box_env_stage_records_power_fields():
    s = BOX.read_text()
    for k in ("power_limit_w", "power_default_limit_w", "power_max_limit_w"):
        assert f'"{k}"' in s


def gate_for(tmp_path, output, devices):
    smi = tmp_path / "nvidia-smi"
    smi.write_text(f"#!/usr/bin/env bash\ncat <<'EOF'\n{output}\nEOF\n")
    smi.chmod(0o755)
    r = subprocess.run(["bash", str(GATE), devices], capture_output=True, text=True, cwd=ROOT, timeout=60,
                       env={**os.environ, "NVIDIA_SMI": str(smi)})
    return r.returncode, r.stdout


FOUR = "0, 250.00, 350.00, 400.00\n1, 350.00, 350.00, 400.00\n2, 350.00, 350.00, 400.00\n3, 350.00, 350.00, 400.00"


def test_two_of_four_judges_only_the_chosen_pair(tmp_path):
    rc, out = gate_for(tmp_path, FOUR, "2,3")
    assert rc == 0, out
    assert "gpu 0: limit 250.00 W, default 350.00 W, max 400.00 W (not judged)" in out
    assert "2 GPUs at default power limit" in out
    assert gate(tmp_path, FOUR)[0] == 1  # without a pair every GPU counts: unchanged


def test_two_of_four_fails_on_a_limited_gpu_of_the_pair(tmp_path):
    rc, out = gate_for(tmp_path, FOUR, "0,3")
    assert rc == 1 and "1 of 2 GPUs" in out


def test_two_of_four_fails_when_the_pair_is_not_listed(tmp_path):
    assert gate_for(tmp_path, FOUR, "4,5")[0] == 1
