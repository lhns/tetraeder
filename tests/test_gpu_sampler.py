"""scripts/idle_gate.sh and scripts/gpu_sampler.sh with a fake nvidia-smi on PATH (ADR 0065 addendum)."""

import os
import pathlib
import shutil
import subprocess
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="needs bash on Linux (runs on the ubuntu CI job)"
)


def fake_bin(tmp_path, reasons, sampler=False):
    """A bin dir with an nvidia-smi that prints the next line of `reasons` per call (the last one repeats) and, for
    `-lms`, a CSV line per 0.2 s in the sampler's query format. flock is faked when the box lacks it."""
    b = tmp_path / "bin"
    b.mkdir()
    (tmp_path / "reasons").write_text("\n".join(reasons) + "\n")
    smi = b / "nvidia-smi"
    smi.write_text(
        f"""#!/usr/bin/env bash
case "$*" in
  *-lms*)
    while true; do echo "2026/11/17 10:00:00.123, 0x0000000000000404, 160.5, 1410, 9001, 45, 1"; sleep 0.2; done ;;
  *)
    f={tmp_path.as_posix()}/reasons; n={tmp_path.as_posix()}/n; i=$(cat $n 2>/dev/null || echo 0); echo $((i + 1)) > $n
    sed -n "$((i + 1))p" $f | grep . || tail -n1 $f ;;
esac
"""
    )
    smi.chmod(0o755)
    if shutil.which("flock") is None:
        (b / "flock").write_text("#!/usr/bin/env bash\nexit 0\n")
        (b / "flock").chmod(0o755)
    return b


def gate(tmp_path, reasons):
    b = fake_bin(tmp_path, reasons)
    env = {**os.environ, "PATH": f"{b}{os.pathsep}{os.environ['PATH']}", "INTERVAL": "0"}
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / "idle_gate.sh")], capture_output=True, text=True, env=env, timeout=60
    )


def test_idle_gate_passes_with_nine_of_ten(tmp_path):
    r = gate(tmp_path, ["0x0000000000000001"] * 3 + ["0x0000000000000004", "0x0000000000000600"] + ["0x8"] + ["0x0000000000000003"] * 4)
    assert r.returncode == 0, r.stdout
    assert "9/10" in r.stdout and r.stdout.count("sample ") == 10 and "BAD" in r.stdout


def test_idle_gate_fails_with_eight_of_ten(tmp_path):
    r = gate(tmp_path, ["0x0"] * 3 + ["0x0000000000000020", "0x8"] + ["0x0"] * 5)
    assert r.returncode == 1 and "8/10" in r.stdout


def test_idle_gate_fails_without_nvidia_smi_values(tmp_path):
    r = gate(tmp_path, [""])
    assert r.returncode == 1


def test_sampler_writes_csv_and_stops_on_the_stop_file(tmp_path):
    b = fake_bin(tmp_path, ["0x0"])
    out = tmp_path / "out"
    env = {**os.environ, "PATH": f"{b}{os.pathsep}{os.environ['PATH']}", "OUT": str(out), "SAMPLER_LOCK": str(tmp_path / "lock")}
    p =subprocess.Popen(["bash", str(ROOT / "scripts" / "gpu_sampler.sh")], env=env)
    csv = out / "gpu-samples.csv"
    for _ in range(50):
        if csv.exists() and csv.read_text().count("\n") >= 2:
            break
        time.sleep(0.2)
    (out / "gpu-sampler.stop").touch()
    assert p.wait(timeout=20) == 0
    first = csv.read_text().splitlines()[0].split(",")
    assert len(first) == 8 and float(first[0]) > 1e9
    assert first[1] == "2026/11/17 10:00:00.123" and first[2].strip() == "0x0000000000000404"
    assert first[7] == "1"
