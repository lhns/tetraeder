"""scripts/s3d_diag_box.sh in DRY_RUN=1 mode: the run list, trace modes and clock logs (ADR 0080, s3d diagnostic)."""

import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None, reason="needs bash on Linux")
def test_dry_run_lists_the_diagnostic_runs(tmp_path):
    env = {**os.environ, "DRY_RUN": "1", "SESSION": "s1", "OUT_DIR": str(tmp_path)}
    r = subprocess.run(["bash", str(ROOT / "scripts" / "s3d_diag_box.sh")], capture_output=True, text=True,
                       env=env, cwd=ROOT, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    plans = [ln for ln in r.stdout.splitlines() if ln.startswith("PLAN ") and ln.endswith(".json")]
    by = {re.search(r"> (\S+)\.json$", ln).group(1): ln for ln in plans}
    for m in ("qwen3-30b-a3b", "qwen3-32b"):
        for mb in (1, 4):
            host = by[f"{m}-tp2pp2-mb{mb}-pp-trace"]
            dev = by[f"{m}-tp2pp2-mb{mb}-pp-sync"]
            assert f"tp2-pp2-mb{mb}.toml" in host and "-p 2048 -n 0" in host
            assert "TETRAEDER_EXEC_TRACE=" in host and "TETRAEDER_EXEC_TRACE_SYNC" not in host
            assert "TETRAEDER_EXEC_TRACE_SYNC=1" in dev
            assert "TETRAEDER_EXEC_TRACE" not in by[f"{m}-tp2pp2-mb{mb}-pp-plain"]
        sync = by[f"{m}-layer4-tg-sync"]
        assert "layer-4.toml" in sync and "TETRAEDER_EXEC_TRACE_SYNC=1" in sync and "-p 0 -n 128" in sync
        assert "single.toml" in by[f"{m}-single-tg-sync"]
        assert "-sm layer" in by[f"{m}-llama-layer-tg"]
    # decode toggles are left at their defaults (on)
    assert not any("TETRAEDER_RANGE_UID" in p or "TETRAEDER_ASYNC_INPUTS" in p for p in plans)
    # the 30B comes first; every part B run has a 20 ms clock log
    assert next(iter(by)).startswith("qwen3-30b-a3b")
    assert len([ln for ln in r.stdout.splitlines() if "-lms 20" in ln]) == 12
