"""scripts/s3f_confirm_box.sh in DRY_RUN=1 mode: stage A command and the stage B run list (ADR 0081, s3f confirmation)."""

import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ("qwen3-30b-a3b", "qwen3-32b")
ARM_ORDER = ["default-r1", "nograph-r1", "nograph-r2", "default-r2"]


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None, reason="needs bash on Linux")
def test_dry_run_lists_stage_a_and_the_interleaved_arms(tmp_path):
    env = {**os.environ, "DRY_RUN": "1", "SESSION": "s1", "OUT_DIR": str(tmp_path)}
    r = subprocess.run(["bash", str(ROOT / "scripts" / "s3f_confirm_box.sh")], capture_output=True, text=True,
                       env=env, cwd=ROOT, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    lines = r.stdout.splitlines()
    stage_a = [ln for ln in lines if "phase5_box.sh" in ln and ln.startswith("PLAN ")]
    assert len(stage_a) == 1
    assert "NGPUS=4" in stage_a[0] and "ROUNDS=1" in stage_a[0] and "BUDGET_MIN=50" in stage_a[0]
    assert stage_a[0].endswith("phase5_box.sh s1 4")
    runs = [ln for ln in lines if ln.startswith("PLAN env") and ln.endswith(".json")]
    by = {re.search(r"> (\S+)\.json$", ln).group(1): ln for ln in runs}
    assert len(by) == len(runs) == 16
    names = list(by)
    expect = [f"{m}-{c}-mb4-{a}" for m in MODELS for c in ("tp2pp2", "layer4") for a in ARM_ORDER]
    assert names == expect
    # stage A comes before every stage B run
    assert lines.index(stage_a[0]) < lines.index(next(ln for ln in lines if ln.startswith("PLAN env")))
    for name, ln in by.items():
        assert "-p 2048 -n 128 -r 5 " in ln
        assert ("GGML_CUDA_DISABLE_GRAPHS=1" in ln) == ("-nograph-" in name)
        # no debug env and no trace in a timed run
        assert "GRAPH_DEBUG" not in ln and "EXEC_TRACE" not in ln
        plan = "tp2-pp2" if "tp2pp2" in name else "layer-4"
        model = name.split("-mb4")[0].rsplit("-", 1)[0]
        assert f"/plans/{model}-{plan}-mb4.toml " in ln
        assert ("--engine moe " in ln) == (model == "qwen3-30b-a3b")
    assert any("prefill_micro_batches = 4" in ln for ln in lines)
    assert lines[-1].endswith("S3F_CONFIRM_DONE rc=0")
