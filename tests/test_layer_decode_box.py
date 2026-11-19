"""scripts/layer_decode_box.sh in DRY_RUN=1 mode (the run list and its toggles) and scripts/exec_trace_summary.py
(ADR 0069 History 2026-11-19)."""

import importlib.util
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("exec_trace_summary", ROOT / "scripts" / "exec_trace_summary.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None, reason="needs bash on Linux")
def test_dry_run_lists_the_pairs_and_toggles(tmp_path):
    env = {**os.environ, "DRY_RUN": "1", "SESSION": "s1", "OUT_DIR": str(tmp_path)}
    r = subprocess.run(["bash", str(ROOT / "scripts" / "layer_decode_box.sh")], capture_output=True, text=True,
                       env=env, cwd=ROOT, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    plans = [ln for ln in r.stdout.splitlines() if ln.startswith("PLAN ")]
    names = [re.search(r"> (\S+)\.json$", ln).group(1) for ln in plans]
    m = "qwen3-30b-a3b"
    assert names[:2] == [f"{m}-llama-none-r1", f"{m}-llama-layer-r1"]
    # round 2 of the 30B is round 1's untraced runs reversed
    r1 = [n[:-3] for n in names if n.startswith(m) and n.endswith("-r1") and "trace" not in n]
    r2 = [n[:-3] for n in names if n.startswith(m) and n.endswith("-r2")]
    assert r2 == r1[::-1]
    assert any(n.startswith("qwen3-32b-") for n in names)
    by = dict(zip(names, plans))
    both = by[f"{m}-layer4-uid-async-r1"]
    assert "TETRAEDER_RANGE_UID=1" in both and "TETRAEDER_ASYNC_INPUTS=1" in both and "layer-4.toml" in both
    base = by[f"{m}-layer4-r1"]
    assert "TETRAEDER_" not in base
    sync = by[f"{m}-layer4-tracesync-r1"]
    assert "TETRAEDER_EXEC_TRACE=" in sync and "TETRAEDER_EXEC_TRACE_SYNC=1" in sync
    assert "-sm none" in by[f"{m}-llama-none-r1"]
    assert all("-p 0 -n 128" in p for p in plans)


def test_summary_takes_the_most_common_sequence():
    decode = '{"step":%d,"sync":false,"ev":[["upload",0,-1,%d],["compute",0,-1,3000],["copy",0,1,1000],' \
             '["compute",1,-1,2000],["sync",1,-1,5000],["logits",1,-1,4000]]}'
    lines = ['{"step":0,"sync":false,"ev":[["compute",0,-1,99999]]}']
    lines += [decode % (i + 1, 1000 + 1000 * i) for i in range(3)]
    s = summary.summarize(lines)
    assert s["steps"] == 3
    assert [e["kind"] for e in s["events"]] == ["upload", "compute", "copy", "compute", "sync", "logits"]
    assert s["events"][0]["us"] == 2.0
    assert s["kinds"]["compute"] == 5.0
    assert s["total_us"] == 2.0 + 3.0 + 1.0 + 2.0 + 5.0 + 4.0
    assert summary.summarize([])["steps"] == 0
