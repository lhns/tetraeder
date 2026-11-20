"""scripts/s3e_diag_box.sh in DRY_RUN=1 mode: the run list and the environment per arm (ADR 0080, s3e diagnostic)."""

import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS = ("qwen3-30b-a3b", "qwen3-32b")
LENGTHS = (2048, 2040, 2032, 2024, 2016)


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None, reason="needs bash on Linux")
def test_dry_run_lists_the_arms(tmp_path):
    env = {**os.environ, "DRY_RUN": "1", "SESSION": "s1", "OUT_DIR": str(tmp_path)}
    r = subprocess.run(["bash", str(ROOT / "scripts" / "s3e_diag_box.sh")], capture_output=True, text=True,
                       env=env, cwd=ROOT, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    plans = [ln for ln in r.stdout.splitlines() if ln.startswith("PLAN ") and ln.endswith(".json")]
    by = {re.search(r"> (\S+)\.json$", ln).group(1): ln for ln in plans}
    assert len(by) == len(plans) == 64
    names = list(by)
    assert names[0].startswith(MODELS[0]) and names.index(f"{MODELS[1]}-tp2pp2-mb4-pp-nccl-nograph") > max(
        i for i, k in enumerate(names) if k.startswith(MODELS[0]))
    for m in MODELS:
        # the decisive pair first, then varykey, graphdebug, eager
        order = [k for k in names if k.startswith(m)]
        kinds = [k.rsplit("-", 1)[-1] if "varykey" not in k else "varykey" for k in order]
        assert kinds == ["nograph"] * 4 + ["default"] * 4 + ["varykey"] * 20 + ["graphdebug"] * 2 + ["eager"] * 2
        for mb in (4, 1):
            for ar in ("nccl", "internal"):
                base = f"{m}-tp2pp2-mb{mb}-pp-{ar}"
                for arm in ("nograph", "default"):
                    ln = by[f"{base}-{arm}"]
                    assert f"tp2-pp2-mb{mb}.toml" in ln and "-p 2048 -n 0 -r 5 " in ln
                    assert f"TETRAEDER_EXEC_TRACE={tmp_path}/{base}-{arm}.trace.jsonl" in ln
                    assert ("GGML_CUDA_DISABLE_GRAPHS=1" in ln) == (arm == "nograph")
                    assert ("GGML_CUDA_ALLREDUCE=internal" in ln) == (ar == "internal")
                    assert "GRAPH_DEBUG" not in ln and "CUDA_MODULE_LOADING" not in ln
                for p in LENGTHS:
                    ln = by[f"{base}-varykey-{p}"]
                    assert f"-p {p} -n 0 -r 1 " in ln and "TETRAEDER_EXEC_TRACE=" in ln
                    assert ("GGML_CUDA_ALLREDUCE=internal" in ln) == (ar == "internal")
                    assert "DISABLE_GRAPHS" not in ln
        for mb in (4, 1):
            dbg = by[f"{m}-tp2pp2-mb{mb}-pp-nccl-graphdebug"]
            assert "GGML_CUDA_GRAPH_DEBUG=1" in dbg and "-r 5 " in dbg and "DISABLE_GRAPHS" not in dbg
            eager = by[f"{m}-tp2pp2-mb{mb}-pp-nccl-eager"]
            assert "CUDA_MODULE_LOADING=EAGER" in eager and "NCCL_RUNTIME_CONNECT=0" in eager
            assert "ALLREDUCE" not in dbg + eager
    # the probe variables reach no other run
    for k, v in by.items():
        assert ("GRAPH_DEBUG" in v) == k.endswith("graphdebug")
        assert ("NCCL_RUNTIME_CONNECT" in v) == k.endswith("eager")
    # only tp2-pp2 pp: no decode, no sync trace, no other plan
    assert all("-n 0 " in v and "TETRAEDER_EXEC_TRACE_SYNC" not in v and "-pp-" in k for k, v in by.items())
