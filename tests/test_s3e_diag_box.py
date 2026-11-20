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
    assert len(by) == len(plans) == 40
    clocks = [re.search(r"> (\S+)\.clocks\.csv$", ln).group(1) for ln in r.stdout.splitlines()
              if ln.startswith("PLAN nvidia-smi ") and "-lms 20" in ln]
    # part C: 30B only, after both models' arms; each run has a clock log started right before it
    part_c = [f"{MODELS[0]}-{p}{lim}-tg128" for p in ("single", "layer-4") for lim in ("-L12", "-L24", "")]
    assert clocks == part_c
    names = list(by)
    assert names[-6:] == part_c
    arms = names[:-6]
    assert arms[0].startswith(MODELS[0]) and arms.index(f"{MODELS[1]}-tp2pp2-mb4-pp-nccl-nograph") > max(
        i for i, k in enumerate(arms) if k.startswith(MODELS[0]))
    for m in MODELS:
        # the decisive pair first, then varykey (mb4, NCCL), graphdebug, eager
        order = [k for k in arms if k.startswith(m)]
        kinds = [k.rsplit("-", 1)[-1] if "varykey" not in k else "varykey" for k in order]
        assert kinds == ["nograph"] * 4 + ["default"] * 4 + ["varykey"] * 5 + ["graphdebug"] * 2 + ["eager"] * 2
        for mb in (4, 1):
            for ar in ("nccl", "internal"):
                base = f"{m}-tp2pp2-mb{mb}-pp-{ar}"
                for arm in ("nograph", "default"):
                    ln = by[f"{base}-{arm}"]
                    assert f"tp2-pp2-mb{mb}.toml" in ln and "-p 2048 -n 0 -r 5 " in ln
                    assert f"TETRAEDER_EXEC_TRACE={tmp_path}/{base}-{arm}.trace.jsonl" in ln
                    assert ("GGML_CUDA_DISABLE_GRAPHS=1" in ln) == (arm == "nograph")
                    assert ("GGML_CUDA_ALLREDUCE=internal" in ln) == (ar == "internal")
                    assert "CUDA_MODULE_LOADING" not in ln
                    # only the very first nograph run (30B, mb4, NCCL) also logs graph events: the graphs-off proof
                    first = (m, mb, ar, arm) == (MODELS[0], 4, "nccl", "nograph")
                    assert ("GRAPH_DEBUG" in ln) == first
        for p in LENGTHS:
            ln = by[f"{m}-tp2pp2-mb4-pp-nccl-varykey-{p}"]
            assert f"-p {p} -n 0 -r 1 " in ln and "tp2-pp2-mb4.toml" in ln and "TETRAEDER_EXEC_TRACE=" in ln
            assert "ALLREDUCE" not in ln and "DISABLE_GRAPHS" not in ln
        for mb in (4, 1):
            dbg = by[f"{m}-tp2pp2-mb{mb}-pp-nccl-graphdebug"]
            assert "GGML_CUDA_GRAPH_DEBUG=1" in dbg and "-r 5 " in dbg and "DISABLE_GRAPHS" not in dbg
            eager = by[f"{m}-tp2pp2-mb{mb}-pp-nccl-eager"]
            assert "CUDA_MODULE_LOADING=EAGER" in eager and "NCCL_RUNTIME_CONNECT=0" in eager
            assert "ALLREDUCE" not in dbg + eager
    # the probe variables reach no other run
    assert [k for k in by if "varykey" in k and ("mb1" in k or "internal" in k)] == []
    for k, v in by.items():
        if k not in arms:
            continue
        assert ("GRAPH_DEBUG" in v) == (k.endswith("graphdebug") or k == f"{MODELS[0]}-tp2pp2-mb4-pp-nccl-nograph")
        assert ("NCCL_RUNTIME_CONNECT" in v) == k.endswith("eager")
    # only tp2-pp2 pp: no decode, no sync trace, no other plan
    assert all("-n 0 " in v and "TETRAEDER_EXEC_TRACE_SYNC" not in v and "-pp-" in k for k, v in by.items() if k in arms)
    for k in part_c:
        v = by[k]
        assert "-p 0 -n 128 -r 5 " in v and "TETRAEDER_EXEC_TRACE" not in v and "GGML_CUDA" not in v
        plan = k.split("-tg128")[0].removeprefix(f"{MODELS[0]}-").split("-L")[0]
        assert f"/plans/{MODELS[0]}-{plan}.toml " in v
        if "-L" in k:
            assert f"--layer-limit {k.split('-L')[1].split('-')[0]} " in v
        assert ("--layer-limit" in v) == ("-L" in k)
