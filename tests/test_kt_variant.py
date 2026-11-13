"""KT_KERNEL_CPU_VARIANT forcing and the KT thread headroom (m1-open point 3, 20261113d).

kt-kernel v0.7.2 accepts amx, avx512_bf16, avx512_vbmi, avx512_vnni, avx512_base and avx2 and silently ignores any
other value, so the "avx512" stage must pass a real name.
"""

import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs a POSIX userland (Linux)")


def sh(*args, env=None):
    e = {**os.environ, **(env or {})}
    return subprocess.run(["bash", *args], cwd=ROOT, env=e, capture_output=True, text=True)


def dry(stages, **env):
    r = sh("scripts/exit_run.sh", env={"DRY_RUN": "1", "STAGES": stages, **env})
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_avx512_stage_forces_a_real_variant():
    out = dry("ktransformers ktransformers-repeat ktransformers-avx2 ktransformers-avx512 ktransformers-avx512-repeat")
    assert "ktransformers forces KT_KERNEL_CPU_VARIANT=\n" in out  # own pick, nothing forced
    assert "ktransformers-repeat forces KT_KERNEL_CPU_VARIANT=\n" in out
    assert "ktransformers-avx2 forces KT_KERNEL_CPU_VARIANT=avx2\n" in out
    assert "ktransformers-avx512 forces KT_KERNEL_CPU_VARIANT=avx512_vbmi\n" in out
    assert "ktransformers-avx512-repeat forces KT_KERNEL_CPU_VARIANT=avx512_vbmi\n" in out


def test_avx512_variant_override():
    out = dry("ktransformers-avx512", KT_AVX512_VARIANT="avx512_bf16")
    assert "ktransformers-avx512 forces KT_KERNEL_CPU_VARIANT=avx512_bf16\n" in out


def test_unknown_variant_fails_before_the_run():
    r = sh("scripts/ktransformers_exit.sh", "m.gguf", "8", "1", "out.json", env={"KT_KERNEL_CPU_VARIANT": "avx512"})
    assert r.returncode == 2
    assert "not a kt-kernel variant" in r.stderr


@pytest.mark.parametrize(
    ("stages", "quota", "avx512", "want"),
    [
        ("ktransformers", "30", "1", "26"),  # 20261113d: 28 threads on 30.7 cores were throttled
        ("main-overlap ktransformers-avx512", "30", "1", "26"),
        ("main-overlap", "30", "1", "30"),  # no KT stage
        ("ktransformers", "30", "0", "30"),  # no AVX-512
        ("ktransformers", "4", "1", "4"),  # too small to give up cores
    ],
)
def test_kt_thread_headroom(stages, quota, avx512, want):
    r = sh("scripts/kt_threads.sh", stages, quota, avx512)
    assert r.stdout.strip() == want
