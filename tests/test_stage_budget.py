"""scripts/stage_budget.sh: measured minutes for the deadline guard (bash only; skipped without bash)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")


def run(code: str) -> str:
    src = (ROOT / "scripts" / "stage_budget.sh").as_posix()
    out = subprocess.run([BASH, "-c", f'. "{src}"; {code}'], capture_output=True, text=True, check=True)
    return out.stdout.strip()


@pytest.mark.parametrize(
    "default,sec,want",
    [(10, 0, 10), (10, 300, 10), (10, 540, 10), (10, 600, 11), (10, 1170, 22), (10, 1560, 29), (30, 1800, 33), (30, 600, 30)],
)
def test_budget_min(default: int, sec: int, want: int) -> None:
    assert run(f"budget_min {default} {sec}") == str(want)


def test_classes_and_recording() -> None:
    assert run("stage_class main-overlap-poll0-thp-repeat") == "builder"
    assert run("stage_class main-repeat") == "builder"
    assert run("stage_class llamacpp-cache-fp8") == "cache"
    assert run("stage_class smoke") == ""
    assert run("record_stage_sec main-overlap 1200; record_stage_sec smoke 9999; echo $LAST_BUILDER_SEC $LAST_CACHE_SEC") == "1200 0"
    assert run("record_stage_sec llamacpp-cache 2400; echo $LAST_CACHE_SEC") == "2400"
