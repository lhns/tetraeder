"""scripts/planner_builder_campaign.sh DRY_RUN=1: the fixture, the candidate list and the estimate without a GPU.

The candidate list needs the tetraeder-plan binary (target/{release,debug}); without it the dry run says so and the
list checks skip. The unit tests of the set itself are in crates/planner (builder.rs, campaign.rs).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="needs bash")
SCRIPT = "scripts/planner_builder_campaign.sh"


def plan_bin() -> Path | None:
    for d in ("release", "debug"):
        for n in ("tetraeder-plan", "tetraeder-plan.exe"):
            p = ROOT / "target" / d / n
            if p.is_file():
                return p
    return None


def dry(**env: str) -> str:
    e = {**os.environ, "DRY_RUN": "1", "THREADS": "15", "REPS": "2", **env}
    for k in ("STAGES", "OUT_DIR", "START", "DEADLINE", "FREE_MOE_MIB", "CARD_MIB"):
        if k not in env:
            e.pop(k, None)
    p = plan_bin()
    if p is not None:
        e["PLAN_BIN"] = p.as_posix()
    out = subprocess.run([BASH, SCRIPT], cwd=ROOT, env=e, capture_output=True, text=True, check=True)
    return out.stdout


def tiers(out: str) -> list[int]:
    line = next(ln for ln in out.splitlines() if ln.startswith("vram_slots "))
    return [int(x) for x in line.split()[1].split(",")]


def test_scripts_parse() -> None:
    for s in ("planner_builder_campaign.sh", "session.sh"):
        subprocess.run([BASH, "-n", str(ROOT / "scripts" / s)], check=True)


def test_session_profile_exists() -> None:
    src = (ROOT / "scripts" / "session.sh").read_text(encoding="utf-8")
    assert "b3-builder)" in src
    assert "planner_builder_campaign.sh" in src.split("b3-builder)")[1].split("b3-detector)")[0]


def test_dry_run_prints_the_fixture_and_writes_nothing(tmp_path: Path) -> None:
    out_dir = tmp_path / "planner-moe-builder"
    out = dry(OUT_DIR=str(out_dir.as_posix()))
    assert out.startswith("DRY_RUN planner_builder_campaign")
    assert "free_mib=14336" in out and "threads=15" in out and "reps=2" in out
    assert "stages: disk-curve runs trace" in out
    assert not out_dir.exists()


@pytest.mark.skipif(plan_bin() is None, reason="needs the tetraeder-plan binary")
def test_dry_run_lists_candidates_with_the_fill_tier() -> None:
    out = dry()
    t = tiers(out)
    # the balloon fixture (14336 MiB free): the doubling tiers below the budget, then the fill tier above 64
    assert t[-4:] == [64, 32, 16, 8] and 64 < t[0] < 128, t
    ids = re.findall(r"^\s*\d+ (anchor-bld-\S+|cand-\S+) +vram_slots=", out, re.M)
    assert ids[:4] == ["anchor-bld-ram-all", "anchor-bld-vram-small", "anchor-bld-vram-large", "anchor-bld-opo-off"]
    assert 20 <= len(ids) - 4 <= 24
    large = next(ln for ln in out.splitlines() if " anchor-bld-vram-large " in ln)
    assert f"vram_slots={t[0]} " in large
    assert "min in all" in out and "one rep is" in out


@pytest.mark.skipif(plan_bin() is None, reason="needs the tetraeder-plan binary")
@pytest.mark.parametrize("card_mib", [24576, 49152])
def test_unballooned_cards_fill_with_all_experts(card_mib: int) -> None:
    out = dry(FREE_MOE_MIB="0", CARD_MIB=str(card_mib))
    assert tiers(out) == [128, 64, 32, 16, 8]


@pytest.mark.skipif(plan_bin() is None, reason="needs the tetraeder-plan binary")
def test_stages_change_the_estimate() -> None:
    full = dry()
    runs_only = dry(STAGES="runs")
    assert "stages: runs" in runs_only
    assert full != runs_only
