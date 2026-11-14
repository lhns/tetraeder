"""vramfill_bytes (scripts/vramfill.sh): expert slot bytes that fill the free VRAM, floored at the matched slots."""

import os
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs a POSIX userland (Linux)")

FLOOR = 10_970_000_000


def fill(total_mib, used_mib=0, **env):
    e = {**os.environ, **env}
    r = subprocess.run(
        ["bash", "-c", f". scripts/vramfill.sh; vramfill_bytes {FLOOR} {total_mib} {used_mib}"],
        cwd=ROOT, env=e, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    return int(r.stdout)


def test_48gb_card_gets_about_29gb():
    assert fill(49140) == 49140 * 1048576 - 22_500_000_000
    assert 28.9e9 < fill(49140) < 29.1e9


def test_24gb_card_stays_at_the_floor():
    assert fill(24564) == FLOOR


def test_used_vram_is_subtracted():
    assert fill(49140, 4000) == fill(49140) - 4000 * 1048576


def test_missing_smi_reading_is_the_floor():
    assert fill(0, 0) == FLOOR


def test_extra_overrides():
    assert fill(24564, VRAMFILL_EXTRA="18000000000") == FLOOR + 18_000_000_000


def test_reserve_override():
    assert fill(49140, VRAMFILL_RESERVE="30000000000") == 49140 * 1048576 - 30_000_000_000
