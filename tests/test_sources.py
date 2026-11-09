"""docs/sources.md lists every patch in the queue and every pinned upstream in vendor/PIN."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (ROOT / "docs" / "sources.md").read_text(encoding="utf-8")


def test_every_patch_has_a_row():
    series = (ROOT / "patches" / "series").read_text(encoding="utf-8").splitlines()
    patches = [line.split("#")[0].strip() for line in series]
    missing = [p for p in patches if p and f"`{p}`" not in SOURCES]
    assert not missing, f"patches without a row in docs/sources.md: {missing}"


def test_every_pin_key_is_documented():
    keys = [line.split("=")[0] for line in (ROOT / "vendor" / "PIN").read_text(encoding="utf-8").splitlines() if "=" in line]
    missing = [k for k in keys if k != "image_tag" and f"`{k}`" not in SOURCES]
    assert not missing, f"vendor/PIN keys not in docs/sources.md: {missing}"
