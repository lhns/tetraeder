"""scripts/balloon_guard.py: the RAM balloon check of exit_run.sh."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("balloon_guard", ROOT / "scripts" / "balloon_guard.py")
bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bg)

TARGET = 50_000_000_000


def status(state: str, rss_kb: int) -> str:
    return f"Name:\tpython3\nState:\t{state} (x)\nVmRSS:\t{rss_kb} kB\n"


def test_ok() -> None:
    assert bg.evaluate(status("S", TARGET // 1024), TARGET) == (True, "ok")
    assert bg.evaluate(status("S", int(TARGET * 0.95) // 1024), TARGET)[0]


def test_gone_zombie_shrunk() -> None:
    assert not bg.evaluate(None, TARGET)[0]
    assert "dead" in bg.evaluate(status("Z", 0), TARGET)[1]
    ok, why = bg.evaluate(status("S", TARGET // 2048), TARGET)
    assert not ok and "below" in why


def test_merge_object_and_sidecar(tmp_path: Path) -> None:
    f = tmp_path / "r.json"
    f.write_text(json.dumps({"decode_tok_s": 1.5}))
    assert bg.merge(str(f), False, "gone") == str(f)
    d = json.loads(f.read_text())
    assert d == {"decode_tok_s": 1.5, "balloon_ok": False, "balloon_reason": "gone"}
    bg.merge(str(f), True, "")
    assert json.loads(f.read_text())["balloon_ok"] is True
    arr = tmp_path / "a.json"
    arr.write_text("[1]")
    side = bg.merge(str(arr), False, "x")
    assert side.endswith(".balloon.json") and json.loads(arr.read_text()) == [1]
    assert json.loads(Path(side).read_text())["balloon_ok"] is False
    assert bg.merge(str(tmp_path / "missing.json"), False, "x").endswith(".balloon.json")
