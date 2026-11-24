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

HW = """[[devices]]
id = "cpu0"
kind = "cpu"
capacity_bytes = 1032772911104
threads = 61

[[devices]]
id = "gpu0"
kind = "gpu"
capacity_bytes = 48305799168
"""


def fake_cgroup(tmp_path: Path, mx: str, cur: int, inactive: int) -> str:
    d = tmp_path / "cg"
    d.mkdir(exist_ok=True)
    (d / "memory.max").write_text(f"{mx}\n")
    (d / "memory.current").write_text(f"{cur}\n")
    (d / "memory.stat").write_text(f"anon 1\nfile 2\ninactive_file {inactive}\nactive_file 7\n")
    return str(d)


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


def test_budget_is_the_headroom_with_droppable_cache_less_one_gib(tmp_path: Path) -> None:
    # the A40 box of stream-predict-20261124: limit 1032.8 GB, balloon 968.8 GB and the rest of the container
    cg = fake_cgroup(tmp_path, "1032772911104", 970_000_000_000 + 2_500_000_000, 1_000_000_000)
    c = bg.read_cgroup(cg)
    assert c == {"memory_max": 1032772911104, "memory_current": 972_500_000_000, "inactive_file": 1_000_000_000}
    assert bg.ram_budget(c) == 1032772911104 - 972_500_000_000 + 1_000_000_000 - (1 << 30)


def test_no_memory_limit_has_no_budget(tmp_path: Path) -> None:
    cg = fake_cgroup(tmp_path, "max", 5, 0)
    try:
        bg.read_cgroup(cg)
    except ValueError as e:
        assert "no memory limit" in str(e)
    else:
        raise AssertionError("an unlimited cgroup has no budget")


def test_record_writes_env_json_and_the_node_graph(tmp_path: Path) -> None:
    cg = fake_cgroup(tmp_path, "100000000000", 38_000_000_000, 500_000_000)
    env = tmp_path / "env.json"
    env.write_text(json.dumps({"host_id": 7}))
    hw = tmp_path / "hw.toml"
    hw.write_text(HW)
    out = bg.record_budget(cg, str(env), str(hw))
    want = 100_000_000_000 - 38_000_000_000 + 500_000_000 - (1 << 30)
    assert out["ram_budget_bytes"] == want
    e = json.loads(env.read_text())
    assert e["host_id"] == 7
    assert e["ram_budget"]["ram_budget_bytes"] == want
    assert {"memory_max", "memory_current", "inactive_file", "reserve_bytes"} <= set(e["ram_budget"])
    text = hw.read_text()
    assert f"capacity_bytes = {want}" in text
    # only cpu0 changed
    assert "capacity_bytes = 48305799168" in text and "threads = 61" in text


def test_wait_returns_when_the_balloon_is_full_and_gives_up_otherwise() -> None:
    calls = iter([(False, "below"), (False, "below"), (True, "ok")])
    assert bg.wait_full(lambda: next(calls), timeout=5, interval=0.01) == (True, "ok")
    ok, why = bg.wait_full(lambda: (False, "below"), timeout=0.05, interval=0.01)
    assert not ok and why == "below"
    # full means 100%: 95% passes the stage check and fails the wait
    s = status("S", int(TARGET * 0.95) // 1024)
    assert bg.evaluate(s, TARGET, 1.0)[0] is False
    assert bg.evaluate(s, TARGET)[0] is True
