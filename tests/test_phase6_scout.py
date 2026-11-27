"""Case S scout probe (scripts/phase6_scout.py): the raw first-SSH readings and their verdicts. No network. Hardware label: none."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import phase6_scout as sc

GOOD = """[17:01:02] box running 123
K memmax 91587870720
K cpumax 2304000 100000
K model AMD Ryzen 9 5950X 16-Core Processor
K cpus 32
K sockets 1
K numa 1
K smi NVIDIA GeForce RTX 3090, 390.00, 390.00, 400.00
"""


def test_a_one_socket_box_at_default_power_passes():
    r = sc.judge_raw(sc.parse_raw(GOOD))
    assert all(ok for ok, _ in r.values()), r


def test_each_host_property_fails_on_its_own():
    capped = GOOD.replace("390.00, 390.00", "300.00, 420.00")
    two = GOOD.replace("K sockets 1", "K sockets 2").replace("K numa 1", "K numa 2")
    small = GOOD.replace("91587870720", "60000000000")
    r = sc.judge_raw(sc.parse_raw(capped))
    assert not r["power"][0] and "LIMITED" in r["power"][1] and r["numa"][0]
    r = sc.judge_raw(sc.parse_raw(two))
    assert not r["sockets"][0] and not r["numa"][0] and r["power"][0]
    r = sc.judge_raw(sc.parse_raw(small))
    assert not r["memory"][0] and r["power"][0]
    assert "memory" not in sc.HOST_CHECKS  # an offer property: never a skip-list reason


def test_missing_readings_fail():
    r = sc.judge_raw(sc.parse_raw("K memmax max\n"))
    assert not any(ok for ok, _ in r.values())
    assert sc.created_id("[62870946] created instance on offer 1") == "62870946" and sc.created_id("no offer") is None
