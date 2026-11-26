"""Case S pair finder (scripts/phase6_pair_finder.py) on a recorded offers file: the rules, the RAM floor, the pairing and the
ranking, a pair dropped by the covering plan. No network. Hardware label: none."""

import json
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import phase6_pair_finder as pf

FIXTURE = ROOT / "tests" / "fixtures" / "phase6" / "pair_finder_offers.json"


def args(*extra):
    return pf.parse_args(["--offers-json", str(FIXTURE), *extra])


def test_ram_floor_is_need_plus_box_use():
    assert pf.RAM_FLOOR_GB == pytest.approx(pf.NODE_B_HOST_GB + pf.BOX_USE_GB) == pytest.approx(61.7)
    # the box use is what the committed pick's 59.4 GB budget leaves of the 64 GB balloon
    assert pf.BALLOON_GB - pf.BOX_USE_GB == pytest.approx(59.4)
    # the floor box still budgets the need without a balloon; a box over 64 GB gets the balloon's budget
    assert pf.ram_budget_gb(pf.RAM_FLOOR_GB) >= pf.NODE_B_HOST_GB
    assert pf.ram_budget_gb(200.0) == pytest.approx(59.4)


def test_each_rule_excludes_its_offer_and_counts():
    a = args()
    offers = pf.load_offers([FIXTURE])
    keep, c = pf.screen(offers, a)
    assert [o["id"] for o in keep] == ["1", "2", "3", "4"]
    assert c["excluded"] == {"gpu_class": 2, "skip_host": 1, "cores": 1, "ram": 2, "disk": 1, "inet": 1, "price": 1}
    assert c["offers"] == 13 and c["survivors"] == 4
    # independent counts: the skip host fails nothing else, the unknown-RAM offer fails ram and disk and inet
    assert c["fails"]["skip_host"] == 1 and c["fails"]["ram"] == 2
    by_id = {o["id"]: pf.failing(o, a) for o in offers}
    assert by_id["7"] == ["ram"] and by_id["9"] == ["inet"] and by_id["10"] == ["price"] and by_id["8"] == ["disk"]
    assert by_id["11"][:1] == ["ram"]


def test_price_includes_traffic_over_the_session():
    o = {"price_per_h": 0.40, "inet_down_cost": 0.0026}
    assert pf.all_in(o) == pytest.approx(0.40 + 0.0026 * 150 / 3)
    assert pf.all_in({"price_per_h": 0.4}) is None  # unknown traffic cannot be shown to fit


def test_aclass_only_with_the_flag():
    keep, _ = pf.screen(pf.load_offers([FIXTURE]), args("--allow-aclass"))
    assert "12" in [o["id"] for o in keep]


def test_pairs_need_different_hosts_and_rank_by_bulk_then_price():
    a = args()
    keep, _ = pf.screen(pf.load_offers([FIXTURE]), a)
    pairs = pf.make_pairs(keep)
    assert [(p["a"], p["b"]) for p in pairs] == [("1", "3"), ("2", "3"), ("1", "4"), ("2", "4"), ("3", "4")]
    assert all(p["hosts"][0] != p["hosts"][1] for p in pairs)  # offers 1 and 2 share host 1001
    assert pairs[0]["bulk_mbps"] == 2600 and pairs[2]["bulk_mbps"] == 1000
    # the same bulk bucket: datacenter hosts first
    assert pairs[3]["datacenter"] == 1 and pairs[4]["datacenter"] == 0


def test_no_plan_binary_reports_the_static_pass(capsys):
    res = pf.find(args())
    assert len(res["pairs"]) == 5 and res["counts"]["pairs_same_host_dropped"] == 1
    assert all("covering plan not run" in " ".join(p["flags"]) for p in res["pairs"])
    assert any("power limit unknown" in f for f in res["pairs"][0]["flags"])
    assert any("no balloon" in f for f in res["pairs"][0]["flags"])  # offer 3 has 62 GB
    text = pf.table(res, 3)
    assert "excluded first by rule" in text and "ram" in text


def test_pair_graph_sets_ram_budget_and_threads():
    keep, _ = pf.screen(pf.load_offers([FIXTURE]), args())
    x, y = keep[2], keep[3]  # offer 3 (62 GB, 16 cores) and offer 4 (96 GB, 12 cores)
    text = pf.pair_graph(x, y)
    a_cpu = re.search(r'id = "a/cpu0".*?capacity_bytes = (\d+).*?threads = (\d+)', text, re.DOTALL).groups()
    b_cpu = re.search(r'id = "b/cpu0".*?capacity_bytes = (\d+).*?threads = (\d+)', text, re.DOTALL).groups()
    assert a_cpu == (str(int(57.4e9)), "16") and b_cpu == (str(int(59.4e9)), "12")


STUB = """
import json, pathlib, re, sys
a = sys.argv
hw = pathlib.Path(a[a.index("--hw") + 1]).read_text()
out = pathlib.Path(a[a.index("--out") + 1])
caps = [int(c) for c in re.findall(r'id = "./cpu0".*?capacity_bytes = (\\d+)', hw, re.S)]
if min(caps) < 58_000_000_000:  # the node cannot hold its covering segment
    print("no covering plan", file=sys.stderr)
    sys.exit(1)
out.write_text(json.dumps({"candidates": [{"id": "c0", "notes": ["pick"]}]}))
"""


def test_pair_without_covering_plan_is_dropped(tmp_path):
    stub = tmp_path / "plan_stub.py"
    stub.write_text(STUB)
    res = pf.find(args(), [sys.executable, str(stub)])
    # offer 3 has 62 GB: a budget of 57.4 GB, under the stub's 58 GB covering need
    assert [(p["a"], p["b"]) for p in res["pairs"]] == [("1", "4"), ("2", "4")]
    assert res["counts"]["pairs_no_covering_plan"] == 3
    assert res["pairs"][0]["plan"]["covering"] and res["pairs"][0]["plan"]["best"]["id"] == "c0"


def test_exit_code_and_json(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert pf.main(["--offers-json", str(FIXTURE), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["rules"]["ram_floor_gb"] == pytest.approx(61.7) and len(doc["pairs"]) == 5
    # one box left per host: no pair, exit 1 and the message to create nothing
    single = tmp_path / "one.json"
    single.write_text(json.dumps({"offers": pf.load_offers([FIXTURE])[:2]}))
    assert pf.main(["--offers-json", str(single)]) == 1
    assert "NO PAIR" in capsys.readouterr().out


def test_skip_hosts_are_in_the_box_driver_doc():
    doc = (ROOT / "docs" / "box-driver.md").read_text(encoding="utf-8")
    missing = [h for h in pf.SKIP_HOSTS if not re.search(rf"\b{h}\b", doc)]
    assert missing == []
