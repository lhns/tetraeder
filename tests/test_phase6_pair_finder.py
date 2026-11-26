"""Case S pair finder (scripts/phase6_pair_finder.py) on a recorded offers file: the rules, the RAM floor, the pairing and the
ranking, a pair dropped by the covering plan. No network. Hardware label: none."""

import json
import os
import pathlib
import re
import sys
import types

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
    assert [o["id"] for o in keep] == ["1", "2", "3", "4", "12"]
    assert c["excluded"] == dict.fromkeys(pf.RULES, 1)
    assert c["offers"] == 13 and c["survivors"] == 5
    assert c["per_class"]["RTX A6000"] == 1 and c["per_class"]["L40"] == 0  # a wrong class name would show as 0
    by_id = {o["id"]: pf.failing(o, a) for o in offers}
    assert by_id["7"] == ["ram"] and by_id["9"] == ["inet"] and by_id["10"] == ["price"] and by_id["8"] == ["disk"]
    assert by_id["11"] == ["unknown"]  # unknown data is its own count, not a RAM failure
    assert by_id["13"] == ["gpu_class"] and by_id["5"] == ["skip_host"] and by_id["6"] == ["cores"]


def test_price_includes_traffic_over_the_session():
    o = {"price_per_h": 0.40, "inet_down_cost": 0.0026}
    assert pf.all_in(o) == pytest.approx(0.40 + 0.0026 * 150 / 3)
    assert pf.all_in({"price_per_h": 0.4}) is None  # unknown traffic cannot be shown to fit


def test_aclass_pairs_are_listed_but_not_plannable_without_the_flag():
    res = pf.find(args())
    aclass = [p for p in res["pairs"] if "RTX A6000" in p["gpus"]]
    assert len(aclass) == 4 and not any(p["plannable"] for p in aclass)
    assert all("not plannable" in " ".join(p["flags"]) for p in aclass)
    assert all(p["plannable"] for p in pf.find(args("--plan-aclass"))["pairs"])


def test_pairs_need_different_hosts_and_rank_by_bulk_then_price():
    a = args()
    keep, _ = pf.screen(pf.load_offers([FIXTURE]), a)
    pairs = pf.make_pairs(keep)
    plain = [p for p in pairs if p["plannable"]]
    assert [(p["a"], p["b"]) for p in plain] == [("1", "3"), ("2", "3"), ("1", "4"), ("2", "4"), ("3", "4")]
    assert pairs[: len(plain)] == plain  # A-class pairs come after every plannable one
    assert len(pairs) == 9 and all(p["hosts"][0] != p["hosts"][1] for p in pairs)  # offers 1 and 2 share host 1001
    pairs = plain
    assert pairs[0]["bulk_mbps"] == 2600 and pairs[2]["bulk_mbps"] == 1000
    # the same bulk bucket: datacenter hosts first
    assert pairs[3]["datacenter"] == 1 and pairs[4]["datacenter"] == 0


def test_no_plan_binary_reports_the_static_pass(capsys):
    res = pf.find(args())
    assert len(res["pairs"]) == 9 and res["counts"]["pairs_same_host_dropped"] == 1
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
    assert [(p["a"], p["b"]) for p in res["pairs"]] == [("1", "4"), ("2", "4"), ("1", "12"), ("2", "12"), ("4", "12")]
    assert res["counts"]["pairs_no_covering_plan"] == 4
    assert res["pairs"][0]["plan"]["covering"] and res["pairs"][0]["plan"]["best"]["id"] == "c0"


def test_exit_code_and_json(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert pf.main(["--offers-json", str(FIXTURE), "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["rules"]["ram_floor_gb"] == pytest.approx(61.7) and len(doc["pairs"]) == 9
    # one box left per host: no pair, exit 1 and the message to create nothing
    single = tmp_path / "one.json"
    single.write_text(json.dumps({"offers": pf.load_offers([FIXTURE])[:2]}))
    assert pf.main(["--offers-json", str(single)]) == 1
    assert "NO PLANNABLE PAIR" in capsys.readouterr().out


def test_skip_hosts_are_in_the_box_driver_doc():
    doc = (ROOT / "docs" / "box-driver.md").read_text(encoding="utf-8")
    missing = [h for h in pf.SKIP_HOSTS if not re.search(rf"\b{h}\b", doc)]
    assert missing == []


def test_provider_row_is_normalized_and_reads_are_merged():
    raw = json.loads((ROOT / "tests" / "fixtures" / "phase6" / "pair_finder_raw_row.json").read_text())
    off = types.SimpleNamespace(id="62579299", gpu_name="RTX 3090", num_gpus=1, host_id="192026", machine_id="153018",
                                location="Florida, US", cpu_cores=16.0, price_per_h=0.4078, inet_down_cost=0.0026, raw=raw)  # fmt: skip
    row = pf.row_from_provider(off)
    assert row["ram_gb"] == pytest.approx(125.829) and row["disk_gb"] == 426.5  # cpu_ram MB read as 1e6 bytes
    assert row["inet_down_mbps"] == 7995.2 and row["inet_up_mbps"] == 3137.2 and row["hosting_type"] == 1
    # a row without the provider's RAM stays unknown
    gone = types.SimpleNamespace(**{**vars(off), "raw": {k: v for k, v in raw.items() if k != "cpu_ram"}})
    assert pf.row_from_provider(gone)["ram_gb"] is None

    other = types.SimpleNamespace(**{**vars(off), "host_id": "7", "machine_id": "8", "price_per_h": 0.5})
    rotated = types.SimpleNamespace(**{**vars(off), "id": "9"})

    class Provider:  # offers appear, vanish and rotate ids between searches
        reads = iter([[off], [rotated, other], []])

        def offers(self, q):
            return next(self.reads)

    rows = pf.fetch_class(Provider(), "RTX 3090", args("--reads", "3"))
    assert sorted(r["host_id"] for r in rows) == ["192026", "7"]  # the rotated id of the same machine is not a second offer


def planner_bin():
    explicit = os.environ.get("TETRAEDER_PLAN_BIN")
    if explicit:
        return explicit
    tb = os.environ.get("TETRAEDER_BIN")
    if tb:
        cand = pathlib.Path(tb).with_name("tetraeder-plan" + pathlib.Path(tb).suffix)
        if cand.exists():
            return str(cand)
    return None


@pytest.mark.skipif(planner_bin() is None, reason="needs a built tetraeder-plan (TETRAEDER_PLAN_BIN or beside TETRAEDER_BIN)")
@pytest.mark.parametrize("ram_gb", [200.0, pf.RAM_FLOOR_GB])
def test_real_planner_places_a_covering_plan_at_the_budget(tmp_path, ram_gb):
    """The RAM floor box (no balloon, budget 57.1 GB) and a balloon box (59.4 GB) both get a covering plan from the real planner."""
    box = {"ram_gb": ram_gb, "cpu_cores": 12}
    res = pf.run_plan([planner_bin()], pf.pair_graph(box, box), ROOT / pf.LAYOUT, tmp_path)
    assert res["covering"], res
