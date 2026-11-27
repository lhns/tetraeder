"""The gates of the segment confirmation as code (ADR 0091 answers 7 and 8, ADR 0092 section 6): scripts/sc_gates.py,
the summary's clean rules in scripts/sc_summary.py on fixture JSON, the co-tenant reading of engine_sample.py and the
driver's --download-gb. Hardware label: none."""

import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import sc_gates as sg  # noqa: E402

spec = importlib.util.spec_from_file_location("engine_sample", SCRIPTS / "engine_sample.py")
es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(es)

LSCPU_1 = "Architecture: x86_64\nNUMA node(s):                      1\nNUMA node0 CPU(s):                 0-31\n"
LSCPU_2 = "Architecture: x86_64\nNUMA node(s):                      2\n"


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj), encoding="utf-8")


# ---- host gates ---------------------------------------------------------------------------------------------------


def test_exactly_one_numa_node_passes():
    assert sg.check_numa(LSCPU_1) == []
    assert "2 NUMA nodes" in sg.check_numa(LSCPU_2)[0]
    assert "does not say" in sg.check_numa("Architecture: x86_64\n")[0]


def reading(outside, cpus=128, source=sg.COTENANT_SOURCE, **kw):
    return {"source": source, "outside_cores": outside, "host_cpus": cpus, "limit": 99.0, **kw}


def test_the_cotenant_reading_must_come_from_the_sampler_and_be_under_the_limit(tmp_path):
    f = tmp_path / "c.json"
    limit = 19.2  # max(4, 15% of 128 physical cores), cotenant_guard
    write(f, reading(15.6, 256))
    assert sg.check_cotenant(str(f)) == []
    write(f, reading(22.5, 256))  # the 2026-11-24 reading that was waved through
    assert "22.5" in sg.check_cotenant(str(f))[0] and f"{limit:.1f}" in sg.check_cotenant(str(f))[0]
    write(f, reading(1.0, 256, source="/root/cot.py"))  # an ad-hoc sampler
    assert "no ad-hoc sampler" in sg.check_cotenant(str(f))[0]
    write(f, reading(30.0, 256, ok=True))  # a file cannot state its own pass
    assert sg.check_cotenant(str(f))
    write(f, {"source": sg.COTENANT_SOURCE})
    assert "no CPU counters" in sg.check_cotenant(str(f))[0]
    assert "no co-tenant reading" in sg.check_cotenant(str(tmp_path / "missing.json"))[0]


def cpus(busy_per_cpu, n=8):
    """/proc/stat rows: (busy, total) jiffies per CPU after one second at 100 jiffies per CPU."""
    return [(int(100 * busy_per_cpu), 100) for _ in range(n)]


def test_the_sampler_reading_counts_busy_cores_outside_the_container():
    # 8 CPUs, 3 busy host cores in all, the container used 1 core: 2 busy outside
    def row(t, jiffy, cg):
        return {"t": t, "cpus": [(int(jiffy * 3 / 8), jiffy)] * 8, "cg_cpu_usec": cg, "own_cpu_s": None, "cg_io": None, "rss_kb": 0}

    rows = [row(0.0, 0, 0), row(10.0, 1000, 10_000_000)]
    r = es.cotenant_reading(rows)
    assert r["source"] == sg.COTENANT_SOURCE
    assert r["host_cpus"] == 8 and abs(r["host_busy_cores"] - 3.0) < 1e-9
    assert abs(r["container_cores"] - 1.0) < 1e-9 and abs(r["outside_cores"] - 2.0) < 1e-9
    assert r["limit"] == 4.0 and r["ok"] is True
    busy = [row(0.0, 0, 0), {**row(10.0, 1000, 0), "cpus": [(1000, 1000)] * 8}]  # 8 busy cores, none ours
    assert es.cotenant_reading(busy)["ok"] is False
    assert es.cotenant_reading(rows[:1])["ok"] is None  # one sample: no reading, not a pass


# ---- the freeze ---------------------------------------------------------------------------------------------------


def sha(b):
    return hashlib.sha256(b).hexdigest()


def pred(band=0.071):
    return {"routing": "trace", "routing_trace": {}, "hw_sha256": "x",
            "prediction": {"pp_tps": 200.0, "tg_tps": 14.0, "uncalibrated_pp_tps": 300.0, "uncalibrated_tg_tps": 13.0,
                           "band": band, "anchors": [], "residuals": [], "replay_vram_share": 0.39,
                           "replay_full_hit_share": 0.005, "replay_ram_hit_rate": 1.0}}


def anchor(seed=False):
    hidden = {"seed": 20261124} if seed else {"file": "/root/tetraeder/runs/sc/hidden.bin"}
    return {"runs": [{"prefill_tok_s": 900.0, "decode_tok_s": 40.0, "segment": {"hidden_in": hidden}}]}


def run_json(pp=190.0, tg=14.5, share=0.39, full=0.005, ram=1.0, covering=True, unclean=False, prefill_bytes=0, **kw):
    r = {"prefill_tok_s": pp, "decode_tok_s": tg, "warm": {"covering": covering, "unclean": unclean},
         "prefill_io": {"bytes": prefill_bytes},
         "decode_layers": {"vram_pair_share": share, "full_hit_rate_mean": full, "ram_hit_rate_mean": ram}}
    return {"runs": [r], "balloon_ok": True, "cotenant_ok": True, **kw}


def gpu_txt(bits="0x0"):
    return f"before: {bits}, 100.0 W, 350.0 W, 350.0 W\nafter: {bits}, 300.0 W, 350.0 W, 350.0 W\n"


def make_session(tmp_path, **o):
    """A clean session directory; keyword overrides break one thing each."""
    res = tmp_path / "res"
    write(res / "env" / "lscpu.txt", o.get("lscpu", LSCPU_1))
    write(res / "env" / "cotenant.json", o.get("cotenant", reading(1.0, 256)))
    write(res / "predictions" / "pred.json", pred(o.get("band", 0.071)))
    for a in ("a1-r1", "a1-r2", "a2-r1", "a2-r2"):
        write(res / "anchors" / f"{a}.json", anchor(o.get("seed_anchors", False)))
    write(res / "layout.json", {"n_layer": 94})
    names = ["layout.json", "predictions/pred.json"] + [f"anchors/{a}.json" for a in ("a1-r1", "a1-r2", "a2-r1", "a2-r2")]
    sums = "".join(f"{sha((res / n).read_bytes())}  {n}\n" for n in names)
    write(res / "predictions" / "SHA256SUMS", sums)
    frozen = sha((res / "predictions" / "SHA256SUMS").read_bytes())
    write(res / "predictions" / "FROZEN.sha256", f"{frozen}  SHA256SUMS\n")
    rounds = o.get("rounds", [{}, {}, {}])
    for i, kw in enumerate(rounds, 1):
        write(res / "judged" / f"r{i}.json", run_json(**kw))
        write(res / "judged" / f"r{i}.gpu.txt", gpu_txt(o.get("bits", "0x0")))
    if o.get("aa_last", {}) is not None:
        write(res / "judged" / "aa-last.json", run_json(**o.get("aa_last", {})))
    return res, frozen


def summary(res):
    out = subprocess.run([sys.executable, str(SCRIPTS / "sc_summary.py"), str(res)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return json.loads((res / "verdict.json").read_text(encoding="utf-8"))


def failed(v):
    return {r["name"] for r in v["rules"] if not r["ok"]}


def test_judged_start_passes_on_a_clean_session(tmp_path):
    res, frozen = make_session(tmp_path)
    assert sg.check_judged_start(str(res), frozen) == []


@pytest.mark.parametrize("what, over, key", [
    ("two numa nodes", {"lscpu": LSCPU_2}, "numa"),
    ("co-tenant over the limit", {"cotenant": reading(22.5, 256)}, "co-tenant"),
    ("ad-hoc co-tenant sampler", {"cotenant": reading(0.5, 256, source="/root/cot.py")}, "co-tenant"),
    ("fit band over the limit", {"band": 1.226}, "fit band"),
    ("fixed-seed anchors", {"seed_anchors": True}, "anchors"),
])
def test_each_host_or_calibration_gate_refuses_the_judged_stages(tmp_path, what, over, key):
    res, frozen = make_session(tmp_path, **over)
    why = sg.check_judged_start(str(res), frozen)
    assert why and all(w.startswith(key) for w in why), (what, why)


def test_the_frozen_hash_must_match_and_a_frozen_file_must_not_change(tmp_path):
    res, frozen = make_session(tmp_path)
    assert any(w.startswith("freeze") and "FROZEN_SHA" in w for w in sg.check_judged_start(str(res), "0" * 64))
    assert any("FROZEN_SHA" in w for w in sg.check_judged_start(str(res), ""))
    (res / "predictions" / "pred.json").write_text(json.dumps(pred(0.05)), encoding="utf-8")
    assert any("changed: predictions/pred.json" in w for w in sg.check_judged_start(str(res), frozen))
    (res / "predictions" / "FROZEN.sha256").unlink()
    assert any("not written" in w for w in sg.check_judged_start(str(res), frozen))


def test_the_fit_band_limit_is_the_one_the_planner_enforces():
    src = (ROOT / "crates" / "planner" / "src" / "segment.rs").read_text(encoding="utf-8")
    m = re.search(r"pub const FIT_BAND_LIMIT: f64 = ([0-9.]+);", src)
    assert m and float(m.group(1)) == sg.FIT_BAND_LIMIT
    # the measured bands: real hidden rows 0.071 pass, fixed-seed input 1.226 does not
    assert 0.071 <= sg.FIT_BAND_LIMIT < 1.226


def test_the_cli_exits_non_zero_with_the_reason(tmp_path):
    f = tmp_path / "lscpu.txt"
    f.write_text(LSCPU_2, encoding="utf-8")
    out = subprocess.run([sys.executable, str(SCRIPTS / "sc_gates.py"), "numa", str(f)], capture_output=True, text=True)
    assert out.returncode == 1 and "GATE FAILED numa" in out.stdout
    f.write_text(LSCPU_1, encoding="utf-8")
    out = subprocess.run([sys.executable, str(SCRIPTS / "sc_gates.py"), "numa", str(f)], capture_output=True, text=True)
    assert out.returncode == 0


# ---- A/A ----------------------------------------------------------------------------------------------------------


def test_aa_is_on_t_with_the_band_0_08():
    a, b = run_json(pp=200.0, tg=14.0), run_json(pp=200.0, tg=14.0)
    assert sg.aa(a, b)["ok"] and sg.aa(a, b)["diff"] == 0.0
    slow = run_json(pp=200.0, tg=14.0)
    slow["runs"][0]["prefill_tok_s"] = 100.0  # T 19.4 s -> 29.7 s
    r = sg.aa(a, slow)
    assert not r["ok"] and r["diff"] > 0.5
    # the 2026-11-24 session: 13.40 s then 12.07 s, -0.099
    t = lambda s: {"runs": [{"prefill_tok_s": 2048 / s, "decode_tok_s": 1e9}]}  # noqa: E731
    assert not sg.aa(t(13.40), t(12.07))["ok"]
    assert sg.aa(t(13.40), t(12.50))["ok"]


# ---- the summary ---------------------------------------------------------------------------------------------------


def test_a_clean_session_gets_a_verdict(tmp_path):
    res, _ = make_session(tmp_path)
    v = summary(res)
    assert v["clean"] and failed(v) == set() and v["verdict"] == "pass"
    assert v["aa_early"]["ok"] and v["aa"]["ok"]


def test_a_clean_miss_is_a_miss_not_no_verdict(tmp_path):
    res, _ = make_session(tmp_path, rounds=[{"pp": 100.0}] * 3, aa_last={"pp": 100.0})
    v = summary(res)
    assert v["clean"] and v["verdict"] == "miss"


@pytest.mark.parametrize("over, held", [
    # the replay of the frozen prediction says share 0.39, full hit 0.005, RAM hit 1.0 (ADR 0092 section 1)
    ({"share": 0.40}, True),           # 0.01 off: judged
    ({"share": 0.38}, True),
    ({"share": 0.42}, False),          # 0.03 off: no verdict
    ({"share": 0.36}, False),
    ({"full": 0.035}, False),          # the full-hit line is 0.02 as well
    ({"full": 0.015}, True),
    ({"ram": 0.996}, True),            # the RAM hit line is 0.005
    ({"ram": 0.994}, False),
])
def test_the_routing_gate_holds_inside_its_lines_and_misses_outside(tmp_path, over, held):
    res, _ = make_session(tmp_path, rounds=[over] * 3, aa_last=over)
    v = summary(res)
    assert v["gate_held"] is held, v["gate"]
    if held:
        assert v["clean"] and v["verdict"] in ("pass", "miss") and "routing_gate" not in failed(v)
    else:
        assert failed(v) == {"routing_gate"}
        assert v["verdict"] == "no verdict" and not v["clean"]


def test_there_is_no_prefill_noise_band(tmp_path):
    # identical rounds that differ in prefill by 8x but agree on A/A would be unclean only through A/A; a spread among
    # rounds 2 and 3 that does not touch the A/A pair is not a rule
    res, _ = make_session(tmp_path, rounds=[{"pp": 190.0}, {"pp": 190.0}, {"pp": 40.0}])
    v = summary(res)
    assert "rounds_clean" not in failed(v) and v["verdict"] in ("pass", "miss")


@pytest.mark.parametrize("over, rule", [
    ({"aa_last": None}, "aa"),
    ({"aa_last": {"pp": 120.0}}, "aa"),
    ({"rounds": [{}, {"pp": 120.0}, {}]}, "aa_early"),
    ({"lscpu": LSCPU_2}, "one_numa_node"),
    ({"cotenant": reading(22.5, 256)}, "cotenant_ssh"),
    ({"cotenant": reading(0.1, 256, source="/root/cot.py")}, "cotenant_ssh"),
    ({"band": 1.226}, "fit_band"),
    ({"seed_anchors": True}, "anchors_real_hidden"),
    ({"bits": "0x20"}, "rounds_clean"),
    ({"rounds": [{}, {"unclean": True}, {}]}, "rounds_clean"),
    ({"rounds": [{}, {"prefill_bytes": 5}, {}]}, "rounds_clean"),
    ({"rounds": [{}, {"covering": False}, {}]}, "rounds_clean"),
    ({"rounds": [{}, {"cotenant_ok": False}, {}]}, "rounds_clean"),
    ({"rounds": [{"share": 0.30}] * 3}, "routing_gate"),
    ({"rounds": [{}, {}]}, "rounds_clean"),
])
def test_each_clean_rule_gives_no_verdict(tmp_path, over, rule):
    res, _ = make_session(tmp_path, **over)
    v = summary(res)
    assert rule in failed(v), failed(v)
    assert v["verdict"] == "no verdict" and not v["clean"]
    assert v["status"].startswith("no verdict")


def test_a_broken_freeze_gives_no_verdict(tmp_path):
    res, _ = make_session(tmp_path)
    (res / "layout.json").write_text('{"n_layer": 95}', encoding="utf-8")
    assert "freeze" in failed(summary(res))


# ---- the session script and the driver ----------------------------------------------------------------------------


def test_the_session_script_wires_the_gates():
    s = (SCRIPTS / "sc_box.sh").read_text(encoding="utf-8")
    assert 'stages="${STAGES:-gates ' in s
    for needle in ("sc_gates.py numa", "engine_sample.py cotenant", "sc_gates.py judged-start", "sc_gates.py aa ",
                   "sc_gates.py fitband", "sc_gates.py anchors-real", "--max-band", "st_aalast", "check_order"):
        assert needle in s, needle
    # the anchors run on the dumped hidden rows
    anchors = s[s.index("st_anchors()"):s.index("anchor_args()")]
    assert anchors.count('"${hin[@]}"') == 2 and 'hin=("TETRAEDER_EXIT_HIDDEN_IN=$hidden")' in anchors  # a span from layer 0 takes tokens instead
    # no ad-hoc sampler in the script
    assert "/proc/stat" not in s


@pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="needs a POSIX userland")
def test_the_driver_prices_the_traffic_into_offers_and_create():
    def dry(*args, **env):
        return subprocess.run(["bash", str(SCRIPTS / "sc_offers.sh"), *args], capture_output=True, text=True,
                              env={**os.environ, "DRY_RUN": "1", **env})

    for cmd in ("offers", "create"):
        out = dry(cmd, "--min-cpu-cores", "24")
        assert out.returncode == 0
        assert out.stdout.split() == ["uv", "run", "taxameter", cmd, "--download-gb", "150", "--max-price", "0.90",
                                      "--min-disk", "230", "--min-cpu-cores", "24"]
    assert "--min-disk" not in dry("offers", "--min-disk", "300").stdout.replace("--min-disk 300", "")
    assert "--download-gb 142" in dry("offers", DOWNLOAD_GB="142").stdout
    assert dry("offers", "--download-gb", "20").returncode == 2
    assert dry("offers", DOWNLOAD_GB="20").returncode == 2
    assert dry("sell").returncode == 2


# ---- the engine_sample.py cotenant subcommand and the committed real session --------------------------------------


def fake_host(monkeypatch, busy_cpus, cg_cores):
    """A clock that advances 1 s per sleep and a host whose `busy_cpus` CPUs are fully busy, 8 CPUs in all."""
    now = [1000.0]
    monkeypatch.setattr(es.time, "time", lambda: now[0])
    monkeypatch.setattr(es.time, "sleep", lambda s: now.__setitem__(0, now[0] + s))

    def jiffies():
        t = int((now[0] - 1000.0) * 100)
        return [(t if i < busy_cpus else 0, t) for i in range(8)]

    monkeypatch.setattr(es, "stat_jiffies", jiffies)
    monkeypatch.setattr(es.ds, "cgroup_cpu_usec", lambda: int((now[0] - 1000.0) * 1e6 * cg_cores))


@pytest.mark.parametrize("busy, rc", [(1, 0), (6, 1)])
def test_the_cotenant_subcommand_exits_with_the_verdict(monkeypatch, tmp_path, busy, rc):
    fake_host(monkeypatch, busy, 0.5)
    out = tmp_path / "c.json"
    args = types.SimpleNamespace(seconds=5.0, out=str(out))
    assert es.cotenant(args) == rc
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["source"] == sg.COTENANT_SOURCE and r["host_cpus"] == 8 and r["limit"] == 4.0
    assert abs(r["outside_cores"] - (busy - 0.5)) < 1e-6 and r["ok"] is (rc == 0)
    assert (sg.check_cotenant(str(out)) == []) is (rc == 0)  # the gate reads the same verdict


def test_the_cotenant_subcommand_without_a_window_is_not_a_pass(monkeypatch, tmp_path):
    fake_host(monkeypatch, 1, 0.5)
    assert es.cotenant(types.SimpleNamespace(seconds=0.0, out=None)) == 2


def test_main_passes_only_the_cotenant_code_on(monkeypatch, tmp_path):
    fake_host(monkeypatch, 1, 0.5)
    monkeypatch.setattr(sys, "argv", ["engine_sample.py", "cotenant", "--seconds", "3"])
    assert es.main() == 0


def test_the_committed_two_socket_session_fails_the_numa_and_fit_gates():
    res = ROOT / "bench" / "results" / "segment-confirm-20261124"
    if not (res / "env" / "lscpu.txt").exists():
        pytest.skip("no bench/results checkout")
    assert "2 NUMA nodes" in sg.check_numa((res / "env" / "lscpu.txt").read_text(encoding="utf-8"))[0]
    assert sg.check_freeze(str(res)) == []  # real sha256sum output, whatever the checkout's line endings
    assert sg.check_fit_band(str(res / "predictions" / "pred.json"))  # 1.226
    assert sg.check_fit_band(str(res / "report" / "pred-real-anchors.json")) == []  # 0.071
    assert sg.check_anchors_real(sg.anchor_files(str(res)))  # fixed-seed anchors
    assert sg.check_anchors_real([str(res / "report" / "real-a1-r1.json")]) == []
    aa = sg.aa(sg.load(str(res / "judged" / "r1.json")), sg.load(str(res / "report" / "extra-r5.json")))
    assert not aa["ok"] and abs(aa["diff"] + 0.099) < 0.001
