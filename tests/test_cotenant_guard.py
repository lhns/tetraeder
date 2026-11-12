"""Co-tenant fields of phase() and the stage guard (ADR 0045, 20261112g), on fake /proc and /sys trees with two SMT
siblings per core, and the OTLP path against the sampler path."""

import importlib.util
import json
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("engine_sample", SCRIPTS / "engine_sample.py")
es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(es)
import cotenant_guard as cg  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
import test_otlp_host as th  # noqa: E402

NCPU = 16  # 8 cores, CPUs 2k and 2k+1 are siblings


def sys_tree(tmp_path, with_topology=True):
    root = tmp_path / "cpu"
    for c in range(NCPU):
        (root / f"cpu{c}").mkdir(parents=True)
        if with_topology:
            k = c // 2 * 2
            (root / f"cpu{c}" / "topology").mkdir()
            (root / f"cpu{c}" / "topology" / "thread_siblings_list").write_text(f"{k},{k + 1}\n")
    return str(root)


def proc_stat(busy):
    """/proc/stat text with `busy` jiffies out of 1000 per CPU (user time)."""
    lines = ["cpu  1 1 1 1 1 1 1 1"] + [f"cpu{i} {b} 0 0 {1000 - b} 0 0 0 0" for i, b in enumerate(busy)]
    return "\n".join(lines) + "\nintr 0\n"


def rows(busy_pct, cg_cores, own_cores, dur=10.0):
    """Two samples dur apart: per-CPU busy percent, cgroup and engine tree cores."""
    out = []
    for k, t in enumerate((0.0, dur)):
        out.append({"t": t, "rss_kb": 1, "cg_io": None,
                    "cpus": [(int(p / 100 * 100 * dur) * k, int(100 * dur) * k) for p in busy_pct],
                    "cg_cpu_usec": int(cg_cores * dur * 1e6) * k, "own_cpu_s": own_cores * dur * k})
    return out


SIB = [[2 * i, 2 * i + 1] for i in range(NCPU // 2)]


def test_smt_siblings_read_from_sys(tmp_path):
    assert es.smt_siblings(sys_tree(tmp_path)) == SIB


def test_missing_topology_gives_null_pairs(tmp_path):
    sib = es.smt_siblings(sys_tree(tmp_path, with_topology=False))
    assert sib == []
    p = es.phase(rows([100] * 4 + [0] * 12, 4, 4), 0.0, 10.0, 10, 4, sib)
    assert p["smt_pairs_busy"] is None
    assert p["host_cpus"] == NCPU


def test_range_syntax_in_siblings(tmp_path):
    root = sys_tree(tmp_path, with_topology=False)
    for c in range(4):
        d = pathlib.Path(root) / f"cpu{c}" / "topology"
        d.mkdir()
        (d / "thread_siblings_list").write_text("0-3\n")
    assert es.smt_siblings(root) == [[0, 1, 2, 3]]


def test_stat_jiffies_of_a_fake_proc(monkeypatch):
    monkeypatch.setattr(es.ds, "read", lambda p, default="": proc_stat([500] * 4) if p == "/proc/stat" else default)
    assert es.stat_jiffies() == [(500, 1000)] * 4


def test_no_cotenant_fields():
    # our 4 threads on 2 cores (4 siblings busy), nothing else
    busy = [100] * 4 + [0] * 12
    p = es.phase(rows(busy, 4.0, 4.0), 0.0, 10.0, 10, 4, SIB)
    assert p["host_busy_cores"] == pytest.approx(4.0)
    assert p["container_cores"] == pytest.approx(4.0)
    assert p["engine_cores"] == pytest.approx(4.0)
    assert p["outside_cores"] == pytest.approx(0.0)
    assert p["smt_pairs_busy"] == 2
    assert p["host_cpus"] == NCPU
    assert cg.evaluate(p, 4) == (True, "")


def test_cotenant_present():
    # every CPU busy, we use 4 cores: 12 outside
    p = es.phase(rows([100] * NCPU, 4.0, 4.0), 0.0, 10.0, 10, 4, SIB)
    assert p["outside_cores"] == pytest.approx(12.0)
    assert p["smt_pairs_busy"] == 8
    ok, why = cg.evaluate(p, 4)
    assert ok is False
    assert "outside the container" in why and "SMT pairs" in why


def test_small_outside_load_is_under_the_threshold():
    # 1.5 outside cores on 16 CPUs: limit is max(4, 1.2) = 4
    busy = [100] * 9 + [0] * 7
    p = es.phase(rows(busy, 7.5, 7.5), 0.0, 10.0, 10, 4, SIB)
    assert p["outside_cores"] == pytest.approx(1.5)
    assert cg.evaluate(p, 4)[0] is True


def test_threshold_scales_with_physical_cores():
    assert cg.evaluate({"outside_cores": 9.0, "host_cpus": 128}, None)[0] is True  # limit 9.6
    assert cg.evaluate({"outside_cores": 10.0, "host_cpus": 128}, None)[0] is False


def test_missing_inputs_are_null_not_false():
    empty = [{"t": 0.0, "rss_kb": 1, "cpus": [], "cg_io": None}, {"t": 10.0, "rss_kb": 1, "cpus": [], "cg_io": None}]
    p = es.phase(empty, 0.0, 10.0, 10, 4, None)
    for k in ("host_busy_cores", "container_cores", "engine_cores", "outside_cores", "smt_pairs_busy", "host_cpus"):
        assert p[k] is None, k
    assert cg.evaluate(p, 4)[0] is None


def test_merge_into_stage_json(tmp_path):
    f = tmp_path / "r.json"
    bad = es.phase(rows([100] * NCPU, 4.0, 4.0), 0.0, 10.0, 10, 4, SIB)
    f.write_text(json.dumps({"host": {"decode": bad}, "host_windows": {"threads": 4}}))
    line = cg.merge(str(f))
    d = json.loads(f.read_text())
    assert d["cotenant_ok"] is False and "outside" in d["cotenant_reason"]
    assert line.startswith("cotenant CO-TENANT")
    f.write_text(json.dumps({"x": 1}))
    cg.merge(str(f))
    d = json.loads(f.read_text())
    assert d["cotenant_ok"] is None and "cotenant_reason" in d
    f.write_text("[1]")
    cg.merge(str(f))
    assert json.loads((tmp_path / "r.json.cotenant.json").read_text())["cotenant_ok"] is None


def test_otlp_path_equals_sampler_path(tmp_path, monkeypatch):
    sib = [[0, 1], [2, 3]]
    monkeypatch.setattr(th.es, "smt_siblings", lambda root=None: sib)
    samples, otel, results, builder = th.write_inputs(tmp_path)
    th.summarize_sampler(samples, builder, tmp_path)
    es.otlp_merge(types.SimpleNamespace(otel_dir=str(otel), results=str(results)))
    d = json.loads(builder.read_text())
    assert d["env"]["smt_siblings"] == sib
    keys = ("host_busy_cores", "container_cores", "engine_cores", "outside_cores", "smt_pairs_busy", "host_cpus")
    assert d["host"]["decode"]["host_cpus"] == th.CPUS
    assert d["host"]["decode"]["host_busy_cores"] > 0
    for ph in ("prefill", "decode"):
        for k in keys:
            assert d["host_otlp"][ph][k] == pytest.approx(d["host"][ph][k]), (ph, k)
    for i, w in enumerate(d["host"]["decode_windows"]):
        for k in keys:
            assert d["host_otlp"]["decode_windows"][i][k] == pytest.approx(w[k]), (i, k)
