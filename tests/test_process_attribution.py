"""Context switches, AnonHugePages, THP mode and the membw probe of scripts/engine_sample.py (ADR 0058 addendum 2026-11-12)."""

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("engine_sample", SCRIPTS / "engine_sample.py")
es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(es)


def fake(monkeypatch, files, tasks=None):
    monkeypatch.setattr(es.ds, "read", lambda p, default="": files.get(p, default))
    monkeypatch.setattr(es, "task_ids", lambda pid: (tasks or {}).get(pid, []))


def status(v, i):
    return f"Name:\tx\nvoluntary_ctxt_switches:\t{v}\nnonvoluntary_ctxt_switches:\t{i}\n"


def test_ctxsw_sums_tasks_of_all_pids(monkeypatch):
    fake(monkeypatch, {"/proc/1/task/1/status": status(10, 1), "/proc/1/task/2/status": status(20, 2),
                       "/proc/5/task/5/status": status(100, 7)}, {1: ["1", "2"], 5: ["5"]})
    assert es.ctxsw([1, 5]) == {"vol": 130, "invol": 10}


def test_ctxsw_falls_back_to_process_line_and_null(monkeypatch):
    fake(monkeypatch, {"/proc/1/status": status(3, 4)})
    assert es.ctxsw([1]) == {"vol": 3, "invol": 4}
    fake(monkeypatch, {})
    assert es.ctxsw([1]) is None


def test_anon_huge_and_thp_mode(monkeypatch):
    fake(monkeypatch, {"/proc/1/smaps_rollup": "Rss: 5 kB\nAnonHugePages:  2048 kB\n", "/proc/2/smaps_rollup": "AnonHugePages: 1024 kB\n",
                       "/sys/kernel/mm/transparent_hugepage/enabled": "always [madvise] never\n"})
    assert es.anon_huge_kb([1, 2]) == 3072
    assert es.thp_mode() == "madvise"
    fake(monkeypatch, {})
    assert es.anon_huge_kb([1]) is None
    assert es.thp_mode() is None


def test_smaps_rollup_only_for_thp_stages_unless_forced():
    assert not es.read_smaps("main-overlap", {})
    assert es.read_smaps("main-overlap-thp-repeat", {})
    assert es.read_smaps("main-overlap", {"SAMPLER_SMAPS": "1"})
    assert not es.read_smaps("main-overlap-thp", {"SAMPLER_SMAPS": "0"})
    assert not es.read_smaps(None, {})
    assert es.read_smaps(None, {"TETRAEDER_STAGE": "main-overlap-poll0-thp"})


def row(t, **kw):
    r = {"t": t, "rss_kb": 1, "cpus": [], "cg_io": None}
    r.update(kw)
    return r


def test_phase_reports_ctxsw_rates_and_anon_huge():
    rows = [row(0.0, ctxsw={"vol": 100, "invol": 10}, anon_huge_kb=0),
            row(10.0, ctxsw={"vol": 600, "invol": 50}, anon_huge_kb=2_000_000)]
    p = es.phase(rows, 0.0, 10.0, 10, 4)
    assert p["ctxsw_vol_per_s"] == pytest.approx(50.0)
    assert p["ctxsw_invol_per_s"] == pytest.approx(4.0)
    assert p["anon_huge_gb"] == pytest.approx(2.048)


def test_phase_without_the_files_is_null():
    p = es.phase([row(0.0), row(10.0)], 0.0, 10.0, 10, 4)
    assert p["ctxsw_vol_per_s"] is None and p["ctxsw_invol_per_s"] is None and p["anon_huge_gb"] is None


def test_membw_probe_json_shape():
    r = subprocess.run([sys.executable, str(SCRIPTS / "engine_sample.py"), "membw", "--seconds", "0.2", "--threads", "2", "--mib", "4"],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert set(d) == {"membw_gb_s", "threads", "seconds"}
    assert d["threads"] == 2 and d["seconds"] == 0.2 and d["membw_gb_s"] > 0


def test_ctxsw_rate_survives_an_exiting_thread(monkeypatch):
    """Threads 2 and 3 exit mid-window (their counters vanish) and thread 4 appears: the rate is never negative."""
    snaps = [
        {"/proc/1/task/1/status": status(100, 1), "/proc/1/task/2/status": status(5000, 50), "/proc/1/task/3/status": status(900, 9)},
        {"/proc/1/task/1/status": status(160, 2), "/proc/1/task/2/status": status(5200, 60), "/proc/1/task/3/status": status(950, 9)},
        {"/proc/1/task/1/status": status(200, 3), "/proc/1/task/4/status": status(30, 1)},
    ]
    layouts = [{1: ["1", "2", "3"]}, {1: ["1", "2", "3"]}, {1: ["1", "4"]}]
    rows = []
    for t, (files, tasks) in enumerate(zip(snaps, layouts)):
        fake(monkeypatch, files, tasks)
        rows.append(row(float(t), ctxsw=es.ctxsw([1]), ctxsw_tasks=es.ctxsw_tasks([1])))
    assert rows[2]["ctxsw"]["vol"] < rows[0]["ctxsw"]["vol"]  # the plain sum falls
    p = es.phase(rows, 0.0, 2.0, 2, 4)
    # t0->t1: 60 + 200 + 50; t1->t2: 40 + 30 (new thread, full count)
    assert p["ctxsw_vol_per_s"] == pytest.approx((60 + 200 + 50 + 40 + 30) / 2)
    assert p["ctxsw_invol_per_s"] == pytest.approx((1 + 10 + 0 + 1 + 1) / 2)


def stat_line(cpu, comm="engine (x)"):
    """A /proc/PID/task/TID/stat line: pid, (comm), state, then fields 4..52 with processor (39) = cpu."""
    rest = ["0"] * 49
    rest[35] = str(cpu)  # rest[0] is field 4, so field 39 is rest[35]
    return "1 (" + comm + ") S " + " ".join(rest)


def test_placement_reads_processor_and_node(monkeypatch):
    fake(monkeypatch, {"/proc/1/task/1/stat": stat_line(0), "/proc/1/task/2/stat": stat_line(65),
                       "/proc/1/task/3/stat": stat_line(66)}, {1: ["1", "2", "3"]})
    pl = es.placement([1], {0: 0, 65: 1})
    assert pl == {"cpus": [0, 65, 66], "nodes": [0, 1, None]}


def test_placement_without_files_is_none(monkeypatch):
    fake(monkeypatch, {}, {1: ["1"]})
    assert es.placement([1], {}) is None


def test_cpu_nodes_from_sysfs(monkeypatch):
    tree = {"/sys/devices/system/cpu": ["cpu0", "cpu1", "cpufreq", "online"],
            "/sys/devices/system/cpu/cpu0": ["node0", "topology"], "/sys/devices/system/cpu/cpu1": ["node1"]}
    monkeypatch.setattr(es, "sys_ls", lambda p: tree.get(p, []))
    assert es.cpu_nodes() == {0: 0, 1: 1}
    monkeypatch.setattr(es, "sys_ls", lambda p: [])
    assert es.cpu_nodes() == {}


def test_phase_reports_placement():
    rows = [row(0.0), row(1.0, placement={"cpus": [0, 1, 64], "nodes": [0, 0, 1]}),
            row(2.0, placement={"cpus": [0, 2, 3], "nodes": [0, 0, 0]}), row(3.0)]
    p = es.phase(rows, 1.0, 2.0, 2, 4)
    assert p["cpus_used"] == 5 and p["numa_nodes_used"] == 2
    assert p["top_node_share"] == pytest.approx(5 / 6)


def test_phase_placement_null_without_data_or_nodes():
    p = es.phase([row(0.0), row(1.0), row(2.0)], 0.5, 1.5, 1, 4)
    assert p["cpus_used"] is None and p["numa_nodes_used"] is None and p["top_node_share"] is None
    rows = [row(0.0), row(1.0, placement={"cpus": [0, 1], "nodes": [None, None]}), row(2.0)]
    p = es.phase(rows, 0.5, 1.5, 1, 4)
    assert p["cpus_used"] == 2 and p["numa_nodes_used"] is None and p["top_node_share"] is None
