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
