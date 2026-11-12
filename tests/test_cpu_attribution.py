"""CPU attribution of the exit results (scripts/engine_sample.py, ADR 0058 addendum 2026-11-12): CPU pressure,
cgroup throttling (v2 and v1), steal share, mean MHz, the per-window fields, and the thread clamp."""

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("engine_sample", SCRIPTS / "engine_sample.py")
es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(es)


def fake(monkeypatch, files):
    monkeypatch.setattr(es.ds, "read", lambda p, default="": files.get(p, default))


PSI = "some avg10=0.00 avg60=0.00 avg300=0.00 total={s}\nfull avg10=0.00 avg60=0.00 avg300=0.00 total={f}\n"


def test_cpu_pressure_prefers_cgroup_then_proc_then_none(monkeypatch):
    fake(monkeypatch, {"/sys/fs/cgroup/cpu.pressure": PSI.format(s=1500, f=700), "/proc/pressure/cpu": PSI.format(s=9, f=1)})
    assert es.cpu_pressure() == {"src": "/sys/fs/cgroup/cpu.pressure", "some": 1500, "full": 700}
    fake(monkeypatch, {"/proc/pressure/cpu": "some avg10=0.00 avg60=0.00 avg300=0.00 total=9\n"})
    assert es.cpu_pressure() == {"src": "/proc/pressure/cpu", "some": 9, "full": None}
    fake(monkeypatch, {})
    assert es.cpu_pressure() == {"src": None, "some": None, "full": None}


def test_throttle_v2(monkeypatch):
    fake(monkeypatch, {"/sys/fs/cgroup/cpu.stat": "usage_usec 5\nnr_periods 100\nnr_throttled 20\nthrottled_usec 3000\n"})
    assert es.cgroup_throttle() == {"src": "/sys/fs/cgroup/cpu.stat", "periods": 100, "throttled": 20, "usec": 3000}


def test_throttle_v1_converts_ns_to_usec(monkeypatch):
    fake(monkeypatch, {"/sys/fs/cgroup/cpu,cpuacct/cpu.stat": "nr_periods 10\nnr_throttled 4\nthrottled_time 7000000\n"})
    assert es.cgroup_throttle() == {"src": "/sys/fs/cgroup/cpu,cpuacct/cpu.stat", "periods": 10, "throttled": 4, "usec": 7000}
    fake(monkeypatch, {"/sys/fs/cgroup/cpu/cpu.stat": "nr_periods 10\nnr_throttled 4\nthrottled_time 1000\n"})
    assert es.cgroup_throttle()["src"] == "/sys/fs/cgroup/cpu/cpu.stat"


def test_throttle_missing_or_without_quota_is_null(monkeypatch):
    null = {"src": None, "periods": None, "throttled": None, "usec": None}
    fake(monkeypatch, {})
    assert es.cgroup_throttle() == null
    fake(monkeypatch, {"/sys/fs/cgroup/cpu.stat": "usage_usec 5\nuser_usec 3\n"})
    assert es.cgroup_throttle() == null


def test_steal_and_mhz(monkeypatch):
    fake(monkeypatch, {"/proc/stat": "cpu  10 0 10 70 0 0 0 10 0 0\ncpu0 1 0 1 7 0 0 0 1 0 0\n",
                       "/proc/cpuinfo": "processor : 0\ncpu MHz\t\t: 2000.0\nprocessor : 1\ncpu MHz\t\t: 3000.0\n"})
    assert es.cpu_steal() == {"steal": 10, "total": 100}
    assert es.cpu_mhz() == 2500.0
    fake(monkeypatch, {})
    assert es.cpu_steal() is None
    assert es.cpu_mhz() is None


def row(t, **kw):
    r = {"t": t, "rss_kb": 1, "cpus": [], "cg_io": None}
    r.update(kw)
    return r


def test_phase_reports_cpu_deltas():
    def psi(s, f):
        return {"src": "p", "some": s, "full": f}

    rows = [
        row(0.0, cpu_psi=psi(0, 0), cg_throttle={"src": "s", "periods": 0, "throttled": 0, "usec": 0},
            cpu_steal={"steal": 0, "total": 100}, cpu_mhz=2000.0),
        row(10.0, cpu_psi=psi(2_000_000, 500_000), cg_throttle={"src": "s", "periods": 100, "throttled": 25, "usec": 4_000_000},
            cpu_steal={"steal": 5, "total": 200}, cpu_mhz=3000.0),
    ]
    p = es.phase(rows, 0.0, 10.0, 10, 4)
    assert p["cpu_psi_src"] == "p"
    assert p["cpu_some_share"] == pytest.approx(0.2)
    assert p["cpu_full_share"] == pytest.approx(0.05)
    assert p["throttled_share"] == pytest.approx(0.25)
    assert p["throttled_ms_per_s"] == pytest.approx(400.0)
    assert p["cpu_steal_share"] == pytest.approx(0.05)
    assert p["cpu_mhz_mean"] == pytest.approx(2500.0)


def test_phase_without_cpu_counters_is_null_not_an_error():
    p = es.phase([row(0.0), row(10.0)], 0.0, 10.0, 10, 4)
    for k in ("cpu_psi_src", "cpu_some_share", "cpu_full_share", "throttle_src", "throttled_share",
              "throttled_ms_per_s", "cpu_steal_share", "cpu_mhz_mean"):
        assert p[k] is None, k


def test_phase_mixed_sources_are_null():
    a = row(0.0, cpu_psi={"src": "a", "some": 0, "full": 0}, cg_throttle={"src": "a", "periods": 0, "throttled": 0, "usec": 0})
    b = row(10.0, cpu_psi={"src": "b", "some": 9, "full": 9}, cg_throttle={"src": "b", "periods": 9, "throttled": 9, "usec": 9})
    p = es.phase([a, b], 0.0, 10.0, 10, 4)
    assert p["cpu_some_share"] is None and p["throttled_share"] is None


def test_decode_windows_carry_the_cpu_fields():
    rows = [row(float(t), cpu_psi={"src": "p", "some": t * 1_000_000, "full": 0}) for t in range(0, 11)]
    out = es.host_summary(rows, (0.0, 2.0, 5), (2.0, 10.0, 8), 4, [])
    assert len(out["decode_windows"]) == es.N_DECODE_WINDOWS
    assert all(w["cpu_some_share"] == pytest.approx(1.0) for w in out["decode_windows"])
    assert "throttled_share" in out["decode"] and "cpu_steal_share" in out["prefill"]


# thread clamp (scripts/clamp_threads.sh)

BASH = shutil.which("bash")


def clamp(n, quota, **env):
    e = {**os.environ, **env}
    e.pop("THREADS_ALLOW_OVER_QUOTA", None) if "THREADS_ALLOW_OVER_QUOTA" not in env else None
    r = subprocess.run([BASH, str(SCRIPTS / "clamp_threads.sh"), str(n), str(quota)], capture_output=True, text=True, env=e)
    return r.stdout.strip(), r.stderr


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_clamp_lowers_threads_above_quota_and_logs():
    out, err = clamp(24, 23)
    assert out == "23" and "clamp_threads.sh" in err and "exceeds" in err


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_clamp_keeps_threads_at_or_below_quota():
    assert clamp(23, 23) == ("23", "")
    assert clamp(8, 23) == ("8", "")


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_clamp_override_keeps_threads():
    assert clamp(24, 23, THREADS_ALLOW_OVER_QUOTA="1") == ("24", "")
