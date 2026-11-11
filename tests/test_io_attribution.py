"""I/O attribution of the exit results (scripts/engine_sample.py, ADR 0056 addendum): io_class, disk_util, the I/O
pressure sampler, the KTransformers host merge, and four regressions from committed results."""

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

CURVE = [(1.0, 2.0e9), (2.0, 4.0e9), (4.0, 6.0e9), (16.0, 7.0e9)]
RESULTS = ROOT / "bench" / "results"


def cls(**kw):
    return es.io_class(kw)


# io_class branches

def test_class_not_io_bound_below_threshold():
    assert cls(io_some_share=0.1, disk_util=None) == ("not_io_bound", [])


def test_class_disk_bandwidth():
    assert cls(io_some_share=0.5, disk_util=0.9, disk_queue_depth_mean=8)[0] == "disk_bandwidth"


def test_class_io_latency_on_serial_reads():
    assert cls(io_some_share=0.5, disk_util=0.1, disk_queue_depth_mean=0.4)[0] == "io_latency"


def test_class_io_under_issue():
    assert cls(io_some_share=0.5, disk_util=0.3, disk_queue_depth_mean=6)[0] == "io_under_issue"


def test_class_uses_the_larger_share():
    assert cls(io_some_share=0.05, read_wait_share=0.6, disk_util=0.1, disk_queue_depth_mean=5)[0] == "io_under_issue"


def test_class_disk_util_decides_before_the_stall_share():
    assert cls(io_some_share=0.01, disk_util=0.95)[0] == "disk_bandwidth"
    assert cls(disk_util=0.95)[0] == "disk_bandwidth"


def test_class_unknown_lists_missing_inputs():
    assert cls() == ("unknown", ["io_some_share"])
    assert cls(disk_util=0.2) == ("unknown", ["io_some_share"])
    assert cls(io_some_share=0.5, disk_queue_depth_mean=3) == ("unknown", ["disk_util"])
    assert cls(io_some_share=0.5, disk_util=0.2) == ("unknown", ["disk_queue_depth_mean"])


def test_classify_adds_the_numbers_and_the_ceiling():
    h = {"read_mb_s": 2500.0, "disk_queue_depth_mean": 3.0, "read_bytes_per_token": 5e8}
    es.classify(h, CURVE, (0.4, False))
    assert h["bytes_per_token"] == 5e8 and h["disk_util"] == pytest.approx(0.5)
    assert h["tok_s_ceiling_disk"] == pytest.approx(10.0)  # 5e9 B/s over 5e8 B
    assert h["io_class"] == "io_under_issue"
    no_curve = es.classify({"read_mb_s": 1.0, "disk_queue_depth_mean": 3.0, "read_bytes_per_token": 5e8}, [])
    assert no_curve["tok_s_ceiling_disk"] is None and no_curve["disk_util"] is None


def test_prefetch_only_read_wait_is_unreliable_not_zero():
    prof = {"read_wait_ms": 0.0, "total_ms": 900.0}
    assert es.read_wait_share(prof, {"bytes": 5e9}) == (None, True)
    assert es.read_wait_share(prof, {"bytes": 0}) == (0.0, False)
    assert es.read_wait_share({"read_wait_ms": 90.0, "total_ms": 900.0}, {"bytes": 5e9}) == (pytest.approx(0.1), False)
    h = es.classify({"read_mb_s": 1.0, "disk_queue_depth_mean": 3.0}, CURVE, es.read_wait_share(prof, {"bytes": 5e9}))
    assert h["read_wait_share"] is None and h["read_wait_share_unreliable"] is True
    assert h["io_class"] == "unknown"


def test_host_summary_splits_baseline_decode_into_four_estimated_windows():
    rows = [rec(t, some=t * 300_000, full=0, wms=t * 1000, rb=t * 10**8) for t in range(0, 41)]
    out = es.host_summary(rows, (0, 10, 100), (10, 30, 8), 4, CURVE)
    assert len(out["decode_windows"]) == 4
    assert all(w["tokens_estimated"] and w["tokens"] == 2 for w in out["decode_windows"])
    assert out["decode_windows"][0]["io_some_share"] == pytest.approx(0.3)


# disk_util

def test_disk_util_interpolates_between_points():
    # depth 3 is halfway between 4e9 and 6e9: 5e9 B/s
    assert es.disk_util(2500.0, 3.0, CURVE) == pytest.approx(0.5)


def test_disk_util_clamps_to_the_curve_range():
    assert es.disk_util(1000.0, 0.37, CURVE) == pytest.approx(0.5)  # depth 1: 2e9
    assert es.disk_util(3500.0, 64.0, CURVE) == pytest.approx(0.5)  # depth 16: 7e9


def test_disk_util_is_null_without_inputs():
    assert es.disk_util(100.0, 2.0, []) is None
    assert es.disk_util(None, 2.0, CURVE) is None
    assert es.disk_util(100.0, None, CURVE) is None


def test_load_and_find_curve(tmp_path):
    (tmp_path / "disk-curve-loaded-1.json").write_text(json.dumps({"points": [{"depth": 4, "bytes_per_s": 3e9},
                                                                              {"depth": 1, "bytes_per_s": 1e9}]}))
    assert es.load_curve(es.find_curve(str(tmp_path))) == [(1.0, 1e9), (4.0, 3e9)]
    assert es.find_curve(str(tmp_path / "none")) is None
    assert es.load_curve(str(tmp_path / "missing.json")) == []


# I/O pressure sampler

def test_io_pressure_prefers_the_cgroup_file(monkeypatch):
    files = {
        "/sys/fs/cgroup/io.pressure": "some avg10=0.00 avg60=0.00 avg300=0.00 total=1500\n"
                                       "full avg10=0.00 avg60=0.00 avg300=0.00 total=700\n",
        "/proc/pressure/io": "some avg10=0.00 avg60=0.00 avg300=0.00 total=9\n",
    }
    monkeypatch.setattr(es.ds, "read", lambda p, default="": files.get(p, default))
    assert es.io_pressure() == {"src": "/sys/fs/cgroup/io.pressure", "some": 1500, "full": 700}


def test_io_pressure_falls_back_to_proc_then_none(monkeypatch):
    files = {"/proc/pressure/io": "some avg10=0.00 avg60=0.00 avg300=0.00 total=9\n"}
    monkeypatch.setattr(es.ds, "read", lambda p, default="": files.get(p, default))
    assert es.io_pressure() == {"src": "/proc/pressure/io", "some": 9, "full": None}
    monkeypatch.setattr(es.ds, "read", lambda p, default="": default)
    assert es.io_pressure() == {"src": None, "some": None, "full": None}


def rec(t, some, full, wms=0, rb=0):
    return {"t": t, "disk_read_bytes": rb, "disk_read_ops": 0, "disk_inflight": 1, "disk_weighted_ms": wms,
            "cg_io": None, "majflt": 0, "cpus": [], "cg_cpu_usec": 0, "own_cpu_s": 0, "rss_kb": 1,
            "cg_mem": {"file": 1, "anon": 1}, "gpu": None,
            "io_psi": {"src": "/sys/fs/cgroup/io.pressure", "some": some, "full": full}}


def test_phase_reports_pressure_shares():
    rows = [rec(t, some=t * 300_000, full=t * 100_000, wms=t * 1000, rb=t * 10**8) for t in range(0, 11)]
    h = es.phase(rows, 0, 10, 10, 4)
    assert h["io_some_share"] == pytest.approx(0.3)
    assert h["io_full_share"] == pytest.approx(0.1)
    assert h["io_psi_src"] == "/sys/fs/cgroup/io.pressure"


def test_phase_without_pressure_is_null():
    rows = [rec(t, None, None) for t in range(0, 11)]
    for r in rows:
        r["io_psi"] = {"src": None, "some": None, "full": None}
    assert es.phase(rows, 0, 10, 10, 4)["io_some_share"] is None
    rows = [{k: v for k, v in r.items() if k != "io_psi"} for r in rows]  # samples from before this field
    assert es.phase(rows, 0, 10, 10, 4)["io_some_share"] is None


def test_summarize_end_to_end_on_a_kt_result(tmp_path):
    rows = [rec(t, some=t * 600_000, full=0, wms=t * 400, rb=t * 7 * 10**7) for t in range(0, 21)]
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "res.json").write_text(json.dumps({"t_start": 0, "t_first": 5, "t_end": 20, "decode_tokens": 11,
                                                   "prompt_tokens": 100}))
    (tmp_path / "disk-curve-loaded-1.json").write_text(
        json.dumps({"points": [{"depth": 1, "bytes_per_s": 2e9}, {"depth": 16, "bytes_per_s": 7e9}]}))
    out = tmp_path / "sum.json"
    es.summarize(types.SimpleNamespace(samples=str(tmp_path / "s.jsonl"), threads=4, builder_json=None,
                                       kt_json=str(tmp_path / "res.json"), out=str(out), merge_into=None,
                                       curve_dir=str(tmp_path), curve=None))
    d = json.loads(out.read_text())["host"]["decode"]
    assert d["io_some_share"] == pytest.approx(0.6)
    assert d["disk_util"] == pytest.approx(70.0 / 2000.0)
    assert d["io_class"] == "io_latency"


# KTransformers host merge

def test_kt_summary_fills_host_from_the_summary(tmp_path):
    f = tmp_path / "host-deep.json"
    f.write_text(json.dumps({"host": {"decode": {"read_mb_s": 69.39}}, "env": {"cpu_model": "x"}}))
    got = es.kt_summary(str(f))
    assert got["host"]["decode"]["read_mb_s"] == 69.39 and got["env"] == {"cpu_model": "x"}


def test_kt_summary_without_a_summary_is_an_error_not_empty(tmp_path):
    for content in (None, "not json", "{}"):
        f = tmp_path / "h.json"
        f.unlink(missing_ok=True)
        if content is not None:
            f.write_text(content)
        got = es.kt_summary(str(f))
        assert got["host"].get("error")


# regressions from committed results

def builder_share(date, name):
    d = json.loads((RESULTS / date / name).read_text())
    return es.read_wait_share(d["runs"][0]["decode_profile_per_token"])[0]


def test_regression_slow_disk_overlap_is_not_not_io_bound():
    s = builder_share("20261111b", "exit-main-overlap-100117.json")
    assert s == pytest.approx(0.80, abs=0.01)
    # no committed curve for this box: unknown (missing disk_util), never not_io_bound
    host = {"read_wait_share": s, "disk_queue_depth_mean": 17.2, "read_mb_s": 514.6}
    assert es.io_class(host) == ("unknown", ["disk_util"])


def test_regression_fast_disk_overlap_is_not_io_bound():
    s = builder_share("20261111a", "exit-main-overlap-025210.json")
    assert s == pytest.approx(0.11, abs=0.01)
    assert es.io_class({"read_wait_share": s})[0] == "not_io_bound"


def test_regression_w4_is_a_known_miss_without_a_disk_curve():
    # share 0.24, slower than the default because 4 workers under-issue: the rule gives not_io_bound; only a curve's
    # disk_util and tok_s_ceiling_disk would show it
    s = builder_share("20261111a", "exit-main-overlap-w4-025210.json")
    assert s == pytest.approx(0.24, abs=0.01)
    assert es.io_class({"read_wait_share": s})[0] == "not_io_bound"


def test_regression_kt_20261111c_from_its_committed_host_block():
    d = json.loads((RESULTS / "20261111c" / "exit-ktransformers-123631.json").read_text())["host"]["decode"]
    assert d["read_mb_s"] == pytest.approx(69.39, abs=0.01)
    assert d["disk_queue_depth_mean"] == pytest.approx(0.371, abs=0.001)
    curve = es.load_curve(str(RESULTS / "20261111c" / "disk-curve-loaded-123631.json"))
    es.classify(d, curve)
    assert d["disk_util"] < 0.1
    assert d["tok_s_ceiling_disk"] > 5  # the disk could feed far more tokens/s than the run made
    # no stall share was logged: unknown (missing io_some_share), disk_util does not decide
    assert (d["io_class"], d["io_class_missing"]) == ("unknown", ["io_some_share"])
    d["io_some_share"] = 0.5  # with a stall share the serial reads show
    assert es.io_class(d) == ("io_latency", [])
