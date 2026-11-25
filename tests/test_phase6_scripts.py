"""scripts/phase6_casef.py applies the client-side Case F rules; scripts/phase6_cluster_toml.py writes loopback self entries."""

import importlib.util
import pathlib

S = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, S / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def row(t0, t1, **kw):
    r = {"t_submit": t0, "t_end": t1, "t_first": t0 + 0.1, "status": 200, "chunks": 5, "finish": "length",
         "error": None, "done": True, "after_error": 0}
    r.update(kw)
    return r


def err(t0, t1):
    return row(t0, t1, finish=None, error='{"message":"replica failed"}', done=False)


def test_in_flight_errors_within_bound_pass():
    m = load("phase6_casef")
    rows = [row(0, 5), err(8, 11.5), row(12, 20), row(13, 21)]
    r = m.analyse(rows, 10.0, 3.0)
    assert r["in_flight_at_fault"] == 1 and r["in_flight_errors"] == 1
    assert r["rules"]["inflight_errors_in_bound"] and r["rules"]["one_clean_error_each"]
    assert r["submitted_after_bound"] == 1 and r["rules"]["none_after_bound_failed"]
    assert r["rules"]["other_failures_zero"] and r["rules"]["exactly_once"]


def test_late_error_and_later_failure_fail_the_rules():
    m = load("phase6_casef")
    rows = [err(8, 14.0), err(14, 15), row(1, 2)]
    r = m.analyse(rows, 10.0, 3.0)
    assert not r["rules"]["inflight_errors_in_bound"]
    assert not r["rules"]["none_after_bound_failed"]


def test_cluster_toml_self_entry_is_loopback():
    m = load("phase6_cluster_toml")
    t = m.render("c", "a", {"a": "1.2.3.4:5", "b": "6.7.8.9:10"}, "b", 7000, 2000)
    assert 'addr = "1.2.3.4:5"' in t and 'addr = "127.0.0.1:7000"' in t and "key" not in t.lower()
