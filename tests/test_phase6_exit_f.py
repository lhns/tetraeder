"""Case F summary of the Phase 6 exit (scripts/phase6_exit_f_summary.py): every pass rule of ADR 0091 Case F as code,
read on the rehearsal's real data and on synthetic sessions (one rule broken at a time). Hardware label: none."""

import copy
import gzip
import json
import pathlib
import shutil
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import phase6_exit_f_summary as fs  # noqa: E402
from test_sc_gates import LSCPU_1, LSCPU_2, reading, write  # noqa: E402

REHEARSAL = ROOT / "bench" / "results" / "phase6-rehearsal-20261125"
REPLICA_B_ERR = '{"message": "engine error: the connection to replica 1 was lost", "type": "server_error", "code": 500}'


def good_session(**o):
    s = {
        "model_sha256": "0d00" * 16,
        "replica_b": "replica 1",
        "bound_s": 3.0,
        "rejoin_limit_s": 60.0,
        "trials_per_mode": 3,
        "boxes": {"a": {"model_sha256": "0d00" * 16}, "b": {"model_sha256": "0d00" * 16}},
        "certs": {"a": "c1" * 32, "b": "c2" * 32, "pinned_on": ["a", "b"]},
        "t_created": {"a": 1000.0, "b": 1010.0},
        "t_cluster_toml": 1500.0,
        "load": {"concurrency": 8, "duration_s": 180, "prompt_tokens": 510, "max_tokens": 256, "stream": True},
        "clock": {"b_minus_a_s": 0.2, "uncertainty_s": 0.1},
    }
    s.update(o)
    return s


def write_gates(res, lscpu=LSCPU_1, cot=None, gates=None):
    for box in ("a", "b"):
        d = res / "gates" / box
        write(d / "lscpu.txt", lscpu if box == "b" else LSCPU_1)
        write(d / "cotenant.json", cot or reading(1.0, 256))
        write(d / "gates.json", gates or {"power_ok": True, "idle_ok": True})


def gen_rows(label, t0, tf, delay=1.8, dur=4.0, total=180.0, msg=REPLICA_B_ERR, extra_fail_after=False):
    """Eight closed-loop workers; the even ones run on B and fail `delay` s after the fault, then go on with A."""
    rows, n = [], 0
    for w in range(8):
        t = t0 + 0.05 * w
        while t < t0 + total:
            n += 1
            end = t + dur
            row = {"t_submit": t, "t_end": end, "t_first": t + 0.2, "status": 200, "chunks": 255, "finish": "length", "error": None,
                   "done": True, "after_error": 0, "n": n, "worker": w, "label": label}
            if w % 2 == 0 and t < tf <= end:
                row.update(t_end=tf + delay, chunks=40, finish=None, error=msg, done=False)
                end = tf + delay
            if extra_fail_after and w == 1 and t >= tf + 5 and not any(r.get("_x") for r in rows):
                row.update(finish=None, error="engine error: something else", done=False, _x=True)
            rows.append(row)
            t = end + 0.01
    return rows


def gateway_entries(rows):
    return [{"ts_ms": int(r["t_submit"] * 1000), "path": "/v1/chat/completions", "status": 200, "stream": True,
             "error_type": "internal" if r["error"] else None,
             "request": {"model": "x", "messages": [{"role": "user", "content": f"[{r['label']}-{r['n']}] The quick brown fox"}]}}
            for r in rows]


def write_trial(res, mode, n, *, t0=5000.0, delay=1.8, rejoin=12.0, admits_before=True, restore_after=2.3, meta_extra=None, **kw):
    meta_extra = meta_extra or {}
    label = f"{mode}-t{n}"
    tf = t0 + 60.0
    tr = tf + restore_after
    rows = gen_rows(label, t0, tf, delay=delay, **{k: v for k, v in kw.items() if k in ("dur", "total", "msg", "extra_fail_after")})
    meta = {"mode": mode, "trial": n, "t_fault": tf, "t_restore": tr, "load_label": label, "load_end": t0 + 190.0,
            "fault": {"source": "trigger", "live_b": 4, "uncertainty_s": 0.03}, **meta_extra}
    d = res / "trials" / f"{mode}-{n}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "load.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    off = 0.2  # node b's clock is 0.2 s ahead of the client's
    adm = ([tf - 50 + off, tf - 20 + off] if admits_before else [])
    if rejoin is not None:
        adm.append(tr + rejoin + off)
    (d / "b-admits.txt").write_text("".join(f"{a:.3f}\n" for a in adm))
    return rows, meta


def build(tmp_path, *, session=None, mutate=None, gw_drop=None, skip=()):
    """A passing session of nine trials; `mutate(mode, n, kwargs)` changes one trial's generator arguments."""
    res = tmp_path / "res"
    write(res / "session.json", session or good_session())
    write_gates(res)
    gw = []
    for i, mode in enumerate(fs.MODES):
        for n in (1, 2, 3):
            if (mode, n) in skip:
                continue
            kw = {"t0": 5000.0 + 1000 * (3 * i + n)}
            if mutate:
                mutate(mode, n, kw)
            rows, _ = write_trial(res, mode, n, **kw)
            gw += gateway_entries(rows)
    if gw_drop:
        gw = gw_drop(gw)
    (res / "head-requests.jsonl").write_text("".join(json.dumps(e) + "\n" for e in gw))
    return res


def summary(res):
    return fs.summarize(str(res))


def failing(o):
    return sorted({f"{t['mode']}-{t['trial']}.{k}" for t in o["trials"] for k, r in t["rules"].items() if not r["ok"]}
                  | {r["name"] for r in o["session_rules"] if not r["ok"]})


# ---- a pass, and each rule failing alone --------------------------------------------------------------------------------


def test_a_clean_session_of_nine_trials_passes(tmp_path):
    o = summary(build(tmp_path))
    assert failing(o) == [] and o["verdict"] == "pass" and o["pass"] is True
    assert len(o["trials"]) == 9 and o["reading"] == "every rule holds in every mode"
    assert all(t["stats"]["in_flight_errors"] == 4 for t in o["trials"])


def one_trial(**k):
    return lambda mode, n, kw: kw.update(k) if (mode, n) == ("daemon", 2) else None


@pytest.mark.parametrize("name,mutate,expect", [
    ("late error", one_trial(delay=3.5), "daemon-2.clean_error_in_bound"),
    ("error that names replica 0", one_trial(msg="engine error: the connection to replica 0 was lost"), "daemon-2.no_failure_on_r_a"),
    ("a failure after the bound", one_trial(extra_fail_after=True), "daemon-2.ok_after_bound"),
    ("B not rejoined", one_trial(rejoin=61.0), "daemon-2.rejoin"),
    ("B never serving before", one_trial(admits_before=False), "daemon-2.b_served_before"),
])
def test_each_trial_rule_fails_alone(tmp_path, name, mutate, expect):
    o = summary(build(tmp_path, mutate=mutate))
    assert expect in failing(o), name
    assert o["verdict"] == "fail" and o["pass"] is False
    # only the one trial changed
    assert {f.split(".")[0] for f in failing(o) if "." in f} == {"daemon-2"}


def test_a_fault_that_hits_nothing_fails_the_clean_error_rule(tmp_path):
    def m(mode, n, kw):
        if (mode, n) == ("stop", 3):
            kw.update(msg=REPLICA_B_ERR, dur=0.5)  # requests too short to be in flight: only two or fewer cross the fault

    o = summary(build(tmp_path, mutate=m))
    t = next(t for t in o["trials"] if (t["mode"], t["trial"]) == ("stop", 3))
    assert t["stats"]["in_flight_at_fault"] <= 8
    if not t["stats"]["in_flight_errors"]:
        assert not t["rules"]["clean_error_in_bound"]["ok"]


def test_queued_requests_that_fail_without_a_clean_b_error_fail(tmp_path):
    res = build(tmp_path)
    d = res / "trials" / "worker-1"
    rows = [json.loads(x) for x in (d / "load.jsonl").read_text().splitlines()]
    tf = json.loads((d / "meta.json").read_text())["t_fault"]
    victim = next(r for r in rows if tf <= r["t_submit"] < tf + 3.0)
    victim.update(error="engine error: queue overflow", finish=None, done=False, t_end=victim["t_submit"] + 1)
    (d / "load.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    o = summary(res)
    assert "worker-1.queued_move" in failing(o)


def test_gateway_and_client_counts_must_agree(tmp_path):
    o = summary(build(tmp_path, gw_drop=lambda gw: gw[:-1]))  # one request without a log entry
    assert any(k.endswith(".exactly_once") for k in failing(o))
    res = build(tmp_path / "x", gw_drop=lambda gw: gw + [dict(gw[0])])  # one request counted twice
    assert any(k.endswith(".exactly_once") for k in failing(summary(res)))
    res = build(tmp_path / "y")  # the gateway says ok where the client saw an error
    lines = [json.loads(x) for x in (res / "head-requests.jsonl").read_text().splitlines()]
    for e in lines:
        e["error_type"] = None
    (res / "head-requests.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
    assert any(k.endswith(".exactly_once") for k in failing(summary(res)))


def test_tokens_after_an_error_or_a_missing_done_break_exactly_once(tmp_path):
    res = build(tmp_path)
    d = res / "trials" / "stop-2"
    rows = [json.loads(x) for x in (d / "load.jsonl").read_text().splitlines()]
    next(r for r in rows if r["error"])["after_error"] = 2
    (d / "load.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert "stop-2.exactly_once" in failing(summary(res))


def test_the_head_log_is_read_gzipped_too(tmp_path):
    res = build(tmp_path)
    raw = (res / "head-requests.jsonl").read_bytes()
    (res / "head-requests.jsonl").unlink()
    (res / "head-requests.jsonl.gz").write_bytes(gzip.compress(raw))
    assert summary(res)["verdict"] == "pass"


def test_trial_one_needs_no_manual_action_but_later_trials_may_have_one(tmp_path):
    res = build(tmp_path)
    tf = 5000.0 + 1000 * (3 * 1 + 1) + 60.0  # daemon, trial 1
    write(res / "manual.jsonl", json.dumps({"t": tf + 50, "what": "head restarted by hand"}) + "\n")
    o = summary(res)
    assert failing(o) == ["daemon-1.trial1_no_manual"] and o["verdict"] == "fail"
    tf2 = 5000.0 + 1000 * (3 * 1 + 2) + 60.0  # daemon, trial 2: an action there is not a trial 1 rule
    write(res / "manual.jsonl", json.dumps({"t": tf2 + 50, "what": "head restarted by hand"}) + "\n")
    assert failing(summary(res)) == []
    # an action after the deadline of trial 1 does not count either
    write(res / "manual.jsonl", json.dumps({"t": tf + 70, "what": "late"}) + "\n")
    assert failing(summary(res)) == []


def test_an_admit_after_the_window_does_not_count_as_a_rejoin(tmp_path):
    o = summary(build(tmp_path, mutate=one_trial(rejoin=126.5)))  # the rehearsal's +126 s after a manual head restart
    assert "daemon-2.rejoin" in failing(o)


# ---- missing evidence never passes ---------------------------------------------------------------------------------------


def test_missing_trials_and_files_do_not_pass(tmp_path):
    o = summary(build(tmp_path, skip={("stop", 3)}))
    assert o["verdict"] == "incomplete" and o["pass"] is False and o["missing_trials"] == ["stop-3"]
    res = build(tmp_path / "n")
    (res / "head-requests.jsonl").unlink()
    o = summary(res)
    assert o["verdict"] == "fail" and any(k.endswith(".exactly_once") for k in failing(o))
    res = build(tmp_path / "m")
    (res / "trials" / "worker-2" / "b-admits.txt").unlink()
    assert "worker-2.rejoin" in failing(summary(res))
    res = build(tmp_path / "k")
    (res / "gates" / "b" / "gates.json").unlink()
    assert "gates_b" in failing(summary(res))


@pytest.mark.parametrize("what,session,rule", [
    ("model differs", good_session(boxes={"a": {"model_sha256": "0d00" * 16}, "b": {"model_sha256": "ffff" * 16}}), "same_model"),
    ("certificates not pinned on both", good_session(certs={"a": "c1", "b": "c2", "pinned_on": ["a"]}), "certificates_pinned"),
    ("cluster.toml before create", good_session(t_cluster_toml=1005.0), "cluster_toml_after_create"),
    ("prompt of 395 tokens", good_session(load={"concurrency": 8, "duration_s": 180, "prompt_tokens": 395, "max_tokens": 256, "stream": True}), "load_shape"),
    ("concurrency 4", good_session(load={"concurrency": 4, "duration_s": 180, "prompt_tokens": 512, "max_tokens": 256, "stream": True}), "load_shape"),
    ("clock offset unknown", good_session(clock={}), "clock_offset"),
    ("clock uncertainty 1 s", good_session(clock={"b_minus_a_s": 0.0, "uncertainty_s": 1.0}), "clock_offset"),
])
def test_session_rules(tmp_path, what, session, rule):
    o = summary(build(tmp_path, session=session))
    assert failing(o) == [rule], what
    assert o["verdict"] == "fail"


def test_gates_on_both_boxes(tmp_path):
    res = build(tmp_path)
    write(res / "gates" / "b" / "lscpu.txt", LSCPU_2)
    o = summary(res)
    assert failing(o) == ["gates_b"] and "2 NUMA nodes" in next(r["detail"] for r in o["session_rules"] if r["name"] == "gates_b")
    res = build(tmp_path / "c")
    write(res / "gates" / "a" / "cotenant.json", reading(30.0, 256))
    assert failing(summary(res)) == ["gates_a"]
    res = build(tmp_path / "p")
    write(res / "gates" / "a" / "gates.json", {"power_ok": False, "idle_ok": True})
    assert failing(summary(res)) == ["gates_a"]


def test_the_clock_offset_moves_node_b_s_admits_onto_the_clients_clock(tmp_path):
    # b's clock is 0.2 s ahead: an admit at restore + 59.9 on the client's clock is written as +60.1 on b's and must count
    res = build(tmp_path, mutate=one_trial(rejoin=59.9))
    assert failing(summary(res)) == []
    res = build(tmp_path / "late", mutate=one_trial(rejoin=60.1))
    assert failing(summary(res)) == ["daemon-2.rejoin"]


# ---- the rehearsal's real data ---------------------------------------------------------------------------------------------


def rehearsal_session(tmp_path):
    """The rehearsal's one trial per mode (a/caseF) as a session directory."""
    adm = json.loads((ROOT / "tests" / "fixtures" / "phase6" / "rehearsal-b-admits.json").read_text())["b_admits"]
    res = tmp_path / "rehearsal"
    # the box's own facts: one trial per mode, a prompt of 395 tokens, clocks agreeing within the ssh latency (about 1 s)
    s = good_session(trials_per_mode=1, load={"concurrency": 8, "duration_s": 180, "prompt_tokens": 395, "max_tokens": 256, "stream": True},
                     clock={"b_minus_a_s": 0.0, "uncertainty_s": 1.0})
    write(res / "session.json", s)
    write_gates(res)
    shutil.copy(REHEARSAL / "a" / "head-requests.jsonl.gz", res / "head-requests.jsonl.gz")
    for mode in fs.MODES:
        r = json.loads((REHEARSAL / "a" / "caseF" / f"result-{mode}.json").read_text())
        rows = (REHEARSAL / "a" / "caseF" / f"load-{mode}.jsonl").read_text()
        d = res / "trials" / f"{mode}-1"
        d.mkdir(parents=True)
        (d / "load.jsonl").write_text(rows)
        tf = r["t_fault"]
        # restore: the daemon's restart of the engine at +2.3 s for the worker kill (README), the daemon start and the SIGCONT
        tr = r.get("t_restore") or tf + 2.3
        end = max(json.loads(x)["t_end"] for x in rows.splitlines())
        (d / "meta.json").write_text(json.dumps({"mode": mode, "trial": 1, "t_fault": tf, "t_restore": tr,
                                                 "load_label": mode, "load_end": end,
                                                 "fault": {"source": "rehearsal", "live_b": 4, "uncertainty_s": 0.0}}))
        (d / "b-admits.txt").write_text("".join(f"{a:.6f}\n" for a in adm))
    return res


def test_the_rehearsal_reads_modes_1_and_2_fail_on_rejoin(tmp_path):
    o = summary(rehearsal_session(tmp_path))
    assert o["reading"] == "modes 1 and 2 fail on rejoin"
    by = {t["mode"]: t for t in o["trials"]}
    for mode, delay in (("worker", 2.786), ("daemon", 1.865), ("stop", 2.078)):  # the 3 s rule holds in all three
        t = by[mode]
        assert t["rules"]["clean_error_in_bound"]["ok"], mode
        assert max(t["stats"]["in_flight_error_delay_s"]) <= 3.0 and abs(max(t["stats"]["in_flight_error_delay_s"]) - delay) < 0.01
        assert t["stats"]["in_flight_errors"] == 4
    assert by["stop"]["rules"]["rejoin"]["ok"] and abs(by["stop"]["stats"]["rejoin_s"] - 0.78) < 0.01  # README: 0.78 s after SIGCONT
    for mode in ("worker", "daemon"):
        assert not by[mode]["rules"]["rejoin"]["ok"]
    # the client and the head's request log agree in all three (the README had not compared them)
    assert all(t["rules"]["exactly_once"]["ok"] for t in o["trials"])
    assert all(t["rules"]["no_failure_on_r_a"]["ok"] and t["rules"]["ok_after_bound"]["ok"] for t in o["trials"])
    assert o["verdict"] == "fail" and not o["pass"]


def test_the_cli_exits_one_and_writes_the_verdict(tmp_path):
    import subprocess

    res = rehearsal_session(tmp_path)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_f_summary.py"), str(res)], capture_output=True, text=True)
    assert r.returncode == 1 and "modes 1 and 2 fail on rejoin" in r.stdout
    assert json.loads((res / "verdict-f.json").read_text())["reading"] == "modes 1 and 2 fail on rejoin"
    ok = build(tmp_path / "ok")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_f_summary.py"), str(ok)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_admit_epochs_of_a_node_log():
    text = ("\x1b[2m2026-11-25T04:03:02.751715Z\x1b[0m \x1b[34mDEBUG\x1b[0m \x1b[2mtetraeder::step\x1b[0m\x1b[2m:\x1b[0m admit seq=0 queued_ms=0.13\n"
            "2026-11-25T04:03:03.500000Z  INFO tetraeder::node: other line\n"
            "2026-11-25T04:03:38.901221Z DEBUG tetraeder::step: admit seq=1 prompt=393\n")
    e = fs.admit_epochs(text)
    assert len(e) == 2 and abs(e[0] - 1795579382.751715) < 1e-6 and abs(e[1] - e[0] - 36.149506) < 1e-6


def test_the_clock_offset_between_the_boxes_keeps_the_best_sample():
    # driver clock 100.0..100.4 around A's reading 90.2 (A is 10.0 s behind) and 100.5..100.7 around B's 100.6 (B is 0.0 s off)
    good = [100.0, 90.2, 100.4, 100.5, 100.6, 100.7]
    noisy = [200.0, 190.2, 201.0, 201.5, 201.6, 203.5]
    r = fs.clock_offset([noisy, good])
    assert abs(r["b_minus_a_s"] - 10.0) < 1e-6 and abs(r["uncertainty_s"] - 0.3) < 1e-6 and r["samples"] == 2
    assert fs.clock_offset([]) == {}


# ---- the trigger's evidence: attempts, repeats and the clock ----------------------------------------------------------------


def shift_attempts(res, mode, by):
    """Renumbers the attempts of a mode up by `by` and writes `by` new first attempts whose fault found nothing in flight on B."""
    for n in (3, 2, 1):
        (res / "trials" / f"{mode}-{n}").rename(res / "trials" / f"{mode}-{n + by}")
    for k in range(1, by + 1):
        write_trial(res, mode, 90 + k, t0=900000.0 + 1000 * k, meta_extra={"fault": {"source": "trigger", "live_b": 0, "uncertainty_s": 0.02}})
        (res / "trials" / f"{mode}-{90 + k}").rename(res / "trials" / f"{mode}-{k}")


def test_an_attempt_whose_fault_found_nothing_on_b_is_repeated_and_reported(tmp_path):
    res = build(tmp_path)
    shift_attempts(res, "daemon", 1)
    o = summary(res)
    assert failing(o) == [] and o["verdict"] == "pass" and len(o["trials"]) == 9
    skipped = [a for a in o["attempts"] if not a["judged"]]
    assert [(a["mode"], a["attempt"]) for a in skipped] == [("daemon", 1)] and skipped[0]["live_b"] == 0
    assert [t["attempt"] for t in o["trials"] if t["mode"] == "daemon"] == [2, 3, 4]
    assert len(o["attempts"]) == 10
    # trial 1 is the first valid attempt: the manual-action rule reads it, not the repeated one
    tf = 5000.0 + 1000 * (3 * 1 + 1) + 60.0
    write(res / "manual.jsonl", json.dumps({"t": tf + 50, "what": "head restarted by hand"}) + "\n")
    assert failing(summary(res)) == ["daemon-1.trial1_no_manual"]


def test_two_repeats_in_a_mode_are_allowed_and_three_are_not_a_pass(tmp_path):
    res = build(tmp_path)
    shift_attempts(res, "stop", 2)
    o = summary(res)
    assert failing(o) == [] and o["verdict"] == "pass" and len(o["attempts"]) == 11
    res = build(tmp_path / "three")
    shift_attempts(res, "stop", 3)
    o = summary(res)
    assert o["verdict"] == "fail" and "repeats_within_limit" in failing(o)
    assert "3 invalid attempts against 2" in next(r["detail"] for r in o["session_rules"] if r["name"] == "repeats_within_limit")


def test_a_mode_without_three_valid_attempts_is_incomplete(tmp_path):
    res = build(tmp_path)
    shift_attempts(res, "worker", 2)
    for n in (4, 5):  # only one valid attempt is left
        for f in (res / "trials" / f"worker-{n}").iterdir():
            f.unlink()
        (res / "trials" / f"worker-{n}").rmdir()
    o = summary(res)
    assert o["verdict"] == "incomplete" and o["pass"] is False and o["missing_trials"] == ["worker-2", "worker-3"]


def test_missing_fault_evidence_fails_and_is_not_relabelled_invalid(tmp_path):
    res = build(tmp_path)
    d = res / "trials" / "worker-1"
    meta = json.loads((d / "meta.json").read_text())
    del meta["fault"]
    (d / "meta.json").write_text(json.dumps(meta))
    o = summary(res)
    assert failing(o) == ["worker-1.fault_evidence"] and o["verdict"] == "fail" and len(o["trials"]) == 9
    meta["fault"] = {"live_b": 4, "uncertainty_s": 0.8}  # the time is known only within 0.8 s
    (d / "meta.json").write_text(json.dumps(meta))
    assert failing(summary(res)) == ["worker-1.fault_evidence"]


def test_a_failure_with_requests_in_flight_on_b_is_judged_not_repeated(tmp_path):
    # B's own reading says 4 in flight, yet no request failed: the clean-error rule fails, the attempt is not called invalid
    res = build(tmp_path)
    d = res / "trials" / "daemon-2"
    rows = [json.loads(x) for x in (d / "load.jsonl").read_text().splitlines()]
    for r in rows:
        if r["error"]:
            r.update(error=None, finish="length", done=True, chunks=255)
    (d / "load.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    o = summary(res)
    assert "daemon-2.clean_error_in_bound" in failing(o) and all(a["judged"] for a in o["attempts"])


def test_the_clock_rule_reads_the_worst_measurement_of_the_attempts(tmp_path):
    res = build(tmp_path, session=good_session(clock={"b_minus_a_s": 0.2, "uncertainty_s": 3.0}))
    for d in (res / "trials").iterdir():
        m = json.loads((d / "meta.json").read_text())
        m.update(clock_b_minus_a_s=0.2, clock_uncertainty_s=0.06)
        (d / "meta.json").write_text(json.dumps(m))
    assert failing(summary(res)) == []  # the direct exchanges (60 ms) replace the session's old ssh reading
    m = json.loads((res / "trials" / "stop-2" / "meta.json").read_text())
    m["clock_uncertainty_s"] = 2.5
    (res / "trials" / "stop-2" / "meta.json").write_text(json.dumps(m))
    assert failing(summary(res)) == ["clock_offset"]


def test_valid_subcommand_reads_b_s_own_evidence(tmp_path):
    import subprocess

    def run(name, meta):
        d = tmp_path / name
        d.mkdir()
        (d / "meta.json").write_text(json.dumps(meta))
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "phase6_exit_f_summary.py"), "valid", str(d)], capture_output=True, text=True)

    assert run("a", {"fault": {"live_b": 3}}).returncode == 0
    assert run("b", {"fault": {"live_b": 0}}).returncode == 1
    assert run("c", {}).stdout.strip() == "missing"
