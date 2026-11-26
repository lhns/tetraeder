"""The Case F fault trigger (scripts/phase6_trigger.py): B's reading of requests in flight, the token, the fixed list of faults, the clock
offset by direct exchanges and the send/ack midpoint of a fire. Hardware label: none (the fire runs a stub of the fault script and needs bash)."""

import datetime
import importlib.util
import json
import os
import pathlib
import shutil
import threading
import time
from http.server import ThreadingHTTPServer

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("phase6_trigger", ROOT / "scripts" / "phase6_trigger.py")
trig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trig)

TOKEN = "0123456789abcdef0123456789abcdef"
needs_bash = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None, reason="the stub fault script needs bash")


def stamp(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def admit_line(t, live):
    return f"{stamp(t)} DEBUG tetraeder::step: admit seq=3 queued_ms=0.10 prompt=96 common=0 source=live live={live}\n"


def step_line(t, live, ansi=False):
    ts = f"\x1b[2m{stamp(t)}\x1b[0m" if ansi else stamp(t)
    return f"{ts} DEBUG tetraeder::step: step dec_seqs=1 dec_toks=1 pre_seqs=0 live={live} paused=0 idle=3 waiting=0\n"


def server(tmp_path, with_stub=True):
    p6 = tmp_path / "p6"
    (p6 / "logs").mkdir(parents=True)
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    if with_stub:  # the fault script's log lines, as phase6_fault.sh writes them
        (repo / "scripts" / "phase6_fault.sh").write_text(
            '#!/usr/bin/env bash\nnow() { date +%s.%N; }\necho "$(now) armed mode=$1"  >> "$FAULT_LOG"\n'
            'echo "$(now) fault kill -9 stub" >> "$FAULT_LOG"\nsleep 0.05\necho "$(now) fault done" >> "$FAULT_LOG"\nsleep 1\necho "$(now) restore done" >> "$FAULT_LOG"\n')
    srv = ThreadingHTTPServer(("127.0.0.1", 0), trig.make_handler(TOKEN, str(p6), str(repo)))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"127.0.0.1:{srv.server_address[1]}", p6


def test_in_flight_is_the_newest_live_count_while_it_is_fresh(tmp_path):
    log = tmp_path / "node.log"
    now = time.time()
    log.write_text("2026-11-25T04:03:02.1Z  INFO tetraeder::node: up\n" + step_line(now - 3, 4) + step_line(now - 0.1, 3, ansi=True))
    r = trig.inflight(str(log), now)
    assert r["live"] == 3 and r["last_live"] == 3 and 0.05 < r["age_s"] < 0.2
    old = trig.inflight(str(log), now + 2)  # nothing decoded for 2 s: nothing is in flight
    assert old["live"] == 0 and old["last_live"] == 3
    log.write_text(step_line(now - 0.1, 0))
    assert trig.inflight(str(log), now)["live"] == 0
    assert trig.inflight(str(tmp_path / "none.log"), now)["live"] == 0


def test_the_token_and_the_fixed_list_of_faults(tmp_path):
    srv, peer, _ = server(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="403"):
            trig.call(peer, "wrong", "GET", "/time")
        with pytest.raises(RuntimeError, match="400"):
            trig.call(peer, TOKEN, "POST", "/fire", {"mode": "rm -rf", "label": "x"})
        with pytest.raises(RuntimeError, match="400"):
            trig.call(peer, TOKEN, "POST", "/fire", {"mode": "worker", "label": "../x"})
    finally:
        srv.shutdown()


def test_the_clock_offset_is_read_over_the_direct_connection(tmp_path):
    srv, peer, _ = server(tmp_path)
    try:
        c = trig.clock(peer, TOKEN, 5)
    finally:
        srv.shutdown()
    assert abs(c["b_minus_a_s"]) < 0.05 and 0 <= c["uncertainty_s"] < 0.05 and c["samples"] == 5  # one machine: no offset, a loopback round trip


@needs_bash
def test_a_fire_answers_after_the_fault_and_its_time_lies_between_send_and_ack(tmp_path):
    srv, peer, p6 = server(tmp_path)
    (p6 / "logs" / "node.log").write_text(step_line(time.time(), 4))
    try:
        r = trig.fire(peer, TOKEN, "stop", "t1", {"stop_for": 1, "restore_after": 1})
    finally:
        srv.shutdown()
    assert r["live_b"] == 4 and r["t_send"] <= r["t_fault"] <= r["t_ack"] and r["uncertainty_s"] == pytest.approx((r["t_ack"] - r["t_send"]) / 2, abs=1e-3)
    assert r["t_send"] <= r["t_b_fault"] <= r["t_b_done"] <= r["t_ack"]  # one machine: B's epochs are on the same clock
    log = (p6 / "logs" / "fault-t1.log").read_text()
    assert " fault done" in log


@needs_bash
def test_the_load_fires_once_requests_are_in_flight_and_records_it(tmp_path):
    srv, peer, p6 = server(tmp_path)
    node_log = p6 / "logs" / "node.log"
    node_log.write_text(step_line(time.time() - 10, 4))  # stale: nothing in flight yet
    out = tmp_path / "fire.json"
    start = time.time()

    def later():
        time.sleep(0.6)
        node_log.write_text(step_line(time.time(), 2))  # decoding, but admitted long ago: not yet
        time.sleep(0.4)
        node_log.write_text(admit_line(time.time(), 2))

    threading.Thread(target=later, daemon=True).start()
    try:
        rec = trig.fire_when_in_flight(peer, TOKEN, "worker", "t2", {"restore_after": 1}, start, 0.1, str(out), poll_s=0.02, give_up_s=5)
    finally:
        srv.shutdown()
    assert rec["live_b"] == 2 and rec["waited_s"] >= 0.4 and json.loads(out.read_text())["t_fault"] == rec["t_fault"]


@needs_bash
def test_the_load_fires_anyway_after_the_give_up_and_the_evidence_says_nothing_was_in_flight(tmp_path):
    srv, peer, p6 = server(tmp_path)
    (p6 / "logs" / "node.log").write_text(step_line(time.time() - 10, 4))
    try:
        rec = trig.fire_when_in_flight(peer, TOKEN, "worker", "t3", {"restore_after": 1}, time.time(), 0, str(tmp_path / "f.json"), poll_s=0.02, give_up_s=0.3)
    finally:
        srv.shutdown()
    assert rec["live_b"] == 0  # the summary calls such an attempt invalid and the driver repeats it


def test_a_fire_that_cannot_reach_b_is_recorded_as_an_error(tmp_path):
    out = tmp_path / "f.json"
    rec = trig.fire_when_in_flight("127.0.0.1:9", TOKEN, "worker", "t4", {}, time.time(), 0, str(out), poll_s=0.01, give_up_s=0.1)
    assert "error" in rec and "error" in json.loads(out.read_text())
