#!/usr/bin/env python3
"""The Case F fault trigger (ADR 0091, advisor decision 2026-11-25 in docs/notes/phase6-rule-changes.md), standard library only.

The load generator on box A fires the fault on box B over a direct connection, once it sees requests in flight on B, so the fault
time is a time on A's clock (the middle of the send and the ack, uncertain by half the round trip) and no ssh latency enters it.

    phase6_trigger.py serve --port P --token-file F --p6 DIR [--repo DIR] [--host H]      on box B, started by `phase6_box.sh trigger`
    phase6_trigger.py clock --peer HOST:PORT --token-file F [--samples N]                 on box A: B's clock minus A's, as JSON
    phase6_trigger.py fire  --peer HOST:PORT --token-file F --mode M --label L ...        on box A: one fire (phase6_load.py does it in process)

The server answers three calls, each with the header `X-P6-Token`:
    GET  /time       {"t": epoch of B's clock}
    GET  /inflight   {"live": requests in flight on B, "age_s": age of that reading}: the last `live=N` of a `tetraeder::step` line
                     of B's node log (the scheduler's own count), 0 when the last such line is older than FRESH_S
    POST /fire       {"mode": worker|daemon|stop, "label": ...}: reads `live`, runs scripts/phase6_fault.sh MODE at once (detached, its own
                     session, so the restore goes on after the answer), waits for its `fault done` line, then answers with the reading and
                     B's epochs. The signal is delivered between the request and the answer; nothing is slept or restored before it.
The port is public on a rented box: only this fixed list of faults can run, and only with the token (a random file under the box's P6).
"""

from __future__ import annotations

import argparse
import datetime
import http.client
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODES = ("worker", "daemon", "stop")
FRESH_S = 0.5  # a step line older than this means nothing is decoding
ADMIT_FRESH_S = 0.15  # the fire gate takes only a request admitted this recently: it has nearly its whole token budget left
TAIL_BYTES = 256 * 1024
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"^(\d{4}-\d\d-\d\dT[\d:.]+)Z\s+DEBUG tetraeder::step: (step|admit) .*?\blive=(\d+)")
LABEL = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


def read_token(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def inflight(log_path: str, now: float | None = None, kind: str | None = None, fresh_s: float = FRESH_S) -> dict:
    """The newest `live=N` of the node log's tail, as {live, age_s, last_live}. With `kind` ("admit"), only lines of that kind count."""
    now = time.time() if now is None else now
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(f.tell() - TAIL_BYTES, 0))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return {"live": 0, "age_s": None, "last_live": None, "log": "unreadable"}
    for line in reversed(lines):
        m = STEP.match(ANSI.sub("", line))
        if m and (kind is None or m.group(2) == kind):
            t = datetime.datetime.fromisoformat(m.group(1)).replace(tzinfo=datetime.timezone.utc).timestamp()
            age = round(now - t, 3)
            return {"live": int(m.group(3)) if age <= fresh_s else 0, "age_s": age, "last_live": int(m.group(3))}
    return {"live": 0, "age_s": None, "last_live": None}


def make_handler(token: str, p6: str, repo: str):
    node_log = os.path.join(p6, "logs", "node.log")

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # noqa: D401
            pass

        def send_json(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b)

        def authorized(self) -> bool:
            if token and self.headers.get("X-P6-Token") == token:
                return True
            self.send_json({"error": "no"}, 403)
            return False

        def do_GET(self):  # noqa: N802
            if not self.authorized():
                return
            if self.path == "/time":
                return self.send_json({"t": time.time()})
            if self.path == "/inflight":
                return self.send_json({**inflight(node_log), "t": time.time()})
            self.send_json({"error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            if not self.authorized():
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self.send_json({"error": "bad json"}, 400)
            mode, label = req.get("mode"), str(req.get("label", ""))
            if self.path != "/fire" or mode not in MODES or not LABEL.match(label):
                return self.send_json({"error": "mode or label not allowed"}, 400)
            log = os.path.join(p6, "logs", f"fault-{label}.log")
            env = dict(os.environ, P6=p6, FAULT_LOG=log, STOP_FOR=str(req.get("stop_for", 60)),
                       RESTORE_AFTER=str(req.get("restore_after", 40)), POLL_S=str(req.get("poll_s", 0.2)))
            # The gate is here, not on the client: the reading that licenses the fault and the start of the fault are one step, with no
            # round trip between them (a client that polled /inflight and then fired lost the request in that gap, flaky in CI).
            # `wait_s` > 0: wait up to that long for a request admitted in the last ADMIT_FRESH_S (its remaining tokens outlast the
            # fault-to-error window; a daemon fault once hit four requests that had just finished), then fire; at the deadline fire anyway (live 0, which
            # the evidence calls invalid and the driver repeats).
            wait_s = min(float(req.get("wait_s", 0) or 0), 120.0)
            t_wait = time.time()
            seen = inflight(node_log) if wait_s <= 0 else inflight(node_log, kind="admit", fresh_s=ADMIT_FRESH_S)
            while seen["live"] < 1 and time.time() - t_wait < wait_s:
                time.sleep(0.01)
                seen = inflight(node_log, kind="admit", fresh_s=ADMIT_FRESH_S)
            t_recv = time.time()
            os.makedirs(os.path.dirname(log), exist_ok=True)
            with open(log, "a", encoding="utf-8") as out:
                proc = subprocess.Popen(["bash", os.path.join(repo, "scripts", "phase6_fault.sh"), mode, "0"], cwd=repo, env=env,
                                        stdout=out, stderr=out, stdin=subprocess.DEVNULL, start_new_session=True)
            threading.Thread(target=proc.wait, daemon=True).start()
            t_fault = t_done = None
            deadline = time.time() + 10
            while time.time() < deadline and t_done is None:
                try:
                    for line in open(log, encoding="utf-8", errors="replace"):
                        parts = line.split(None, 2)
                        if len(parts) >= 2 and parts[1] == "fault":
                            if parts[2:3] and parts[2].startswith("done"):
                                t_done = float(parts[0])
                            elif t_fault is None:
                                t_fault = float(parts[0])
                except (OSError, ValueError):
                    pass
                if t_done is None:
                    time.sleep(0.005)
            if t_done is None:
                return self.send_json({"error": "the fault script did not report `fault done` within 10 s"}, 500)
            self.send_json({"mode": mode, "label": label, "live": seen["live"], "step_age_s": seen["age_s"], "last_live": seen["last_live"],
                            "waited_s": round(t_recv - t_wait, 3), "t_b_recv": t_recv, "t_b_fault": t_fault, "t_b_done": t_done, "t_b_ack": time.time()})

    return H


def serve(a: argparse.Namespace) -> None:
    token = read_token(a.token_file)
    if len(token) < 16:
        sys.exit("the token file holds fewer than 16 characters")
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(token, a.p6, a.repo))
    srv.daemon_threads = True
    print(f"phase6_trigger listening on {a.host}:{a.port}", flush=True)
    srv.serve_forever()


# ---- client side (box A) ------------------------------------------------------------------------------------------------------


def call(peer: str, token: str, method: str, path: str, body: dict | None = None, timeout: float = 15.0) -> dict:
    host, port = peer.rsplit(":", 1)
    c = http.client.HTTPConnection(host, int(port), timeout=timeout)
    try:
        c.request(method, path, json.dumps(body) if body is not None else None,
                  {"X-P6-Token": token, "Content-Type": "application/json"})
        r = c.getresponse()
        data = r.read()
        if r.status != 200:
            raise RuntimeError(f"{path} answered {r.status}: {data[:200]!r}")
        return json.loads(data)
    finally:
        c.close()


def clock(peer: str, token: str, samples: int = 20) -> dict:
    """B's clock minus A's from direct exchanges: each is wrong by at most half its round trip, the narrowest is kept."""
    best = None
    for _ in range(samples):
        t0 = time.time()
        b = call(peer, token, "GET", "/time")["t"]
        t1 = time.time()
        err = (t1 - t0) / 2
        if best is None or err < best[1]:
            best = (b - (t0 + t1) / 2, err)
        time.sleep(0.02)
    return {"b_minus_a_s": round(best[0], 4), "uncertainty_s": round(best[1], 4), "samples": samples, "source": "direct exchange with the trigger"}


def fire(peer: str, token: str, mode: str, label: str, params: dict, timeout: float = 20) -> dict:
    """One fire: time on A's clock is the middle of the interval from the end of the trigger's wait to the ack, uncertain by half of it."""
    t_send = time.time()
    resp = call(peer, token, "POST", "/fire", {"mode": mode, "label": label, **params}, timeout=timeout)
    t_ack = time.time()
    # the trigger's own wait for a fresh request (`waited_s`) is not part of the fault's timing: the fault lies in the rest of the interval,
    # and a request admitted during the wait must count as in flight at the fault
    t_from = t_send + min(max(float(resp.get("waited_s") or 0), 0.0), t_ack - t_send)
    return {"source": "trigger", "t_send": t_send, "t_ack": t_ack, "t_fault": round((t_from + t_ack) / 2, 4),
            "uncertainty_s": round((t_ack - t_from) / 2, 4), "live_b": resp["live"], "step_age_s": resp["step_age_s"],
            "t_b_fault": resp["t_b_fault"], "t_b_done": resp["t_b_done"], "mode": mode, "label": label}


def fire_when_in_flight(peer: str, token: str, mode: str, label: str, params: dict, start: float, after_s: float,
                        out_path: str, poll_s: float = 0.05, give_up_s: float = 30.0) -> dict:
    """Waits until `after_s` after `start`, then has the trigger wait for requests in flight on B (at most `give_up_s`: then it fires
    anyway, and the trial is the one the evidence calls invalid) and fire, and writes the record to out_path."""
    try:
        time.sleep(max(start + after_s - time.time(), 0))
        t_wait = time.time()
        # the trigger waits for the in-flight reading itself and fires in the same step (see the /fire handler)
        rec = fire(peer, token, mode, label, {**params, "wait_s": give_up_s}, timeout=give_up_s + 30)
        rec["waited_s"] = round(time.time() - t_wait, 3)
    except Exception as e:  # noqa: BLE001 - the record says why there is no fault
        rec = {"error": f"{type(e).__name__}: {e}", "mode": mode, "label": label}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--token-file", required=True)
    s.add_argument("--p6", required=True)
    s.add_argument("--repo", default=os.getcwd())
    c = sub.add_parser("clock")
    c.add_argument("--peer", required=True)
    c.add_argument("--token-file", required=True)
    c.add_argument("--samples", type=int, default=20)
    f = sub.add_parser("fire")
    f.add_argument("--peer", required=True)
    f.add_argument("--token-file", required=True)
    f.add_argument("--mode", required=True, choices=MODES)
    f.add_argument("--label", required=True)
    f.add_argument("--stop-for", type=float, default=60)
    f.add_argument("--restore-after", type=float, default=40)
    f.add_argument("--poll-s", type=float, default=0.2)
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(a)
        return 0
    token = read_token(a.token_file)
    if a.cmd == "clock":
        print(json.dumps(clock(a.peer, token, a.samples)))
        return 0
    print(json.dumps(fire(a.peer, token, a.mode, a.label, {"stop_for": a.stop_for, "restore_after": a.restore_after, "poll_s": a.poll_s})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
