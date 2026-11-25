#!/usr/bin/env python3
"""A stand-in for `tetraeder` on one machine, so the Phase 6 exit drivers (phase6_exit_f.sh, phase6_exit_s.sh) run end to end in
their loopback dry run (LOOPBACK=1) with two "boxes" as two sets of processes. It speaks only what the drivers use:

    cluster cert --node ID --out DIR          writes key.pem and cert.pem (text, no cryptography)
    node --cluster F --key K --cert C --state-dir D ...      the daemon of one node: control port, supervises one replica engine
    serve --cluster F --key K --cert C --replicas R --port P --request-log L ...     the head: gateway, router, two replicas
    engine --port P --node ID                 (internal) the replica engine the daemon starts as its child

It is not the engine and proves nothing about it: it exists so that the drivers' fault injection (kill -9 of the engine, kill -9 of
the daemon and its children, SIGSTOP and SIGCONT), the load, the epochs, the trial directories and the summary run in CI. The
behaviour that Case F judges is modelled after the real head: a node is down after `node_down_after_ms` without an answer, its
in-flight requests end with one error naming the replica, a request waits for a free replica, and a restored node rejoins on its own.
Env: LB_LOAD_S (engine load time, 0.5), LB_TOKEN_S (seconds per token, 0.02), LB_MAX_TOKENS (tokens per answer, 40).
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOAD_S = float(os.environ.get("LB_LOAD_S", "0.5"))
TOKEN_S = float(os.environ.get("LB_TOKEN_S", "0.02"))
MAX_TOKENS = int(os.environ.get("LB_MAX_TOKENS", "40"))


def iso_now() -> str:
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int((t % 1) * 1e6):06d}Z"


def log(msg: str) -> None:
    print(msg, flush=True)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Quiet(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # noqa: D401
        pass

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def send_json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64


# ---- the replica engine (a child of the node daemon) -----------------------------------------------------------------------


def run_engine(port: int) -> None:
    seq = [0]
    lock = threading.Lock()

    class H(Quiet):
        def do_GET(self):  # noqa: N802
            self.send_json({"ok": True})

        def do_POST(self):  # noqa: N802
            req = self.body()
            with lock:
                seq[0] += 1
                n = seq[0]
            log(f"{iso_now()}  DEBUG tetraeder::step: admit seq={n} queued_ms=0.10 prompt={len(str(req.get('prompt', ''))) // 4} common=0 source=live")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                for i in range(int(req.get("max_tokens", MAX_TOKENS))):
                    time.sleep(TOKEN_S)
                    self.wfile.write(b'{"t":"x "}\n')
                    self.wfile.flush()
                self.wfile.write(b'{"done":"length"}\n')
                self.wfile.flush()
            except OSError:
                pass

    time.sleep(LOAD_S)  # the model load: the daemon reports the engine ready only after it
    srv = Server(("127.0.0.1", port), H)
    log(f"{iso_now()}  INFO engine listening on {port}")
    srv.serve_forever()


# ---- the node daemon ----------------------------------------------------------------------------------------------------------


def me_from(cluster_path: str, cert_path: str) -> tuple[dict, dict]:
    cfg = tomllib.loads(open(cluster_path, encoding="utf-8").read())
    node = open(cert_path, encoding="utf-8").read().split()[-1]
    entry = next(n for n in cfg["nodes"] if n["id"] == node)
    return cfg, entry


def run_node(a: argparse.Namespace) -> None:
    cfg, me = me_from(a.cluster, a.cert)
    host, port = me.get("listen", me["addr"]).rsplit(":", 1)
    state = {"engine_port": None, "child": None}
    os.makedirs(a.state_dir, exist_ok=True)

    def supervise():
        backoff = 0.5
        while True:
            p = free_port()
            child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "engine", "--port", str(p), "--node", me["id"]])
            state["child"] = child
            log(f"{iso_now()}  INFO started replica engine pid {child.pid} on {p}")
            deadline = time.time() + 30
            while time.time() < deadline and child.poll() is None:
                try:
                    socket.create_connection(("127.0.0.1", p), timeout=0.2).close()
                    state["engine_port"] = p
                    break
                except OSError:
                    time.sleep(0.1)
            while child.poll() is None:
                time.sleep(0.1)
            state["engine_port"] = None
            log(f"{iso_now()}  WARN replica engine exited {child.returncode}, restarting")
            time.sleep(backoff)

    class H(Quiet):
        def do_GET(self):  # noqa: N802
            self.send_json({"node": me["id"], "engine_port": state["engine_port"]})

    threading.Thread(target=supervise, daemon=True).start()
    srv = Server((host, int(port)), H)
    log(f"{iso_now()}  INFO tetraeder node node={me['id']} listen={host}:{port}")
    srv.serve_forever()


# ---- the head --------------------------------------------------------------------------------------------------------------------


class Replica:
    def __init__(self, rid: int, leader: str, cap: int, local: bool):
        self.id, self.leader, self.cap, self.local = rid, leader, cap, local
        self.up = local
        self.out = 0
        self.port = None  # engine port of a remote replica
        self.gen = 0  # bumped whenever the replica goes down, so the streams of the old connection end
        self.conns: set = set()  # sockets of the streams in flight: closed when the replica goes down
        self.cv = threading.Condition()


def run_head(a: argparse.Namespace) -> None:
    cfg = tomllib.loads(open(a.cluster, encoding="utf-8").read())
    reps_file = tomllib.loads(open(a.replicas, encoding="utf-8").read())
    head = cfg["head"]
    down_after = cfg.get("node_down_after_ms", 2000) / 1000.0
    addr = {n["id"]: n["addr"] for n in cfg["nodes"]}
    reps = [Replica(i, r["leader"], int(r.get("cap", 4)), r["leader"] == head) for i, r in enumerate(reps_file["replica"])]
    local_engine = {"port": None}
    cv = threading.Condition()
    req_log = open(a.request_log, "a", buffering=1) if a.request_log else None
    log_lock = threading.Lock()

    # R_A, the head's own replica: an engine of this process's own (a thread-served port)
    def start_local():
        p = free_port()
        threading.Thread(target=run_engine, args=(p,), daemon=True).start()
        local_engine["port"] = p

    start_local()
    for _ in range(100):  # R_A serves from the first request
        try:
            socket.create_connection(("127.0.0.1", local_engine["port"]), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)

    def hello(node: str):
        h, p = addr[node].rsplit(":", 1)
        c = http.client.HTTPConnection(h, int(p), timeout=min(1.0, down_after))
        c.request("GET", "/")
        r = c.getresponse()
        d = json.loads(r.read())
        c.close()
        return d

    def monitor(r: Replica):
        last_ok = time.time()
        while True:
            try:
                d = hello(r.leader)
                last_ok = time.time()
                port = d.get("engine_port")
                if port and not r.up:
                    r.port = port
                    r.up = True
                    log(f"{iso_now()}  INFO replica {r.id} joined (node {r.leader}, engine {port})")
                elif r.up and port != r.port:
                    take_down(r, "engine changed" if port else "engine gone")
                elif not port and r.up:
                    take_down(r, "engine gone")
            except (OSError, ValueError):
                if r.up and time.time() - last_ok >= down_after:
                    take_down(r, f"node {r.leader} silent for {down_after:.1f} s")
            with cv:
                cv.notify_all()
            time.sleep(0.1)

    def take_down(r: Replica, why: str):
        r.up = False
        r.gen += 1
        for s in list(r.conns):  # the in-flight streams end with the one error, whatever the engine does
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        log(f"{iso_now()}  WARN replica {r.id} left: {why}")

    for r in reps:
        if not r.local:
            threading.Thread(target=monitor, args=(r,), daemon=True).start()

    def pick() -> Replica | None:
        ok = [r for r in reps if r.up and r.out < r.cap]
        return min(ok, key=lambda r: (r.out, r.id)) if ok else None

    class H(Quiet):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/health") or self.path.startswith("/v1/health"):
                return self.send_json({"status": "ok"})
            self.send_json({"error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            req = self.body()
            if self.path == "/tokenize":
                return self.send_json({"tokens": list(range(max(1, len(str(req.get("content", ""))) // 4)))})
            if not self.path.endswith("/chat/completions"):
                return self.send_json({"error": "not found"}, 404)
            self.chat(req)

        def sse(self, obj):
            self.wfile.write(b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n")
            self.wfile.flush()

        def chat(self, req):
            t0 = time.time()
            with cv:
                r = pick()
                while r is None:
                    cv.wait(0.2)
                    r = pick()
                    if time.time() - t0 > 60:
                        return self.send_json({"error": {"message": "no replica is up", "type": "server_error"}}, 500)
                r.out += 1
            err = None
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            gen = r.gen
            c = None
            try:
                port = local_engine["port"] if r.local else r.port
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=None)
                c.connect()
                r.conns.add(c.sock)
                c.request("POST", "/gen", json.dumps({"prompt": str(req.get("messages", ""))[:2000], "max_tokens": min(int(req.get("max_tokens", 256)), MAX_TOKENS)}))
                resp = c.getresponse()
                finished = False
                while not finished:
                    if r.gen != gen or not (r.up or r.local):
                        raise ConnectionError("down")
                    line = resp.fp.readline()  # blocks on a silent engine until the monitor shuts the socket
                    if not line:
                        raise ConnectionError("reset")
                    ev = json.loads(line)
                    if "done" in ev:
                        self.sse({"choices": [{"delta": {}, "finish_reason": ev["done"]}]})
                        self.sse(b"[DONE]")
                        finished = True
                    else:
                        self.sse({"choices": [{"delta": {"content": ev["t"]}}]})
                c.close()
            except (OSError, ConnectionError, ValueError):
                err = f"engine error: the connection to replica {r.id} was lost"
                try:
                    self.sse({"error": {"message": err, "type": "server_error", "param": None, "code": 500}})
                except OSError:
                    pass
            finally:
                if c is not None and c.sock is not None:
                    r.conns.discard(c.sock)
                with cv:
                    r.out -= 1
                    cv.notify_all()
                if req_log is not None:
                    with log_lock:
                        req_log.write(json.dumps({"v": "1", "ts_ms": int(t0 * 1000), "method": "POST", "path": self.path, "status": 200,
                                                  "stream": True, "error_type": "internal" if err else None, "request": req}) + "\n")

    srv = Server((a.host, a.port), H)
    log(f"{iso_now()}  INFO head listening on {a.host}:{a.port}, replicas {[(r.id, r.leader) for r in reps]}")
    srv.serve_forever()


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["cluster"] and argv[1:2] == ["cert"]:
        ap = argparse.ArgumentParser()
        ap.add_argument("--node", required=True)
        ap.add_argument("--out", required=True)
        a = ap.parse_args(argv[2:])
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, "key.pem"), "w").write(f"FAKE KEY {a.node}\n")
        open(os.path.join(a.out, "cert.pem"), "w").write(f"FAKE CERT {a.node}\n")
        print(os.path.join(a.out, "cert.pem"))
        return 0
    cmd, rest = (argv[0], argv[1:]) if argv else ("", [])
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--cluster")
    ap.add_argument("--key")
    ap.add_argument("--cert")
    ap.add_argument("--state-dir")
    ap.add_argument("--replicas")
    ap.add_argument("--request-log")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--node")
    a, _ = ap.parse_known_args(rest)  # the real flags the drivers pass (--models-dir, --worker-arg=..., -m ...) are ignored
    if cmd == "engine":
        run_engine(a.port)
    elif cmd == "node":
        run_node(a)
    elif cmd == "serve":
        run_head(a)
    else:
        print(f"phase6_loopback.py: unknown command {cmd!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
