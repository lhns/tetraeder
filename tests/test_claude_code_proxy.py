"""Claude Code benchmark kit (bench/claude-code-proxy, scripts/claude_code_proxy*.py, scripts/anthropic_proxy.py).

No Claude and no model involved: every task check must fail on the seeded project and pass with the reference
solution; the summary math and the logging proxy are tested against a fake upstream.
"""

import http.client
import http.server
import importlib.util
import json
import pathlib
import socket
import subprocess
import sys
import threading
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
KIT = ROOT / "bench" / "claude-code-proxy"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


runner = load("claude_code_proxy")
summary = load("claude_code_proxy_summary")
TASKS = runner.load_tasks()


# ---- tasks ---------------------------------------------------------------------------------------------------


def test_task_table_is_consistent():
    ids = [t["id"] for t in TASKS]
    assert len(ids) == len(set(ids)) == 10
    for t in TASKS:
        assert len(t["prompt"]) > 60 and t["check"]
        assert (KIT / "solutions" / t["id"]).is_dir(), t["id"]
        for src in list((t.get("hidden") or {}).values()) + list(((t.get("setup") or {}).get("files") or {}).values()):
            assert (KIT / src).is_file(), src
        assert chr(0x2014) not in t["prompt"]


def test_fixture_size():
    def lines(sub):
        return sum(len(p.read_text(encoding="utf-8").splitlines()) for p in (KIT / "fixture" / sub).rglob("*.py"))

    assert 600 <= lines("invoicing") + lines("tests") <= 1200 + 600  # package plus tests (legacy tests included)
    assert 2500 <= lines("legacy") <= 4000


def run_pytest(cwd):
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=cwd, capture_output=True, text=True
    )


def test_pristine_fixture_passes(tmp_path):
    repo = tmp_path / "repo"
    runner.prepare_workdir({"id": "none"}, repo)
    r = run_pytest(repo)
    assert r.returncode == 0, r.stdout[-2000:]


def needs_git(task):
    return any(c in cmd for cmd in task["check"] for c in ("explain_notes", "new_tests"))


@pytest.mark.parametrize("task", TASKS, ids=[t["id"] for t in TASKS])
def test_check_fails_on_seeded_copy_and_passes_with_reference(task, tmp_path):
    repo = runner.prepare_workdir(task, tmp_path / "repo", init_git=needs_git(task))
    before = runner.run_check(task, repo)
    assert not before["passed"], "check passes without any work"
    runner.apply_overlay(KIT / "solutions" / task["id"], repo)
    after = runner.run_check(task, repo)
    assert after["passed"], json.dumps(after["steps"], indent=1)


def test_block_order_is_abba():
    assert runner.block_order(2) == [(1, "tetraeder"), (1, "llama-server"), (2, "llama-server"), (2, "tetraeder")]
    assert runner.block_order(3)[4:] == [(3, "tetraeder"), (3, "llama-server")]


def test_parse_claude_json():
    assert runner.parse_claude_json('{"type":"result","num_turns":3}')["num_turns"] == 3
    assert runner.parse_claude_json('noise\n{"type":"result","num_turns":4}\n')["num_turns"] == 4
    assert runner.parse_claude_json('[{"type":"system"},{"type":"result","num_turns":5}]')["num_turns"] == 5
    assert runner.parse_claude_json("not json") is None


# ---- summary -------------------------------------------------------------------------------------------------


def session(engine, rep, task, ok, turns=5, wall=100.0, is_error=False):
    return {"engine": engine, "rep": rep, "task": task, "success": ok, "wall_s": wall, "timed_out": False,
            "claude": {"num_turns": turns, "is_error": is_error}}


def req(engine, task, ttft=1.0, tps=50.0, status=200, inp=1000, out=100, cache=0, err=None):
    return {"engine": engine, "task": task, "rep": "1", "status": status, "ttft_s": ttft, "decode_tok_s": tps,
            "input_tokens": inp, "output_tokens": out, "cache_read_tokens": cache, "error_type": err}


def test_percentile_and_median():
    assert summary.percentile([], 90) is None
    assert summary.percentile(list(range(1, 11)), 90) == 9
    assert summary.percentile([5], 90) == 5
    assert summary.median([3, 1, 2]) == 2
    assert summary.median([1, 2, 3, 4]) == 2.5
    assert summary.median([]) is None


def test_summarize_numbers():
    sessions = [
        session("tetraeder", 1, "a", True, turns=4, wall=100),
        session("tetraeder", 2, "a", False, turns=10, wall=200, is_error=True),
        session("tetraeder", 1, "b", True, turns=6, wall=300),
        session("llama-server", 1, "a", True, turns=5, wall=50),
        session("llama-server", 1, "b", True, turns=7, wall=150),
    ]
    requests = [
        req("tetraeder", "a", ttft=1.0, tps=40, inp=100, out=10, cache=50),
        req("tetraeder", "a", ttft=3.0, tps=60, inp=200, out=20),
        req("tetraeder", "b", ttft=2.0, tps=50, status=529, err="overloaded_error"),
        req("llama-server", "a", ttft=0.5, tps=100),
        req("llama-server", "b", ttft=1.5, tps=120),
    ]
    s = summary.summarize(sessions, requests)
    t = s["engines"]["tetraeder"]
    assert (t["runs"], t["successes"], t["success_rate"]) == (3, 2, 0.667)
    assert t["per_task"]["a"] == {"runs": 2, "successes": 1, "rate": 0.5}
    assert t["turns_median"] == 6
    assert (t["non_200_requests"], t["is_error_sessions"], t["protocol_errors"]) == (1, 1, 2)
    assert t["requests"] == 3 and t["ttft_median_s"] == 2.0 and t["ttft_p90_s"] == 3.0
    assert t["decode_tok_s_median"] == 50
    assert (t["input_tokens_total"], t["output_tokens_total"], t["cache_read_tokens_total"]) == (1300, 130, 50)
    assert t["wall_s_median"] == 200 and t["wall_s_total"] == 600
    cmp_a = s["comparison"]["a"]
    assert cmp_a["tetraeder"]["successes"] == 1 and cmp_a["llama-server"]["successes"] == 1
    assert cmp_a["wall_ratio_tetraeder_over_llama"] == 3.0  # median(100, 200) = 150 over 50
    assert s["engines"]["llama-server"]["protocol_errors"] == 0


def test_summary_main_writes_file(tmp_path):
    (tmp_path / "session-tetraeder-r1-a.json").write_text(json.dumps(session("tetraeder", 1, "a", True)))
    (tmp_path / "proxy-tetraeder-r1-a.jsonl").write_text(json.dumps(req("tetraeder", "a")) + "\n")
    assert summary.main([str(tmp_path)]) == 0
    data = json.loads((tmp_path / "summary.json").read_text())
    assert data["engines"]["tetraeder"]["successes"] == 1


# ---- logging proxy -------------------------------------------------------------------------------------------

SSE = [
    ("message_start", {"type": "message_start", "message": {"id": "m", "usage": {"input_tokens": 1234, "output_tokens": 1,
                                                                                 "cache_read_input_tokens": 1000}}}),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    ("ping", {"type": "ping"}),
]
DELTAS = 6


class FakeUpstream(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = b'{"data":[]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        req_body = json.loads(self.rfile.read(n) or b"{}")
        FakeUpstream.seen.append((self.path, dict(self.headers), req_body))
        if req_body.get("model") == "boom":
            body = json.dumps({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}).encode()
            self.send_response(529)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if not req_body.get("stream"):
            body = json.dumps({"type": "message", "usage": {"input_tokens": 7, "output_tokens": 3}, "stop_reason": "end_turn"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(event, data):
            payload = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
            self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
            self.wfile.flush()

        for ev, data in SSE:
            chunk(ev, data)
        time.sleep(0.3)  # slow prompt processing: TTFT
        for i in range(DELTAS):
            chunk("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": f"t{i} "}})
            time.sleep(0.1)
        chunk("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": DELTAS}})
        chunk("message_stop", {"type": "message_stop"})
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


@pytest.fixture
def proxy(tmp_path):
    FakeUpstream.seen = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    log = tmp_path / "proxy.jsonl"
    p = subprocess.Popen(
        [sys.executable, str(SCRIPTS / "anthropic_proxy.py"), "--listen-port", str(port), "--upstream",
         f"http://127.0.0.1:{srv.server_address[1]}", "--label", "eng", "--task", "t1", "--rep", "2", "--log", str(log)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    yield port, log
    p.kill()
    p.wait()
    srv.shutdown()


def post(port, body, path="/v1/messages?beta=true"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json", "x-api-key": "k", "Accept-Encoding": "gzip"})
    r = c.getresponse()
    return r, c


def log_lines(log):
    for _ in range(50):
        if log.exists() and log.read_text().strip():
            break
        time.sleep(0.05)
    return [json.loads(l) for l in log.read_text().splitlines() if l.strip()]


def test_proxy_streams_and_logs(proxy):
    port, log = proxy
    body = json.dumps({"model": "m", "stream": True, "messages": []})
    sock = socket.create_connection(("127.0.0.1", port), timeout=20)
    head = ["POST /v1/messages?beta=true HTTP/1.1", "Host: x", "Content-Type: application/json", "x-api-key: k",
            f"Content-Length: {len(body)}", "", ""]
    sock.sendall("\r\n".join(head).encode() + body.encode())
    t0 = time.monotonic()
    data, t_first = b"", None
    while chunk := sock.recv(65536):
        data += chunk
        if t_first is None and b"message_start" in data:
            t_first = time.monotonic() - t0  # the first events arrive long before the stream ends
    sock.close()
    assert data.startswith(b"HTTP/1.1 200")
    assert data.count(b"event: content_block_delta") == DELTAS
    assert t_first is not None and t_first < 0.25, "proxy buffered the stream"
    (rec,) = log_lines(log)
    assert (rec["engine"], rec["task"], rec["rep"], rec["status"], rec["stream"]) == ("eng", "t1", "2", 200, True)
    assert (rec["input_tokens"], rec["output_tokens"], rec["cache_read_tokens"]) == (1234, DELTAS, 1000)
    assert 0.25 <= rec["ttft_s"] < 1.0
    assert rec["n_deltas"] == DELTAS and rec["stop_reason"] == "end_turn"
    span = rec["last_delta_s"] - rec["ttft_s"]
    assert 0.3 <= span < 1.0
    assert rec["decode_tok_s"] == pytest.approx((DELTAS - 1) / span, rel=0.02)
    assert rec["total_s"] >= rec["last_delta_s"] and rec["error_type"] is None
    # request reached the upstream unchanged, with the client's headers
    path, headers, req_body = FakeUpstream.seen[0]
    assert path == "/v1/messages?beta=true" and req_body["model"] == "m" and headers["x-api-key"] == "k"
    assert headers["Accept-Encoding"] == "identity" and headers["Connection"] == "close"


def test_proxy_json_and_error_and_other_paths(proxy):
    port, log = proxy
    r, c = post(port, {"model": "m", "stream": False})
    assert r.status == 200 and json.loads(r.read())["usage"]["output_tokens"] == 3
    c.close()
    r, c = post(port, {"model": "boom", "stream": True})
    assert r.status == 529 and json.loads(r.read())["error"]["type"] == "overloaded_error"
    c.close()
    r, c = post(port, {"model": "m"}, path="/v1/messages/count_tokens")
    c.close()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("GET", "/v1/models")
    assert json.loads(c.getresponse().read()) == {"data": []}
    c.close()
    ok, err = log_lines(log)[:2]
    assert (ok["status"], ok["stream"], ok["input_tokens"], ok["output_tokens"]) == (200, False, 7, 3)
    assert (err["status"], err["error_type"], err["error_message"]) == (529, "overloaded_error", "busy")
    time.sleep(0.2)
    assert len(log_lines(log)) == 2, "count_tokens and GET must not be logged"
