"""Server fixture: an existing server (TETRAEDER_BASE_URL, TETRAEDER_MODEL) or a spawned `tetraeder serve --fake`."""

import os
import pathlib
import queue
import re
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
READY = re.compile(r"listening on (http://\S+)")


@dataclass
class Server:
    base_url: str
    model: str
    fake: bool
    request_log: pathlib.Path | None


def find_binary() -> pathlib.Path | None:
    env = os.environ.get("TETRAEDER_BIN")
    names = ["tetraeder.exe"] if sys.platform == "win32" else ["tetraeder"]
    candidates = [pathlib.Path(env)] if env else [ROOT / "target" / "debug" / n for n in names]
    return next((c for c in candidates if c.is_file()), None)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    base = os.environ.get("TETRAEDER_BASE_URL")
    if base:
        log = os.environ.get("TETRAEDER_REQUEST_LOG")
        yield Server(base.rstrip("/"), os.environ.get("TETRAEDER_MODEL", "default"), False, pathlib.Path(log) if log else None)
        return
    binary = find_binary()
    if binary is None:
        pytest.skip("no tetraeder binary: run `cargo build -p tetraeder` first, as CI does (or set TETRAEDER_BIN / TETRAEDER_BASE_URL)")
    log = tmp_path_factory.mktemp("serve") / "requests.jsonl"
    port = free_port()
    proc = subprocess.Popen(
        [str(binary), "serve", "--fake", "--port", str(port), "--request-log", str(log)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines: queue.Queue = queue.Queue()
    tail: list[str] = []

    def pump():  # keeps draining stderr so the server never blocks on a full pipe
        for line in proc.stderr:
            tail.append(line)
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    try:
        while True:
            try:
                line = lines.get(timeout=60)
            except queue.Empty:
                pytest.fail("tetraeder serve --fake did not print 'listening on' within 60 s")
            if line is None:
                pytest.fail(
                    "tetraeder serve --fake exited early (stale binary? rebuild with `cargo build -p tetraeder`; "
                    "CI builds it before pytest):\n" + "".join(tail[-20:])
                )
            m = READY.search(line)
            if m:
                break
        yield Server(m.group(1).rstrip("/"), "fake", True, log)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def timeout(server):
    return 600.0 if not server.fake else 30.0


@pytest.fixture(scope="session")
def oa(server, timeout):
    import openai

    return openai.OpenAI(base_url=server.base_url + "/v1", api_key="sk-secret-conformance-key", timeout=timeout, max_retries=0)


@pytest.fixture(scope="session")
def an(server, timeout):
    import anthropic

    return anthropic.Anthropic(base_url=server.base_url, api_key="sk-secret-conformance-key", timeout=timeout, max_retries=0)


@pytest.fixture(scope="session")
def model(server):
    return server.model
