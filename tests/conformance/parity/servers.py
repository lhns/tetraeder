"""Start llama-server (the baseline) and `tetraeder serve` on the tiny fixtures with matching settings."""

from __future__ import annotations

import os
import pathlib
import socket
import subprocess
import sys
import time
from dataclasses import dataclass

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures"
# The tiny GGUFs carry no chat template; both servers get this one, so the prompts are the same.
TEMPLATE = FIXTURES / "chatml.jinja"
# tetraeder engine -> model family; llama-server is keyed by family (auto and dense share the qwen3 fixture)
FAMILY = {"auto": "qwen3", "dense": "qwen3", "moe": "qwen3moe"}
MODEL_FILE = {"qwen3": "tiny-qwen3-f16.gguf", "qwen3moe": "tiny-qwen3moe-f16.gguf"}

# Matching settings of both servers: context and slots (llama-server splits -c over slots unless -kvu,
# tetraeder always shares one pool), threads and sampling defaults.
CTX, SLOTS, THREADS = 4096, 2, 2
SAMPLING = ["--temp", "0.8", "--top-k", "40", "--top-p", "0.95"]

EXE = ".exe" if sys.platform == "win32" else ""


class ServerStartError(RuntimeError):
    pass


def find_llama_server() -> pathlib.Path | None:
    env = os.environ.get("LLAMA_SERVER_BIN")
    if env:
        return pathlib.Path(env) if pathlib.Path(env).is_file() else None
    base = ROOT / "target" / "llama-server"
    for rel in ("bin", "bin/Release", "bin/RelWithDebInfo", "bin/Debug"):
        p = base / rel / f"llama-server{EXE}"
        if p.is_file():
            return p
    return None


def find_tetraeder() -> pathlib.Path | None:
    env = os.environ.get("TETRAEDER_BIN")
    p = pathlib.Path(env) if env else ROOT / "target" / "debug" / f"tetraeder{EXE}"
    return p if p.is_file() else None


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Running:
    name: str
    base_url: str
    proc: subprocess.Popen
    log: pathlib.Path

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def launch(name, argv, ready_path, port, log_dir: pathlib.Path, timeout=180) -> Running:
    log = log_dir / f"{name}.log"
    out = open(log, "wb")
    proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT)
    srv = Running(name, f"http://127.0.0.1:{port}", proc, log)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            tail = log.read_text(errors="replace")[-1500:]
            raise ServerStartError(f"{name} exited with {proc.returncode} before it was ready:\n{tail}")
        try:
            if httpx.get(srv.base_url + ready_path, timeout=2).status_code == 200:
                return srv
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    srv.stop()
    raise ServerStartError(f"{name} not ready after {timeout} s:\n{log.read_text(errors='replace')[-1500:]}")


def start_llama(binary, model: str, log_dir, extra=(), tag="") -> Running:
    port = free_port()
    argv = [
        str(binary), "-m", str(FIXTURES / MODEL_FILE[model]), "--host", "127.0.0.1", "--port", str(port),
        "-c", str(CTX), "-np", str(SLOTS), "-kvu", "-t", str(THREADS), *SAMPLING,
        "--chat-template-file", str(TEMPLATE), "--metrics", "--no-warmup", *extra,
    ]  # fmt: skip
    return launch(f"llama-server-{model}{tag}", argv, "/health", port, log_dir)


def start_tetraeder(binary, engine: str, log_dir, extra=(), tag="") -> Running:
    port = free_port()
    argv = [
        str(binary), "serve", "-m", str(FIXTURES / MODEL_FILE[FAMILY[engine]]), "--engine", engine,
        "--host", "127.0.0.1", "--port", str(port), "--ctx", str(CTX), "--parallel", str(SLOTS), "-t", str(THREADS),
        *SAMPLING, "--chat-template-file", str(TEMPLATE), "--accept-any-model", *extra,
    ]  # fmt: skip
    return launch(f"tetraeder-{engine}{tag}", argv, "/v1/models", port, log_dir)

