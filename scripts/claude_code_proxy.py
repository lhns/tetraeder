#!/usr/bin/env python3
"""Runs real Claude Code sessions (headless) against local LLM servers and scores them.

For every (engine, rep, task): a fresh copy of the fixture project in a temp dir (git repo), the task's setup, a
logging proxy (scripts/anthropic_proxy.py) in front of the engine, `claude -p <prompt>` with an empty config dir,
then the hidden tests and the task's check. Engines run in ABBA order over the reps, one server start per
(engine, rep) block. Kit: bench/claude-code-proxy (fixture, tasks.json, checks). Scoring: claude_code_proxy_summary.py.

Spawn the servers (box):
    python scripts/claude_code_proxy.py --out DIR --model M.gguf --ctx 131072 --threads 8 --reps 2
Use a server that is already running (one engine, e.g. local smoke test):
    python scripts/claude_code_proxy.py --out DIR --engine tetraeder --url http://127.0.0.1:8080 --tasks fix-eom-leap

Permissions: `--dangerously-skip-permissions` with IS_SANDBOX=1 (the box runs as root, where the flag is refused
without it). Use --permissions accept for `--permission-mode acceptEdits` plus an allowed-tools list instead.
"""

import argparse
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIT = ROOT / "bench" / "claude-code-proxy"
ENGINES = ("tetraeder", "llama-server")
GIT_ENV = {
    "GIT_AUTHOR_NAME": "bench",
    "GIT_AUTHOR_EMAIL": "bench@example.invalid",
    "GIT_COMMITTER_NAME": "bench",
    "GIT_COMMITTER_EMAIL": "bench@example.invalid",
}
ALLOWED_TOOLS = "Bash,Edit,MultiEdit,Write,Read,Glob,Grep"


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- the task kit: fixture copy, setup, hidden files, check (no Claude, no network) --------------------------------


def load_tasks(kit: Path = KIT) -> list[dict]:
    return json.loads((kit / "tasks.json").read_text(encoding="utf-8"))["tasks"]


def _git(dest: Path, *args):
    env = {**os.environ, **GIT_ENV}
    subprocess.run(["git", *args], cwd=dest, env=env, check=True, capture_output=True)


def apply_setup(task: dict, dest: Path, kit: Path = KIT) -> None:
    """Seed bugs and failing tests: text replacements and extra files from the kit."""
    setup = task.get("setup") or {}
    for rep in setup.get("replace", []):
        p = dest / rep["path"]
        text = p.read_text(encoding="utf-8")
        if text.count(rep["old"]) != 1:
            raise RuntimeError(f"{task['id']}: setup text not found exactly once in {rep['path']}")
        p.write_text(text.replace(rep["old"], rep["new"]), encoding="utf-8", newline="\n")
    for target, src in (setup.get("files") or {}).items():
        t = dest / target
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(kit / src, t)


def prepare_workdir(task: dict, dest: Path, kit: Path = KIT, init_git: bool = True) -> Path:
    """Fresh copy of the fixture with the task's setup applied and committed as the only commit."""
    fixture = kit / "fixture"
    shutil.copytree(fixture, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    apply_setup(task, dest, kit)
    if init_git:
        _git(dest, "init", "-q", "-b", "main")
        _git(dest, "add", "-A")
        _git(dest, "commit", "-q", "-m", "initial")
    return dest


def apply_overlay(src: Path, dest: Path) -> None:
    """Copy a reference solution (files relative to the repo root) over the work copy."""
    shutil.copytree(src, dest, dirs_exist_ok=True)


def copy_hidden(task: dict, dest: Path, kit: Path = KIT) -> None:
    for target, src in (task.get("hidden") or {}).items():
        t = dest / target
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(kit / src, t)


def run_check(task: dict, dest: Path, kit: Path = KIT, python: str | None = None, timeout: int = 900) -> dict:
    """Copy the hidden files, then run every check command in the repo; all must exit 0."""
    copy_hidden(task, dest, kit)
    python = python or sys.executable
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    steps = []
    for cmd in task["check"]:
        cmd = cmd.replace("{python}", Path(python).as_posix()).replace("{kit}", kit.as_posix())
        try:
            r = subprocess.run(
                shlex.split(cmd), cwd=dest, env=env, capture_output=True, text=True, timeout=timeout, errors="replace"
            )
            code, out = r.returncode, (r.stdout + r.stderr)[-3000:]
        except subprocess.TimeoutExpired:
            code, out = -1, "check timed out"
        steps.append({"cmd": cmd, "exit_code": code, "output_tail": out})
        if code != 0:
            break
    return {"passed": all(s["exit_code"] == 0 for s in steps) and len(steps) == len(task["check"]), "steps": steps}


# ---- processes -------------------------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def rmtree(path: Path) -> None:
    """Remove a temp dir; git object files are read-only on Windows."""

    def writable(func, p, _exc):
        os.chmod(p, 0o700)
        func(p)

    shutil.rmtree(path, onexc=writable) if sys.version_info >= (3, 12) else shutil.rmtree(path, onerror=writable)


def popen(cmd, **kw) -> subprocess.Popen:
    if os.name != "nt":
        kw["start_new_session"] = True
    return subprocess.Popen(cmd, **kw)


def kill_tree(p: subprocess.Popen, grace: float = 10.0) -> None:
    """Stop one process we started (and its children), by pid."""
    if p.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True)
    else:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        p.wait(grace)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        p.kill()
        p.wait(grace)


def wait_http(url: str, secs: float, proc: subprocess.Popen | None = None) -> bool:
    end = time.time() + secs
    while time.time() < end:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


class Server:
    """An engine process we started. `with Server(...)` stops it by pid."""

    def __init__(self, engine: str, cmd: list[str], port: int, log: Path):
        self.engine, self.cmd, self.port, self.log = engine, cmd, port, log
        self.proc = None
        self.url = f"http://127.0.0.1:{port}"

    def start(self, load_secs: int = 900) -> bool:
        say(f"start {self.engine}: {' '.join(self.cmd)}")
        self._fh = open(self.log, "w", encoding="utf-8")
        self.proc = popen(self.cmd, stdout=self._fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        ready = f"{self.url}/health" if self.engine == "llama-server" else f"{self.url}/v1/models"
        ok = wait_http(ready, load_secs, self.proc)
        say(f"{self.engine} {'ready' if ok else 'NOT ready'}")
        return ok

    def stop(self):
        if self.proc is not None:
            kill_tree(self.proc, 30)
            self._fh.close()
            time.sleep(3)  # GPU memory is released after the process is gone


def server_cmd(args, engine: str, port: int, reqlog: Path) -> list[str]:
    if engine == "tetraeder":
        cmd = [args.tet_bin, "serve", "-m", args.model, "--ctx", str(args.ctx), "--port", str(port), "--alias", args.alias,
               "--accept-any-model", "--ngl", "-1", "-t", str(args.threads), "--request-log", str(reqlog),
               "--cache-ram", str(args.cache_ram)]
        return cmd + shlex.split(args.tet_args)
    cmd = [args.llama_bin, "-m", args.model, "-c", str(args.ctx), "-ngl", "99", "-t", str(args.threads),
           "--host", "127.0.0.1", "--port", str(port), "--jinja", "-np", "1", "--alias", args.alias,
           "--cache-ram", str(args.cache_ram)]
    return cmd + shlex.split(args.llama_args)


# ---- one Claude Code session -----------------------------------------------------------------------------------


def claude_env(args, base_url: str, home: Path) -> dict:
    """Environment of one session; `home` is a fresh empty HOME so no login, memory file or settings are read."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_", "CLAUDECODE"))}
    # the venv first on PATH: the model's `python -m pytest` must find pytest
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    env.update(
        ANTHROPIC_BASE_URL=base_url,
        ANTHROPIC_AUTH_TOKEN="local",
        ANTHROPIC_API_KEY="",
        ANTHROPIC_MODEL=args.alias,
        ANTHROPIC_DEFAULT_HAIKU_MODEL=args.alias,
        ANTHROPIC_SMALL_FAST_MODEL=args.alias,
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1",
        DISABLE_AUTOUPDATER="1",
        DISABLE_TELEMETRY="1",
        CLAUDE_CONFIG_DIR=str(home / ".claude"),
        HOME=str(home),
        USERPROFILE=str(home),
        PYTHONDONTWRITEBYTECODE="1",
        **GIT_ENV,
    )
    if args.permissions == "skip":
        env["IS_SANDBOX"] = "1"
    for kv in args.claude_env:
        k, _, v = kv.partition("=")
        env[k] = v
    return env


def claude_cmd(args, prompt: str) -> list[str]:
    cmd = [args.claude, "-p", prompt, "--output-format", "json", "--model", args.alias, "--max-turns", str(args.max_turns)]
    if args.permissions == "skip":
        cmd.append("--dangerously-skip-permissions")
    else:
        cmd += ["--permission-mode", "acceptEdits", "--allowedTools", ALLOWED_TOOLS]
    return cmd


def parse_claude_json(text: str):
    text = text.strip()
    for candidate in [text] + text.splitlines()[::-1]:
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, list):  # stream-style array: take the result entry
            obj = next((o for o in reversed(obj) if isinstance(o, dict) and o.get("type") == "result"), None)
        if isinstance(obj, dict):
            return obj
    return None


def ancestor_memory_files(path: Path) -> list[Path]:
    """CLAUDE.md files Claude Code would pick up from the directories above `path`."""
    found = []
    for d in Path(path).resolve().parents:
        for rel in ("CLAUDE.md", ".claude/CLAUDE.md"):
            if (d / rel).is_file():
                found.append(d / rel)
    return found


def session_name(engine, rep, task_id):
    return f"{engine}-r{rep}-{task_id}"


def run_session(args, engine: str, rep: int, task: dict, upstream: str, out: Path) -> dict:
    name = session_name(engine, rep, task["id"])
    proxy_log = out / f"proxy-{name}.jsonl"
    proxy_log.write_text("")
    work = Path(tempfile.mkdtemp(prefix=f"cc-{task['id']}-", dir=args.work_root))
    home = Path(tempfile.mkdtemp(prefix="cc-home-", dir=args.work_root))
    repo = prepare_workdir(task, work / "repo")
    pport = free_port()
    proxy = popen(
        [sys.executable, str(ROOT / "scripts" / "anthropic_proxy.py"), "--listen-port", str(pport), "--upstream", upstream,
         "--label", engine, "--task", task["id"], "--rep", str(rep), "--log", str(proxy_log)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
    )
    result = {"engine": engine, "rep": rep, "task": task["id"], "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "timed_out": False, "exit_code": None, "claude": None}
    try:
        for _ in range(50):
            try:
                socket.create_connection(("127.0.0.1", pport), timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.1)
        t0 = time.time()
        p = popen(claude_cmd(args, task["prompt"]), cwd=repo, env=claude_env(args, f"http://127.0.0.1:{pport}", home),
                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True, errors="replace")
        try:
            stdout, stderr = p.communicate(timeout=args.task_timeout)
            result["exit_code"] = p.returncode
        except subprocess.TimeoutExpired:
            result["timed_out"] = True
            kill_tree(p)
            try:
                stdout, stderr = p.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
        result["wall_s"] = round(time.time() - t0, 2)
        result["claude"] = parse_claude_json(stdout or "")
        (out / f"stderr-{name}.log").write_text((stderr or "")[-20000:], encoding="utf-8")
        if result["claude"] is None:
            result["stdout_tail"] = (stdout or "")[-2000:]
        result["check"] = run_check(task, repo, python=sys.executable)
        result["success"] = bool(result["check"]["passed"])
        # leave the model's diff behind for review (small, not fetched: .patch)
        diff = subprocess.run(["git", "diff", "HEAD"], cwd=repo, capture_output=True, text=True, errors="replace").stdout
        (out / f"diff-{name}.patch").write_text(diff[-200000:], encoding="utf-8")
    finally:
        kill_tree(proxy, 5)
        if not args.keep:
            for d in (work, home):
                try:
                    rmtree(d)
                except OSError as e:
                    say(f"could not remove {d}: {e}")
    (out / f"session-{name}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    c = result.get("claude") or {}
    say(f"{name}: success={result['success']} turns={c.get('num_turns')} is_error={c.get('is_error')} "
        f"wall={result['wall_s']}s{' TIMEOUT' if result['timed_out'] else ''}")
    return result


# ---- orchestration ---------------------------------------------------------------------------------------------


def block_order(reps: int, engines=ENGINES) -> list[tuple[int, str]]:
    """ABBA: odd reps run tetraeder first, even reps llama-server first."""
    order = []
    for rep in range(1, reps + 1):
        es = list(engines) if rep % 2 else list(reversed(engines))
        order += [(rep, e) for e in es]
    return order


def pending(out: Path, engine: str, rep: int, tasks: list[dict]) -> list[dict]:
    return [t for t in tasks if not (out / f"session-{session_name(engine, rep, t['id'])}.json").exists()]


def run_all(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks()
    if args.tasks:
        wanted = args.tasks.split(",")
        tasks = [t for t in tasks if t["id"] in wanted]
    args.work_root = args.work_root or tempfile.gettempdir()
    for f in ancestor_memory_files(Path(args.work_root)):
        say(f"WARNING: {f} is above the work dir and will be loaded by Claude Code; use --work-root elsewhere")
    t_start = time.time()
    budget, hard = args.budget_min * 60, (args.budget_min + args.grace_min) * 60
    if args.url:
        engines = [args.engine]
    else:
        engines = list(ENGINES) if args.engine == "both" else [args.engine]
    for rep, engine in block_order(args.reps, engines):
        todo = pending(out, engine, rep, tasks)
        if not todo:
            continue
        if time.time() - t_start > budget:
            say(f"budget spent: skip block {engine} r{rep}")
            continue
        server = None
        if args.url:
            upstream = args.url
        else:
            port = args.tet_port if engine == "tetraeder" else args.llama_port
            server = Server(engine, server_cmd(args, engine, port, out / f"{engine}-r{rep}.reqlog.jsonl"), port,
                            out / f"server-{engine}-r{rep}.log")
            if not server.start():
                server.stop()
                continue
            upstream = server.url
        try:
            for task in todo:
                if time.time() - t_start > hard:
                    say(f"hard limit: skip {session_name(engine, rep, task['id'])}")
                    continue
                run_session(args, engine, rep, task, upstream, out)
        finally:
            if server:
                server.stop()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="result directory (bench/results/<date>/claude-code-proxy)")
    ap.add_argument("--engine", default="both", choices=["both", *ENGINES])
    ap.add_argument("--url", help="use this running server instead of spawning one (needs a single --engine)")
    ap.add_argument("--reps", type=int, default=int(os.environ.get("REPS", "2")))
    ap.add_argument("--tasks", help="comma separated task ids (default: all)")
    ap.add_argument("--model", help="GGUF file")
    ap.add_argument("--alias", default="qwen3-coder")
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--cache-ram", type=int, default=8192, help="prompt cache in host RAM, MiB, for both engines (ADR 0049)")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--tet-bin", default="target/release/tetraeder")
    ap.add_argument("--llama-bin", default="/opt/llama.cpp/bin/llama-server")
    ap.add_argument("--tet-args", default="", help="extra tetraeder serve arguments")
    ap.add_argument("--llama-args", default="", help="extra llama-server arguments")
    ap.add_argument("--tet-port", type=int, default=18080)
    ap.add_argument("--llama-port", type=int, default=18081)
    ap.add_argument("--claude", default=shutil.which("claude") or "claude")
    ap.add_argument("--permissions", choices=["skip", "accept"], default="skip")
    ap.add_argument("--max-turns", type=int, default=40)
    ap.add_argument("--task-timeout", type=int, default=900, help="seconds per session")
    ap.add_argument("--budget-min", type=float, default=240, help="no block or session starts after this")
    ap.add_argument("--grace-min", type=float, default=30, help="a block that started in budget may run this longer")
    ap.add_argument("--claude-env", action="append", default=[], help="KEY=VALUE for the claude process")
    ap.add_argument("--work-root", help="parent of the per-session temp dirs (default: system temp)")
    ap.add_argument("--keep", action="store_true", help="keep the temp work dirs")
    args = ap.parse_args(argv)
    if not args.url and not args.model:
        ap.error("--model is required unless --url is given")
    if args.url and args.engine == "both":
        ap.error("--url needs --engine tetraeder or llama-server")
    return run_all(args)


if __name__ == "__main__":
    sys.exit(main())
