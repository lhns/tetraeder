"""scripts/phase4_box.sh in DRY_RUN=1 mode: the stage order, the pair order and the server flags of ADR 0065."""

import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "phase4_box.sh"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="needs bash on Linux (runs on the ubuntu CI job)"
)

STAGES = ["record-log", "build", "gpu-parity", "env", "prepass", "pairs", "ttft", "summary"]


def dry(tmp_path, **env):
    e = {**os.environ, "DRY_RUN": "1", "P4_OUT": str(tmp_path / "out"), "THREADS": "8", **env}
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=e, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.splitlines()


def runs(lines):
    """Run names in order: `run <name> start`."""
    return [m.group(1) for ln in lines if (m := re.search(r"\brun (\S+) start", ln))]


def test_stage_order_and_names(tmp_path):
    lines = dry(tmp_path)
    stages = [m.group(1) for ln in lines if (m := re.search(r"STAGE (\S+)", ln))]
    assert stages == STAGES
    # every stage the script documents in its header is one it runs
    header = SCRIPT.read_text().split("\n\n")[0]
    assert all(re.search(rf"^#   {s}\s", header, re.M) for s in STAGES)
    first_server = next(i for i, ln in enumerate(lines) if "PLAN server" in ln)
    assert next(i for i, ln in enumerate(lines) if "claude_code_proxy.sh" in ln) < first_server


def test_existing_log_skips_recording(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text('{"a":1}\n')
    lines = dry(tmp_path, PHASE4_LOG=str(log))
    assert not any("claude_code_proxy.sh" in ln for ln in lines)
    assert any("exists, sha256" in ln for ln in lines)


def test_pairs_interleaved_with_control(tmp_path):
    names = runs(dry(tmp_path))
    q8b = [n for n in names if n.startswith("q8b-")]
    assert q8b == [
        "q8b-llama-server-r1", "q8b-tetraeder-dense-r1", "q8b-tetraeder-llama-r1",
        "q8b-tetraeder-dense-r2", "q8b-llama-server-r2", "q8b-tetraeder-llama-r2",
    ]
    # q30b without the own MoE builder: the judged run is libllama, there is no separate control
    q30b = [n for n in names if n.startswith("q30b-")]
    assert q30b == ["q30b-llama-server-r1", "q30b-tetraeder-llama-r1", "q30b-tetraeder-llama-r2", "q30b-llama-server-r2"]


def test_q30b_moe_has_a_control(tmp_path):
    q30b = [n for n in runs(dry(tmp_path, Q30B_ENGINE="moe")) if n.startswith("q30b-")]
    assert q30b == [
        "q30b-llama-server-r1", "q30b-tetraeder-moe-r1", "q30b-tetraeder-llama-r1",
        "q30b-tetraeder-moe-r2", "q30b-llama-server-r2", "q30b-tetraeder-llama-r2",
    ]


def test_prepass_before_pairs_and_ttft_after(tmp_path):
    lines = dry(tmp_path)
    pre = [i for i, ln in enumerate(lines) if "prepass q8b" in ln and "start" in ln]
    pair = next(i for i, ln in enumerate(lines) if "run q8b-llama-server-r1 start" in ln)
    ttft = next(i for i, ln in enumerate(lines) if "run replay-q8b-on-r1 start" in ln)
    assert len(pre) == 2 and max(pre) < pair < ttft
    assert [n for n in runs(lines) if n.startswith("replay")] == [
        "replay-q8b-on-r1", "replay-q8b-off-r1", "replay-q8b-off-r2", "replay-q8b-on-r2", "replay-q8b-llama-server-r1"]


def server_lines(lines, tag):
    return [ln for ln in lines if "PLAN server" in ln and tag in ln.split(":", 1)[0]]


def test_server_flags(tmp_path):
    lines = dry(tmp_path)
    base = server_lines(lines, "server-q8b-llama-server-r1.log")[0]
    for flag in ("/opt/llama.cpp-baseline/bin/llama-server", "-ngl 99", "-np 4", "-c 32768", "-fa on", "-ctk f16",
                 "-ctv f16", "-t 8", "-kvu"):
        assert flag in base
    assert "-no-kvu" in server_lines(lines, "server-prepass-q8b-nokvu.log")[0]
    tet = server_lines(lines, "server-q8b-tetraeder-dense-r1.log")[0]
    for flag in ("serve", "--engine dense", "--ngl -1", "--parallel 4", "--ctx 32768", "-t 8"):
        assert flag in tet
    assert "--flash-attn on" in server_lines(lines, "server-q8b-tetraeder-llama-r1.log")[0]
    on = server_lines(lines, "server-replay-q8b-on-r1.log")[0]
    off = server_lines(lines, "server-replay-q8b-off-r1.log")[0]
    assert "--ctx 40960" in on and "--parallel 4" in on and "--no-prefix-reuse" not in on and "--cache-ram" not in on
    assert "--ctx 40960" in off and "--no-prefix-reuse" in off and "--cache-ram 0" in off
    info = server_lines(lines, "server-replay-q8b-llama-server-r1.log")[0]
    assert "-np 4" in info and "-kvu" in info and "--cache-ram 8192" in info


def test_workload_seed_and_files(tmp_path):
    lines = dry(tmp_path)
    c4 = next(ln for ln in lines if "bench-q8b-tetraeder-dense-c4-r2.json" in ln)
    assert "--prompt-tokens 512 --max-tokens 128 --concurrency 4 --requests 16 --seed 2" in c4
    c1 = next(ln for ln in lines if "bench-q8b-llama-server-c1-r1.json" in ln)
    assert "--concurrency 1 --requests 8 --seed 1" in c1
    assert any("gpu-q8b-llama-server-r1.txt" in ln and "before" in ln for ln in lines)
    assert any("gpu-q8b-llama-server-r1.txt" in ln and "after" in ln for ln in lines)
    assert any("prepass-q30b-nokvu.json" in ln for ln in lines)
    replay = next(ln for ln in lines if "-> replay-q8b-on-r1.json" in ln)
    for flag in ("--speed 0", "--concurrency 1", "--max-prompt-tokens 36000", "--expect-context-length 40960"):
        assert flag in replay
