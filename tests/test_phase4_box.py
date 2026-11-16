"""scripts/phase4_box.sh in DRY_RUN=1 mode: the stage order, the pair order and the server flags of ADR 0065."""

import json
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

STAGES = ["build", "gpu-parity", "gpu-shapes", "prepass", "probe", "record-log", "env", "pairs", "ttft", "summary"]


def dry(tmp_path, rc=0, **env):
    e = {**os.environ, "DRY_RUN": "1", "P4_OUT": str(tmp_path / "out"), "THREADS": "8", **env}
    r = subprocess.run(["bash", str(SCRIPT)], capture_output=True, text=True, env=e, cwd=ROOT, timeout=120)
    assert r.returncode == rc, r.stdout + r.stderr
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
    # the gates run before anything else: no record-log before the probe ends
    rec = next(i for i, ln in enumerate(lines) if "STAGE record-log" in ln)
    assert next(i for i, ln in enumerate(lines) if "claude_code_proxy.sh" in ln) > rec
    assert not any("claude_code_proxy.sh" in ln for ln in lines[:rec])


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


def test_q30b_on_libllama_only_by_override(tmp_path):
    # Q30B_ENGINE=llama (a control run): the judged run is libllama, there is no separate control
    q30b = [n for n in runs(dry(tmp_path, Q30B_ENGINE="llama")) if n.startswith("q30b-")]
    assert q30b == ["q30b-llama-server-r1", "q30b-tetraeder-llama-r1", "q30b-tetraeder-llama-r2", "q30b-llama-server-r2"]


def test_q30b_engine_defaults_to_moe(tmp_path):
    lines = dry(tmp_path)
    assert any("q30b=moe" in ln for ln in lines)
    moe = [ln for ln in lines if "PLAN server" in ln and "server-q30b-tetraeder-moe-r1.log" in ln]
    assert moe and "--engine moe" in moe[0]
    assert [n for n in runs(lines) if n.startswith("q30b-tetraeder")] == [
        "q30b-tetraeder-moe-r1", "q30b-tetraeder-llama-r1", "q30b-tetraeder-moe-r2", "q30b-tetraeder-llama-r2"]


def test_gpu_gates_in_the_plan(tmp_path):
    lines = dry(tmp_path)
    shapes = next(ln for ln in lines if "step_shapes" in ln)
    for x in ("TETRAEDER_SHAPES_NGL=99", "TETRAEDER_SHAPES_ENGINES=llama,llama_streams,dense", "--features cuda",
              "gpu-shapes.txt"):
        assert x in shapes
    probes = [ln for ln in lines if "PLAN server" in ln and "server-probe-" in ln]
    assert [re.search(r"server-probe-(\S+)\.log", ln).group(1) for ln in probes] == [
        "q8b-llama-server", "q8b-tetraeder-dense", "q30b-llama-server", "q30b-tetraeder-moe"]
    assert "--engine dense" in probes[1] and "--engine moe" in probes[3]
    assert "-fa on" in probes[0] and "-kvu" in probes[0]  # the pre-pass default (DRY picks kvu)
    c4 = next(ln for ln in lines if "-> probe-q8b-tetraeder-dense-c4.json" in ln)
    c1 = next(ln for ln in lines if "-> probe-q8b-tetraeder-dense-c1.json" in ln)
    assert "--concurrency 4 --requests 8 --seed 94" in c4 and "--concurrency 1 --requests 8 --seed 91" in c1
    assert any("probe.json" in ln and "0.85" in ln for ln in lines)


def write_probe(out, own_c4, model="q8b", engine="tetraeder-dense"):
    out.mkdir(parents=True, exist_ok=True)
    for name, c4 in (("llama-server", 100.0), (engine, own_c4)):
        for test, tps in (("c1", 10.0), ("c4", c4)):
            (out / f"probe-{model}-{name}-{test}.json").write_text(
                json.dumps({"summary": {"errors": 0, "throughput_tps": tps}}))


def test_low_probe_ends_the_run_before_record_log_and_pairs(tmp_path):
    write_probe(tmp_path / "out", 80.0)
    write_probe(tmp_path / "out", 70.0, "q30b", "tetraeder-moe")
    lines = dry(tmp_path, rc=3, PHASE4_LOG=str(tmp_path / "none.jsonl"))
    stages = [m.group(1) for ln in lines if (m := re.search(r"STAGE (\S+)", ln))]
    assert stages == STAGES[: STAGES.index("probe") + 1]
    assert any("PHASE4_BOX_DONE rc=3" in ln for ln in lines)
    assert not any("claude_code_proxy.sh" in ln or "run q8b-" in ln for ln in lines)
    probe = json.loads((tmp_path / "out" / "probe.json").read_text())
    assert probe["stop"] is True and probe["models"]["q8b"]["c4_ratio"] == 0.8
    assert probe["stopped"] == ["q8b", "q30b"]
    assert (tmp_path / "out" / "DONE").exists()


def test_good_probe_continues(tmp_path):
    write_probe(tmp_path / "out", 92.0)
    write_probe(tmp_path / "out", 95.0, "q30b", "tetraeder-moe")
    lines = dry(tmp_path)
    assert [m.group(1) for ln in lines if (m := re.search(r"STAGE (\S+)", ln))] == STAGES
    assert json.loads((tmp_path / "out" / "probe.json").read_text())["stop"] is False


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
    assert "--prompt-tokens 512 --max-tokens 128 --concurrency 4 --requests 16 --seed 24" in c4
    c1 = next(ln for ln in lines if "bench-q8b-llama-server-c1-r1.json" in ln)
    assert "--concurrency 1 --requests 8 --seed 11" in c1
    assert any("gpu-q8b-llama-server-r1.txt" in ln and "before" in ln for ln in lines)
    assert any("gpu-q8b-llama-server-r1.txt" in ln and "after" in ln for ln in lines)
    assert any("prepass-q30b-nokvu.json" in ln for ln in lines)
    replay = next(ln for ln in lines if "-> replay-q8b-on-r1.json" in ln)
    for flag in ("--speed 0", "--concurrency 1", "--max-prompt-tokens 36000", "--expect-context-length 40960"):
        assert flag in replay


def bench_calls(lines):
    return [ln for ln in lines if ln.startswith("PLAN gateway_bench.py")]


def test_every_bench_call_has_no_keepalive(tmp_path):
    calls = bench_calls(dry(tmp_path))
    assert len(calls) > 30 and all("--no-keepalive" in ln for ln in calls)


def test_seeds_per_test_equal_across_engines(tmp_path):
    seeds = {}  # (model, test, rep) -> {engine: seed}
    for ln in bench_calls(dry(tmp_path)):
        m = re.search(r"--label (q\d+b)-(\S+)-r(\d)-(c[14]) .*--seed (\d+)", ln)
        if m:
            model, engine, rep, test, seed = m.groups()
            seeds.setdefault((model, test, rep), {})[engine] = seed
    assert seeds
    for key, by_engine in seeds.items():
        assert len(set(by_engine.values())) == 1, (key, by_engine)
    # c1 and c4 of one server start never share a seed
    for model, rep in {(k[0], k[2]) for k in seeds}:
        assert set(seeds[(model, "c1", rep)].values()) != set(seeds[(model, "c4", rep)].values())


def test_failed_gpu_parity_of_every_model_ends_after_shapes(tmp_path):
    lines = dry(tmp_path, rc=3, DRY_GPU_PARITY="fail", PHASE4_LOG=str(tmp_path / "none.jsonl"))
    stages = [m.group(1) for ln in lines if (m := re.search(r"STAGE (\S+)", ln))]
    assert stages == STAGES[: STAGES.index("probe") + 1]
    assert any("PHASE4_BOX_DONE rc=3" in ln for ln in lines)
    # no server of a stopped model starts: the pre-pass and probe skip it
    assert not any("PLAN server" in ln for ln in lines)
    assert not any("claude_code_proxy.sh" in ln or "run q8b-" in ln for ln in lines)
    probe = json.loads((tmp_path / "out" / "probe.json").read_text())
    assert probe["stop"] is True and probe["gpu_parity"] == "failed" and probe["stopped"] == ["q8b", "q30b"]


def test_gpu_parity_commands_per_model(tmp_path):
    lines = dry(tmp_path)
    q8b = next(ln for ln in lines if "gpu-parity-q8b.log" in ln)
    q30b = next(ln for ln in lines if "gpu-parity-q30b.log" in ln)
    assert "--test gpu_dense" in q8b and "--test gpu_real_batched" in q8b and "--skip gpu_qwen3_30b" in q8b
    assert "--test gpu_multi_seq" in q30b and "--test gpu_real_batched" in q30b and "--skip gpu_qwen3_8b" in q30b
    assert "TETRAEDER_REQUIRE_GPU=1" in q8b and "TETRAEDER_REQUIRE_GPU=1" in q30b


def test_failed_parity_of_the_30b_skips_only_the_30b(tmp_path):
    write_probe(tmp_path / "out", 92.0)
    lines = dry(tmp_path, DRY_GPU_PARITY="q30b")
    names = runs(lines)
    assert [n for n in names if n.startswith("q8b-")] and not [n for n in names if n.startswith("q30b-")]
    assert any(n.startswith("replay-q8b") for n in names)
    assert not any("q30b" in ln and "PLAN server" in ln for ln in lines)
    assert any("STOPPED q30b: gpu-parity failed" in ln for ln in lines)
    assert any("PHASE4_BOX_DONE rc=0" in ln for ln in lines)
    probe = json.loads((tmp_path / "out" / "probe.json").read_text())
    assert probe["stop"] is False and probe["stopped"] == ["q30b"]
    assert probe["models"]["q30b"]["gpu_parity"] == "failed" and probe["models"]["q8b"]["gpu_parity"] == "ok"


def test_failed_parity_of_the_8b_skips_its_pairs_only(tmp_path):
    write_probe(tmp_path / "out", 95.0, "q30b", "tetraeder-moe")
    lines = dry(tmp_path, DRY_GPU_PARITY="q8b", PHASE4_LOG=str(tmp_path / "none.jsonl"))
    names = runs(lines)
    assert [n for n in names if n.startswith("q30b-")] and not [n for n in names if n.startswith("q8b-")]
    assert [n for n in names if n.startswith("replay")]  # the TTFT pairs still run
    assert any("claude_code_proxy.sh" in ln for ln in lines)


def test_low_probe_of_the_30b_skips_only_the_30b(tmp_path):
    write_probe(tmp_path / "out", 92.0)
    write_probe(tmp_path / "out", 80.0, "q30b", "tetraeder-moe")
    lines = dry(tmp_path)
    names = runs(lines)
    assert [n for n in names if n.startswith("q8b-")] and not [n for n in names if n.startswith("q30b-")]
    assert [n for n in names if n.startswith("replay")]  # the TTFT pairs run
    assert not any("prepass q30b" in ln and "start" in ln for ln in lines[lines.index(next(ln for ln in lines if "STAGE probe" in ln)):])
    probe = json.loads((tmp_path / "out" / "probe.json").read_text())
    assert probe["stop"] is False and probe["stopped"] == ["q30b"]


def test_low_probe_of_the_8b_keeps_the_30b_and_the_ttft(tmp_path):
    write_probe(tmp_path / "out", 80.0)
    write_probe(tmp_path / "out", 95.0, "q30b", "tetraeder-moe")
    names = runs(dry(tmp_path))
    assert [n for n in names if n.startswith("q30b-")] and not [n for n in names if n.startswith("q8b-")]
    assert [n for n in names if n.startswith("replay")]


def write_run(out, model, name, rep, c4=100.0, c1=10.0, errors=0):
    for test, key, v in (("c1", "decode_tps_median", c1), ("c4", "throughput_tps", c4)):
        (out / f"bench-{model}-{name}-{test}-r{rep}.json").write_text(
            json.dumps({"summary": {"errors": errors, key: v}}))
    (out / f"gpu-{model}-{name}-r{rep}.txt").write_text("before: 0x1\nafter: 0x1\n")


def test_unclean_pair_is_rerun_as_rep_3(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    for rep in (1, 2):
        for name in ("llama-server", "tetraeder-dense", "tetraeder-llama"):
            write_run(out, "q8b", name, rep, errors=1 if (rep, name) == (2, "tetraeder-dense") else 0)
    names = runs(dry(tmp_path))
    q8b = [n for n in names if n.startswith("q8b-")]
    # rep 2 ran (judged, baseline), so the rerun has that order; the control follows
    assert q8b == ["q8b-tetraeder-dense-r3", "q8b-llama-server-r3", "q8b-tetraeder-llama-r3"]
    assert not [n for n in names if n.startswith("q30b-llama-server-r3")]


def test_clean_pairs_are_not_rerun(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    for rep in (1, 2):
        for name in ("llama-server", "tetraeder-dense", "tetraeder-llama"):
            write_run(out, "q8b", name, rep)
    assert not [n for n in runs(dry(tmp_path)) if n.endswith("-r3") and n.startswith("q8b")]
