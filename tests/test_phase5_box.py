"""scripts/phase5_box.sh in DRY_RUN=1 mode: the stage order, the gates before timed runs, predictions before candidates,
the rotation of the rounds and resume (ADR 0068)."""

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "phase5_box.sh"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="needs bash on Linux (runs on the ubuntu CI job)"
)

STAGES = ["gpus", "env", "build", "models", "gpu-parity", "discover", "enumerate", "probe", "prepass", "anchors",
          "calibrate", "predict", "rounds", "vllm", "summary"]
D = "qwen3-32b"
M = "qwen3-30b-a3b"
# the case order of D2 (candidates, then the baselines of the prepass), the order summary --round-order rotates
D2 = ["single", "layer-2", "tp-2", "llama-bench-layer", "llama-bench-row", "llama-bench-tensor",
      "ik-llama-bench-layer", "ik-llama-bench-row", "ik-llama-bench-graph"]


def dry(tmp_path, ngpu="2", rc=0, args=None, **env):
    e = {**os.environ, "DRY_RUN": "1", "P5_OUT": str(tmp_path / "out"), "THREADS": "8", "PY": sys.executable, **env}
    argv = ["bash", str(SCRIPT)] + (["s1", ngpu] if args is None else args)
    r = subprocess.run(argv, capture_output=True, text=True, env=e, cwd=ROOT, timeout=300)
    assert r.returncode == rc, r.stdout + r.stderr
    return r.stdout.splitlines()


def stages(lines):
    return [m.group(1) for ln in lines if (m := re.search(r"STAGE (\S+)", ln))]


def runs(lines):
    """Run names in order: `run <name> start`."""
    return [m.group(1) for ln in lines if (m := re.search(r"\brun (\S+) start", ln))]


def timed(lines, model=D):
    """Candidate and baseline runs of the rounds (not anchors, not the A/A)."""
    return [n for n in runs(lines) if n.startswith(model + "-") and "-anchor-" not in n and not n.endswith("-aa")]


def test_stage_order_and_names(tmp_path):
    lines = dry(tmp_path)
    assert stages(lines) == STAGES
    # every stage the script documents in its header is one it runs
    header = SCRIPT.read_text().split("\n\n")[0]
    assert all(re.search(rf"^#   {s}\s", header, re.M) for s in STAGES)
    assert lines[-1].endswith("PHASE5_BOX_DONE rc=0")


def test_lock_guard_is_the_first_thing(tmp_path):
    src = SCRIPT.read_text()
    guard = src.index('exec 8> "${P5_LOCK:-/root/tetraeder_phase5_box.lock}"')
    assert src.index("flock -n 8 || exit 0") > guard
    assert guard < src.index("cd \"$(dirname")  # before any work
    # a held lock: the script ends at once with nothing run
    fcntl = pytest.importorskip("fcntl")
    lock = tmp_path / "held.lock"
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        e = {**os.environ, "P5_LOCK": str(lock), "P5_OUT": str(tmp_path / "out2")}
        r = subprocess.run(["bash", str(SCRIPT), "s1", "2"], capture_output=True, text=True, env=e, cwd=ROOT, timeout=60)
    assert r.returncode == 0 and r.stdout == "" and not (tmp_path / "out2").exists()


def test_usage_needs_the_gpu_count(tmp_path):
    r = subprocess.run(["bash", str(SCRIPT), "s1"], capture_output=True, text=True, cwd=ROOT, timeout=60,
                       env={**os.environ, "DRY_RUN": "1", "P5_OUT": str(tmp_path / "o")})
    assert r.returncode == 2 and "usage" in r.stderr
    assert subprocess.run(["bash", str(SCRIPT), "s1", "3"], capture_output=True, cwd=ROOT, timeout=60,
                          env={**os.environ, "DRY_RUN": "1", "P5_OUT": str(tmp_path / "o")}).returncode == 2


def test_wrong_gpu_count_stops_before_anything_else(tmp_path):
    lines = dry(tmp_path, rc=1, DRY_GPUS="1")
    assert stages(lines) == ["gpus"]
    assert any("has 1 GPU(s), the session needs 2" in ln for ln in lines)
    assert any("PHASE5_BOX_DONE rc=1" in ln for ln in lines)
    assert any("nvidia-smi -L" in ln for ln in lines) and (tmp_path / "out" / "DONE").exists()
    assert dry(tmp_path, ngpu="4", rc=1, DRY_GPUS="2")  # a 2x box for a 4x session


def test_gates_and_probe_before_any_timed_run(tmp_path):
    lines = dry(tmp_path)
    idx = {s: next(i for i, ln in enumerate(lines) if f"STAGE {s}" in ln) for s in STAGES}
    parity = next(i for i, ln in enumerate(lines) if "TETRAEDER_REQUIRE_GPU=1 cargo test" in ln)
    probe = next(i for i, ln in enumerate(lines) if "comm-probe" in ln)
    assert idx["gpu-parity"] < parity < idx["discover"] and idx["probe"] < probe < idx["prepass"]
    first_run = next(i for i, ln in enumerate(lines) if re.search(r"\brun \S+ start", ln))
    assert first_run > idx["anchors"] > idx["prepass"] > probe > parity
    enum = [ln for ln in lines if "candidates --multi" in ln]
    assert len(enum) == 2 and all("--plans-dir plans" in ln for ln in enum)


def test_failed_parity_runs_the_cheap_stages_then_ends_with_rc_3(tmp_path):
    lines = dry(tmp_path, rc=3, DRY_GPU_PARITY="fail")
    assert stages(lines) == STAGES[: STAGES.index("prepass") + 1]
    assert any("PHASE5_BOX_DONE rc=3" in ln for ln in lines) and not runs(lines)
    assert (tmp_path / "out" / "DONE").exists()


def test_failed_comm_probe_ends_the_session_too(tmp_path):
    lines = dry(tmp_path, rc=3, DRY_PROBE="fail")
    assert stages(lines)[-1] == "prepass" and not runs(lines) and any("rc=3" in ln for ln in lines)


def test_anchors_are_truncated_two_rounds_and_precede_predictions(tmp_path):
    lines = dry(tmp_path)
    names = [n for n in runs(lines) if n.startswith(D + "-anchor-")]
    assert names == [f"{D}-anchor-{p}-L{lim}-r{k}" for k in (1, 2) for p in ("single", "tp-2") for lim in (16, 32)]
    cmd = next(ln for ln in lines if "qwen3-32b-tp-2.toml" in ln and "--layer-limit 16" in ln)
    assert "-p 2048 -n 128 -r 1" in cmd
    calib = next(i for i, ln in enumerate(lines) if "tetraeder-plan calibrate" in ln)
    assert calib > max(i for i, ln in enumerate(lines) if "run " in ln and "-anchor-" in ln and "start" in ln)
    assert any("anchors-qwen3-32b-r1.json" in ln for ln in lines)


def test_prediction_is_written_and_hashed_before_the_first_candidate(tmp_path):
    lines = dry(tmp_path)
    pred = [i for i, ln in enumerate(lines) if "tetraeder-plan predict" in ln]
    sha = [ln for ln in lines if "sha256 of predict-" in ln]
    assert len(pred) == 2 and len(sha) == 2 and "predict-qwen3-32b-2x.json" in lines[pred[0]]
    assert "--calib calib.json" in lines[pred[0]] and "--comm calib-comm.json" in lines[pred[0]]
    first_candidate = next(i for i, ln in enumerate(lines) if re.search(rf"\brun {D}-(single|layer-2|tp-2)-r1 start", ln))
    assert max(pred) < first_candidate
    assert not [n for n in timed(lines) if lines.index(next(ln for ln in lines if f"run {n} start" in ln)) < max(pred)]


def test_rounds_rotate_and_round_2_is_reversed(tmp_path):
    names = timed(dry(tmp_path), D)
    per_round = {k: [n[len(D) + 1 : -len(f"-r{k}")] for n in names if n.endswith(f"-r{k}")] for k in (1, 2, 3)}
    assert per_round[1] == D2[1:] + D2[:1]
    assert per_round[2] == D2[::-1]
    assert per_round[3] == D2[3:] + D2[:3]
    assert len(names) == 27 and not [n for n in names if n.endswith("-r4")]


def test_aa_is_the_last_run_of_each_model(tmp_path):
    lines = dry(tmp_path)
    names = runs(lines)
    assert names[-1] == f"{M}-single-aa"
    d_runs = [n for n in names if n.startswith(D + "-")]
    assert d_runs[-1] == f"{D}-single-aa" and d_runs.count(f"{D}-single-aa") == 1
    assert names.index(f"{D}-single-aa") < min(i for i, n in enumerate(names) if n.startswith(M + "-") and "-anchor-" not in n)
    cmd = next(ln for ln in lines if "> bench-qwen3-32b-single-aa.json" in ln)
    assert "qwen3-32b-single.toml" in cmd


def test_command_lines(tmp_path):
    lines = dry(tmp_path)
    tet = next(ln for ln in lines if "> bench-qwen3-32b-tp-2-r1.json" in ln)
    for x in ("tetraeder bench", "--engine dense", "--plan ", "plans/qwen3-32b-tp-2.toml", "-p 2048 -n 128 -r 1", "-t 8",
              "--ubatch 512", "--flash-attn on", "-o json"):
        assert x in tet
    assert "--engine moe" in next(ln for ln in lines if "> bench-qwen3-30b-a3b-ep-2-r1.json" in ln)
    base = next(ln for ln in lines if "> baseline-qwen3-32b-llama-bench-tensor-r1.json" in ln)
    for x in ("/opt/llama.cpp-baseline/bin/llama-bench", "-ngl 99", "-fa 1", "-p 2048 -n 128 -r 1", "-t 8", "-sm tensor"):
        assert x in base
    ik = next(ln for ln in lines if "> baseline-qwen3-32b-ik-llama-bench-graph-r1.json" in ln)
    assert "/opt/ik_llama/bin/llama-bench" in ik and "-sm graph" in ik
    # the same -t everywhere
    assert all("-t 8" in ln for ln in lines if ln.startswith("PLAN") and (" > bench-" in ln or " > baseline-" in ln))
    # the prepass is short
    pre = next(ln for ln in lines if "> prepass-qwen3-32b-llama-bench-row.json" in ln)
    assert "-p 64 -n 8" in pre and "-sm row" in pre
    assert any("gpu-qwen3-32b-tp-2-r1.txt" in ln and "before" in ln for ln in lines)
    assert any("gpu-qwen3-32b-tp-2-r1.txt" in ln and "after" in ln for ln in lines)


def test_four_gpu_box(tmp_path):
    lines = dry(tmp_path, ngpu="4")
    assert any("comm-probe --gpus 0,1,2,3" in ln for ln in lines)
    assert any("gpu4_" in ln for ln in lines if "cargo test" in ln)
    anchor = next(ln for ln in lines if "plans-anchor/qwen3-32b-tp-2.toml" in ln and "--layer-limit 16" in ln)
    assert "CUDA_VISIBLE_DEVICES=0,1" in anchor
    names = timed(dry(tmp_path, ngpu="4"), D)
    assert any(n.startswith(f"{D}-tp2-pp2") for n in names) and any(n.startswith(f"{D}-tp-4") for n in names)
    assert (tmp_path / "out" / "candidates-qwen3-32b.json").exists()
    assert any("predict-qwen3-32b-4x.json" in ln for ln in lines)


def test_one_model_per_session(tmp_path):
    names = runs(dry(tmp_path, MODELS=M))
    assert names and not [n for n in names if n.startswith(D + "-")]
    assert names[-1] == f"{M}-single-aa"
    assert len([ln for ln in dry(tmp_path, MODELS=M) if "candidates --multi" in ln]) == 1


def write_done(out, run, kind="bench", throttle="before: 0x1\nafter: 0x1\n"):
    rows = [{"n_prompt": 2048, "n_gen": 0, "avg_ts": 1000.0}, {"n_prompt": 0, "n_gen": 128, "avg_ts": 40.0}]
    (out / f"{kind}-{run}.json").write_text(json.dumps(rows))
    (out / f"run-{run}.rc").write_text("0\n")
    (out / f"gpu-{run}.txt").write_text(throttle)


def test_resume_skips_runs_that_are_done(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    write_done(out, f"{D}-tp-2-r1")
    write_done(out, f"{D}-llama-bench-row-r1", "baseline")
    (out / f"run-{D}-single-r2.rc").write_text("1\n")  # a failed run is evidence, not a hole: not rerun by resume
    lines = dry(tmp_path)
    assert f"{D}-tp-2-r1" not in runs(lines) and f"{D}-llama-bench-row-r1" not in runs(lines)
    assert f"{D}-single-r2" not in runs(lines)
    assert any(f"run {D}-tp-2-r1: done before" in ln for ln in lines)
    assert f"{D}-tp-2-r2" in runs(lines)


def test_prepass_and_hw_outputs_are_skipped_on_resume(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / f"prepass-{D}-llama-bench-row.json").write_text(json.dumps({"ok": True}))
    (out / f"anchors-{D}-r1.json").write_text("[]")
    lines = dry(tmp_path)
    assert not any("prepass-qwen3-32b-llama-bench-row.json" in ln and "PLAN" in ln for ln in lines)
    assert not any("anchor-single-L16-r1 start" in ln and D in ln for ln in lines)
    assert any("anchor-single-L16-r2 start" in ln and D in ln for ln in lines)


def test_unclean_configuration_is_rerun_in_a_later_round(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    dry(tmp_path, MODELS=D)  # writes the synthetic enumeration and prepass files
    ids = [n for n in D2]
    def cfg_run(c, k):
        if c.startswith(("llama-bench", "ik-llama-bench")):
            return "baseline", f"{D}-{c}-r{k}"
        return "bench", f"{D}-{c}-r{k}"
    for k in (1, 2, 3):
        for c in ids:
            kind, run = cfg_run(c, k)
            bad = c == "tp-2" and k in (2, 3)
            write_done(out, run, kind, "before: 0x1\nafter: 0x8\n" if bad else "before: 0x1\nafter: 0x1\n")
    lines = dry(tmp_path, MODELS=D)
    names = timed(lines, D)
    assert names and set(n[len(D) + 1 :] for n in names) == {"tp-2-r4", "tp-2-r5"}
    assert any("rerun round 4 for" in ln for ln in lines)
    # nothing short: no rerun round
    for k in (4,):
        write_done(out, f"{D}-tp-2-r{k}")
    write_done(out, f"{D}-tp-2-r5")
    assert not [ln for ln in dry(tmp_path, MODELS=D) if "rerun round" in ln]


def test_vllm_stage_is_report_only_and_optional(tmp_path):
    src = SCRIPT.read_text()
    assert 'left_s)" -lt 1500' in src and "under 25" in src  # skipped when under 25 minutes of the budget remain
    assert "15m" in src  # install and load within 15 minutes


def test_one_round_has_no_reruns(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    dry(tmp_path, MODELS=D, ROUNDS="1")
    # no run is clean after one round, so the rerun loop of a default session would start; ROUNDS=1 has none
    (out / f"run-{D}-single-r1.rc").write_text("1\n")
    lines = dry(tmp_path, MODELS=D, ROUNDS="1")
    assert not [ln for ln in lines if "rerun round" in ln]
    assert {n.rsplit("-r", 1)[1] for n in timed(lines, D)} == {"1"}
    assert [ln for ln in dry(tmp_path, MODELS=D, ROUNDS="1", RERUNS="1") if "rerun round 2" in ln]


def test_prepass_without_a_baseline_mode_is_a_gate(tmp_path):
    lines = dry(tmp_path, rc=3, DRY_PREPASS="fail")
    assert stages(lines) == STAGES[: STAGES.index("prepass") + 1]
    assert any("no llama-bench split mode accepted" in ln for ln in lines)
    assert any("baseline_modes=failed" in ln for ln in lines) and any("rc=3" in ln for ln in lines) and not runs(lines)


def test_ctk_flags_are_not_passed_to_llama_bench(tmp_path):
    # f16 is the default; a rejected flag would look like three rejected split modes
    base = [ln for ln in dry(tmp_path) if "llama-bench -m" in ln]
    assert base and not any("-ctk" in ln or "-ctv" in ln for ln in base)
