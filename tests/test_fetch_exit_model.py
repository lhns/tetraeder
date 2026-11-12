"""scripts/fetch_exit_model.sh with a fake downloader: skip complete shards, mirror choice, stall guard (20261112c)."""

import os
import pathlib
import shutil
import stat
import subprocess

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "fetch_exit_model.sh"
NEEDS = ("bash", "du", "stat")  # flock, setsid and pkill are shimmed below
# Box-side Linux script; on Windows runners `bash` may resolve to WSL, which cannot run Windows paths and PATH shims.
pytestmark = pytest.mark.skipif(os.name == "nt" or any(shutil.which(t) is None for t in NEEDS),
                                reason="needs a POSIX userland (Linux)")

SIZES = "Q4_K_M/a-1.gguf 100;Q4_K_M/a-2.gguf 200"
# stands in for taxameter's box.sh: records its source and spec, and writes the files named in the spec unless the
# source is listed in HANG (then it sleeps like a stalled mirror)
BOX = """#!/usr/bin/env bash
echo "$TAXAMETER_MODEL_SOURCE $2" >> "$CALLS"
case " $HANG " in *" $TAXAMETER_MODEL_SOURCE "*) sleep 60; exit 0 ;; esac
dir="$TAXAMETER_DIR/.hf_cache/Qwen__Qwen3-235B-A22B-GGUF"
for f in $(echo "${2#*:}" | tr '+' ' '); do
  mkdir -p "$dir/$(dirname "$f")"
  head -c "${SIZE_OF[$f]:-0}" /dev/zero > "$dir/$f" 2>/dev/null
done
"""


def run(tmp_path, pre=(), hang="", ms="0", hf="0", extra=None):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, root / "scripts" / "fetch_exit_model.sh")
    tax = tmp_path / "tax"
    tax.mkdir()
    box = tax / "box.sh"
    box.write_text(BOX.replace('"${SIZE_OF[$f]:-0}"', '$(case "$f" in *a-1*) echo 100;; *) echo 200;; esac)'), newline="\n")
    fake = tmp_path / "bin"
    fake.mkdir()
    for name, body in {"flock": "exit 0", "setsid": 'exec "$@"', "pkill": "exit 0"}.items():
        shim = fake / name
        shim.write_text(f"#!/bin/sh\n{body}\n", newline="\n")
        shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    shard_dir = root / ".hf_cache" / "Qwen__Qwen3-235B-A22B-GGUF" / "Q4_K_M"
    for name, size in pre:
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / name).write_bytes(b"x" * size)
    calls = tmp_path / "calls"
    env = dict(os.environ, PATH=f"{fake}{os.pathsep}{os.environ['PATH']}", TAXAMETER_HOME=str(tax), CALLS=str(calls),
               FETCH_LOCK=str(tmp_path / "lock"), FETCH_SIZES=SIZES, HANG=hang, FETCH_PROBE_MS_KBPS=ms,
               FETCH_PROBE_HF_KBPS=hf, FETCH_SAMPLE_S="1", FETCH_STALL_S="2", FETCH_MIN_MBPS="20", **(extra or {}))
    subprocess.run(["bash", str(root / "scripts" / "fetch_exit_model.sh")], env=env, timeout=60, check=False)
    log = (root / "runs" / "vast" / "fetch_exit.log").read_text()
    return log, calls.read_text().splitlines() if calls.exists() else [], shard_dir


def test_complete_shard_is_skipped_and_untouched(tmp_path):
    log, calls, shard_dir = run(tmp_path, pre=[("a-1.gguf", 100)], hf="5")
    assert len(calls) == 1 and "a-2.gguf" in calls[0] and "a-1.gguf" not in calls[0]
    assert (shard_dir / "a-1.gguf").read_bytes() == b"x" * 100  # not rewritten
    assert "FETCH_DONE rc=0" in log


def test_everything_complete_runs_no_downloader(tmp_path):
    log, calls, _ = run(tmp_path, pre=[("a-1.gguf", 100), ("a-2.gguf", 200)])
    assert calls == [] and "FETCH_DONE rc=0" in log


def test_zero_probes_fall_back_to_huggingface(tmp_path):
    _, calls, _ = run(tmp_path, ms="0", hf="0")
    assert calls[0].startswith("huggingface ")


def test_modelscope_when_it_is_not_slower(tmp_path):
    _, calls, _ = run(tmp_path, ms="5000", hf="5000")
    assert calls[0].startswith("modelscope ")


def test_stall_switches_mirror(tmp_path):
    log, calls, shard_dir = run(tmp_path, ms="5000", hf="5000", hang="modelscope")
    assert [c.split()[0] for c in calls] == ["modelscope", "huggingface"]
    assert "stall on modelscope" in log and "FETCH_DONE rc=0" in log
