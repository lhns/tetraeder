"""scripts/engine_sample.py probe-disk: a failed read or a device that did not move gives "none", not a guess."""

import importlib.util
import pathlib
import subprocess
import sys
import types

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("engine_sample", SCRIPTS / "engine_sample.py")
es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(es)


def probe(monkeypatch, capsys, before, after, rc):
    snaps = iter([before, after])
    monkeypatch.setattr(es.ds, "read", lambda path: next(snaps))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=rc))
    es.probe_disk(types.SimpleNamespace(model="m.gguf"))
    return capsys.readouterr().out.strip()


def stats(sectors):
    return "\n".join(f"259 {i} nvme{i} 1 0 {s} 0" for i, s in enumerate(sectors))


def test_probe_picks_the_device_that_read_the_model(monkeypatch, capsys):
    two_gb = 2_000_000_000 // 512
    assert probe(monkeypatch, capsys, stats([0, 0]), stats([10, two_gb]), 0) == "259:1"


def test_probe_without_a_moving_device_or_with_a_failed_read_is_none(monkeypatch, capsys):
    assert probe(monkeypatch, capsys, stats([0, 0]), stats([0, 0]), 0) == "none"
    assert probe(monkeypatch, capsys, stats([0, 0]), stats([0, 4_000_000]), 1) == "none"
