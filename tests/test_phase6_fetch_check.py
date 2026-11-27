"""The check before a Case S destroy --force (scripts/phase6_fetch_check.py). No network. Hardware label: none."""

import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import phase6_fetch_check as fc


def session(tmp_path):
    out = tmp_path / "s"
    (out / "predictions").mkdir(parents=True)
    (out / "session.json").write_text("{}")
    (out / "predictions" / "pred.json").write_text('{"pp": 1}')
    h = hashlib.sha256(b'{"pp": 1}').hexdigest()
    (out / "predictions" / "SHA256SUMS").write_text(f"{h}  predictions/pred.json\n")
    sums = hashlib.sha256((out / "predictions" / "SHA256SUMS").read_bytes()).hexdigest()
    (out / "predictions" / "FROZEN.sha256").write_text(f"{sums}  SHA256SUMS\n")
    (out / "verdict-s.json").write_text("{}")
    for b in "ab":
        (tmp_path / f"s-n{b}").mkdir()
        (tmp_path / f"s-n{b}" / "DONE").write_text("SC_BOX_DONE rc=0")
    return out


def test_a_fetched_session_passes(tmp_path):
    session(tmp_path)
    assert fc.main(["s", "--root", str(tmp_path)]) == 0


def test_an_unfetched_box_or_a_changed_frozen_file_fails(tmp_path, capsys):
    out = session(tmp_path)
    (tmp_path / "s-nb" / "DONE").unlink()
    assert fc.main(["s", "--root", str(tmp_path)]) == 1
    assert "fetch box b first" in capsys.readouterr().out
    (tmp_path / "s-nb" / "DONE").write_text("x")
    (out / "predictions" / "pred.json").write_text('{"pp": 2}')
    assert fc.main(["s", "--root", str(tmp_path)]) == 1


def test_a_session_stopped_at_the_gates_needs_only_its_gate_files(tmp_path):
    out = tmp_path / "s"
    for b in "ab":
        (out / "gates" / b).mkdir(parents=True)
        (out / "gates" / b / "gates.json").write_text("{}")
    (out / "session.json").write_text("{}")
    assert fc.main(["s", "--root", str(tmp_path), "--stage-reached", "gates"]) == 0
    assert fc.main(["s", "--root", str(tmp_path), "--stage-reached", "setup"]) == 1
