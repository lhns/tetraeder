"""The clock of a Case S session (scripts/phase6_clock.py): the stages after the download must fit the 3 h cap of the older box. Hardware: none."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import phase6_clock as pc  # noqa: E402


def test_the_session_fits_three_hours_at_the_padded_estimate_with_a_little_to_spare():
    pad = sum(pc.minutes(r) for r in pc.STAGES if not r[3])
    nominal = sum(r[1] + r[2] for r in pc.STAGES if not r[3])
    assert 165 <= pad <= 175 and 145 <= nominal <= 155, (pad, nominal)


def test_the_check_before_the_download_passes_on_schedule_and_fails_when_the_box_is_late():
    now = 1_000_000.0
    pre = sum(pc.minutes(r) for r in pc.STAGES[: pc.names().index("fetch")] if not r[3])
    ok, msg = pc.check([now - pre * 60, now - (pre - 5) * 60], now, "fetch")
    assert ok and "older box is" in msg, msg
    ok, _ = pc.check([now - (pre + 15) * 60, now], now, "fetch")  # the OLDER box counts, 15 min late
    assert not ok


def test_optional_stages_count_only_from_their_own_start():
    assert pc.remaining("fetch")[0] == sum(pc.minutes(r) for r in pc.STAGES[pc.names().index("fetch"):] if not r[3])
    with_report = pc.remaining("report")[0]
    assert with_report == pc.minutes(pc.STAGES[pc.names().index("report")]) + pc.minutes(pc.STAGES[-1])


def test_the_cli_exit_status_follows_the_check():
    run = lambda age_min: subprocess.run(  # noqa: E731
        [sys.executable, "-I", str(ROOT / "scripts" / "phase6_clock.py"), "check", "--created", "0,100", "--now", str(age_min * 60.0), "--from", "fetch"],
        capture_output=True, text=True, check=False)
    assert run(60).returncode == 0 and run(120).returncode == 1 and "CLOCK SHORT" in run(120).stdout
