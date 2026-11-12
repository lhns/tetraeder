"""scripts/poll_bench_summary.py: medians, paired ratios and the throttled share from fake runs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("poll_bench_summary", ROOT / "scripts" / "poll_bench_summary.py")
pbs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pbs)


def add_run(d: Path, n: int, rnd: int, arm: str, quota: bool, tok: float, throttled: int, vol: int) -> None:
    i = f"{n:02d}"
    with (d / "manifest.jsonl").open("a") as f:
        f.write(json.dumps({"id": i, "round": rnd, "arm": arm, "quota": quota}) + "\n")
    h = {"runs": [{"decode_tok_s": tok, "decode_profile_per_token": {"compute_ms": 1000 / tok}}]}
    (d / f"harness-{i}.json").write_text(json.dumps(h))
    (d / f"cg-{i}.txt").write_text(
        "cpu.max 200000 100000\n"
        "pre nr_periods 10\npre nr_throttled 0\n"
        f"post nr_periods 110\npost nr_throttled {throttled}\n"
    )
    (d / f"time-{i}.txt").write_text(f"{vol} 5 10.0\n")


def test_summary(tmp_path: Path) -> None:
    n = 0
    for rnd, (a, b, c) in enumerate([(2.0, 3.0, 2.5), (4.0, 4.0, 5.0), (3.0, 6.0, 3.0)], start=1):
        for arm, tok, thr in (("poll50", a, 80), ("poll0", b, 5), ("pause", c, 40)):
            add_run(tmp_path, n, rnd, arm, True, tok, thr, 1000 if arm == "poll0" else 100)
            n += 1
    for arm in ("poll50", "poll0", "pause"):
        add_run(tmp_path, n, 0, arm, False, 5.0, 0, 50)
        n += 1
    runs = [pbs.load_run(tmp_path, m) for m in map(json.loads, (tmp_path / "manifest.jsonl").read_text().splitlines())]
    s = pbs.summarize(runs)
    q = s["quota"]
    assert q["arms"]["poll50"]["decode_tok_s_median"] == 3.0
    assert q["arms"]["poll50"]["throttled_share_median"] == 0.8
    assert q["arms"]["poll0"]["throttled_share_median"] == 0.05
    assert q["arms"]["poll0"]["ctxsw_voluntary_per_s_median"] == 100.0
    r = q["paired_ratio_vs_poll50"]["poll0"]
    assert r["n"] == 3 and r["median"] == 1.5 and r["min"] == 1.0 and r["max"] == 2.0 and r["rounds_above_1"] == 2
    assert s["no_quota"]["arms"]["pause"]["n"] == 1
    assert s["no_quota"]["paired_ratio_vs_poll50"]["pause"]["median"] == 1.0
    assert "poll0 / poll50" in pbs.render(s)
