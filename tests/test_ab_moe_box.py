"""scripts/ab_moe_box.sh (DRY_RUN plan: run order, server flags, workload) and scripts/ab_moe_summary.py (the reading rule)."""

import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("ab_moe_summary", ROOT / "scripts" / "ab_moe_summary.py")
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


@pytest.mark.skipif(sys.platform == "win32" or shutil.which("bash") is None, reason="needs bash on Linux")
def test_dry_run_order_and_workload(tmp_path):
    e = {**os.environ, "DRY_RUN": "1", "AB_OUT": str(tmp_path / "out"), "THREADS": "8", "A_COMMIT": "a", "COMMIT": "b"}
    r = subprocess.run(["bash", str(ROOT / "scripts" / "ab_moe_box.sh")], capture_output=True, text=True, env=e, cwd=ROOT, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    order = ["A-r1", "B-r1", "B-r2", "A-r2", "A-r3", "B-r3"]
    assert re.findall(r"run (q\w+-[AB]-r\d) start", out) == [f"{m}-{x}" for m in ("q30b", "q8b") for x in order]
    assert out.count("--engine moe --ngl -1 --parallel 4 --ctx 32768 -t 8") == 6
    assert out.count("--engine dense --ngl -1 --parallel 4 --ctx 32768 -t 8") == 6
    assert out.count("--concurrency 4 --requests 16 --seed 14") == 4 and out.count("--concurrency 1 --requests 8 --seed 21") == 4
    assert all("--prompt-tokens 512 --max-tokens 128" in ln for ln in out.splitlines() if "gateway_bench.py http" in ln)
    # c4 before c1 in each run; the TTFT replays (dense, ctx 40960, A then B) come after the throughput runs
    assert out.index("-c4 ") < out.index("-c1 ")
    assert re.findall(r"run (ttft-[AB]) start", out) == ["ttft-A", "ttft-B"]
    assert out.count("--engine dense --ngl -1 --parallel 4 --ctx 40960 -t 8") == 2
    assert out.index("run q8b-B-r3 start") < out.index("run ttft-A start")


T0 = datetime(2026, 11, 22, 12, 0, 0)


def write_run(d, side, rnd, c4, c1, errors=0, mask=0x4, start=0, prefix=""):
    """Two bench files whose windows are 10 s each, plus one sampler line per second."""
    for i, (test, field, v) in enumerate((("c4", "throughput_tps", c4), ("c1", "decode_tps_median", c1))):
        end = T0 + timedelta(seconds=start + 10 * (i + 1))
        (d / (f"ab-{prefix}{side}-r{rnd}-{test}.json" if prefix else f"ab-{side}-{test}-r{rnd}.json")).write_text(json.dumps(
            {"finished": end.strftime("%Y-%m-%dT%H:%M:%S"), "summary": {"errors": errors if test == "c4" else 0, "wall_s": 9.0, field: v}}))
    with (d / "gpu-samples.csv").open("a") as f:
        for s in range(start, start + 21):
            t = (T0 + timedelta(seconds=s)).strftime("%Y/%m/%d %H:%M:%S.000")
            f.write(f"0,{t},{mask:#x},300,1,1,50,0\n")


def make(d, ratios, bad=None):
    """Round n: A at 100 / 100, B at ratio * 100; `bad` maps (side, round) to write_run kwargs."""
    for n, (rc4, rc1) in enumerate(ratios, 1):
        for side in "AB":
            k = (bad or {}).get((side, n), {})
            f = 1.0 if side == "A" else None
            write_run(d, side, n, 100 * (f or rc4), 100 * (f or rc1), start=(2 * n + (side == "B")) * 30, **k)


def test_no_regression(tmp_path):
    make(tmp_path, [(0.99, 1.0), (1.0, 0.99), (0.98, 1.01)])
    s = ab.summarize(tmp_path)
    assert s["reading"] == "no regression evidence" and s["clean_rounds"] == [1, 2, 3]
    assert s["median_ratio"]["c4"] == pytest.approx(0.99)


def test_regression_needs_every_round_below_one(tmp_path):
    make(tmp_path, [(0.95, 0.99), (0.96, 1.0), (0.94, 0.98)])
    assert ab.summarize(tmp_path)["reading"] == "regression"
    d = tmp_path / "b"
    d.mkdir()
    make(d, [(0.95, 0.99), (1.01, 1.0), (0.94, 0.98)])
    assert ab.summarize(d)["reading"] == "inconclusive"


def test_unclean_round_is_excluded_and_named(tmp_path):
    make(tmp_path, [(0.99, 1.0), (0.5, 0.5), (0.98, 1.01)], {("B", 2): {"mask": 0x20}, ("A", 3): {"errors": 1}})
    s = ab.summarize(tmp_path)
    assert s["clean_rounds"] == [1] and s["excluded_rounds"] == [2, 3]
    assert s["reading"] == "inconclusive"
    assert any("0x20" in r for r in s["rounds"][1]["why_unclean"]) and any("errors=1" in r for r in s["rounds"][2]["why_unclean"])


def test_model_prefix_files(tmp_path):
    for n in (1, 2):
        for side in "AB":
            write_run(tmp_path, side, n, 100, 100, start=(2 * n + (side == "B")) * 30, prefix="q8b-")
    s = ab.summarize(tmp_path, 2, "q8b-")
    assert s["clean_rounds"] == [1, 2] and s["median_ratio"]["c4"] == pytest.approx(1.0)
    assert ab.summarize(tmp_path, 2)["clean_rounds"] == []


def write_ttft(d, side, ttfts, errors=0):
    results = [{"index": i, "status": 200, "ttft_s": t} for i, t in enumerate(ttfts)]
    (d / f"ab-ttft-{side}.json").write_text(json.dumps({"results": results, "summary": {"errors": errors}}))


def test_ttft_ratio_on_common_requests(tmp_path):
    base = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0]
    write_ttft(tmp_path, "A", base)
    write_ttft(tmp_path, "B", [t * 1.04 for t in base])
    t = ab.ttft_summary(tmp_path)
    assert t["reading"] == "no regression" and t["ratio_median"] == pytest.approx(1.04) and t["ratio_p90"] == pytest.approx(1.04)
    write_ttft(tmp_path, "B", [t * 1.2 for t in base])
    assert ab.ttft_summary(tmp_path)["reading"] == "report"
    write_ttft(tmp_path, "B", base, errors=1)
    assert ab.ttft_summary(tmp_path)["reading"] == "report"
    assert ab.ttft_summary(tmp_path / "none") is None


def test_reading_per_metric(tmp_path):
    # c4 regresses on every round, c1 is level: the combined reading is a regression, the metric readings say where
    make(tmp_path, [(0.95, 1.0), (0.96, 1.0), (0.94, 0.99)])
    s = ab.summarize(tmp_path)
    assert s["metric_readings"] == {"c4": "regression", "c1": "no regression evidence"}
