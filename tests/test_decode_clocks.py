"""scripts/decode_clocks.py: the clock a GPU runs at over its timed decode tail (ADR 0082)."""

from __future__ import annotations

import datetime
import importlib.util
import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("decode_clocks", ROOT / "scripts" / "decode_clocks.py")
dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dc)

S3E = ROOT / "bench" / "results" / "s3e-20261120"
EPOCH = datetime.datetime(2026, 11, 20, 8, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
HEADER = "timestamp, index, clocks.current.sm [MHz], clocks.current.memory [MHz], power.draw [W], clocks_event_reasons.active\n"


def stamp(t: float) -> str:
    return datetime.datetime.fromtimestamp(EPOCH + t, datetime.timezone.utc).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]


def write_run(d: Path, run: str, rows: list, window: tuple[float, float], reps: int = 5, tg: float = 100.0,
              n_gen: int = 128) -> None:
    """A run's clock csv (`rows`: (t, gpu, sm, reasons)), window file and rates; the tail is reps x n_gen / tg."""
    lines = [HEADER]
    for t, gpu, sm, why in rows:
        lines.append(f"{stamp(t)}, {gpu}, {sm} MHz, 9501 MHz, 150.00 W, 0x{why:016x}\n")
    (d / f"clocks-{run}.csv").write_text("".join(lines))
    (d / f"window-{run}.txt").write_text(f"start={EPOCH + window[0]:.6f} end={EPOCH + window[1]:.6f}\n")
    (d / "anchor-parts").mkdir(exist_ok=True)
    m, rest = run.split("-anchor-")
    plan, lim, k = rest.rsplit("-L", 1)[0], rest.rsplit("-L", 1)[1].split("-r")[0], rest.rsplit("-r", 1)[1]
    (d / "anchor-parts" / f"{m}-{plan}-L{lim}-r{k}.json").write_text(json.dumps([
        {"n_prompt": 2048, "n_gen": 0, "avg_ts": 5000.0, "samples_ts": [5000.0]},
        {"n_prompt": 0, "n_gen": n_gen, "avg_ts": tg, "samples_ts": [tg] * reps}]))


def samples(gpu: int, t0: float, t1: float, sm: float, why: int = 0, step: float = 0.02) -> list:
    n = int(round((t1 - t0) / step))
    return [(t0 + i * step, gpu, sm, why) for i in range(n)]


@pytest.mark.skipif(not (S3E / "qwen3-30b-a3b-single-tg128.clocks.csv").is_file(), reason="s3e not committed")
def test_s3e_part_c_runs_give_the_clocks_of_adr_0082():
    """The 20 ms logs of s3e part C cut to the tail: single boosts on gpu0, layer-4 sits at 1695 on all four GPUs,
    L12 and L24 keep the clocks of the stages that compute (ADR 0082's table)."""
    r = dc.summarize(S3E, "diag")["runs"]
    med = lambda run: {int(g): v["median"] for g, v in r[run]["gpus"].items()}  # noqa: E731
    assert med("qwen3-30b-a3b-single-tg128")[0] == 1905
    assert med("qwen3-30b-a3b-layer-4-tg128") == {0: 1695, 1: 1695, 2: 1695, 3: 1695}
    assert med("qwen3-30b-a3b-layer-4-L12-tg128") == {0: 1890, 1: 1695, 2: 1695, 3: 1695}
    l24 = med("qwen3-30b-a3b-layer-4-L24-tg128")
    assert (l24[0], l24[1], l24[2], l24[3]) == (1965, 1995, 1695, 1695)
    assert med("qwen3-30b-a3b-single-L12-tg128")[0] == 1890 and med("qwen3-30b-a3b-single-L24-tg128")[0] == 1905
    out = dc.summarize(S3E, "diag")
    assert out["low_mhz"] == 1695 and out["busy_mhz"] == {"Qwen3 30B Gptq Fp16": 1905}
    assert out["withheld"] == []


def test_the_tail_is_the_timed_decode_only(tmp_path):
    """reps x n_gen / tg = 5 x 128 / 100 = 6.4 s before the window end; the load before it at another clock is cut."""
    run = "m-anchor-single-L12-r1"
    rows = samples(0, 0.0, 14.0, 1500) + samples(0, 14.0, 20.0, 1900)
    write_run(tmp_path, run, rows, (0.0, 20.0))
    e = dc.summarize(tmp_path)["runs"][run]
    assert e["tail_s"] == pytest.approx(6.4)
    assert e["gpus"]["0"]["median"] == 1900 and e["gpus"]["0"]["samples"] >= 15 and e["withheld"] == []
    # the whole window would read the load
    assert statistics.median(r[2] for r in rows) == 1500


def test_busy_is_the_median_of_the_single_runs_and_low_of_the_layer_anchor_without_the_head(tmp_path):
    for k, sm in ((1, 1905), (2, 1935)):
        for lim in (12, 24):
            write_run(tmp_path, f"m-anchor-single-L{lim}-r{k}", samples(0, 0.0, 10.0, sm), (0.0, 10.0))
    # layer-4@L24: gpus 0 to 2 at 1695, the head's gpu 3 at 1800 (not read)
    rows = [s for g, sm in ((0, 1695), (1, 1695), (2, 1695), (3, 1800)) for s in samples(g, 0.0, 10.0, sm)]
    write_run(tmp_path, "m-anchor-layer-4@L24-L24-r1", rows, (0.0, 10.0))
    out = dc.summarize(tmp_path)
    assert out["busy_mhz"] == {"m": 1920.0} and out["low_mhz"] == 1695 and out["withheld"] == []
    assert out["duty_threshold"] == 0.32


@pytest.mark.parametrize("why,bit", [(0x8, "0x8"), (0x20, "0x20"), (0x40, "0x40"), (0x80, "0x80")])
def test_a_throttle_reason_in_the_tail_withholds_the_run(tmp_path, why, bit):
    rows = samples(0, 0.0, 10.0, 1900) + [(9.5, 0, 1900, why)]
    write_run(tmp_path, "m-anchor-single-L12-r1", rows, (0.0, 10.0))
    out = dc.summarize(tmp_path)
    assert out["busy_mhz"] == {} and any(bit in w for w in out["withheld"])


def test_a_power_cap_alone_is_not_withheld(tmp_path):
    write_run(tmp_path, "m-anchor-single-L12-r1", samples(0, 0.0, 10.0, 1900, why=0x4), (0.0, 10.0))
    assert dc.summarize(tmp_path)["busy_mhz"] == {"m": 1900}


def test_under_15_tail_samples_on_a_gpu_withholds_the_run(tmp_path):
    rows = samples(0, 0.0, 10.0, 1900, step=0.02)[-14:] + samples(1, 0.0, 10.0, 1695)
    write_run(tmp_path, "m-anchor-single-L12-r1", rows, (0.0, 10.0))
    out = dc.summarize(tmp_path)
    assert out["busy_mhz"] == {} and any("under 15" in w for w in out["withheld"])


def test_no_valid_layer_run_leaves_low_clock_null(tmp_path):
    write_run(tmp_path, "m-anchor-single-L12-r1", samples(0, 0.0, 10.0, 1900), (0.0, 10.0))
    out = dc.summarize(tmp_path)
    assert out["low_mhz"] is None and out["busy_mhz"] == {"m": 1900}


def test_busy_clocks_are_keyed_by_the_layout_model(tmp_path):
    """`features_multi` carries `layout.model`, not the file stem; the clocks file must use the same key."""
    (tmp_path / "layouts").mkdir()
    (tmp_path / "layouts" / "m.json").write_text(json.dumps({"model": "M Long Name"}))
    write_run(tmp_path, "m-anchor-single-L12-r1", samples(0, 0.0, 10.0, 1900), (0.0, 10.0))
    assert dc.summarize(tmp_path)["busy_mhz"] == {"M Long Name": 1900}
