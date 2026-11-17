"""scripts/phase4_summary.py: ADR 0065's verdict on synthetic result directories."""

import importlib.util
import json
import pathlib
from datetime import datetime, timedelta

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("phase4_summary", ROOT / "scripts" / "phase4_summary.py")
p4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p4)

IDLE = "0x0000000000000001"
APPS = "0x0000000000000003"
POWER = "0x0000000000000004"
HW_SLOW = "0x0000000000000008"


def bench(tps=None, decode=None, errors=0, completion_tokens=None, **extra):
    s = {"errors": errors}
    if completion_tokens is not None:
        s["completion_tokens"] = completion_tokens
    if tps is not None:
        s["throughput_tps"] = tps
    if decode is not None:
        s["decode_tps_median"] = decode
    return {"summary": s, **extra}


class Session:
    """Writes a phase4 directory; each run is (c1 decode_tps, c4 throughput_tps)."""

    def __init__(self, tmp_path, gpu="NVIDIA A40"):
        self.d = tmp_path / "phase4"
        self.d.mkdir()
        self.write("env.json", {"gpu": gpu})
        self.noise = tmp_path / "noise"
        self.noise.mkdir()
        self.spec = ROOT / "bench" / "spec" / "phase0.toml"

    def write(self, name, obj):
        (self.d / name).write_text(json.dumps(obj))

    def run(self, model, engine, rep, c4, c1, errors=0, throttle=(IDLE, IDLE), **extra):
        self.write(f"bench-{model}-{engine}-c4-r{rep}.json", bench(tps=c4, errors=errors, **extra))
        self.write(f"bench-{model}-{engine}-c1-r{rep}.json", bench(decode=c1, **extra))
        (self.d / f"gpu-{model}-{engine}-r{rep}.txt").write_text(f"before: {throttle[0]}\nafter: {throttle[1]}\n")

    def prepass(self, model, kvu, nokvu):
        self.write(f"prepass-{model}-kvu.json", bench(tps=kvu))
        self.write(f"prepass-{model}-nokvu.json", bench(tps=nokvu))

    def q8b(self, r1=(1.0, 1.0), r2=(1.0, 1.0), base=(100.0, 80.0), engine="tetraeder-dense"):
        """Baseline (c4 tps 100, c1 80) twice; Tetraeder at ratio r1 / r2 = (c4, c1) of baseline."""
        for rep, r in ((1, r1), (2, r2)):
            self.run("q8b", "llama-server", rep, base[0], base[1])
            self.run("q8b", engine, rep, base[0] * r[0], base[1] * r[1])
        self.prepass("q8b", 90.0, 100.0)

    def replay(self, mode, rep, ttft, status=None, errors=None, throttle=(IDLE, IDLE)):
        status = status or [200] * len(ttft)
        results = [
            {"index": i, "status": st, "ttft_s": t if st == 200 else None}
            for i, (t, st) in enumerate(zip(ttft, status))
        ]
        errs = sum(1 for st in status if st != 200) if errors is None else errors
        self.write(
            f"replay-q8b-{mode}-r{rep}.json",
            {"params": {"log_sha256": "ab" * 32, "requests": len(ttft)}, "results": results, "summary": {"errors": errs}},
        )
        (self.d / f"gpu-replay-q8b-{mode}-r{rep}.txt").write_text(f"before: {throttle[0]}\nafter: {throttle[1]}\n")

    def ttft(self, on_ratio=0.5, off_ttft=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)):
        for rep in (1, 2):
            self.replay("off", rep, list(off_ttft))
            self.replay("on", rep, [t * on_ratio for t in off_ttft])

    def verdict(self):
        return p4.verdict(self.d, self.noise, self.spec)


@pytest.fixture
def s(tmp_path):
    return Session(tmp_path)


def test_pass(s):
    s.q8b(r1=(1.05, 1.0), r2=(0.95, 1.02))
    s.ttft()
    v = s.verdict()
    m = v["models"]["q8b"]
    assert v["band"] == 0.08 and v["band_source"].startswith("provisional cuda-default")
    assert m["pass"] is True and m["counts"] is True and m["engine"] == "tetraeder-dense"
    assert m["baseline_variant"] == "nokvu"
    assert m["worst"]["c4"] == pytest.approx(0.95) and m["worst"]["c1"] == pytest.approx(1.0)
    assert m["aa"] == {"c4": 0.0, "c1": 0.0}
    assert [p["clean"] for p in m["pairs"]] == [True, True]
    assert v["ttft"]["pass"] is True and v["ttft"]["log_sha256"] == "ab" * 32 and v["ttft"]["requests"] == 10
    assert v["phase4_pass"] is True


def test_exact_boundary_passes_and_one_pair_below_fails(s):
    s.ttft()
    s.q8b(r1=(1.0, 1.0), r2=(0.92, 1.0))  # r = 1 - b exactly: a tie inside the band passes
    assert s.verdict()["models"]["q8b"]["pass"] is True
    s.q8b(r1=(1.0, 1.0), r2=(0.919, 1.0))  # one pair at 1 - b - 0.001
    v = s.verdict()
    assert v["models"]["q8b"]["pass"] is False and v["models"]["q8b"]["worst"]["c4"] == pytest.approx(0.919)
    assert v["phase4_pass"] is False


def test_c1_gates_too(s):
    s.q8b(r1=(1.0, 0.9), r2=(1.0, 0.9))
    s.ttft()
    assert s.verdict()["models"]["q8b"]["pass"] is False


def test_errors_make_a_pair_unclean_and_no_verdict(s):
    s.q8b()
    s.run("q8b", "tetraeder-dense", 2, 100.0, 80.0, errors=1)
    s.ttft()
    v = s.verdict()
    m = v["models"]["q8b"]
    assert [p["clean"] for p in m["pairs"]] == [True, False]
    assert any("errors=1" in r for r in m["pairs"][1]["reasons"])
    assert m["pass"] is None and v["phase4_pass"] is None  # fewer than two clean pairs: never true


def test_unclean_pair_with_a_fail_is_still_no_verdict(s):
    s.q8b(r1=(0.5, 1.0))
    s.run("q8b", "llama-server", 2, 100.0, 80.0, throttle=(IDLE, HW_SLOW))
    s.ttft()
    assert s.verdict()["models"]["q8b"]["pass"] is None


@pytest.mark.parametrize(
    "throttle,clean",
    [
        ((IDLE, APPS), True),
        ((IDLE, POWER), True),  # SW power cap after a run is benign (ADR 0065 addendum)
        ((POWER, IDLE), False),  # but not before
        ((IDLE, "0x0000000000000024"), False),
        (("0x0000000000000600", IDLE), False),
        (("0x0", "0x8"), False),
    ],
)
def test_throttle_reasons(s, throttle, clean):
    s.q8b()
    s.run("q8b", "tetraeder-dense", 1, 100.0, 80.0, throttle=throttle)
    s.ttft()
    m = s.verdict()["models"]["q8b"]
    assert m["pairs"][0]["clean"] is clean
    assert m["pass"] is (True if clean else None)
    assert m["pairs"][0]["throttle"]["judged"]["before"] == [throttle[0]]
    assert m["pairs"][0]["throttle"]["judged"]["after"] == [throttle[1]]


def test_power_lines_are_recorded_and_ignored_by_the_gate(s):
    s.q8b()
    s.run("q8b", "tetraeder-dense", 1, 100.0, 80.0)
    f = s.d / "gpu-q8b-tetraeder-dense-r1.txt"
    f.write_text(f.read_text() + "power-before: 350.00 W, 30.10 W\npower-after: 350.00 W, 340.00 W\n")
    s.ttft()
    pair = s.verdict()["models"]["q8b"]["pairs"][0]
    assert pair["clean"] is True
    assert pair["throttle"]["judged"]["power_after"] == ["350.00 W, 340.00 W"]


def test_missing_throttle_record_is_unclean(s):
    s.q8b()
    (s.d / "gpu-q8b-tetraeder-dense-r1.txt").unlink()
    assert s.verdict()["models"]["q8b"]["pairs"][0]["clean"] is False


def test_cotenant_false_is_unclean_null_is_fine(s):
    s.q8b()
    s.run("q8b", "tetraeder-dense", 1, 100.0, 80.0, cotenant_ok=None)
    assert s.verdict()["models"]["q8b"]["pairs"][0]["clean"] is True
    s.run("q8b", "tetraeder-dense", 1, 100.0, 80.0, cotenant_ok=False, cotenant_reason="busy")
    assert s.verdict()["models"]["q8b"]["pairs"][0]["clean"] is False


def test_unclean_session_when_baselines_disagree(s):
    s.q8b()
    s.run("q8b", "llama-server", 2, 100.0 * 0.9, 80.0)  # B2 / B1 - 1 = -0.10, outside 0.08
    s.ttft()
    v = s.verdict()
    m = v["models"]["q8b"]
    assert m["aa"]["c4"] == pytest.approx(-0.10) and [p["clean"] for p in m["pairs"]] == [False, False]
    assert m["pass"] is None and v["phase4_pass"] is None
    s.run("q8b", "llama-server", 2, 100.0 * 0.93, 80.0)  # inside the band
    assert s.verdict()["models"]["q8b"]["aa"]["c4"] == pytest.approx(-0.07)


def test_band_from_class_file_needs_three_sessions(s):
    (s.noise / "a40.json").write_text(json.dumps({"class": "a40", "n": 1, "band": 0.02}))
    v = s.verdict()
    assert v["band"] == 0.08 and "measured n=1 < 3" in v["band_source"]
    (s.noise / "a40.json").write_text(json.dumps({"class": "a40", "n": 3, "band": 0.02}))
    v = s.verdict()
    assert v["band"] == 0.02 and v["band_source"] == "measured a40 n=3"
    s.q8b(r1=(0.97, 1.0), r2=(1.0, 1.0))  # 0.97 passes at 0.08, fails at 0.02
    assert s.verdict()["models"]["q8b"]["pass"] is False


def test_class_slug():
    assert p4.class_slug("NVIDIA GeForce RTX 3090") == "rtx-3090"
    assert p4.class_slug("NVIDIA RTX A6000") == "rtx-a6000"
    assert p4.class_slug(None) == "cuda-unknown"


def test_repo_class_files_give_a_band(tmp_path):
    s = Session(tmp_path, gpu="NVIDIA GeForce RTX 3090")
    v = p4.verdict(s.d)  # real bench/noise and spec
    assert v["band_source"].startswith("measured rtx-3090 n=") and 0 < v["band"] < 0.2
    (tmp_path / "a40").mkdir()
    s = Session(tmp_path / "a40", gpu="NVIDIA A40")
    assert p4.verdict(s.d)["band"] == 0.08  # the A40 file holds one session


def test_q30b_on_llama_does_not_count(s):
    s.q8b()
    s.ttft()
    for rep in (1, 2):
        s.run("q30b", "llama-server", rep, 100.0, 100.0)
        s.run("q30b", "tetraeder-llama", rep, 10.0, 10.0)  # would fail badly if it counted
    v = s.verdict()
    q = v["models"]["q30b"]
    assert q["counts"] is False and q["engine"] == "tetraeder-llama" and q["pass"] is False
    assert v["phase4_pass"] is True


def test_q30b_on_moe_counts(s):
    s.q8b()
    s.ttft()
    for rep in (1, 2):
        s.run("q30b", "llama-server", rep, 100.0, 100.0)
        s.run("q30b", "tetraeder-moe", rep, 50.0, 100.0)
        s.run("q30b", "tetraeder-llama", rep, 100.0, 100.0)  # control, ignored
    v = s.verdict()
    assert v["models"]["q30b"]["counts"] is True and v["models"]["q30b"]["engine"] == "tetraeder-moe"
    assert v["phase4_pass"] is False


def test_control_never_replaces_q8b_judged(s):
    s.q8b(engine="tetraeder-llama")  # dense files missing: q8b has no verdict, the control is not it
    s.ttft()
    v = s.verdict()
    assert v["models"]["q8b"]["engine"] == "tetraeder-dense" and v["models"]["q8b"]["pass"] is None
    assert v["phase4_pass"] is None


def test_pick_variant_and_kv_unified(s):
    s.prepass("q8b", 120.0, 100.0)
    assert p4.pick_variant(s.d, "q8b") == "kvu"
    s.prepass("q8b", 90.0, 100.0)
    assert p4.pick_variant(s.d, "q8b") == "nokvu"
    assert p4.pick_variant(s.d, "q30b") is None
    (s.d / "server-prepass-q8b-nokvu.log").write_text("x\nllama_context: kv_unified = 'false'\n")
    assert p4.kv_unified(s.d, "q8b", "nokvu") == "false"
    assert p4.kv_unified(s.d, "q8b", "kvu") is None
    assert p4.main([str(s.d), "--pick-variant", "q8b"]) == 0


def test_ttft_pass_and_fail(s):
    s.q8b()
    s.ttft(on_ratio=0.92)  # exactly 1 - b: passes
    assert s.verdict()["ttft"]["pass"] is True
    s.ttft(on_ratio=0.93)
    assert s.verdict()["ttft"]["pass"] is False


def test_ttft_p90_fail_with_good_median(s):
    s.q8b()
    off = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    on = [t * 0.5 for t in off[:8]] + [9.0, 10.0]
    for rep in (1, 2):
        s.replay("off", rep, off)
        s.replay("on", rep, on)
    t = s.verdict()["ttft"]
    assert t["pairs"][0]["median_ratio"] == pytest.approx(0.5) and t["pairs"][0]["p90_ratio"] == 1.0
    assert t["pass"] is False


def test_ttft_excludes_requests_failing_in_one_run(s):
    s.q8b()
    for rep in (1, 2):
        s.replay("off", rep, [1.0, 2.0, 3.0, 4.0])
        # request 3 fails in the on run only: it is left out of both runs' medians
        s.replay("on", rep, [0.5, 1.0, 1.5, 999.0], status=[200, 200, 200, 500], errors=0)
    t = s.verdict()["ttft"]
    assert t["pairs"][0]["median_ratio"] == pytest.approx(0.5)  # request 1,2,3 -> median(on) 1.0 / median(off) 2.0
    assert t["pass"] is True


def test_ttft_unequal_errors_is_unclean(s):
    s.q8b()
    for rep in (1, 2):
        s.replay("off", rep, [1.0, 2.0])
        s.replay("on", rep, [0.5, 1.0], status=[200, 500], errors=1)
    t = s.verdict()["ttft"]
    assert [p["clean"] for p in t["pairs"]] == [False, False] and t["pass"] is None


def test_ttft_throttled_run_is_unclean(s):
    s.q8b()
    s.ttft()
    s.replay("on", 2, [0.5] * 10, throttle=(IDLE, HW_SLOW))
    t = s.verdict()["ttft"]
    assert t["pairs"][1]["clean"] is False and t["pass"] is None


def test_missing_everything_is_null_not_true(tmp_path):
    v = Session(tmp_path).verdict()
    assert v["phase4_pass"] is None and v["models"]["q8b"]["pass"] is None and v["ttft"]["pass"] is None


def test_main_writes_verdict_json(s):
    s.q8b()
    s.ttft()
    assert p4.main([str(s.d), "--noise-dir", str(s.noise)]) == 0
    v = json.loads((s.d / "verdict.json").read_text())
    assert set(v) == {"band", "band_source", "models", "ttft", "phase4_pass"}
    assert set(v["models"]["q8b"]) >= {
        "engine", "counts", "baseline_variant", "baseline_kv_unified", "aa", "pairs", "worst", "pass"}
    assert v["phase4_pass"] is True


def probe_files(tmp_path, base_c4, own_c4, engine="tetraeder-dense", model="q8b"):
    d = tmp_path / "p"
    d.mkdir(exist_ok=True)
    for name, c4 in (("llama-server", base_c4), (engine, own_c4)):
        (d / f"probe-{model}-{name}-c1.json").write_text(json.dumps(bench(tps=10.0)))
        if c4 is not None:
            (d / f"probe-{model}-{name}-c4.json").write_text(json.dumps(bench(tps=c4)))
    return d


@pytest.mark.parametrize("own,stop", [(86.0, False), (85.0, False), (84.9, True), (50.0, True)])
def test_probe_stop_line_is_085(tmp_path, own, stop):
    d = probe_files(tmp_path, 100.0, own)
    args = [str(d), "--probe", "--probe-run", "q8b:tetraeder-dense"]
    assert p4.main(args) == (3 if stop else 0)
    pr = json.loads((d / "probe.json").read_text())
    assert pr["stop"] is stop and pr["threshold_c4"] == 0.85
    assert pr["models"]["q8b"]["c4_ratio"] == pytest.approx(own / 100.0)


def test_probe_missing_c4_or_failed_parity_stops(tmp_path):
    d = probe_files(tmp_path, 100.0, None)
    assert p4.probe(d, {"q8b": "tetraeder-dense"})["stop"] is True
    d = probe_files(tmp_path, 100.0, 99.0)
    assert p4.probe(d, {"q8b": "tetraeder-dense"})["stop"] is False
    assert p4.probe(d, {"q8b": "tetraeder-dense"}, "failed")["reasons"] == ["q8b tetraeder-dense: gpu-parity failed"]


def test_probe_each_judged_engine_counts(tmp_path):
    d = probe_files(tmp_path, 100.0, 99.0)
    probe_files(tmp_path, 100.0, 60.0, engine="tetraeder-moe", model="q30b")
    pr = p4.probe(d, {"q8b": "tetraeder-dense", "q30b": "tetraeder-moe"})
    # one model below the line stops that model only; the session goes on
    assert pr["stop"] is False and pr["stopped"] == ["q30b"] and len(pr["reasons"]) == 1 and "q30b" in pr["reasons"][0]
    pr = p4.probe(d, {"q8b": "tetraeder-dense", "q30b": "tetraeder-moe"}, {"q8b": "failed"})
    assert pr["stop"] is True and pr["stopped"] == ["q8b", "q30b"]  # every model stopped: the session ends


def test_rerun_pair_replaces_the_unclean_one(s):
    s.q8b(r1=(1.0, 1.0), r2=(1.0, 1.0))
    s.run("q8b", "tetraeder-dense", 2, 100.0, 80.0, errors=1)  # rep 2 unclean
    assert p4.rerun_reps(s.d, "q8b") == [2]
    s.run("q8b", "llama-server", 3, 100.0, 80.0)
    s.run("q8b", "tetraeder-dense", 3, 95.0, 80.0)
    s.ttft()
    m = s.verdict()["models"]["q8b"]
    assert m["reps_used"] == [1, 3] and [p["reps"] for p in m["pairs"]] == [[1], [3]]
    assert m["pass"] is True and m["worst"]["c4"] == pytest.approx(0.95)
    assert [d["rep"] for d in m["dropped_pairs"]] == [2] and any("errors=1" in r for r in m["dropped_pairs"][0]["reasons"])
    assert p4.rerun_reps(s.d, "q8b") == [2]  # unchanged after rep 3: a resumed session finishes the pair


def test_unclean_rep_3_does_not_help(s):
    s.q8b()
    s.run("q8b", "llama-server", 1, 100.0, 80.0, errors=1)
    s.run("q8b", "llama-server", 3, 100.0, 80.0)
    s.run("q8b", "tetraeder-dense", 3, 100.0, 80.0, throttle=(IDLE, HW_SLOW))
    s.ttft()
    m = s.verdict()["models"]["q8b"]
    assert m["reps_used"] == [1, 2] and m["pass"] is None


def test_rerun_reps_needs_exactly_one_unclean(s):
    s.q8b()
    assert p4.rerun_reps(s.d, "q8b") == []
    s.run("q8b", "tetraeder-dense", 1, 100.0, 80.0, errors=1)
    s.run("q8b", "tetraeder-dense", 2, 100.0, 80.0, errors=1)
    assert p4.rerun_reps(s.d, "q8b") == []  # one more pair cannot give two clean ones
    assert p4.main([str(s.d), "--rerun-reps", "q8b"]) == 0


def test_clean_pairs_ignore_extra_rep(s):
    s.q8b()
    s.run("q8b", "llama-server", 3, 100.0, 80.0)
    s.run("q8b", "tetraeder-dense", 3, 10.0, 80.0)  # a bad rep 3 is not used when 1 and 2 are clean
    s.ttft()
    m = s.verdict()["models"]["q8b"]
    assert m["reps_used"] == [1, 2] and m["pass"] is True


def test_completion_token_totals_must_match(s):
    s.q8b()
    s.ttft()
    assert s.verdict()["models"]["q8b"]["pass"] is True
    for rep in (1, 2):  # the same totals in both engines: clean
        for eng in ("llama-server", "tetraeder-dense"):
            s.run("q8b", eng, rep, 100.0, 80.0, completion_tokens=4096)
    assert s.verdict()["models"]["q8b"]["pass"] is True
    s.run("q8b", "tetraeder-dense", 2, 100.0, 80.0, completion_tokens=4000)  # stopped early: not the same work
    m = s.verdict()["models"]["q8b"]
    assert m["pass"] is None and not m["pairs"][1]["clean"]
    assert any("completion tokens differ" in r for r in m["pairs"][1]["reasons"])
    assert p4.rerun_reps(s.d, "q8b") == [2]  # one more pair replaces it


def test_stopped_model_is_null_and_the_other_is_still_judged(s):
    s.q8b()
    s.ttft()
    for rep in (1, 2):
        s.run("q30b", "llama-server", rep, 100.0, 100.0)
        s.run("q30b", "tetraeder-moe", rep, 100.0, 100.0)
    s.write("probe.json", {"models": {"q30b": {"engine": "tetraeder-moe", "stop": True, "reasons": ["c4 ratio 0.800 < 0.85"]},
                                       "q8b": {"engine": "tetraeder-dense", "stop": False, "reasons": []}}})
    v = s.verdict()
    q = v["models"]["q30b"]
    assert q["pass"] is None and q["stopped"] is True and q["counts"] is True and q["pairs"] == []
    assert q["reasons"] == ["c4 ratio 0.800 < 0.85"]
    assert v["models"]["q8b"]["pass"] is True and v["ttft"]["pass"] is True
    assert v["phase4_pass"] is None  # a stopped counting model never lets the rest pass
    s.q8b(r1=(0.8, 1.0), r2=(1.0, 1.0))
    assert s.verdict()["phase4_pass"] is False  # but a failure elsewhere still fails


def test_stopped_30b_on_llama_does_not_count(s):
    s.q8b()
    s.ttft()
    s.write("probe.json", {"models": {"q30b": {"engine": "tetraeder-llama", "stop": True, "reasons": ["x"]}}})
    v = s.verdict()
    assert v["models"]["q30b"]["counts"] is False and v["phase4_pass"] is True


def test_stopped_cli(tmp_path):
    d = probe_files(tmp_path, 100.0, 60.0)
    assert p4.main([str(d), "--probe", "--probe-run", "q8b:tetraeder-dense", "--parity", "q8b:failed"]) == 3
    assert json.loads((d / "probe.json").read_text())["models"]["q8b"]["gpu_parity"] == "failed"
    assert p4.main([str(d), "--stopped"]) == 0


# ---- sampler (gpu-samples.csv, ADR 0065 addendum) -------------------------------------------------------------

T0 = datetime(2026, 11, 17, 10, 0, 0)
BOARD = 0x400
BOARD_BEFORE = "0x0000000000000400"


def stamp_runs(s, *, replay=False):
    """Give run k (q8b: r1 baseline, r1 judged, r2 baseline, r2 judged; replay k=4..7: off1, on1, off2, on2) a window of
    its own: bench c1 ends at t0+10 and c4 at t0+30, 10 s each (the stamp is the end); a replay ends at t0+30 after 30 s."""
    if not replay:
        runs = [("llama-server", 1), ("tetraeder-dense", 1), ("llama-server", 2), ("tetraeder-dense", 2)]
        for k, (eng, rep) in enumerate(runs):
            t0 = T0 + timedelta(seconds=100 * k)
            for test, off in (("c1", 10), ("c4", 30)):
                name = f"bench-q8b-{eng}-{test}-r{rep}.json"
                j = json.loads((s.d / name).read_text())
                j["summary"]["wall_s"] = 10.0
                j["started"] = (t0 + timedelta(seconds=off)).strftime("%Y-%m-%dT%H:%M:%S")
                s.write(name, j)
    else:
        for k, (mode, rep) in enumerate([("off", 1), ("on", 1), ("off", 2), ("on", 2)]):
            t0 = T0 + timedelta(seconds=100 * (k + 4))
            name = f"replay-q8b-{mode}-r{rep}.json"
            j = json.loads((s.d / name).read_text())
            j["wall_s"] = 30.0
            j["finished"] = (t0 + timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%S")
            s.write(name, j)


def write_samples(s, shares, *, first=0, extra=None, gap=None):
    """gpu-samples.csv with 20 samples inside each run k's windows (the first round(share * 20) carry 0x400, the rest
    0x4), an optional `gap` mask at t0+15 (between the c1 and c4 windows), and `extra` {k: mask} OR-ed into one sample."""
    lines = []
    for i, share in enumerate(shares):
        k = first + i
        t0 = T0 + timedelta(seconds=100 * k)
        sec = list(range(1, 11)) + list(range(21, 31))
        for n, off in enumerate(sec):
            m = BOARD if n < round(share * 20) else 0x4
            if n == 5:
                m |= (extra or {}).get(k, 0)
            t = t0 + timedelta(seconds=off)
            lines.append(f"{1.7e9 + 100 * k + off:.3f},{t:%Y/%m/%d %H:%M:%S}.250,{m:#018x},120.5,1410,9001,45")
        if gap is not None:
            t = t0 + timedelta(seconds=15)
            lines.append(f"0.0,{t:%Y/%m/%d %H:%M:%S}.500,{gap:#018x},120.5,1410,9001,45")
    (s.d / "gpu-samples.csv").write_text("\n".join(lines) + "\n")


def board_session(s, shares, **kw):
    """Every run reads 0x400 before it, as in 20261117e."""
    s.q8b()
    for eng in ("llama-server", "tetraeder-dense"):
        for rep in (1, 2):
            (s.d / f"gpu-q8b-{eng}-r{rep}.txt").write_text(f"before: {BOARD_BEFORE}\nafter: {POWER}\n")
    stamp_runs(s)
    write_samples(s, shares, **kw)
    return s.verdict()["models"]["q8b"]


def test_equal_shares_pass_despite_board_bits_before(s):
    m = board_session(s, [0.5, 0.5, 0.5, 0.5])
    assert [p["clean"] for p in m["pairs"]] == [True, True] and m["pass"] is True
    sam = m["pairs"][0]["sampler"]
    assert sam["rule"] == "sampler" and sam["baseline"]["samples"] == 20
    assert sam["baseline"]["share"] == pytest.approx(0.5) and sam["judged"]["disqualifying"] == 0
    # raw masks are kept
    assert m["pairs"][0]["throttle"]["judged"]["before"] == [BOARD_BEFORE]


def test_shares_within_ten_points_pass_and_a_gap_beyond_fails(s):
    assert board_session(s, [0.5, 0.4, 0.5, 0.5])["pass"] is True
    m = board_session(s, [0.3, 0.0, 0.3, 0.3])
    assert m["pairs"][0]["clean"] is False and m["pass"] is None
    assert any("share differs" in r for r in m["pairs"][0]["reasons"])


def test_a_disqualifying_bit_in_a_window_sample_fails(s):
    m = board_session(s, [0.5] * 4, extra={1: 0x8})
    assert m["pairs"][0]["clean"] is False and m["pairs"][1]["clean"] is True and m["pass"] is None
    assert m["pairs"][0]["sampler"]["judged"]["disqualifying"] == 1
    assert m["pairs"][0]["sampler"]["judged"]["bad_bits"] == "0x8"
    # an unknown bit counts too
    assert board_session(s, [0.5] * 4, extra={3: 0x1000})["pairs"][1]["clean"] is False


def test_samples_outside_the_windows_are_ignored(s):
    m = board_session(s, [0.5] * 4, gap=0x8)
    assert m["pass"] is True and m["pairs"][0]["sampler"]["baseline"]["samples"] == 20


def test_before_bits_other_than_board_still_disqualify_with_sampler_data(s):
    s.q8b()
    (s.d / "gpu-q8b-tetraeder-dense-r1.txt").write_text(f"before: {POWER}\nafter: {IDLE}\n")
    stamp_runs(s)
    write_samples(s, [0.0] * 4)
    assert s.verdict()["models"]["q8b"]["pairs"][0]["clean"] is False


def test_no_samples_file_keeps_the_old_rule(s):
    board_session(s, [0.5] * 4)
    (s.d / "gpu-samples.csv").unlink()
    m = s.verdict()["models"]["q8b"]
    assert [p["clean"] for p in m["pairs"]] == [False, False] and m["pass"] is None
    assert m["pairs"][0]["sampler"]["rule"] == "single-sample"
    assert any("0x400" in r for r in m["pairs"][0]["reasons"])


def test_samples_for_one_side_only_keep_the_old_rule(s):
    s.q8b()
    stamp_runs(s)
    write_samples(s, [0.0], first=0)  # only run 0 (rep 1 baseline) has samples
    for eng in ("llama-server", "tetraeder-dense"):
        (s.d / f"gpu-q8b-{eng}-r1.txt").write_text(f"before: {BOARD_BEFORE}\nafter: {IDLE}\n")
    p = s.verdict()["models"]["q8b"]["pairs"][0]
    assert p["sampler"]["rule"] == "single-sample" and p["clean"] is False


def ttft_session(s, shares, **kw):
    s.q8b()
    s.ttft()
    for mode in ("on", "off"):
        for rep in (1, 2):
            (s.d / f"gpu-replay-q8b-{mode}-r{rep}.txt").write_text(f"before: {BOARD_BEFORE}\nafter: {POWER}\n")
    stamp_runs(s, replay=True)
    write_samples(s, shares, first=4, **kw)
    return s.verdict()["ttft"]


def test_ttft_pairs_use_the_sampler_rule(s):
    t = ttft_session(s, [0.5] * 4)
    assert [p["clean"] for p in t["pairs"]] == [True, True] and t["pass"] is True
    assert t["pairs"][0]["sampler"]["rule"] == "sampler"
    assert t["pairs"][0]["sampler"]["on"]["samples"] == 20


def test_ttft_share_gap_and_bad_bit_fail(s):
    t = ttft_session(s, [0.3, 0.0, 0.3, 0.3])  # runs 4..7 = off1, on1, off2, on2
    assert [p["clean"] for p in t["pairs"]] == [False, True] and t["pass"] is None
    t = ttft_session(s, [0.5] * 4, extra={7: 0x20})
    assert [p["clean"] for p in t["pairs"]] == [True, False]


def test_ttft_without_samples_keeps_the_old_rule(s):
    ttft_session(s, [0.5] * 4)
    (s.d / "gpu-samples.csv").unlink()
    t = s.verdict()["ttft"]
    assert [p["clean"] for p in t["pairs"]] == [False, False] and t["pairs"][0]["sampler"]["rule"] == "single-sample"


def test_window_ends_at_started_or_finished():
    end = "2026-11-17T10:00:30"
    a, b = p4.result_window({"started": end, "summary": {"wall_s": 10.0}})
    assert (a, b) == (datetime(2026, 11, 17, 10, 0, 19), datetime(2026, 11, 17, 10, 0, 31))
    assert p4.result_window({"started": "2026-11-17T09:00:00", "finished": end, "wall_s": 10.0}) == (a, b)
    assert p4.result_window({"started": end}) is None
