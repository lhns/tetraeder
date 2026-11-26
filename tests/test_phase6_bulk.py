"""The bulk screen of the Case S pair (scripts/phase6_bulk.py, ADR 0091 History 2026-11-26): its constant comes from the committed prices of the
covering pick, rehearsal 1's link passes and rehearsal 2's fails, and a real transfer over loopback TCP runs. Hardware label: none."""

import json
import pathlib
import subprocess
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import phase6_bulk as pb  # noqa: E402
import phase6_exit_s_summary as ss  # noqa: E402

R1 = "bench/results/phase6-rehearsal-20261125/a/"
R2 = "crates/planner/fixtures/phase6-rehearsal/link-probes/probe-%s.json"


def hops(path):
    r = ss.link_readings(ss.load(str(ROOT / path)))
    return r["a->b 33554432"], r["b->a 33554432"]


def test_the_constant_is_the_one_both_committed_prices_give():
    d = pb.derive()
    assert abs(d["c_r1_s"] - pb.C_S) < 0.01 and abs(d["c_r2_s"] - pb.C_S) < 0.01
    assert abs(d["slope"] - 1.0) < 0.002  # the prediction adds the two hops one for one


def test_the_closed_form_reproduces_the_prices_of_the_planner_test():
    # pp moves 0.2433 on rehearsal 1 as probed and 0.428 on rehearsal 2's mid-1 probe (cluster_discovered_ram_experts.rs)
    pp = lambda p: 2048 / (pb.C_S + sum(hops(p)))  # noqa: E731
    assert abs(pp(R1 + "probe-explicit-16777216.json") / pp(R1 + "probe-auto.json") - 1 + 0.2433) < 0.001
    assert abs(pp(R2 % "mid-1") / pp(R2 % "start") - 1 - 0.428) < 0.001
    assert abs(pp(R1 + "probe-auto.json") - 196.0) < 0.1 and abs(pp(R2 % "start") - 81.4) < 0.1


def runs_of(path, spread):
    """64 MiB transfers a hop of `path` apart: 2 x the 32 MiB hop each way, the slow repeat `spread` times the fast one."""
    ab, ba = hops(path)
    return {"mib": 64, "a->b": [2 * ab, 2 * ab * spread, 2 * ab * (1 + (spread - 1) / 2)], "b->a": [2 * ba, 2 * ba * spread]}


def test_rehearsal_1s_link_passes_at_its_own_4_6_percent_spread_and_a_3_5_times_spread_of_rehearsal_2_fails():
    ok = pb.evaluate(runs_of(R1 + "probe-auto.json", 1.047))
    assert ok["pass"] and ok["move"] < 0.01, ok
    # 64 MiB in about 1.3 s: both directions may spread by about 1.6 times before the prediction moves 0.08, so the screen is about drift
    assert pb.evaluate(runs_of(R1 + "probe-auto.json", 1.5))["pass"]
    assert not pb.evaluate(runs_of(R1 + "probe-auto.json", 2.0))["pass"]
    assert not pb.evaluate(runs_of(R1 + "probe-auto.json", 3.5))["pass"]
    bad = pb.evaluate(runs_of(R2 % "mid-1", 3.5))
    assert not bad["pass"] and bad["move"] > 0.08 and "move" in bad["why"][0]


def test_too_few_repeats_fail():
    assert not pb.evaluate({"mib": 64, "a->b": [1.3], "b->a": [1.2, 1.2]})["pass"]


def test_a_real_transfer_over_loopback_tcp_is_timed_and_acked(tmp_path):
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    t = threading.Thread(target=pb.serve, args=(port, 2, 20.0), daemon=True)
    t.start()
    import time

    time.sleep(0.3)
    outs = [subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "phase6_bulk.py"), "send", "--addr", f"127.0.0.1:{port}", "--mib", "4"],
                           capture_output=True, text=True, timeout=60) for _ in range(2)]
    t.join(20)
    secs = [json.loads(o.stdout)["seconds"] for o in outs]
    assert all(o.returncode == 0 for o in outs) and all(0 < x < 10 for x in secs), [o.stderr for o in outs]
    runs = tmp_path / "runs.json"
    runs.write_text(json.dumps({"mib": 4, "a->b": secs, "b->a": secs}))
    r = subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "phase6_bulk.py"), "eval", str(runs), "--out", str(tmp_path / "bulk.json")],
                       capture_output=True, text=True)
    assert "bulk screen" in r.stdout and (tmp_path / "bulk.json").exists()
