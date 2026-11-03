"""Pure parts of scripts/gateway_bench.py and scripts/gateway_replay.py (no network)."""

import importlib.util
import json
import pathlib
import random

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = load("gateway_bench")
replay = load("gateway_replay")


def test_prompts_unique_unless_shared_prefix():
    rng = random.Random(1)
    a = bench.make_prompt(512, bench.new_nonce(rng))
    b = bench.make_prompt(512, bench.new_nonce(rng))
    assert a != b and a[:20] != b[:20]
    s1 = bench.make_prompt(512, "x", shared_prefix=True)
    s2 = bench.make_prompt(512, "y", shared_prefix=True)
    assert s1 == s2
    assert abs(len(a) - 512 * bench.CHARS_PER_TOKEN) <= 1


def test_percentile_and_decode_rate():
    assert bench.percentile([], 90) is None
    assert bench.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90) == 9
    assert bench.percentile([5], 90) == 5
    assert bench.decode_rate(11, 1.0, 2.0) == 10.0
    assert bench.decode_rate(1, 1.0, 2.0) is None
    assert bench.decode_rate(5, 1.0, 1.0) is None


def test_aggregate():
    results = [
        {"ttft_s": 0.1, "completion_tokens": 10, "decode_tps": 50.0},
        {"ttft_s": 0.3, "completion_tokens": 20, "decode_tps": 30.0},
        {"error": "HTTP 500"},
    ]
    a = bench.aggregate(results, 2.0)
    assert a["requests"] == 3 and a["errors"] == 1
    assert a["ttft_median_s"] == 0.2
    assert a["decode_tps_median"] == 40.0
    assert a["throughput_tps"] == 15.0


def test_sse_parsing():
    assert bench.parse_sse_line("data: [DONE]") is None
    assert bench.parse_sse_line(": keepalive") is None
    chunk = bench.parse_sse_line('data: {"choices":[{"delta":{"content":"hi"}}]}')
    assert bench.delta_text(chunk) == "hi"
    assert bench.delta_text({"choices": [{"delta": {"role": "assistant", "content": ""}}]}) == ""
    assert bench.delta_text({"choices": [{"text": "abc"}]}) == "abc"
    assert bench.delta_text({"choices": [], "usage": {"completion_tokens": 3}}) == ""


LOG = [
    {"v": 1, "ts_ms": 1000, "method": "POST", "path": "/v1/messages", "query": "beta=true", "headers": {},
     "status": 200, "stream": True, "request": {"model": "m", "stream": True}},
    {"v": 1, "ts_ms": 3000, "method": "POST", "path": "/v1/chat/completions", "query": "", "status": 200,
     "stream": False, "request": {"model": "m"}},
    {"v": 1, "ts_ms": 500, "method": "POST", "path": "/v1/messages/count_tokens", "status": 200, "request": {}},
]


def test_parse_log_orders_and_skips_bad_lines():
    lines = [json.dumps(e) for e in LOG] + ["", "not json", json.dumps({"v": 1, "path": "/x"})]
    entries = replay.parse_log(lines)
    assert [e["ts_ms"] for e in entries] == [500, 1000, 3000]


def test_schedule_and_url():
    entries = replay.parse_log(json.dumps(e) for e in LOG)
    assert replay.schedule(entries, 1) == [0.0, 0.5, 2.5]
    assert replay.schedule(entries, 5) == [0.0, 0.1, 0.5]
    assert replay.schedule(entries, 0) == [0.0, 0.0, 0.0]
    assert replay.schedule([], 1) == []
    assert replay.target_url("http://h:1/", entries[1]) == "http://h:1/v1/messages?beta=true"
    assert replay.target_url("http://h:1", entries[2]) == "http://h:1/v1/chat/completions"


def test_replay_sse_helpers():
    assert replay.sse_has_content('data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"a"}}')
    assert not replay.sse_has_content('data: {"type":"message_start","message":{}}')
    assert replay.sse_has_content('data: {"choices":[{"delta":{"content":"x"}}]}')
    assert not replay.sse_has_content('data: {"choices":[{"delta":{"role":"assistant"}}]}')
    assert replay.usage_from({"message": {"usage": {"input_tokens": 3}}}) == {"input_tokens": 3}
    s = replay.summarize([
        {"status": 200, "ttft_s": 0.1, "total_s": 1.0, "logged_status": 200},
        {"status": 500, "total_s": 0.1, "logged_status": 200},
    ])
    assert s["ok"] == 1 and s["errors"] == 1 and s["status_mismatch"] == 1


def test_summary_paired_ratios(tmp_path):
    summ = load("gateway_summary")

    def write(engine, rep, tps, ttft):
        s = {"errors": 0, "decode_tps_median": tps, "ttft_median_s": ttft, "throughput_tps": tps, "prompt_tokens_median": 512}
        (tmp_path / f"bench-q8b-{engine}-c1-r{rep}.json").write_text(json.dumps({"summary": s}))

    for rep, (a, b) in enumerate([(90, 100), (80, 100), (95, 100)], start=1):
        write("tetraeder", rep, a, 0.2)
        write("llama-server", rep, b, 0.1)
    (tmp_path / "overhead-q8b-llama-bench.json").write_text(
        json.dumps([{"n_prompt": 512, "n_gen": 0, "avg_ts": 5000}, {"n_prompt": 0, "n_gen": 128, "avg_ts": 125}])
    )
    out = summ.summarize(tmp_path)
    m = out["models"]["q8b"]
    r = m["ratios_vs_llama-server"]["tetraeder"]["c1"]
    assert r["decode_tps_median"] == 0.9 and r["ttft_median_s"] == 2.0 and r["pairs"] == 3
    assert m["engines"]["llama-server"]["c1"]["decode_tps_median"] == 100
    assert m["api_overhead"]["llama-server"]["decode_ratio"] == 0.8
    assert abs(m["api_overhead"]["llama-server"]["prefill_ratio"] - (512 / 0.1) / 5000) < 1e-9
