"""The request log of ADR 0046: one JSON line per generation or count_tokens request, no credentials."""

import json
import time

import pytest

FIELDS = {
    "v", "ts_ms", "method", "path", "query", "headers", "status", "stream", "request", "first_byte_ms", "total_ms",
    "generation_id", "usage", "cost", "error_type", "response",
}
SECRET = "sk-secret-conformance-key"


def read_log(path, marker, want=1, wait=5.0):
    """Lines are written when a response body ends, so wait for `want` lines whose request mentions `marker`."""
    deadline = time.time() + wait
    while True:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        mine = [e for e in lines if marker in json.dumps(e["request"])]
        if len(mine) >= want or time.time() > deadline:
            return text, mine
        time.sleep(0.1)


@pytest.fixture
def log(server):
    if server.request_log is None:
        pytest.skip("no request log (set TETRAEDER_REQUEST_LOG for an external server)")
    return server.request_log


def test_log_lines(server, log, oa, an, model):
    oa.chat.completions.create(model=model, messages=[{"role": "user", "content": "FAKE:logmark"}], max_tokens=16)
    list(an.messages.create(model=model, messages=[{"role": "user", "content": "FAKE:logmark too"}], max_tokens=16, stream=True))
    an.messages.count_tokens(model=model, messages=[{"role": "user", "content": "logmark count"}])
    text, lines = read_log(log, "logmark", want=3)
    assert len(lines) >= 3
    for e in lines:
        assert FIELDS <= set(e), FIELDS - set(e)
        assert e["v"] == 1 and e["method"] == "POST" and isinstance(e["ts_ms"], int)
        assert set(e["headers"]) <= {"anthropic-version", "anthropic-beta", "user-agent"}
    by_path = {e["path"]: e for e in lines}
    chat = by_path["/v1/chat/completions"]
    assert chat["status"] == 200 and chat["stream"] is False and chat["request"]["model"] == model
    assert chat["generation_id"] and chat["usage"]["completion_tokens"] > 0
    assert chat["total_ms"] >= chat["first_byte_ms"] >= 0
    assert isinstance(chat["response"], dict)
    sse = by_path["/v1/messages"]
    assert sse["stream"] is True and isinstance(sse["response"], str) and "message_start" in sse["response"]
    assert sse["headers"].get("anthropic-version")
    assert "/v1/messages/count_tokens" in by_path


def test_log_has_no_credentials(server, log, an, oa, model):
    oa.chat.completions.create(model=model, messages=[{"role": "user", "content": "FAKE:credmark"}], max_tokens=8)
    an.messages.create(model=model, messages=[{"role": "user", "content": "FAKE:credmark"}], max_tokens=8)
    text, _ = read_log(log, "credmark", want=2)
    low = text.lower()
    assert SECRET not in text
    assert "x-api-key" not in low and "authorization" not in low


def test_log_records_errors(server, log, oa, model):
    import openai

    with pytest.raises(openai.BadRequestError):
        oa.chat.completions.create(model=model, messages=[])
    deadline = time.time() + 5
    while True:
        bad = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if '"status":400' in x.replace(" ", "")]
        if bad or time.time() > deadline:
            break
        time.sleep(0.1)
    assert bad and bad[-1]["error_type"]
