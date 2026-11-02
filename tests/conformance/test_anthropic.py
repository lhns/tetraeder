"""Anthropic SDK against /v1/messages, /v1/messages/count_tokens and /v1/models (ADR 0028, 0040)."""

import json

import anthropic
import jsonschema
import pytest
from _common import PERSON_JSON, PERSON_SCHEMA, REAL, TOOL_CALL, WEATHER_SCHEMA, ask, echo, fake_only

TOOLS = [{"name": "get_weather", "description": "Weather for a city", "input_schema": WEATHER_SCHEMA}]
ANY = {"type": "any"}
NAMED = {"type": "tool", "name": "get_weather"}
AUTO = ANY if REAL else {"type": "auto"}
WEATHER_ASK = ask(TOOL_CALL, "What is the weather in Paris? Call the tool.")
PERSON_ASK = ask(PERSON_JSON, "Give a person named Ada, age 36.")
LONG_ASK = ask("word " * 40, "Write a very long story about a dragon.")
MAX = 256 if REAL else 128


def msg(an, model, text, **kw):
    kw.setdefault("max_tokens", MAX)
    return an.messages.create(model=model, messages=[{"role": "user", "content": text}], **kw)


def text_of(m):
    return "".join(b.text for b in m.content if b.type == "text")


def check_use(m):
    assert m.stop_reason == "tool_use"
    uses = [b for b in m.content if b.type == "tool_use"]
    assert uses and uses[0].id and uses[0].name == "get_weather"
    jsonschema.validate(uses[0].input, WEATHER_SCHEMA)
    if not REAL:
        assert uses[0].input == {"city": "Paris"}
    return uses[0]


def test_plain_text(an, model):
    m = msg(an, model, echo("Hello there"))
    assert m.type == "message" and m.role == "assistant" and m.id and m.model
    assert m.stop_reason == "end_turn" and text_of(m).strip()
    assert m.usage.input_tokens > 0 and m.usage.output_tokens > 0
    if not REAL:
        assert text_of(m) == "Hello there"


@fake_only
def test_echo_without_marker(an, model):
    assert text_of(msg(an, model, "hi")) == "You said: hi"


def test_multi_turn(an, model):
    msgs = [
        {"role": "user", "content": "Say one."},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": [{"type": "text", "text": echo("second")}]},
    ]
    m = an.messages.create(model=model, messages=msgs, max_tokens=64)
    assert text_of(m).strip()
    if not REAL:
        assert text_of(m) == "second"


def test_system_string_and_blocks(an, model):
    m = msg(an, model, echo("ok"), system="You are terse.")
    assert text_of(m).strip()
    blocks = [
        {"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Answer in English."},
    ]
    m = msg(an, model, echo("ok"), system=blocks)
    assert text_of(m).strip()


@fake_only
def test_stop_sequence(an, model):
    m = msg(an, model, "FAKE:one two STOP three", stop_sequences=["STOP"])
    assert text_of(m) == "one two "
    assert m.stop_reason == "stop_sequence" and m.stop_sequence == "STOP"


def test_max_tokens(an, model):
    m = msg(an, model, LONG_ASK, max_tokens=5)
    assert m.stop_reason == "max_tokens" and m.usage.output_tokens == 5
    if not REAL:
        assert text_of(m) == "word "


def test_tool_use_and_result(an, model):
    m = msg(an, model, WEATHER_ASK, tools=TOOLS, tool_choice=AUTO)
    use = check_use(m)
    msgs = [
        {"role": "user", "content": WEATHER_ASK},
        {"role": "assistant", "content": [{"type": "tool_use", "id": use.id, "name": use.name, "input": use.input}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use.id, "content": "18C and sunny"}]},
    ]
    # a tool result as the last turn is accepted
    m2 = an.messages.create(model=model, messages=msgs, tools=TOOLS, tool_choice={"type": "none"}, max_tokens=32)
    assert m2.stop_reason in ("end_turn", "max_tokens")
    # tool result plus a scripted user turn
    msgs[-1]["content"].append({"type": "text", "text": echo("It is sunny in Paris.")})
    m3 = an.messages.create(model=model, messages=msgs, tools=TOOLS, tool_choice={"type": "none"}, max_tokens=128)
    assert m3.stop_reason == "end_turn" and text_of(m3).strip()
    if not REAL:
        assert text_of(m3) == "It is sunny in Paris."


@pytest.mark.parametrize("choice", [ANY, NAMED], ids=["any", "named"])
def test_forced_tool(an, model, choice):
    check_use(msg(an, model, WEATHER_ASK, tools=TOOLS, tool_choice=choice, max_tokens=128))


def output_config(schema):
    return {"format": {"type": "json_schema", "schema": schema}}


def test_structured_output(an, model):
    m = msg(an, model, PERSON_ASK, output_config=output_config(PERSON_SCHEMA), max_tokens=300)
    data = json.loads(text_of(m))
    jsonschema.validate(data, PERSON_SCHEMA)
    if not REAL:
        assert data == {"name": "Ada", "age": 36}


def test_thinking_adaptive(an, model):
    m = msg(an, model, echo("ok"), thinking={"type": "adaptive"}, max_tokens=512 if REAL else 128)
    assert m.content and m.stop_reason in ("end_turn", "max_tokens")
    assert text_of(m).strip() or any(b.type == "thinking" for b in m.content)


def test_raw_stream_events(an, model):
    events = list(an.messages.create(model=model, messages=[{"role": "user", "content": echo("Hello there")}],
                                     max_tokens=MAX, stream=True))
    types = [e.type for e in events]
    assert types[0] == "message_start" and types[-1] == "message_stop" and types[-2] == "message_delta"
    assert events[0].message.role == "assistant" and events[0].message.usage.input_tokens > 0
    # blocks open, get deltas, close, in order
    open_idx = None
    for e in events[1:-2]:
        if e.type == "content_block_start":
            assert open_idx is None
            open_idx = e.index
        elif e.type == "content_block_delta":
            assert e.index == open_idx
        elif e.type == "content_block_stop":
            assert e.index == open_idx
            open_idx = None
        elif e.type != "ping":
            pytest.fail(f"unexpected event {e.type}")
    assert open_idx is None and "content_block_start" in types
    last = events[-2]
    assert last.delta.stop_reason == "end_turn" and last.usage.output_tokens > 0
    if not REAL:
        assert "".join(e.delta.text for e in events if e.type == "content_block_delta") == "Hello there"


def test_stream_helper(an, model):
    with an.messages.stream(model=model, messages=[{"role": "user", "content": echo("Hello there")}], max_tokens=MAX) as s:
        streamed = "".join(s.text_stream)
        final = s.get_final_message()
    assert streamed.strip() and text_of(final) == streamed
    assert final.stop_reason == "end_turn" and final.usage.output_tokens > 0


def test_stream_max_tokens(an, model):
    with an.messages.stream(model=model, messages=[{"role": "user", "content": LONG_ASK}], max_tokens=4) as s:
        final = s.get_final_message()
    assert final.stop_reason == "max_tokens" and final.usage.output_tokens == 4


def test_stream_tool_use(an, model):
    with an.messages.stream(model=model, messages=[{"role": "user", "content": WEATHER_ASK}], tools=TOOLS,
                            tool_choice=AUTO, max_tokens=128) as s:
        final = s.get_final_message()
    check_use(final)


def test_stream_structured_output(an, model):
    with an.messages.stream(model=model, messages=[{"role": "user", "content": PERSON_ASK}],
                            output_config=output_config(PERSON_SCHEMA), max_tokens=300) as s:
        final = s.get_final_message()
    jsonschema.validate(json.loads(text_of(final)), PERSON_SCHEMA)


def test_count_tokens(an, model):
    a = an.messages.count_tokens(model=model, messages=[{"role": "user", "content": "Hello"}])
    b = an.messages.count_tokens(model=model, messages=[{"role": "user", "content": "Hello " * 50}], system="Be brief.",
                                 tools=TOOLS)
    assert a.input_tokens > 0 and b.input_tokens > a.input_tokens


def test_models(an, model):
    ids = [m.id for m in an.models.list().data]
    assert model in ids
    m = an.models.retrieve(model)
    assert m.id == model and m.type == "model"


def test_errors(an, model):
    with pytest.raises(anthropic.BadRequestError):
        an.messages.create(model=model, messages=[], max_tokens=16)
    with pytest.raises(anthropic.BadRequestError):
        an.messages.create(model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=0)


@fake_only
def test_unknown_model_not_found(an):
    # a real server runs with --accept-any-model, which serves any id
    with pytest.raises(anthropic.NotFoundError):
        an.models.retrieve("no-such-model")
    with pytest.raises(anthropic.NotFoundError):
        msg(an, "no-such-model", "hi")


def test_mid_conversation_system_message(an, model):
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "system", "content": "Reminder: be terse."},
        {"role": "user", "content": echo("ok")},
    ]
    m = an.messages.create(model=model, messages=msgs, max_tokens=64)
    assert text_of(m).strip()
    if not REAL:
        assert text_of(m) == "ok"


def claude_code_body(model, text, stream=False):
    tool = {
        "name": "get_weather",
        "description": "Weather for a city. " * 20,
        "input_schema": WEATHER_SCHEMA,
        "cache_control": {"type": "ephemeral"},
    }
    return {
        "model": model,
        "max_tokens": 32000,
        "stream": stream,
        "system": [
            {"type": "text", "text": "You are Claude Code, an agent. " * 30},
            {"type": "text", "text": "Project rules. " * 200, "cache_control": {"type": "ephemeral"}},
        ],
        "tools": [tool],
        "metadata": {"user_id": "user_abc_account__session_123"},
        "messages": [{"role": "user", "content": [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]}],
        "thinking": {"type": "adaptive"},
    }


@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_claude_code_shaped_request(server, stream):
    import httpx

    headers = {
        "x-api-key": "sk-secret-conformance-key",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "claude-code-20250219,interleaved-thinking-2025-05-14,fine-grained-tool-streaming-2025-05-14",
        "user-agent": "claude-cli/2.0.0 (external, cli)",
        "x-app": "cli",
    }
    body = claude_code_body(server.model, echo("ok"), stream)
    r = httpx.post(server.base_url + "/v1/messages?beta=true", json=body, headers=headers, timeout=600 if REAL else 30)
    assert r.status_code == 200, r.text
    if stream:
        assert "event: message_start" in r.text and "event: message_stop" in r.text
    else:
        assert r.json()["type"] == "message"
