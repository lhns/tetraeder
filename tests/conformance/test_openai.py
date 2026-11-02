"""OpenAI SDK against /v1/chat/completions, /v1/completions and /v1/models (ADR 0028, 0040)."""

import json

import jsonschema
import openai
import pydantic
import pytest
from _common import PERSON_JSON, PERSON_SCHEMA, REAL, TOOL_CALL, WEATHER_SCHEMA, ask, echo, fake_only

TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "description": "Weather for a city", "parameters": WEATHER_SCHEMA},
    }
]
NAMED = {"type": "function", "function": {"name": "get_weather"}}
# Under a real model the call is forced so the check is structural; the fake follows its script.
AUTO = "required" if REAL else "auto"
WEATHER_ASK = ask(TOOL_CALL, "What is the weather in Paris? Call the tool.")
PERSON_ASK = ask(PERSON_JSON, "Give a person named Ada, age 36.")
LONG_ASK = ask("word " * 40, "Write a very long story about a dragon.")


def chat(oa, model, text, **kw):
    kw.setdefault("max_tokens", 256 if REAL else 128)
    return oa.chat.completions.create(model=model, messages=[{"role": "user", "content": text}], **kw)


def check_call(choice):
    assert choice.finish_reason == "tool_calls"
    calls = choice.message.tool_calls
    assert calls and calls[0].type == "function" and calls[0].id
    assert calls[0].function.name == "get_weather"
    args = json.loads(calls[0].function.arguments)
    jsonschema.validate(args, WEATHER_SCHEMA)
    if not REAL:
        assert args == {"city": "Paris"}
    return calls[0]


def finishes(chunks):
    return [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]


def test_plain_text(oa, model):
    r = chat(oa, model, echo("Hello there"))
    c = r.choices[0]
    assert c.finish_reason == "stop" and c.message.role == "assistant"
    assert c.message.content and c.message.content.strip()
    assert r.usage.prompt_tokens > 0 and r.usage.completion_tokens > 0
    assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens
    assert r.id and r.model


@fake_only
def test_scripted_text_and_echo(oa, model):
    assert chat(oa, model, "FAKE:Hello there").choices[0].message.content == "Hello there"
    assert chat(oa, model, "hi").choices[0].message.content == "You said: hi"
    # the text after the LAST marker wins
    assert chat(oa, model, "FAKE:a FAKE:b").choices[0].message.content == "b"


def test_multi_turn(oa, model):
    msgs = [
        {"role": "user", "content": "Say one."},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": echo("second")},
    ]
    r = oa.chat.completions.create(model=model, messages=msgs, max_tokens=64)
    assert r.choices[0].message.content.strip()
    if not REAL:
        assert r.choices[0].message.content == "second"


def test_system_prompt(oa, model):
    msgs = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": echo("ok")}]
    r = oa.chat.completions.create(model=model, messages=msgs, max_tokens=64)
    assert r.choices[0].message.content.strip()


@fake_only
def test_stop_sequence(oa, model):
    r = chat(oa, model, "FAKE:one two STOP three", stop=["STOP"])
    assert r.choices[0].message.content == "one two "
    assert r.choices[0].finish_reason == "stop"


def test_max_tokens_length(oa, model):
    r = chat(oa, model, LONG_ASK, max_tokens=5)
    assert r.choices[0].finish_reason == "length"
    assert r.usage.completion_tokens == 5
    if not REAL:
        assert r.choices[0].message.content == "word "


def test_tool_call_and_result(oa, model):
    r = chat(oa, model, WEATHER_ASK, tools=TOOLS, tool_choice=AUTO)
    call = check_call(r.choices[0])
    # follow-up turn with the tool result (the last user turn scripts the fake's answer)
    msgs = [
        {"role": "user", "content": WEATHER_ASK},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": "get_weather", "arguments": call.function.arguments}}
            ],
        },
        {"role": "tool", "tool_call_id": call.id, "content": '{"temp_c": 18, "sky": "sunny"}'},
        {"role": "user", "content": echo("It is sunny in Paris.")},
    ]
    r2 = oa.chat.completions.create(model=model, messages=msgs, tools=TOOLS, tool_choice="none", max_tokens=128)
    assert r2.choices[0].finish_reason == "stop" and r2.choices[0].message.content.strip()
    if not REAL:
        assert r2.choices[0].message.content == "It is sunny in Paris."
    # a tool result as the last turn is accepted too
    r3 = oa.chat.completions.create(model=model, messages=msgs[:3], tools=TOOLS, tool_choice="none", max_tokens=32)
    assert r3.choices[0].finish_reason in ("stop", "length")


@pytest.mark.parametrize("choice", ["required", NAMED], ids=["required", "named"])
def test_forced_tool(oa, model, choice):
    r = chat(oa, model, WEATHER_ASK, tools=TOOLS, tool_choice=choice, max_tokens=128)
    check_call(r.choices[0])


def test_structured_output(oa, model):
    fmt = {"type": "json_schema", "json_schema": {"name": "person", "strict": True, "schema": PERSON_SCHEMA}}
    r = chat(oa, model, PERSON_ASK, response_format=fmt, max_tokens=300)
    data = json.loads(r.choices[0].message.content)
    jsonschema.validate(data, PERSON_SCHEMA)
    if not REAL:
        assert data == {"name": "Ada", "age": 36}


def test_structured_output_parse(oa, model):
    class Person(pydantic.BaseModel):
        name: str
        age: int

    parse = getattr(oa.chat.completions, "parse", None) or oa.beta.chat.completions.parse
    r = parse(model=model, messages=[{"role": "user", "content": PERSON_ASK}], response_format=Person, max_tokens=300)
    p = r.choices[0].message.parsed
    assert isinstance(p, Person)
    if not REAL:
        assert p == Person(name="Ada", age=36)


def stream(oa, model, text, **kw):
    kw.setdefault("max_tokens", 256 if REAL else 128)
    s = oa.chat.completions.create(
        model=model, messages=[{"role": "user", "content": text}], stream=True,
        stream_options={"include_usage": True}, **kw,
    )
    return list(s)


def test_stream_text(oa, model):
    chunks = stream(oa, model, echo("Hello there"))
    assert chunks[0].choices[0].delta.role == "assistant"
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text.strip()
    assert finishes(chunks) == ["stop"]
    last = chunks[-1]
    assert last.choices == [] and last.usage.completion_tokens > 0 and last.usage.prompt_tokens > 0
    assert len({c.id for c in chunks}) == 1
    if not REAL:
        assert text == "Hello there"


def test_stream_length(oa, model):
    chunks = stream(oa, model, LONG_ASK, max_tokens=4)
    assert finishes(chunks) == ["length"] and chunks[-1].usage.completion_tokens == 4


def test_stream_without_usage_option(oa, model):
    chunks = list(
        oa.chat.completions.create(model=model, messages=[{"role": "user", "content": echo("hi")}], stream=True, max_tokens=32)
    )
    assert all(not c.usage for c in chunks)


def test_stream_tool_call(oa, model):
    chunks = stream(oa, model, WEATHER_ASK, tools=TOOLS, tool_choice=AUTO, max_tokens=128)
    name, args, ids = "", "", set()
    for c in chunks:
        for ch in c.choices:
            for tc in ch.delta.tool_calls or []:
                assert tc.index == 0
                if tc.id:
                    ids.add(tc.id)
                if tc.function and tc.function.name:
                    name += tc.function.name
                if tc.function and tc.function.arguments:
                    args += tc.function.arguments
    assert name == "get_weather" and len(ids) == 1
    jsonschema.validate(json.loads(args), WEATHER_SCHEMA)
    assert finishes(chunks) == ["tool_calls"]


def test_completions_endpoint(oa, model):
    prompt = ask("abc", "Count: one two three")
    r = oa.completions.create(model=model, prompt=prompt, max_tokens=8)
    assert r.choices[0].text and r.choices[0].finish_reason in ("stop", "length")
    chunks = list(
        oa.completions.create(model=model, prompt=prompt, max_tokens=8, stream=True, stream_options={"include_usage": True})
    )
    assert "".join(c.choices[0].text for c in chunks if c.choices)
    assert chunks[-1].usage.completion_tokens > 0


def test_models(oa, model):
    ids = [m.id for m in oa.models.list().data]
    assert model in ids
    m = oa.models.retrieve(model)
    assert m.id == model and m.object == "model"


def test_errors(oa, model):
    with pytest.raises(openai.BadRequestError):
        oa.chat.completions.create(model=model, messages=[])
    with pytest.raises(openai.BadRequestError):
        chat(oa, model, "hi", max_tokens=-3)


@fake_only
def test_unknown_model_not_found(oa):
    # a real server runs with --accept-any-model, which serves any id
    with pytest.raises(openai.NotFoundError):
        oa.models.retrieve("no-such-model")
    with pytest.raises(openai.NotFoundError):
        chat(oa, "no-such-model", "hi")
