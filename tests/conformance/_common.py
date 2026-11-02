"""Shared helpers for the SDK conformance suite (ADR 0040)."""

import os

import pytest

REAL = bool(os.environ.get("TETRAEDER_BASE_URL"))

# Scripted-output assertions need the fake engine; real models get structural checks.
fake_only = pytest.mark.skipif(REAL, reason="needs the scripted fake engine")

WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}
TOOL_CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'

PERSON_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
    "additionalProperties": False,
}
PERSON_JSON = '{"name": "Ada", "age": 36}'


def ask(fake: str, real: str) -> str:
    """User text: the scripted `FAKE:` answer for the fake engine, a natural request for a real model."""
    return real if REAL else "FAKE:" + fake


def echo(text: str) -> str:
    """User text that makes the fake answer `text`; real models just get it as a request to repeat it."""
    return ask(text, f"Repeat exactly this and nothing else: {text}")
