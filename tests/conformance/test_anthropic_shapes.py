"""Request shapes Claude Code sends, which llama-server accepts (ADR 0049): each must get a 200, not a 400."""

import pytest
from _common import REAL, echo

TOOL = {
    "name": "Bash",
    "description": "Run a shell command",
    "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    "cache_control": {"type": "ephemeral"},
}
DEFERRED = {
    "name": "WebFetch",
    "description": "Fetch a URL",
    "input_schema": {"type": "object", "properties": {"url": {"type": "string"}}},
    "defer_loading": True,
}
PNG_1PX = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def use(tid, name="Bash", inp=None):
    return {"type": "tool_use", "id": tid, "name": name, "input": {"command": "ls"} if inp is None else inp}


def result(tid, content, **kw):
    return {"type": "tool_result", "tool_use_id": tid, "content": content, **kw}


def turn(tid, content, **kw):
    """A tool call by the assistant, its result, then the next user request."""
    return [
        {"role": "user", "content": "Look around."},
        {"role": "assistant", "content": [{"type": "text", "text": "Checking."}, use(tid)]},
        {"role": "user", "content": [result(tid, content, **kw), {"type": "text", "text": echo("done")}]},
    ]


SHAPES = {
    "tool_reference": turn("t1", [{"type": "tool_reference", "tool_name": "WebFetch"}]),
    "tool_result_list_is_error": turn("t2", [{"type": "text", "text": "exit 1"}], is_error=True),
    "tool_result_image": turn(
        "t3", [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_1PX}}]
    ),
    "tool_result_empty": turn("t4", []),
    "empty_input": [
        {"role": "user", "content": "Go."},
        {"role": "assistant", "content": [use("t5", inp={})]},
        {"role": "user", "content": [result("t5", "ok"), {"type": "text", "text": echo("done")}]},
    ],
    "parallel_calls": [
        {"role": "user", "content": "Go."},
        {"role": "assistant", "content": [use("t6"), use("t7", inp={"command": "pwd"})]},
        {"role": "user", "content": [result("t6", "a"), result("t7", "b"), {"type": "text", "text": echo("done")}]},
    ],
    "nested_input": [
        {"role": "user", "content": "Go."},
        {"role": "assistant", "content": [use("t8", inp={"command": "ls", "opts": {"all": True, "dirs": ["a", "b"]}})]},
        {"role": "user", "content": [result("t8", "ok"), {"type": "text", "text": echo("done")}]},
    ],
    "empty_text_block": [{"role": "user", "content": [{"type": "text", "text": ""}, {"type": "text", "text": echo("hi")}]}],
    "cache_control_everywhere": [
        {"role": "user", "content": [{"type": "text", "text": echo("hi"), "cache_control": {"type": "ephemeral"}}]}
    ],
    "thinking_blocks": [
        {"role": "user", "content": "First."},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me think.", "signature": "sig"},
                {"type": "redacted_thinking", "data": "opaque"},
                {"type": "text", "text": "one"},
            ],
        },
        {"role": "user", "content": echo("second")},
    ],
    "image_block": [
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_1PX}},
                {"type": "text", "text": echo("seen")},
            ],
        }
    ],
    "document_block": [
        {
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "Notes."}},
                {"type": "text", "text": echo("read")},
            ],
        }
    ],
    "unknown_block": [
        {"role": "user", "content": [{"type": "server_tool_use_future", "x": 1}, {"type": "text", "text": echo("hi")}]}
    ],
    "mid_conversation_system": [
        {"role": "user", "content": "One."},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": echo("two")},
        {"role": "system", "content": "Be brief."},
    ],
}


@pytest.mark.parametrize("name", sorted(SHAPES))
def test_claude_code_shape(an, model, name):
    m = an.messages.create(
        model=model,
        max_tokens=32,
        system=[{"type": "text", "text": "You are a coding agent.", "cache_control": {"type": "ephemeral"}}],
        tools=[TOOL, DEFERRED],
        messages=SHAPES[name],
    )
    assert m.type == "message" and m.stop_reason in ("end_turn", "max_tokens", "tool_use")


def test_assistant_prefill_continues(an, model):
    m = an.messages.create(
        model=model,
        max_tokens=16,
        messages=[{"role": "user", "content": echo("abc")}, {"role": "assistant", "content": "The answer is"}],
    )
    assert m.type == "message"
    if not REAL:
        # The response continues the prefill; it does not repeat it.
        assert not "".join(b.text for b in m.content if b.type == "text").startswith("The answer is")
