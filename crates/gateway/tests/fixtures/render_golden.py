"""Renders the fixture templates with Python jinja2 the way transformers does.

Usage: uv run --no-project --with jinja2 python -I render_golden.py <templates_dir> <out.json>

Writes the case inputs and, per (template, case), the expected output or an error marker.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from jinja2.exceptions import TemplateError
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

FIXED_TIME = 1768478400  # 2026-01-15 12:00:00 UTC

# (bos, eos) per template, the real tokenizer strings.
SPECIAL = {
    "Qwen-Qwen3-0.6B": ("", "<|im_end|>"),
    "Qwen3-Coder": ("", "<|im_end|>"),
    "Qwen-Qwen2.5-7B-Instruct": ("", "<|im_end|>"),
    "meta-llama-Llama-3.1-8B-Instruct": ("<|begin_of_text|>", "<|eot_id|>"),
    "meta-llama-Llama-3.2-3B-Instruct": ("<|begin_of_text|>", "<|eot_id|>"),
    "Mistral-Small-3.2-24B-Instruct-2506": ("<s>", "</s>"),
    "NousResearch-Hermes-3-Llama-3.1-8B-tool_use": ("<|begin_of_text|>", "<|im_end|>"),
    "deepseek-ai-DeepSeek-R1-Distill-Qwen-32B": (
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
    ),
    "google-gemma-2-2b-it": ("<bos>", "<eos>"),
    "microsoft-Phi-3.5-mini-instruct": ("<s>", "<|endoftext|>"),
}

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather in a city (°C) <now> & 'later'",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    },
}


def msg(role, content, **extra):
    return {"role": role, "content": content, **extra}


CASES = {
    "system_user": {
        "messages": [msg("system", "You are terse."), msg("user", "Hello there")],
        "tools": [],
        "add_generation_prompt": True,
        "kwargs": {},
    },
    "multi_turn": {
        "messages": [
            msg("user", "Hi"),
            msg("assistant", "Hello! How can I help?"),
            msg("user", "What is 2+2?"),
        ],
        "tools": [],
        "add_generation_prompt": True,
        "kwargs": {},
    },
    "tool_round": {
        "messages": [
            msg("system", "You can use tools."),
            msg("user", "Weather in Zürich?"),
            msg(
                "assistant",
                "",
                tool_calls=[
                    {
                        "id": "abc123xyz",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Zürich", "unit": "celsius", "days": 2},
                        },
                    }
                ],
            ),
            msg("tool", '{"temp": 3}', tool_call_id="abc123xyz", name="get_weather"),
        ],
        "tools": [WEATHER],
        "add_generation_prompt": True,
        "kwargs": {},
    },
    "thinking_off": {
        "messages": [
            msg("user", "Hi"),
            msg("assistant", "Hello", reasoning_content="The user greets me."),
            msg("user", "Again"),
        ],
        "tools": [],
        "add_generation_prompt": True,
        "kwargs": {"enable_thinking": False},
    },
}


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # As in transformers.utils.chat_template_utils.
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def raise_exception(message):
    raise TemplateError(message)


def strftime_now(fmt):
    return datetime.fromtimestamp(FIXED_TIME, timezone.utc).strftime(fmt)


def main():
    tdir, out = Path(sys.argv[1]), Path(sys.argv[2])
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now

    renders = []
    for name, (bos, eos) in SPECIAL.items():
        tmpl = env.from_string((tdir / f"{name}.jinja").read_text(encoding="utf-8"))
        for case_name, case in CASES.items():
            ctx = {
                "messages": case["messages"],
                "add_generation_prompt": case["add_generation_prompt"],
                "bos_token": bos,
                "eos_token": eos,
                **case["kwargs"],
            }
            if case["tools"]:
                ctx["tools"] = case["tools"]
            entry = {"template": name, "case": case_name, "bos": bos, "eos": eos}
            try:
                entry["expected"] = tmpl.render(**ctx)
            except TemplateError as e:
                entry["error"] = True
                entry["message"] = str(e)
            renders.append(entry)

    doc = {"fixed_time": FIXED_TIME, "cases": CASES, "renders": renders}
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    errs = sum(1 for r in renders if r.get("error"))
    print(f"{len(renders)} renders, {errs} expected errors")


if __name__ == "__main__":
    main()
