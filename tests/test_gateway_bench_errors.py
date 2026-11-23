"""scripts/gateway_bench.py counts mid-stream error frames and empty streams as errors (bench-strata-q36 defect)."""

import asyncio
import importlib.util
import pathlib

import httpx

SPEC = importlib.util.spec_from_file_location(
    "gateway_bench", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "gateway_bench.py"
)
gb = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gb)

NL = b"\n\n"


def run(*frames: bytes) -> dict:
    body = b"".join(b"data: " + f + NL for f in frames)
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=body))

    async def go():
        async with httpx.AsyncClient(transport=transport) as c:
            return await gb.one_request(c, "http://x/v1/chat/completions", {})

    return asyncio.run(go())


def test_error_frame_is_an_error():
    r = run(b'{"error": {"message": "KV cache cannot drop positions"}}', b"[DONE]")
    assert "stream error" in r["error"]


def test_empty_stream_is_an_error():
    assert "empty stream" in run(b"[DONE]")["error"]


def test_tokens_are_a_success():
    r = run(b'{"choices": [{"delta": {"content": "hi"}}]}', b"[DONE]")
    assert "error" not in r and r["completion_tokens"] == 1
