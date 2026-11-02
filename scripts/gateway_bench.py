#!/usr/bin/env python3
"""Load client for an OpenAI-compatible streaming endpoint (llama-server or `tetraeder serve`, ADR 0040).

Per request: TTFT (first non-empty content delta), completion tokens (usage chunk, else delta count) and decode
tok/s = (n - 1) / (t_last - t_first). Aggregates: median/p90 TTFT, median decode tok/s, total throughput
(sum of completion tokens / wall time), errors. `--multi-turn K` runs one growing conversation and reports the
TTFT per turn (prefix reuse shows as a flat curve).

Needs httpx only. Example:
    python scripts/gateway_bench.py http://127.0.0.1:18080 --model m --prompt-tokens 512 --max-tokens 128 \
        --concurrency 4 --requests 16 --label tetraeder-c4 --out out.json
"""

import argparse
import asyncio
import json
import math
import random
import statistics
import string
import sys
import time

FILLER = (
    "The quick brown fox jumps over the lazy dog while the engineers measure how many tokens per second "
    "a gateway can stream. "
)
CHARS_PER_TOKEN = 4.0  # rough, only used to size prompts


def make_prompt(prompt_tokens: int, nonce: str | None, shared_prefix: bool = False, shared_nonce: str = "shared") -> str:
    """Prompt of about `prompt_tokens` tokens. A random nonce prefix makes every request unique unless
    `shared_prefix`, where all requests start with the same text (prefix reuse)."""
    head = f"[{shared_nonce if shared_prefix else nonce}] "
    n_chars = max(int(prompt_tokens * CHARS_PER_TOKEN) - len(head), 0)
    body = (FILLER * (n_chars // len(FILLER) + 1))[:n_chars]
    return head + body


def new_nonce(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(12))


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, q in [0, 100]."""
    if not values:
        return None
    s = sorted(values)
    k = max(math.ceil(q / 100 * len(s)) - 1, 0)
    return s[min(k, len(s) - 1)]


def decode_rate(n_tokens: int, t_first: float | None, t_last: float | None) -> float | None:
    if t_first is None or t_last is None or n_tokens < 2 or t_last <= t_first:
        return None
    return (n_tokens - 1) / (t_last - t_first)


def med_or_none(values: list) -> float | None:
    return statistics.median(values) if values else None


def aggregate(results: list[dict], wall_s: float) -> dict:
    ok = [r for r in results if not r.get("error")]
    ttft = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    rate = [r["decode_tps"] for r in ok if r.get("decode_tps") is not None]
    tokens = sum(r.get("completion_tokens", 0) for r in ok)
    return {
        "requests": len(results),
        "errors": len(results) - len(ok),
        "wall_s": wall_s,
        "completion_tokens": tokens,
        "prompt_tokens_median": med_or_none([r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]),
        "ttft_median_s": statistics.median(ttft) if ttft else None,
        "ttft_p90_s": percentile(ttft, 90),
        "decode_tps_median": statistics.median(rate) if rate else None,
        "throughput_tps": tokens / wall_s if wall_s > 0 else None,
    }


def parse_sse_line(line: str) -> dict | None:
    """The JSON payload of an SSE `data:` line, None for comments, blanks and [DONE]."""
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def delta_text(chunk: dict) -> str:
    """Generated text in a chat or completions chunk (content only; reasoning counts as output too)."""
    out = ""
    for c in chunk.get("choices") or []:
        d = c.get("delta") or {}
        out += d.get("content") or d.get("reasoning_content") or c.get("text") or ""
    return out


async def one_request(client, url: str, body: dict) -> dict:
    t0 = time.perf_counter()
    t_first = t_last = None
    deltas = 0
    usage = None
    text = ""
    try:
        async with client.stream("POST", url, json=body) as resp:
            if resp.status_code != 200:
                err = (await resp.aread()).decode(errors="replace")[:300]
                return {"error": f"HTTP {resp.status_code}: {err}", "total_s": time.perf_counter() - t0}
            async for line in resp.aiter_lines():
                chunk = parse_sse_line(line)
                if chunk is None:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                piece = delta_text(chunk)
                if piece:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    deltas += 1
                    text += piece
    except Exception as e:  # network errors are results, not crashes
        return {"error": f"{type(e).__name__}: {e}", "total_s": time.perf_counter() - t0}
    end = time.perf_counter()
    n = (usage or {}).get("completion_tokens") or deltas
    return {
        "ttft_s": None if t_first is None else t_first - t0,
        "total_s": end - t0,
        "completion_tokens": n,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "usage_from": "usage" if usage else "deltas",
        "decode_tps": decode_rate(n, t_first, t_last),
        "text_head": text[:60],
    }


def request_body(args, content: str | list, chat: bool) -> dict:
    body = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if chat:
        body["messages"] = content
    else:
        body["prompt"] = content
    return body


async def run_load(args, client, url: str, chat: bool) -> dict:
    rng = random.Random(args.seed)
    prompts = [
        make_prompt(args.prompt_tokens, new_nonce(rng), args.shared_prefix) for _ in range(args.requests)
    ]
    sem = asyncio.Semaphore(args.concurrency)

    async def worker(p: str) -> dict:
        async with sem:
            content = [{"role": "user", "content": p}] if chat else p
            return await one_request(client, url, request_body(args, content, chat))

    t0 = time.perf_counter()
    results = await asyncio.gather(*(worker(p) for p in prompts))
    return {"results": results, "summary": aggregate(results, time.perf_counter() - t0)}


async def run_multi_turn(args, client, url: str) -> dict:
    """One conversation grown by `K` turns: each turn resends all earlier turns plus its answers."""
    rng = random.Random(args.seed)
    messages = [{"role": "system", "content": make_prompt(args.prompt_tokens, new_nonce(rng))}]
    turns = []
    for k in range(args.multi_turn):
        messages.append({"role": "user", "content": f"Turn {k + 1}: continue the story in two sentences."})
        r = await one_request(client, url, request_body(args, list(messages), True))
        turns.append(r)
        if r.get("error"):
            break
        # the reply is not kept (stream text is a head only); a fixed assistant turn keeps prompts identical
        messages.append({"role": "assistant", "content": r.get("text_head") or "ok"})
    ttft = [t.get("ttft_s") for t in turns]
    return {
        "results": turns,
        "summary": {
            "turns": len(turns),
            "errors": sum(1 for t in turns if t.get("error")),
            "ttft_per_turn_s": ttft,
            "ttft_first_s": ttft[0] if ttft else None,
            "ttft_later_median_s": statistics.median([t for t in ttft[1:] if t is not None])
            if len([t for t in ttft[1:] if t is not None])
            else None,
        },
    }


async def main_async(args) -> dict:
    import httpx

    chat = args.endpoint == "chat"
    url = args.base_url.rstrip("/") + ("/v1/chat/completions" if chat else "/v1/completions")
    timeout = httpx.Timeout(args.timeout, connect=30)
    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=args.concurrency + 2)) as client:
        if args.multi_turn:
            body = await run_multi_turn(args, client, url)
        else:
            body = await run_load(args, client, url, chat)
    return {
        "label": args.label,
        "base_url": args.base_url,
        "model": args.model,
        "endpoint": args.endpoint,
        "params": {
            "prompt_tokens": args.prompt_tokens,
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "temperature": args.temperature,
            "shared_prefix": args.shared_prefix,
            "multi_turn": args.multi_turn,
        },
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **body,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base_url")
    p.add_argument("--model", default="default")
    p.add_argument("--endpoint", choices=["chat", "completions"], default="chat")
    p.add_argument("--prompt-tokens", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--requests", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--shared-prefix", action="store_true")
    p.add_argument("--multi-turn", type=int, default=0, metavar="K")
    p.add_argument("--seed", type=int, default=None, help="nonce seed (random per run when unset)")
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--label", default="")
    p.add_argument("--out")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out = asyncio.run(main_async(args))
    text = json.dumps(out, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
    print(json.dumps({"label": out["label"], **out["summary"]}))
    return 1 if out["summary"].get("errors") else 0


if __name__ == "__main__":
    sys.exit(main())
