#!/usr/bin/env python3
"""Replay a `tetraeder serve --request-log` file (ADR 0046) against a base URL.

Each logged `request` body is resent to its `path` (+ query) with the logged inter-arrival times divided by
`--speed` (0 = as fast as possible, bounded by `--concurrency`). Records per request status, TTFT, total time and
usage. Headers of the log (anthropic-version, anthropic-beta) are sent again. Needs httpx only.

    python scripts/gateway_replay.py http://127.0.0.1:18080 requests.jsonl --speed 4 --out replay.json
"""

import argparse
import asyncio
import hashlib
import json
import math
import struct
import sys
import time


def parse_log(lines) -> list[dict]:
    """Request-log lines to entries, in arrival order. Blank and malformed lines are skipped; entries without a
    `request` body or `path` are skipped too."""
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(e, dict) or "request" not in e or "path" not in e:
            continue
        out.append(e)
    out.sort(key=lambda e: e.get("ts_ms", 0))
    return out


def entry_prompt_tokens(entry: dict) -> int | None:
    """`usage.prompt_tokens` the request log recorded for the entry (reqlog.rs)."""
    u = entry.get("usage")
    n = u.get("prompt_tokens") if isinstance(u, dict) else None
    return n if isinstance(n, int) else None


def filter_prompt_tokens(entries: list[dict], max_tokens: int) -> tuple[list[dict], int]:
    """(entries whose logged prompt tokens fit max_tokens, number skipped). Entries without a logged count stay."""
    kept = [e for e in entries if (entry_prompt_tokens(e) or 0) <= max_tokens]
    return kept, len(entries) - len(kept)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gguf_context_length(path: str) -> int | None:
    """The `<arch>.context_length` key of a GGUF file (v2/v3 header walk), None when absent."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    fmts = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 10: "<Q", 11: "<q"}

    def skip(f, t):
        if t in sizes:
            return f.read(sizes[t])
        if t == 8:
            f.seek(struct.unpack("<Q", f.read(8))[0], 1)
        elif t == 9:
            et, n = struct.unpack("<IQ", f.read(12))
            for _ in range(n):
                skip(f, et)
        else:
            raise ValueError(f"unknown GGUF type {t}")

    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return None
        _version, _n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        for _ in range(n_kv):
            key = f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")
            (t,) = struct.unpack("<I", f.read(4))
            if key.endswith(".context_length") and t in fmts:
                return struct.unpack(fmts[t], f.read(sizes[t]))[0]
            skip(f, t)
    return None


def schedule(entries: list[dict], speed: float) -> list[float]:
    """Send offset in seconds for each entry: logged gap to the first request, divided by `speed`; all zero when
    speed is 0."""
    if not entries:
        return []
    if speed <= 0:
        return [0.0] * len(entries)
    t0 = entries[0].get("ts_ms", 0)
    return [max(e.get("ts_ms", t0) - t0, 0) / 1000.0 / speed for e in entries]


def target_url(base: str, entry: dict) -> str:
    url = base.rstrip("/") + entry["path"]
    q = entry.get("query")
    return url + ("?" + q) if q else url


def sse_content_kind(line: str) -> str | None:
    """The kind of output an SSE data line carries ("text", "thinking" or "tool"), None without output. A
    content or text delta (OpenAI, delta.content / text) or a content_block_delta (Anthropic)."""
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        c = json.loads(data)
    except json.JSONDecodeError:
        return None
    if c.get("type") == "content_block_delta":
        d = c.get("delta") or {}
        for key, kind in (("text", "text"), ("thinking", "thinking"), ("partial_json", "tool")):
            if d.get(key):
                return kind
        return None
    for ch in c.get("choices") or []:
        d = ch.get("delta") or {}
        if d.get("content") or ch.get("text"):
            return "text"
        if d.get("reasoning_content"):
            return "thinking"
        if d.get("tool_calls"):
            return "tool"
    return None


def sse_has_content(line: str) -> bool:
    """True for an SSE data line that carries output (first content for TTFT)."""
    return sse_content_kind(line) is not None


def usage_from(obj: dict) -> dict | None:
    u = obj.get("usage") if isinstance(obj, dict) else None
    if not u and isinstance(obj, dict):
        u = (obj.get("message") or {}).get("usage")
    return u or None


async def send(client, base: str, entry: dict, index: int) -> dict:
    url = target_url(base, entry)
    headers = dict(entry.get("headers") or {})
    headers.setdefault("x-api-key", "x")
    body = entry["request"]
    stream = bool(entry.get("stream") or (isinstance(body, dict) and body.get("stream")))
    t0 = time.perf_counter()
    res = {"index": index, "path": entry["path"], "stream": stream, "logged_status": entry.get("status")}
    try:
        if stream:
            ttft = None
            usage = None
            async with client.stream(entry.get("method", "POST"), url, json=body, headers=headers) as r:
                res["status"] = r.status_code
                async for line in r.aiter_lines():
                    if ttft is None and (kind := sse_content_kind(line)):
                        ttft = time.perf_counter() - t0
                        # A tool call streamed whole arrives only once it is generated.
                        res["first_kind"] = kind
                    if line.startswith("data:"):
                        try:
                            u = usage_from(json.loads(line[5:].strip()))
                            if u:
                                usage = {**(usage or {}), **u}
                        except json.JSONDecodeError:
                            pass
            res["ttft_s"], res["usage"] = ttft, usage
        else:
            r = await client.request(entry.get("method", "POST"), url, json=body, headers=headers)
            res["status"] = r.status_code
            res["ttft_s"] = None
            try:
                res["usage"] = usage_from(r.json())
            except ValueError:
                res["usage"] = None
    except Exception as e:
        res["status"] = None
        res["error"] = f"{type(e).__name__}: {e}"
    res["total_s"] = time.perf_counter() - t0
    return res


def summarize(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("status") == 200]
    ttft = sorted(r["ttft_s"] for r in ok if r.get("ttft_s") is not None)
    tot = sorted(r["total_s"] for r in ok)

    def med(v):
        return v[len(v) // 2] if v else None

    def p90(v):  # nearest rank, as gateway_bench.percentile
        return v[min(max(math.ceil(0.9 * len(v)) - 1, 0), len(v) - 1)] if v else None

    return {
        "requests": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "status_mismatch": sum(1 for r in results if r.get("logged_status") != r.get("status")),
        "ttft_median_s": med(ttft),
        "ttft_p90_s": p90(ttft),
        "total_median_s": med(tot),
    }


async def replay(args, entries: list[dict]) -> dict:
    import httpx

    offsets = schedule(entries, args.speed)
    sem = asyncio.Semaphore(args.concurrency)
    start = time.perf_counter()

    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout, connect=30)) as client:

        async def one(i: int) -> dict:
            wait = offsets[i] - (time.perf_counter() - start)
            if wait > 0:
                await asyncio.sleep(wait)
            async with sem:
                return await send(client, args.base_url, entries[i], i)

        results = await asyncio.gather(*(one(i) for i in range(len(entries))))
    wall = time.perf_counter() - start
    return {"wall_s": wall, "results": results, "summary": summarize(results)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("base_url")
    p.add_argument("log")
    p.add_argument("--speed", type=float, default=1.0, help="time scale of logged gaps; 0 = as fast as possible")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--limit", type=int, default=0, help="replay only the first N requests")
    p.add_argument("--max-prompt-tokens", type=int, default=0,
                   help="skip entries whose logged usage.prompt_tokens exceed N (0 = keep all)")
    p.add_argument("--gguf", help="model file: its context_length must be at least --max-prompt-tokens, else exit 2")
    p.add_argument("--expect-context-length", type=int, default=0,
                   help="with --gguf: stop unless the file's context_length equals this (ADR 0065: 40960)")
    p.add_argument("--out")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    with open(args.log, encoding="utf-8") as f:
        entries = parse_log(f)
    if args.gguf:
        n_ctx = gguf_context_length(args.gguf)
        if args.expect_context_length and n_ctx != args.expect_context_length:
            print(f"context_length of {args.gguf} is {n_ctx}, expected {args.expect_context_length}", file=sys.stderr)
            return 2
        if args.max_prompt_tokens and (n_ctx is None or n_ctx < args.max_prompt_tokens):
            print(f"context_length {n_ctx} below --max-prompt-tokens {args.max_prompt_tokens}", file=sys.stderr)
            return 2
    skipped = 0
    if args.max_prompt_tokens:
        entries, skipped = filter_prompt_tokens(entries, args.max_prompt_tokens)
    if args.limit:
        entries = entries[: args.limit]
    out = asyncio.run(replay(args, entries))
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")  # box local time, as the nvidia-smi timestamp column
    out["params"] = {
        "base_url": args.base_url, "log": args.log, "speed": args.speed, "concurrency": args.concurrency,
        "max_prompt_tokens": args.max_prompt_tokens, "skipped": skipped, "requests": len(entries),
        "log_sha256": file_sha256(args.log),
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
    print(json.dumps(out["summary"]))
    return 1 if out["summary"]["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
