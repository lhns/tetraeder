#!/usr/bin/env python3
"""Replay a `tetraeder serve --request-log` file (ADR 0040) against a base URL.

Each logged `request` body is resent to its `path` (+ query) with the logged inter-arrival times divided by
`--speed` (0 = as fast as possible, bounded by `--concurrency`). Records per request status, TTFT, total time and
usage. Headers of the log (anthropic-version, anthropic-beta) are sent again. Needs httpx only.

    python scripts/gateway_replay.py http://127.0.0.1:18080 requests.jsonl --speed 4 --out replay.json
"""

import argparse
import asyncio
import json
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


def sse_has_content(line: str) -> bool:
    """True for an SSE data line that carries output (first content for TTFT): a content or text delta (OpenAI,
    delta.content / text) or a content_block_delta (Anthropic)."""
    if not line.startswith("data:"):
        return False
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return False
    try:
        c = json.loads(data)
    except json.JSONDecodeError:
        return False
    if c.get("type") == "content_block_delta":
        d = c.get("delta") or {}
        return bool(d.get("text") or d.get("partial_json") or d.get("thinking"))
    for ch in c.get("choices") or []:
        d = ch.get("delta") or {}
        if d.get("content") or d.get("reasoning_content") or d.get("tool_calls") or ch.get("text"):
            return True
    return False


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
                    if ttft is None and sse_has_content(line):
                        ttft = time.perf_counter() - t0
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

    return {
        "requests": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "status_mismatch": sum(1 for r in results if r.get("logged_status") != r.get("status")),
        "ttft_median_s": med(ttft),
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
    p.add_argument("--out")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    with open(args.log, encoding="utf-8") as f:
        entries = parse_log(f)
    if args.limit:
        entries = entries[: args.limit]
    out = asyncio.run(replay(args, entries))
    out["params"] = {"base_url": args.base_url, "log": args.log, "speed": args.speed, "concurrency": args.concurrency}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
    print(json.dumps(out["summary"]))
    return 1 if out["summary"]["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
