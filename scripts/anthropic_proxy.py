#!/usr/bin/env python3
"""Logging reverse proxy for the Anthropic API (stdlib asyncio only).

Forwards every request to --upstream and relays the response bytes as they arrive (SSE is not buffered). For each
POST /v1/messages it appends one JSON line to --log: token counts from the usage fields, time to first token (first
content_block_delta of any kind), decode speed, total time and the error of a non-200 answer. Both engines of the
Claude Code benchmark are measured through this one code path.

    python scripts/anthropic_proxy.py --listen-port 18090 --upstream http://127.0.0.1:18080 --label tetraeder \
        --task fix-rounding --rep 1 --log proxy.jsonl

Two small changes to the forwarded request: Connection: close (one upstream connection per request) and
Accept-Encoding: identity (the stream must be readable here).
"""

import argparse
import asyncio
import json
import ssl
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding",
    "upgrade", "content-length", "host", "accept-encoding",
}
MAX_BODY_OBSERVED = 64 * 1024 * 1024
# Text of a tool call that the engine's parser failed to turn into a tool_use block (ADR 0051).
LEAK_MARKERS = ("<tool_call>", "<function=", "<parameter=")


# Claude Code's wording for file tool failures that come from the task, not the call's form.
# Seen in bench/results/20261110b.
FILE_ERRORS = ("String to replace not found", "No changes to make", "File does not exist", "EISDIR")


def tool_error_class(text: str) -> str:
    """Class of an is_error tool_result by its text; the patterns are Claude Code's wording.

    input_validation and unknown_tool are call errors (the engine's tool-call output); bash_exit (a command
    exited non-zero, e.g. failing tests) and file (Read/Edit on a missing path or string) are task outcomes.
    bash_exit is checked first: command output may quote the other patterns."""
    if text.lstrip().startswith("Exit code "):
        return "bash_exit"
    if "InputValidationError" in text:
        return "input_validation"
    if "No such tool available" in text:
        return "unknown_tool"
    if any(p in text for p in FILE_ERRORS):
        return "file"
    return "other"


def result_text(content) -> str:
    """Text of a tool_result content: a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and isinstance(b.get("text"), str))
    return ""


class Dechunker:
    """Incremental HTTP/1.1 chunked-transfer decoder, used only to observe the body that is relayed untouched."""

    def __init__(self):
        self.buf = b""
        self.need = None  # bytes left in the current chunk, None = expecting a size line
        self.done = False

    def feed(self, data: bytes) -> bytes:
        self.buf += data
        out = bytearray()
        while not self.done:
            if self.need is None:
                i = self.buf.find(b"\r\n")
                if i < 0:
                    break
                size = int(self.buf[:i].split(b";")[0].strip() or b"0", 16)
                self.buf = self.buf[i + 2 :]
                if size == 0:
                    self.done = True
                    break
                self.need = size
            elif self.need > 0:
                take = self.buf[: self.need]
                if not take:
                    break
                out += take
                self.buf = self.buf[len(take) :]
                self.need -= len(take)
            else:  # chunk data done, expect CRLF
                if len(self.buf) < 2:
                    break
                self.buf = self.buf[2:]
                self.need = None
        return bytes(out)


class Observer:
    """Collects the metrics of one /v1/messages exchange from request body and relayed response."""

    def __init__(self, t0: float, request_body: bytes):
        self.t0 = t0
        self.status = 0
        self.stream = False
        self.model = None
        self.tool_names = set()
        self.tool_errors = {}
        try:
            req = json.loads(request_body)
            self.stream = bool(req.get("stream"))
            self.model = req.get("model")
            self._scan_request(req)
        except (ValueError, AttributeError):
            pass
        self.blocks = []  # content blocks of the response in order: {"type", "name", "text": [], "json": []}
        self._by_index = {}
        self.is_sse = False
        self.text = b""
        self.body = bytearray()
        self.input_tokens = None
        self.output_tokens = None
        self.cache_read = None
        self.cache_creation = None
        self.stop_reason = None
        self.t_first = None
        self.t_last = None
        self.n_deltas = 0
        self.error_type = None
        self.error_message = None

    def _scan_request(self, req: dict):
        """Tool names, and the is_error tool results of the last user message by class (counts only)."""
        self.tool_names = {t["name"] for t in req.get("tools") or [] if isinstance(t, dict) and "name" in t}
        msgs = req.get("messages") or []
        last = msgs[-1] if msgs and isinstance(msgs[-1], dict) else {}
        if last.get("role") != "user" or not isinstance(last.get("content"), list):
            return
        for b in last["content"]:
            if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error") is True:
                cls = tool_error_class(result_text(b.get("content")))
                self.tool_errors[cls] = self.tool_errors.get(cls, 0) + 1

    def _block_start(self, index, block: dict):
        if not isinstance(block, dict):
            return
        b = {"type": block.get("type"), "name": block.get("name"), "text": [], "json": []}
        if isinstance(block.get("text"), str):
            b["text"].append(block["text"])
        if isinstance(block.get("input"), dict) and block["input"]:
            b["input"] = block["input"]
        self.blocks.append(b)
        self._by_index[index] = b

    def _block_delta(self, index, delta: dict):
        b = self._by_index.get(index)
        if b is None or not isinstance(delta, dict):
            return
        if isinstance(delta.get("text"), str):
            b["text"].append(delta["text"])
        if isinstance(delta.get("partial_json"), str):
            b["json"].append(delta["partial_json"])

    def output(self) -> list:
        """The assistant output: text and tool_use blocks."""
        out = []
        for b in self.blocks:
            if b["type"] == "text":
                out.append({"type": "text", "text": "".join(b["text"])})
            elif b["type"] == "tool_use":
                raw = "".join(b["json"])
                try:
                    args = json.loads(raw) if raw else b.get("input", {})
                except ValueError:
                    args = raw
                out.append({"type": "tool_use", "name": b["name"], "input": args})
        return out

    def tool_stats(self) -> dict:
        out = self.output()
        uses = [b for b in out if b["type"] == "tool_use"]
        return {
            "tool_uses": len(uses),
            "tool_errors": dict(sorted(self.tool_errors.items())),
            "tool_names_unknown": sum(1 for b in uses if b["name"] not in self.tool_names),
            "leaked_markers": sum(1 for b in out if b["type"] == "text" and any(m in b["text"] for m in LEAK_MARKERS)),
        }

    def _usage(self, usage: dict):
        if not isinstance(usage, dict):
            return
        for key, attr in (
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("cache_read_input_tokens", "cache_read"),
            ("cache_creation_input_tokens", "cache_creation"),
        ):
            if isinstance(usage.get(key), int):
                setattr(self, attr, usage[key])

    def _event(self, name: str, data: dict, now: float):
        kind = data.get("type", name)
        if kind == "message_start":
            self._usage((data.get("message") or {}).get("usage"))
        elif kind == "content_block_start":
            self._block_start(data.get("index"), data.get("content_block"))
        elif kind == "content_block_delta":
            self._block_delta(data.get("index"), data.get("delta"))
            if self.t_first is None:
                self.t_first = now
            self.t_last = now
            self.n_deltas += 1
        elif kind == "message_delta":
            self._usage(data.get("usage"))
            self.stop_reason = (data.get("delta") or {}).get("stop_reason", self.stop_reason)
        elif kind == "error":
            err = data.get("error") or {}
            self.error_type = err.get("type", "error")
            self.error_message = err.get("message")

    def feed_body(self, data: bytes, now: float):
        if self.is_sse:
            self.text += data.replace(b"\r\n", b"\n")
            while b"\n\n" in self.text:
                block, self.text = self.text.split(b"\n\n", 1)
                name, payload = "", []
                for line in block.decode("utf-8", "replace").split("\n"):
                    if line.startswith("event:"):
                        name = line[6:].strip()
                    elif line.startswith("data:"):
                        payload.append(line[5:].lstrip())
                if payload:
                    try:
                        obj = json.loads("\n".join(payload))
                    except ValueError:
                        continue
                    if isinstance(obj, dict):
                        self._event(name, obj, now)
        elif len(self.body) < MAX_BODY_OBSERVED:
            self.body += data

    def finish(self):
        if self.is_sse or not self.body:
            return
        try:
            obj = json.loads(bytes(self.body))
        except ValueError:
            return
        if not isinstance(obj, dict):
            return
        if self.status != 200 or obj.get("type") == "error":
            err = obj.get("error") if isinstance(obj.get("error"), dict) else {}
            self.error_type = err.get("type") or "error"
            self.error_message = err.get("message") or bytes(self.body)[:500].decode("utf-8", "replace")
        else:
            self._usage(obj.get("usage"))
            self.stop_reason = obj.get("stop_reason")
            for i, block in enumerate(obj.get("content") or []):
                self._block_start(i, block)

    def record(self, label: str, task: str, rep: str, path: str, t_end: float, req_bytes: int) -> dict:
        if self.status != 200 and self.error_type is None:
            self.error_type = f"http_{self.status}"
        tps = None
        if self.output_tokens and self.output_tokens > 1 and self.t_first is not None and self.t_last > self.t_first:
            tps = (self.output_tokens - 1) / (self.t_last - self.t_first)
        rel = lambda t: None if t is None else round(t - self.t0, 4)  # noqa: E731
        return {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "engine": label,
            "task": task,
            "rep": rep,
            "path": path,
            "model": self.model,
            "status": self.status,
            "stream": self.stream,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read,
            "cache_creation_tokens": self.cache_creation,
            "stop_reason": self.stop_reason,
            "ttft_s": rel(self.t_first),
            "last_delta_s": rel(self.t_last),
            "n_deltas": self.n_deltas,
            "decode_tok_s": None if tps is None else round(tps, 3),
            "total_s": round(t_end - self.t0, 4),
            "request_bytes": req_bytes,
            "error_type": self.error_type,
            "error_message": self.error_message,
            **self.tool_stats(),
        }


async def read_request(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head[:-4].split(b"\r\n")
    method, target, version = lines[0].decode("latin-1").split(" ", 2)
    headers = []
    for raw in lines[1:]:
        k, _, v = raw.decode("latin-1").partition(":")
        headers.append((k.strip(), v.strip()))
    low = {k.lower(): v for k, v in headers}
    if low.get("expect", "").lower() == "100-continue":
        writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        await writer.drain()
    if "chunked" in low.get("transfer-encoding", "").lower():
        dec, body = Dechunker(), b""
        while not dec.done:
            data = await reader.read(65536)
            if not data:
                break
            body += dec.feed(data)
    else:
        body = await reader.readexactly(int(low.get("content-length", "0")))
    return method, target, version, headers, body


def make_handler(args, upstream):
    scheme, host, port, base_path = upstream

    async def open_upstream():
        ctx = ssl.create_default_context() if scheme == "https" else None
        return await asyncio.open_connection(host, port, ssl=ctx)

    async def handle(reader, writer):
        obs = None
        t0 = time.monotonic()
        path = ""
        req_bytes = 0
        uwriter = None
        try:
            try:
                method, target, version, headers, body = await read_request(reader, writer)
            except (asyncio.IncompleteReadError, ConnectionError, ValueError):
                return
            t0 = time.monotonic()
            path = target.split("?", 1)[0]
            req_bytes = len(body)
            logged = method == "POST" and path == "/v1/messages"
            obs = Observer(t0, body) if logged else None

            try:
                ureader, uwriter = await open_upstream()
            except OSError as e:
                msg = json.dumps({"type": "error", "error": {"type": "proxy_upstream_error", "message": str(e)}}).encode()
                writer.write(
                    b"HTTP/1.1 502 Bad Gateway\r\nContent-Type: application/json\r\nConnection: close\r\n"
                    + f"Content-Length: {len(msg)}\r\n\r\n".encode()
                    + msg
                )
                await writer.drain()
                if obs:
                    obs.status = 502
                    obs.body += msg
                    obs.finish()
                return

            out = [f"{method} {base_path.rstrip('/')}{target} HTTP/1.1", f"Host: {host}:{port}"]
            out += [f"{k}: {v}" for k, v in headers if k.lower() not in HOP_BY_HOP]
            out += [f"Content-Length: {len(body)}", "Accept-Encoding: identity", "Connection: close", "", ""]
            uwriter.write("\r\n".join(out).encode("latin-1") + body)
            await uwriter.drain()

            rhead = await ureader.readuntil(b"\r\n\r\n")
            rlines = rhead[:-4].split(b"\r\n")
            status = int(rlines[0].split()[1])
            rheaders = []
            for raw in rlines[1:]:
                k, _, v = raw.decode("latin-1").partition(":")
                rheaders.append((k.strip(), v.strip()))
            rlow = {k.lower(): v for k, v in rheaders}
            chunked = "chunked" in rlow.get("transfer-encoding", "").lower()
            length = int(rlow["content-length"]) if "content-length" in rlow else None
            if obs:
                obs.status = status
                obs.is_sse = "text/event-stream" in rlow.get("content-type", "")
            client_head = [rlines[0].decode("latin-1")]
            client_head += [f"{k}: {v}" for k, v in rheaders if k.lower() != "connection"]
            client_head += ["Connection: close", "", ""]
            writer.write("\r\n".join(client_head).encode("latin-1"))
            await writer.drain()

            dec = Dechunker() if chunked else None
            got = 0
            while True:
                data = await ureader.read(65536)
                if not data:
                    break
                now = time.monotonic()
                writer.write(data)
                await writer.drain()
                got += len(data)
                if obs:
                    obs.feed_body(dec.feed(data) if dec else data, now)
                if dec and dec.done:
                    break
                if length is not None and got >= length:
                    break
            if obs:
                obs.finish()
        except (ConnectionError, asyncio.IncompleteReadError) as e:
            if obs:
                obs.error_type = obs.error_type or "connection_error"
                obs.error_message = obs.error_message or f"{type(e).__name__}: {e}"
                if not obs.status:
                    obs.status = 499
        except Exception as e:  # keep serving after a bug in the observer
            print(f"proxy error: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            if obs:
                obs.error_type = obs.error_type or "proxy_error"
                obs.error_message = obs.error_message or str(e)
        finally:
            t_end = time.monotonic()
            if obs:
                rec = obs.record(args.label, args.task, args.rep, path, t_end, req_bytes)
                with open(args.log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            for w in (uwriter, writer):
                if w is not None:
                    try:
                        w.close()
                    except Exception:
                        pass

    return handle


def parse_upstream(url: str):
    u = urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise SystemExit(f"bad --upstream: {url}")
    return u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80), u.path


async def amain(args):
    handler = make_handler(args, parse_upstream(args.upstream))
    server = await asyncio.start_server(handler, args.listen_host, args.listen_port)
    port = server.sockets[0].getsockname()[1]
    print(f"listening on http://{args.listen_host}:{port}", file=sys.stderr, flush=True)
    async with server:
        await server.serve_forever()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen-host", default="127.0.0.1")
    ap.add_argument("--listen-port", type=int, default=0)
    ap.add_argument("--upstream", required=True, help="base URL of the engine, e.g. http://127.0.0.1:18080")
    ap.add_argument("--label", default="engine", help="engine label written to every log line")
    ap.add_argument("--task", default="")
    ap.add_argument("--rep", default="")
    ap.add_argument("--log", required=True, help="JSONL file, one line per POST /v1/messages")
    args = ap.parse_args(argv)
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
