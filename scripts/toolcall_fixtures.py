#!/usr/bin/env python3
"""Extracts failing tool-call outputs from a gateway request log as small, committable fixtures (ADR 0051).

    python scripts/toolcall_fixtures.py bench/results/<date>/claude-code-proxy/tetraeder-r1.reqlog.jsonl fixtures.json

A logged /v1/messages response is kept when its text contains <tool_call>, <function= or <parameter=, or when it calls a
tool that is not in the request's tools. For each one the file gets {generation_id, stop_reason, tools, output}: the
tool names of the request and the assistant output only (text and tool_use blocks). Request messages and system
prompts are never written. Commit the fixtures, not the request log (it holds clear-text prompts, ADR 0046).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anthropic_proxy import Observer  # noqa: E402


def fixtures(lines):
    """Yields one fixture per logged response that shows a tool-call failure."""
    for line in lines:
        if line.get("path") != "/v1/messages" or line.get("status") != 200:
            continue
        request, response = line.get("request"), line.get("response")
        if not isinstance(request, dict):
            continue
        obs = Observer(0.0, json.dumps({"tools": request.get("tools"), "messages": []}).encode())
        if isinstance(response, str):
            obs.is_sse = True
            obs.feed_body(response.encode("utf-8") + b"\n\n", 0.0)
        elif isinstance(response, dict):
            obs.status = 200
            obs.body = bytearray(json.dumps(response).encode("utf-8"))
            obs.finish()
        else:
            continue
        stats = obs.tool_stats()
        if stats["leaked_markers"] or stats["tool_names_unknown"]:
            yield {
                "generation_id": line.get("generation_id"),
                "stop_reason": obs.stop_reason,
                "tools": sorted(obs.tool_names),
                "output": obs.output(),
            }


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    out = list(fixtures(lines))
    with open(argv[1], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"{len(out)} fixtures from {len(lines)} logged requests", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
