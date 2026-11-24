#!/usr/bin/env python3
"""A synthetic multi-session request log for the prefix disk tier (ADR 0087), in the format of scripts/gateway_replay.py.

SESSIONS distinct conversations, each with its own long system prompt (words drawn from a seeded list, so no two sessions
share a prefix), TURNS turns each; the requests go round robin (turn 1 of every session, then turn 2 of every session, ...),
so with more sessions than server slots a session's sequence has been evicted from its slot, then from the RAM budget, when
its next turn arrives. A turn repeats the earlier user messages with a fixed stand-in for the assistant reply, so the common
prefix with the saved sequence is the system prompt plus the first user message.

    python scripts/disk_tier_log.py --sessions 12 --turns 4 --words 1500 --out disk-tier-requests.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

WORDS = (
    "river stone market winter lantern copper harbor meadow engine signal garden bridge falcon orchard velvet thunder "
    "compass candle forest anchor mirror saddle pepper window marble ladder island prairie quartz harvest timber ribbon "
    "kettle summit shadow canyon beacon trellis granite pillar fabric violet lagoon cinder meadowlark bramble tunnel "
    "paddle glacier lattice sparrow bonfire cobweb parcel thistle outpost rampart weaver quarry hollow sundial rooftop "
    "ferry cellar plume basalt ember frontier tapestry lighthouse salvage drizzle monsoon gravel pasture cobalt"
).split()


def system_prompt(session: int, words: int) -> str:
    rng = random.Random(1000 + session)
    body = " ".join(rng.choice(WORDS) for _ in range(words))
    return f"You are assistant number {session}. Background notes, unique to this conversation: {body}."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=12)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--words", type=int, default=1500)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    systems = [system_prompt(s, a.words) for s in range(a.sessions)]
    n = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for t in range(1, a.turns + 1):
            for s in range(a.sessions):
                msgs = [{"role": "system", "content": systems[s]}]
                for k in range(1, t + 1):
                    msgs.append({"role": "user", "content": f"Question {k} of session {s}: name one item from the notes."})
                    if k < t:
                        msgs.append({"role": "assistant", "content": f"Answer {k}: noted."})
                body = {"model": a.model, "messages": msgs, "max_tokens": a.max_tokens, "temperature": 0, "stream": True,
                        "chat_template_kwargs": {"enable_thinking": False}}
                f.write(json.dumps({"ts_ms": n, "path": "/v1/chat/completions", "status": 200, "request": body}) + "\n")
                n += 1


if __name__ == "__main__":
    main()
