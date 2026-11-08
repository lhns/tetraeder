#!/usr/bin/env python3
"""Uncached prompt tokens of a Claude Code proxy run by class, and the share ADR 0049's prefix cache would reach.

Reads the proxy logs (proxy-<engine>-*.jsonl, scripts/anthropic_proxy.py) of a claude-code-proxy result directory.
The logs carry token counts, not bodies, so conversations are inferred per session: a request continues the open
conversation with the longest last prompt that is not longer than its own; a request continuing none starts one.
Classes of uncached tokens (ADR 0051):
  first        first request of a conversation (Claude Code's prompts diverge after about 22 tokens)
  suffix       at most the growth since the conversation's last prompt (+64): new turns, re-rendered output
  divergence   more than the growth, but at least 1000 tokens cached: the prompt changed mid-way
  evicted      more than the growth with almost nothing cached: another conversation evicted this one
With ADR 0049 an evicted conversation is restored from host RAM, so its request costs only the growth.

    python scripts/claude_code_cache_classes.py bench/results/20261103/claude-code-proxy
"""

import collections
import glob
import json
import os
import sys

SLACK = 64
DIVERGENCE_MIN_CACHED = 1000


def classify(rows):
    """Yields (class, prompt tokens, uncached tokens, uncached tokens with ADR 0049) per answered request."""
    convs = []  # last prompt length of each open conversation
    for r in rows:
        if r.get("status") != 200 or r.get("input_tokens") is None:
            continue
        new, cached = r["input_tokens"], r.get("cache_read_tokens") or 0
        prompt = new + cached
        cands = [i for i, p in enumerate(convs) if p <= prompt]
        if not cands:
            convs.append(prompt)
            yield "first", prompt, new, new
            continue
        i = max(cands, key=lambda j: convs[j])
        growth = prompt - convs[i]
        convs[i] = prompt
        if new <= growth + SLACK:
            yield "suffix", prompt, new, new
        elif cached >= DIVERGENCE_MIN_CACHED:
            yield "divergence", prompt, new, new
        else:
            yield "evicted", prompt, new, min(new, growth + SLACK)


def main(argv):
    d = argv[1] if len(argv) > 1 else "."
    engines = sorted({os.path.basename(f).split("-r")[0][len("proxy-"):] for f in glob.glob(f"{d}/proxy-*-r*.jsonl")})
    for eng in engines:
        tokens = collections.Counter()
        prompt_total = new_total = new_0049 = n = 0
        for f in sorted(glob.glob(f"{d}/proxy-{eng}-r*.jsonl")):
            with open(f, encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            for cls, prompt, new, new_sim in classify(rows):
                n += 1
                prompt_total += prompt
                new_total += new
                new_0049 += new_sim
                tokens[cls] += new
                tokens["evicted, recovered by 0049"] += new - new_sim
        if not prompt_total:
            continue
        ceiling = 1 - (tokens["first"] + tokens["suffix"]) / prompt_total
        print(f"{eng}: {n} requests, {prompt_total} prompt tokens, cached share {1 - new_total / prompt_total:.4f}, "
              f"with ADR 0049 {1 - new_0049 / prompt_total:.4f}, ceiling of any prefix cache {ceiling:.4f}")
        for cls in ("first", "suffix", "divergence", "evicted", "evicted, recovered by 0049"):
            print(f"  {cls:28s} {tokens[cls]:9d} uncached ({tokens[cls] / prompt_total:6.2%} of prompt tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
