#!/usr/bin/env python3
"""Greedy token parity of a fixed prompt set for the Phase 6 rehearsal (ADR 0091, rule 2), standard library only.

`run URL --out F` sends the prompts one at a time with temperature 0 and seed 1 through /v1/completions (raw
prompts, no template, `return_tokens` when the server supports it) and stores the completion text (the server returns no token ids),
the finish reason and the token count. `compare A B` lists the prompts whose output differs, with the
first differing token or character. Exit 1 when any differ.
"""

import argparse
import http.client
import json
import sys
import urllib.parse

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Explain in two sentences why the sky is blue.",
    "1, 2, 3, 5, 8, 13, 21,",
    "Write a haiku about distributed systems.\n",
    "Q: What is 17 times 23?\nA:",
    ("Summarize the following text in one line.\n\n"
     + "The cluster has two nodes, each with one GPU. A head routes requests to replicas. " * 24
     + "\n\nSummary:"),
    ("List the planets of the solar system, one per line, with a short fact each.\n"
     + "Context: " + "Mercury is the smallest planet and Jupiter the largest. " * 20 + "\n1."),
]


def post(url, key, path, body, timeout=300):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    c.request("POST", path, json.dumps(body), h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, json.loads(data) if data else {}


def run(a):
    key = ""
    if a.key_file:
        with open(a.key_file) as f:
            key = next((x.strip() for x in f if x.strip() and not x.startswith("#")), "")
    out = []
    for i, p in enumerate(PROMPTS):
        st, j = post(a.url, key, "/v1/completions", {"model": a.model, "prompt": p, "max_tokens": a.max_tokens,
                                                     "temperature": 0, "seed": 1})
        ch = (j.get("choices") or [{}])[0]
        rec = {"i": i, "status": st, "text": ch.get("text"), "tokens": ch.get("tokens"),
               "finish": ch.get("finish_reason"), "usage": j.get("usage"), "error": j.get("error")}
        out.append(rec)
        print(f"prompt {i}: status {st}, {len((rec['text'] or ''))} chars, finish {rec['finish']}")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)


def compare(a):
    x, y = (json.load(open(p)) for p in (a.a, a.b))
    bad = 0
    for ra, rb in zip(x, y):
        ta, tb = ra.get("tokens"), rb.get("tokens")
        if ta is not None and tb is not None:
            same = ta == tb
            where = next((k for k, (m, n) in enumerate(zip(ta, tb)) if m != n), min(len(ta), len(tb)))
            what = f"token {where} of {len(ta)} vs {len(tb)}"
        else:
            same = ra["text"] == rb["text"]
            ca, cb = ra["text"] or "", rb["text"] or ""
            where = next((k for k, (m, n) in enumerate(zip(ca, cb)) if m != n), min(len(ca), len(cb)))
            what = f"char {where} of {len(ca)} vs {len(cb)}"
        ok = same and ra.get("status") == 200 and rb.get("status") == 200
        print(f"prompt {ra['i']}: {'equal' if ok else 'DIFFERENT (' + what + ')'}")
        bad += not ok
    print(f"{len(x) - bad} of {len(x)} prompts equal")
    return 1 if bad or len(x) != len(y) else 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("url")
    r.add_argument("--key-file")
    r.add_argument("--model", default="x")
    r.add_argument("--max-tokens", type=int, default=64)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        sys.exit(compare(a))


if __name__ == "__main__":
    main()
