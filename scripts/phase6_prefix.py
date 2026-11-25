#!/usr/bin/env python3
"""Prefix fetch between replicas through the head (ADR 0091, rehearsal), standard library only.

1. X1: a long prompt P (about --prompt-tokens tokens) + " Question one:" runs alone, greedy; its replica (R1) saves the
   prefix.
2. Four fillers that share P and generate long fill R1 (affinity sends them there); 1.5 s later X2 (the prompt of X1
   again) arrives with R1 full, so the router sends it to the other replica, which has no prefix: it fetches the
   snapshot from R1 or recomputes, as the fetch rule prices it.
   With --first-on-b four blockers fill the head's own replica first, so X1 and the prefix are on the remote one.
3. X3: a cold prompt of the same length on an idle cluster, the recompute time of the same size.
Wall times are of the whole non-streamed request. Which replica ran what is read from the replicas' logs
(`admit` lines with `prompt=` and `source=`/`common=`). Output: JSON with the texts and times; X1 and X2 must be equal.

    python3 scripts/phase6_prefix.py http://127.0.0.1:8080 --key-file keys.txt --out prefix.json
"""

import argparse
import json
import threading
import time

from phase6_parity import post

FILLER = "The quick brown fox jumps over the lazy dog while the engineers measure a cluster of rented machines. "


def long_prompt(tokens, nonce):
    head = f"[{nonce}] "
    n = max(int(tokens * 4.0) - len(head), 0)
    return head + (FILLER * (n // len(FILLER) + 1))[:n]


def complete(url, key, prompt, max_tokens):
    t0 = time.time()
    st, j = post(url, key, "/v1/completions", {"model": "x", "prompt": prompt, "max_tokens": max_tokens,
                                               "temperature": 0, "seed": 1}, timeout=600)
    ch = (j.get("choices") or [{}])[0]
    return {"status": st, "text": ch.get("text"), "finish": ch.get("finish_reason"), "usage": j.get("usage"),
            "t_start": t0, "wall_s": round(time.time() - t0, 3), "error": j.get("error")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--key-file")
    ap.add_argument("--prompt-tokens", type=int, default=3000)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--first-on-b", action="store_true", help="X1 runs on the replica of the other node (holder B)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = ""
    if a.key_file:
        with open(a.key_file) as f:
            key = next((x.strip() for x in f if x.strip() and not x.startswith("#")), "")
    nonce = f"prefix-{int(time.time())}"
    p = long_prompt(a.prompt_tokens, nonce)
    q = p + " Question one: what is the first word of this text?"
    out = {"nonce": nonce, "prompt_tokens_target": a.prompt_tokens}
    blockers = []
    if a.first_on_b:
        # four unrelated long requests keep the head's own replica (A) full, so X1 goes to the other one (B)
        # (they share one prefix, so affinity keeps them together on whichever replica gets the first)
        shared = long_prompt(400, f"{nonce}-block")

        def blocker(i):
            blockers.append(complete(a.url, key, shared + f" Blocker {i}: continue.", 350))

        bts = [threading.Thread(target=blocker, args=(i,)) for i in range(4)]
        for t in bts:
            t.start()
            time.sleep(0.4)
        time.sleep(0.3)
    out["x1"] = complete(a.url, key, q, a.max_tokens)
    print("x1", out["x1"]["status"], out["x1"]["wall_s"])
    if a.first_on_b:
        for t in bts:
            t.join()
    time.sleep(2)  # the save of the finished sequence reaches the router's index
    fill = []

    def filler(i):
        fill.append(complete(a.url, key, p + f" Filler {i}: continue the story.", 400))

    ts = [threading.Thread(target=filler, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    time.sleep(1.5)
    out["x2"] = complete(a.url, key, q, a.max_tokens)
    print("x2", out["x2"]["status"], out["x2"]["wall_s"])
    for t in ts:
        t.join()
    out["fillers"] = [{"status": f["status"], "wall_s": f["wall_s"]} for f in fill]
    time.sleep(1)
    out["x3_cold"] = complete(a.url, key, long_prompt(a.prompt_tokens, nonce + "-cold") + " Question one: what is the first word of this text?", a.max_tokens)
    print("x3", out["x3_cold"]["status"], out["x3_cold"]["wall_s"])
    out["x1_equals_x2"] = out["x1"]["text"] == out["x2"]["text"]
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("x1 == x2:", out["x1_equals_x2"])


if __name__ == "__main__":
    main()
