#!/usr/bin/env python3
"""One timed pp + tg request through the head, for the 30B split measurement (rehearsal 2, ADR 0091), standard library only.

The prompt is the first N tokens of a text file (cut by the head's own /tokenize), sent greedy through /v1/completions with
streaming. Times are the client's: pp = prompt tokens / time to the first streamed token (it includes the first decode step and
the HTTP round trip), tg = generated tokens after the first / (last chunk - first chunk). ignore_eos keeps the 128 tokens.
Prefix reuse is turned off on the head (`serve --no-prefix-reuse`); the admit lines (`common=`) show it.

    python3 scripts/phase6_split_round.py http://127.0.0.1:8080 --key-file K --prompt-file bench/prompts/exit_prompt.txt \
        --prompt-tokens 2048 --gen 128 --label r1 --out round-r1.json
    python3 scripts/phase6_split_round.py URL --key-file K --prompt-file F --prompt-tokens 2048 --cut-only --out prompt.json
"""

import argparse
import hashlib
import http.client
import json
import time
import urllib.parse


def call(host, port, key, path, body, timeout=900, stream=False):
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    conn.request("POST", path, json.dumps(body), headers)
    return conn, conn.getresponse()


def ntok(host, port, key, text):
    conn, r = call(host, port, key, "/tokenize", {"content": text}, 120)
    data = r.read()
    conn.close()
    if r.status != 200:
        raise RuntimeError(f"/tokenize {r.status}: {data[:200]!r}")
    return len(json.loads(data)["tokens"])


def cut_prompt(host, port, key, text, target):
    """The longest prefix of `text` (on a whitespace edge) with at most `target` tokens, then its token count."""
    lo, hi = 0, len(text)
    while hi - lo > 8:
        mid = (lo + hi) // 2
        if ntok(host, port, key, text[:mid]) <= target:
            lo = mid
        else:
            hi = mid
    s = text[:lo]
    return s, ntok(host, port, key, s)


def fnv1a64(b):
    h = 0xCBF29CE484222325
    for x in b:
        h = ((h ^ x) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--key-file")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--label", default="round")
    ap.add_argument("--cut-only", action="store_true", help="write the cut prompt's facts and send nothing")
    ap.add_argument("--prompt-out", help="write the cut prompt text here")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    u = urllib.parse.urlparse(a.url)
    host, port = u.hostname, u.port or 80
    key = ""
    if a.key_file:
        with open(a.key_file) as f:
            key = next((x.strip() for x in f if x.strip() and not x.startswith("#")), "")
    text = open(a.prompt_file, encoding="utf-8").read()
    prompt, n = cut_prompt(host, port, key, text, a.prompt_tokens)
    facts = {"prompt_tokens": n, "chars": len(prompt), "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
             "fnv1a64": fnv1a64(prompt.encode()), "source": a.prompt_file}
    if a.prompt_out:
        with open(a.prompt_out, "w", encoding="utf-8") as f:
            f.write(prompt)
    if a.cut_only:
        json.dump(facts, open(a.out, "w"), indent=2)
        print(json.dumps(facts))
        return
    body = {"model": "x", "prompt": prompt, "max_tokens": a.gen, "temperature": 0, "seed": 1, "stream": True,
            "ignore_eos": True, "stream_options": {"include_usage": True}}
    rec = {"label": a.label, "prompt": facts, "gen_asked": a.gen, "t_submit": time.time(), "t_first": None, "t_last": None,
           "chunks": 0, "usage": None, "error": None, "text": ""}
    conn, r = call(host, port, key, "/v1/completions", body)
    rec["status"] = r.status
    if r.status != 200:
        rec["error"] = r.read(300).decode("utf-8", "replace")
    else:
        while True:
            line = r.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            d = line[5:].strip()
            if d == b"[DONE]":
                break
            ev = json.loads(d)
            if ev.get("usage"):
                rec["usage"] = ev["usage"]
            for ch in ev.get("choices", []):
                if ch.get("text"):
                    now = time.time()
                    rec["chunks"] += 1
                    rec["text"] += ch["text"]
                    if rec["t_first"] is None:
                        rec["t_first"] = now
                    rec["t_last"] = now
    conn.close()
    rec["t_end"] = time.time()
    if rec["t_first"] is not None:
        ttft = rec["t_first"] - rec["t_submit"]
        rec["ttft_s"] = round(ttft, 4)
        rec["pp_tok_s"] = round(n / ttft, 2)
        k = (rec["usage"] or {}).get("completion_tokens", rec["chunks"])
        rec["gen_tokens"] = k
        span = rec["t_last"] - rec["t_first"]
        rec["tg_tok_s"] = round((k - 1) / span, 3) if span > 0 and k > 1 else None
        rec["wall_s"] = round(rec["t_end"] - rec["t_submit"], 3)
    rec["text"] = rec["text"][:200]
    json.dump(rec, open(a.out, "w"), indent=2)
    print(json.dumps({k: rec.get(k) for k in ("label", "ttft_s", "pp_tok_s", "gen_tokens", "tg_tok_s", "error")}))


if __name__ == "__main__":
    main()
