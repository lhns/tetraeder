#!/usr/bin/env python3
"""Closed-loop streaming chat load for the Phase 6 rehearsal (ADR 0091, Case F), standard library only.

Each of --concurrency workers sends one request after another until --duration seconds have passed, then the
in-flight ones finish. One JSON line per request: submit, first token and end as epoch seconds, HTTP status,
chunks received, finish reason, an error string (transport error, an error event in the stream, a missing
[DONE]) and the chunks that arrived after an error event (must be 0). The API key comes from --key-file.

    python3 scripts/phase6_load.py http://127.0.0.1:8080 --key-file keys.txt --out load.jsonl
    python3 scripts/phase6_load.py URL --key-file F --calibrate --prompt-tokens 512 --meta-out cal.json

`--calibrate` sizes the prompt through the head's /tokenize (the rehearsal's 512-token prompt came out at 395) and writes
{"chars_per_token", "prompt_tokens"}; the load then takes `--chars-per-token` from it.

`--fire-peer HOST:PORT` (Case F, scripts/phase6_trigger.py): FIRE_AFTER seconds into the load, once the trigger on box B reports requests
in flight on B, the load fires the fault over that direct connection and writes the record (send, ack, middle, B's reading) to --fire-out.
"""

import argparse
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse

FILLER = "The quick brown fox jumps over the lazy dog while the engineers measure a cluster of rented machines. "
CHARS_PER_TOKEN = 4.0


def prompt(tokens: int, nonce: str, cpt: float = CHARS_PER_TOKEN) -> str:
    head = f"[{nonce}] "
    n = max(int(tokens * cpt) - len(head), 0)
    return head + (FILLER * (n // len(FILLER) + 1))[:n]


def token_count(host, port, key, text, timeout=60):
    """Tokens of `text` by the head's own tokenizer (POST /tokenize, llama.cpp's form)."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    conn.request("POST", "/tokenize", json.dumps({"content": text}), headers)
    r = conn.getresponse()
    data = r.read()
    conn.close()
    if r.status != 200:
        raise RuntimeError(f"/tokenize answered {r.status}: {data[:200]!r}")
    return len(json.loads(data)["tokens"])


def calibrate(host, port, key, target, cpt=CHARS_PER_TOKEN, tol=0.01, rounds=5):
    """Characters per token at which a prompt of `target` tokens is within `tol`; returns (cpt, tokens)."""
    n = 0
    for _ in range(rounds):
        n = token_count(host, port, key, prompt(target, "calibrate-0", cpt))
        if abs(n / target - 1.0) <= tol:
            break
        cpt *= target / n
    return cpt, n


def one_request(host, port, key, model, text, max_tokens, timeout, ignore_eos=False):
    """Returns the record of one streamed request (times in epoch seconds)."""
    rec = {"t_submit": time.time(), "t_first": None, "t_end": None, "status": None, "chunks": 0,
           "finish": None, "error": None, "done": False, "after_error": 0}
    body = json.dumps({"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0.7, **({"ignore_eos": True} if ignore_eos else {}),
                       "messages": [{"role": "user", "content": text}]})
    conn = None
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        conn.request("POST", "/v1/chat/completions", body, headers)
        resp = conn.getresponse()
        rec["status"] = resp.status
        if resp.status != 200:
            rec["error"] = f"http {resp.status}: {resp.read(300).decode('utf-8', 'replace')}"
            return rec
        errored = False
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                rec["done"] = True
                break
            try:
                ev = json.loads(data)
            except ValueError:
                rec["error"] = "malformed event"
                continue
            if errored:
                rec["after_error"] += 1
            if "error" in ev:
                errored = True
                rec["error"] = json.dumps(ev["error"])[:300]
                continue
            for ch in ev.get("choices", []):
                if ch.get("finish_reason"):
                    rec["finish"] = ch["finish_reason"]
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    rec["chunks"] += 1
                    if rec["t_first"] is None:
                        rec["t_first"] = time.time()
        if not rec["done"] and rec["error"] is None:
            rec["error"] = "stream ended without [DONE]"
    except Exception as e:  # noqa: BLE001 - every failure is a result
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        rec["t_end"] = time.time()
        if conn is not None:
            conn.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--key-file")
    ap.add_argument("--model", default="x")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--duration", type=float, default=180)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--ignore-eos", action="store_true", help="every request runs to --max-tokens (the loopback fixture would stop early)")
    ap.add_argument("--out", help="one JSON line per request (required unless --calibrate)")
    ap.add_argument("--label", default="load")
    ap.add_argument("--chars-per-token", type=float, default=CHARS_PER_TOKEN)
    ap.add_argument("--calibrate", action="store_true", help="size the prompt through /tokenize, write --meta-out, send nothing")
    ap.add_argument("--meta-out", help="with --calibrate: {chars_per_token, prompt_tokens}")
    ap.add_argument("--fire-peer", help="HOST:PORT of the fault trigger on box B")
    ap.add_argument("--fire-token-file")
    ap.add_argument("--fire-mode", choices=("worker", "daemon", "stop"))
    ap.add_argument("--fire-after", type=float, default=60.0, help="seconds after the start of the load")
    ap.add_argument("--fire-out")
    ap.add_argument("--fire-stop-for", type=float, default=60.0)
    ap.add_argument("--fire-restore-after", type=float, default=40.0)
    ap.add_argument("--fire-poll-s", type=float, default=0.2)
    a = ap.parse_args()
    if a.fire_peer and not (a.fire_token_file and a.fire_mode and a.fire_out):
        ap.error("--fire-peer needs --fire-token-file, --fire-mode and --fire-out")
    if not a.calibrate and not a.out:
        ap.error("--out is required")
    u = urllib.parse.urlparse(a.url)
    key = ""
    if a.key_file:
        with open(a.key_file) as f:
            key = next((x.strip() for x in f if x.strip() and not x.startswith("#")), "")
    if a.calibrate:
        cpt, n = calibrate(u.hostname, u.port or 80, key, a.prompt_tokens)
        text = json.dumps({"chars_per_token": cpt, "prompt_tokens": n, "target": a.prompt_tokens})
        if a.meta_out:
            with open(a.meta_out, "w") as f:
                f.write(text + "\n")
        print(text)
        return
    start = time.time()
    lock = threading.Lock()
    counter = [0]
    out = open(a.out, "a", buffering=1)

    def worker(w):
        while time.time() - start < a.duration:
            with lock:
                counter[0] += 1
                n = counter[0]
            rec = one_request(u.hostname, u.port or 80, key, a.model, prompt(a.prompt_tokens, f"{a.label}-{n}", a.chars_per_token),
                              a.max_tokens, a.timeout, a.ignore_eos)
            rec.update({"n": n, "worker": w, "label": a.label})
            with lock:
                out.write(json.dumps(rec) + "\n")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(a.concurrency)]
    if a.fire_peer:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import phase6_trigger as trig  # next to this file

        threads.append(threading.Thread(target=trig.fire_when_in_flight, args=(
            a.fire_peer, trig.read_token(a.fire_token_file), a.fire_mode, a.label,
            {"stop_for": a.fire_stop_for, "restore_after": a.fire_restore_after, "poll_s": a.fire_poll_s},
            start, a.fire_after, a.fire_out)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out.close()
    print(f"phase6_load {a.label}: {counter[0]} requests in {time.time() - start:.1f} s")


if __name__ == "__main__":
    main()
