"""llama.cpp side of the ADR 0043 speculative decoding pair: one llama-server run, one greedy completion.

Starts llama-server with the given args (with or without -md), tokenizes the prompt file with the server's vocab,
repeats it to --prompt-tokens tokens (as tests/exit_regime.rs does), sends one /completion of --decode tokens at
temperature 0 and writes the server's timings (prompt, decode, draft_n, draft_n_accepted) and the tokens as JSON.
Per-request speculative settings are compiled out of this llama-server (server-schema.cpp, `#if 0`), so a run
without a draft model is a separate server start.

usage: python3 spec_llama.py --server BIN --prompt-file F --prompt-tokens N --decode N --out F.json -- <server args>
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request


def post(url, body, timeout):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_healthy(base, proc, limit_s):
    t0 = time.time()
    while time.time() - t0 < limit_s:
        if proc.poll() is not None:
            sys.exit(f"llama-server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=5) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(2)
    sys.exit("llama-server not healthy in time")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--prompt-tokens", type=int, required=True)
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("server_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    args = [x for x in a.server_args if x != "--"]
    cmd = [a.server, "--host", "127.0.0.1", "--port", str(a.port), "-np", "1", "--no-webui"] + args
    print("cmd:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    base = f"http://127.0.0.1:{a.port}"
    try:
        load_s = wait_healthy(base, proc, 1800)
        text = open(a.prompt_file, encoding="utf-8").read()
        toks = post(base + "/tokenize", {"content": text, "add_special": False}, 600)["tokens"]
        toks = [t if isinstance(t, int) else t["id"] for t in toks]
        prompt = [toks[i % len(toks)] for i in range(a.prompt_tokens)]
        t0 = time.time()
        r = post(
            base + "/completion",
            {
                "prompt": prompt,
                "n_predict": a.decode,
                "temperature": 0,
                "ignore_eos": True,
                "cache_prompt": False,
                "return_tokens": True,
                "stream": False,
            },
            7200,
        )
        wall = time.time() - t0
        t = r.get("timings", {})
        dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
        out = {
            "engine": "llama.cpp llama-server",
            "note": a.note,
            "cmd": cmd,
            "load_s": load_s,
            "prompt_tokens": len(prompt),
            "wall_s": wall,
            "timings": t,
            "prefill_tok_s": t.get("prompt_per_second"),
            "decode_tok_s": t.get("predicted_per_second"),
            "decode_tokens": t.get("predicted_n"),
            "draft_n": dn,
            "draft_n_accepted": da,
            "acceptance": da / dn if dn else None,
            "tokens": r.get("tokens"),
        }
        json.dump(out, open(a.out, "w"), indent=1)
        print("result:", {k: out[k] for k in ("prefill_tok_s", "decode_tok_s", "draft_n", "acceptance")}, flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
