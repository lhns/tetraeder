#!/usr/bin/env python3
"""Join the admit lines of a tetraeder server log (RUST_LOG=tetraeder::step=debug) with a gateway_replay JSON.

Usage: ttft_join.py REPLAY.json SERVER.log [--out joined.json] [--filter-log FILTERED.log]
Concurrency 1: the n-th admitted request is the n-th replayed request with a real admission, so the join walks both
in order and checks the prompt length (usage input + cache read against `prompt=`). Prints the 15 slowest hits (common
prefix of 1024 tokens or more), the 10 slowest misses (less) (common < 1024 tokens) and totals. The filtered log keeps the admit lines and WARN/ERROR lines only.
"""
import argparse
import json
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ADMIT = re.compile(
    r"admit seq=(\d+) queued_ms=([\d.]+) place_ms=([\d.]+) save_ms=([\d.]+) prompt=(\d+) common=(\d+) "
    r"source=(\S+) live=(\d+) pool_spare=(\d+) pool_fallback=(\d+) pool_ready=(\d+)"
)


def admits(path):
    out, keep = [], []
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = ANSI.sub("", raw.rstrip("\n"))
        m = ADMIT.search(line)
        if m:
            g = m.groups()
            out.append(dict(seq=int(g[0]), queued_ms=float(g[1]), place_ms=float(g[2]), save_ms=float(g[3]),
                            prompt=int(g[4]), common=int(g[5]), source=g[6], live=int(g[7]),
                            pool_spare=int(g[8]), pool_fallback=int(g[9]), pool_ready=int(g[10])))
            keep.append(line)
        elif re.search(r"\b(WARN|ERROR)\b", line):
            keep.append(line)
    return out, keep


def prompt_total(u):
    if not u:
        return None
    return (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)


def pct(v, p):
    v = sorted(v)
    import math
    return v[min(max(math.ceil(p * len(v)) - 1, 0), len(v) - 1)] if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay")
    ap.add_argument("log")
    ap.add_argument("--out")
    ap.add_argument("--filter-log")
    ap.add_argument("--llama", help="llama-server replay JSON of the same round: its TTFT is printed per request")
    a = ap.parse_args()
    res = json.load(open(a.replay))["results"]
    adm, keep = admits(a.log)
    if a.filter_log:
        open(a.filter_log, "w").write("\n".join(keep) + "\n")
    llama = {r["index"]: r.get("ttft_s") for r in json.load(open(a.llama))["results"]} if a.llama else {}
    rows, j, mism = [], 0, 0
    for r in res:
        row = dict(index=r["index"], llama_ttft_s=llama.get(r["index"]), status=r.get("status"), ttft_s=r.get("ttft_s"), total_s=r.get("total_s"),
                   usage=r.get("usage"))
        tot = prompt_total(r.get("usage"))
        # next admit whose prompt matches the request; a request that failed before admission has none
        k = j
        while k < len(adm) and tot is not None and abs(adm[k]["prompt"] - tot) > 16 and k < j + 3:
            k += 1
        if j < len(adm) and tot is not None and abs(adm[k if k < len(adm) else j]["prompt"] - tot) <= 16:
            row["admit"] = adm[k]
            j = k + 1
        elif j < len(adm) and r.get("status") == 200:
            row["admit"] = adm[j]
            row["admit_mismatch"] = True
            mism += 1
            j += 1
        rows.append(row)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
    ok = [x for x in rows if x["status"] == 200 and x.get("ttft_s") is not None]
    t = [x["ttft_s"] for x in ok]
    print(f"requests {len(rows)}, status 200 {len(ok)}, admits {len(adm)}, joined {sum(1 for x in rows if 'admit' in x)}, "
          f"mismatches {mism}")
    print(f"ttft median {statistics.median(t):.3f} s, p90 {pct(t, 0.9):.3f} s")
    with_a = [x for x in ok if "admit" in x]
    print(f"sum ttft {sum(t):.1f} s; sum save_ms {sum(x['admit']['save_ms'] for x in with_a)/1e3:.2f} s; "
          f"sum place_ms {sum(x['admit']['place_ms'] for x in with_a)/1e3:.2f} s; "
          f"sum queued_ms {sum(x['admit']['queued_ms'] for x in with_a)/1e3:.2f} s")
    src = {}
    for x in with_a:
        src[x["admit"]["source"]] = src.get(x["admit"]["source"], 0) + 1
    print("sources", src)
    if llama:
        both = [x for x in ok if x["llama_ttft_s"] is not None]
        lt = [x["llama_ttft_s"] for x in both]
        print(f"llama-server same round, same {len(both)} requests: ttft median {statistics.median(lt):.3f} s, "
              f"p90 {pct(lt, 0.9):.3f} s, sum {sum(lt):.1f} s")

    def line(x):
        ad, u = x["admit"], x["usage"] or {}
        return (f"#{x['index']:>3} prompt {ad['prompt']:>6} cached {u.get('cache_read_input_tokens', 0):>6} "
                f"new {u.get('input_tokens', 0):>6} common {ad['common']:>6} {ad['source']:<8} "
                f"queued {ad['queued_ms']:>7.1f} place {ad['place_ms']:>7.1f} save {ad['save_ms']:>6.1f} "
                f"fb {ad['pool_fallback']} ttft {x['ttft_s']:.2f} ttft-place {x['ttft_s'] - ad['place_ms'] / 1e3:.2f} llama "
                f"{x['llama_ttft_s'] if x['llama_ttft_s'] is None else round(x['llama_ttft_s'], 2)}")

    hits = sorted((x for x in with_a if x["admit"]["common"] >= 1024), key=lambda x: -x["ttft_s"])[:15]
    miss = sorted((x for x in with_a if x["admit"]["common"] < 1024), key=lambda x: -x["ttft_s"])[:10]
    print("slowest hits")
    for x in hits:
        print(" ", line(x))
    print("slowest misses (common < 1024 tokens)")
    for x in miss:
        print(" ", line(x))
    bad = [x for x in rows if x["status"] != 200]
    print("non-200:", [(x["index"], x["status"]) for x in bad])
    return 0


sys.exit(main())
