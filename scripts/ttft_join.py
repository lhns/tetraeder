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
ADMIT = re.compile(r"admit seq=(\d+) (.*)")
STEP_SAVE = re.compile(r"step save (.*)")
FIELD = re.compile(r"(\w+)=(\S+)")
INT_FIELDS = ("prompt", "common", "live", "pool_spare", "pool_fallback", "pool_ready")


def parse_admit(line):
    """The admit fields of a log line, or None. Fields are key=value, so new ones need no change here."""
    m = ADMIT.search(line)
    if not m:
        return None
    f = dict(FIELD.findall(m.group(2)))
    try:
        row = dict(seq=int(m.group(1)), queued_ms=float(f["queued_ms"]), place_ms=float(f["place_ms"]),
                   save_ms=float(f["save_ms"]), prompt=int(f["prompt"]), common=int(f["common"]),
                   source=f["source"])
        for k in INT_FIELDS[2:]:
            row[k] = int(f.get(k, 0))
    except (KeyError, ValueError):
        return None
    # Absent in logs from before the field existed.
    row["restore_ms"] = float(f["restore_ms"]) if "restore_ms" in f else None
    row["skipped"] = f.get("skipped")
    return row


def parse_step_save(line):
    """(save_ms, slots prefilling) of a `step save` line (a save inside a step by make_room, ADR 0074), or None."""
    m = STEP_SAVE.search(line)
    if not m:
        return None
    f = dict(FIELD.findall(m.group(1)))
    try:
        ms = float(f["save_ms"])
    except (KeyError, ValueError):
        return None
    return ms, [int(x) for x in f.get("prefilling", "").split(",") if x]


def admits(path):
    out, keep = [], []
    latest = {}  # slot -> its latest admit row
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = ANSI.sub("", raw.rstrip())
        row = parse_admit(line)
        step = None if row else parse_step_save(line)
        if row:
            out.append(row)
            latest[row["seq"]] = row
            keep.append(line)
        elif step:
            # The saves stall every sequence, but only a request still without its first token loses TTFT to it. The line
            # lists the slots prefilling at that moment, so the time counts against the latest admit of each of them
            # (every waiting request when several are; the log cannot say which one a stall "belongs" to).
            for slot in step[1]:
                if slot in latest:
                    latest[slot]["step_save_ms"] = latest[slot].get("step_save_ms", 0.0) + step[0]
            keep.append(line)
        elif re.search(r"\b(WARN|ERROR)\b", line):
            keep.append(line)
    return out, keep


def save_ms(ad):
    """Saves that delayed this request: its admission's plus those inside steps while it prefilled."""
    return ad["save_ms"] + ad.get("step_save_ms", 0.0)


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
    step_ms = sum(x["admit"].get("step_save_ms", 0.0) for x in with_a)
    in_step = f" (of which in-step {step_ms/1e3:.2f} s)" if step_ms else ""
    print(f"sum ttft {sum(t):.1f} s; sum save_ms {sum(save_ms(x['admit']) for x in with_a)/1e3:.2f} s{in_step}; "
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
                f"queued {ad['queued_ms']:>7.1f} place {ad['place_ms']:>7.1f} save {save_ms(ad):>6.1f} "
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


if __name__ == "__main__":
    sys.exit(main())
