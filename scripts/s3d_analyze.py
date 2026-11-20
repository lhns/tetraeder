"""Analysis of the s3d traces (ADR 0080 "Not decided here"): per op kind and rank, host time against device time,
and the GPU clocks logged beside decode runs.

    python3 scripts/s3d_analyze.py <session dir>

Trace lines: {"step":N,"sync":bool,"ev":[[kind,rank,peer,ns],...]}; kinds upload, compute, all_reduce, copy, sync,
logits, step (whole step, host). Steps of the most common event signature are used (the prefill or decode steps of
the timed runs; warmups share the signature and are kept).
"""

import collections
import csv
import glob
import json
import os
import re
import statistics
import sys


def load(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def common(steps):
    by = collections.defaultdict(list)
    for s in steps:
        by[tuple((e[0], e[1], e[2]) for e in s["ev"])].append(s)
    return max(by.values(), key=len) if by else []


def per_step(group):
    """{(kind, rank): median over steps of the summed ns}, {(kind, rank): median ns per call}, calls per step."""
    tot = collections.defaultdict(list)
    one = collections.defaultdict(list)
    calls = collections.defaultdict(int)
    for s in group:
        acc = collections.defaultdict(int)
        n = collections.defaultdict(int)
        for kind, rank, _peer, ns in s["ev"]:
            acc[(kind, rank)] += ns
            n[(kind, rank)] += 1
            one[(kind, rank)].append(ns)
        for k, v in acc.items():
            tot[k].append(v)
        calls = n
    return ({k: statistics.median(v) for k, v in tot.items()}, {k: statistics.median(v) for k, v in one.items()},
            dict(calls))


def report(path):
    steps = load(path)
    g = common(steps)
    if not g:
        return None
    tot, one, calls = per_step(g)
    return {"steps": len(g), "sync": g[0]["sync"], "tot_us": {k: v / 1e3 for k, v in tot.items()},
            "one_us": {k: v / 1e3 for k, v in one.items()}, "calls": calls}


def show(name, r):
    print(f"\n{name}: {r['steps']} steps, sync={r['sync']}")
    for k in sorted(r["tot_us"]):
        print(f"  {k[0]:10} rank {k[1]}  calls {r['calls'][k]:4}  sum {r['tot_us'][k]:10.1f} us  per call {r['one_us'][k]:8.1f} us")


def clocks(path):
    """Per GPU: median and min of clocks.sm, share of samples in each pstate, and the sample count."""
    rows = collections.defaultdict(list)
    with open(path) as f:
        rd = csv.reader(f)
        next(rd, None)
        for r in rd:
            if len(r) < 5:
                continue
            try:
                rows[int(r[1])].append((r[0].strip(), int(r[2].split()[0]), r[4].strip()))
            except (ValueError, IndexError):
                continue
    out = {}
    for gpu, v in sorted(rows.items()):
        sm = [x[1] for x in v]
        ps = collections.Counter(x[2] for x in v)
        out[gpu] = {"n": len(v), "sm_median": statistics.median(sm), "sm_min": min(sm), "sm_max": max(sm),
                    "sm_p10": sorted(sm)[len(sm) // 10], "pstate": dict(ps)}
    return out


def transports(path):
    """Lines of an NCCL_DEBUG log that name a transport per channel ("... via P2P/CUMEM"), and a count per transport."""
    lines, kinds = [], collections.Counter()
    with open(path, errors="replace") as f:
        for ln in f:
            m = re.search(r" via (\S+)", ln)
            if m:
                lines.append(ln.rstrip())
                kinds[m.group(1)] += 1
    return lines, kinds


def pp_ts(path):
    """Median pp tok/s of a bench json (entries with a prompt and no generation)."""
    with open(path) as f:
        rows = json.load(f)
    ts = [x for r in rows if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0 for x in r.get("samples_ts") or [r["avg_ts"]]]
    return statistics.median(ts) if ts else None


def ratios(d):
    """{(model, variant): {mb: tok/s}} from <model>-tp2pp2-mb<N>-pp-<variant>.json; variant plain is the default run."""
    out = collections.defaultdict(dict)
    for p in glob.glob(os.path.join(d, "*-tp2pp2-mb*-pp-*.json")):
        m = re.match(r"(.+)-tp2pp2-mb(\d+)-pp-(.+)\.json$", os.path.basename(p))
        if not m or os.path.exists(p[:-5] + ".rc") and open(p[:-5] + ".rc").read().strip() != "0":
            continue
        try:
            v = pp_ts(p)
        except (ValueError, KeyError, OSError):
            continue
        if v:
            out[(m.group(1), m.group(3))][int(m.group(2))] = v
    return out


def main(d):
    for p in sorted(glob.glob(os.path.join(d, "*.nccl.log")) or glob.glob(os.path.join(d, "*-nccldebug.log"))):
        lines, kinds = transports(p)
        print(f"\n{os.path.basename(p)}: {len(lines)} transport lines, per transport {dict(kinds)}")
        for ln in lines:
            print("  " + ln)
    lib = os.path.join(d, "env", "nccl-lib.txt")
    if os.path.exists(lib):
        print("\nenv/nccl-lib.txt")
        print("".join("  " + ln for ln in open(lib, errors="replace")))
    rt = ratios(d)
    if rt:
        print("\npp tok/s mb4 / mb1 per variant (plain = default, internal = GGML_CUDA_ALLREDUCE=internal)")
        for (model, variant), v in sorted(rt.items()):
            if 1 in v and 4 in v:
                print(f"  {model:15} {variant:16} mb4 {v[4]:9.1f}  mb1 {v[1]:9.1f}  ratio {v[4] / v[1]:.3f}")
        for (model, variant), v in sorted(rt.items()):
            if 1 not in v:
                print(f"  {model:15} {variant:16} mb4 {v[4]:9.1f}  (mb4 only)")
    for p in sorted(glob.glob(os.path.join(d, "*.trace.jsonl"))):
        r = report(p)
        if r:
            show(os.path.basename(p), r)
    for p in sorted(glob.glob(os.path.join(d, "*.clocks.csv"))):
        print(f"\n{os.path.basename(p)}")
        for gpu, c in clocks(p).items():
            print(f"  gpu {gpu}: n {c['n']} sm median {c['sm_median']} p10 {c['sm_p10']} min {c['sm_min']} max {c['sm_max']} pstate {c['pstate']}")


if __name__ == "__main__":
    main(sys.argv[1])
