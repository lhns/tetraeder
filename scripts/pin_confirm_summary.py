"""Summary of a pin-confirm session (scripts/gpu_pin_confirm.sh): summary.md and summary.json in the results dir.

Usage: python -I scripts/pin_confirm_summary.py <results dir>
Tables only, no interpretation: test results, step shapes, the c1/c4 probe with ratios to llama-server, and per
tetraeder probe run the admit/step log lines (save_ms per admitting step, gap_ms distribution, prompts per prefill step).
"""

import glob
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

ANSI = re.compile(r"\x1b\[[0-9;]*m")
KV = re.compile(r"(\w+)=(-?[\d.]+)")


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def fmt(x, p=1):
    return "-" if x is None else f"{x:.{p}f}"


def ratio(a, b):
    return "-" if a is None or not b else f"{a / b:.3f}"


def read(path):
    try:
        return open(path, errors="replace").read()
    except OSError:
        return ""


def tests(out, w):
    w += ["## Test stages", ""]
    for name, p in (("smoke6 rc", "gpu-smoke5/rc6.json"), ("smoke5 build/tests rc", "gpu-smoke5/rc.json"),
                    ("real-batched rc", "real-batched/rc.json")):
        t = read(os.path.join(out, p)).strip()
        w.append(f"- {name}: {t or 'missing'}")
    for f in ("gpu-smoke5/mul_mat_id_skip.log", "gpu-smoke5/gpu_multi_seq.log", "gpu-smoke5/gpu_dense.log",
              "gpu-smoke5/executor_gpu.log", "real-batched/q8b.log", "real-batched/q30b.log"):
        t = read(os.path.join(out, f))
        res = re.findall(r"^test result: .*$", t, re.M)
        tl = re.findall(r"^test (\S+) \.\.\. (\w+)$", t, re.M)
        w.append(f"- {f}: " + ("; ".join(res) if res else "no result line"))
        for n, r in tl:
            if r != "ok":
                w.append(f"  - {n}: {r}")
    for m in ("q8b", "q30b"):
        t = read(os.path.join(out, f"real-batched/{m}.log"))
        lines = [ANSI.sub("", l) for l in t.splitlines()
                 if re.search(r"worst .* of the largest logit|bound max|argmax .* against|panicked|differs", l)]
        w += ["", f"### real-batched {m} (verbatim lines)", ""] + [f"    {l}" for l in lines[:60]]
    w.append("")


def shapes(out, w, js):
    d = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(out, "shapes-r*.txt"))):
        for m in re.finditer(r"SHAPE engine=(\S+) shape=(\S+) ms=([\d.]+)", read(f)):
            d[(m[2], m[1])].append(float(m[3]))
    w += ["## step_shapes on CUDA (Qwen3-8B Q4_K_M, all layers on GPU), ms per step shape", "",
          "shape | engine | per round | min | median", "--- | --- | --- | --- | ---"]
    for (s, e), v in sorted(d.items()):
        w.append(f"{s} | {e} | {', '.join(f'{x:.1f}' for x in v)} | {min(v):.1f} | {statistics.median(v):.1f}")
        js.setdefault("shapes", {})[f"{s}|{e}"] = v
    w.append("")


def gpu_hex(out, run):
    t = read(os.path.join(out, f"gpu-{run}.txt"))
    b = re.search(r"before: (\S+)", t)
    a = re.search(r"after: (\S+)", t)
    return (b[1] if b else "?") + "/" + (a[1] if a else "?")


def probe(out, w, js):
    rows = {}
    for f in sorted(glob.glob(os.path.join(out, "probe-*-c[14].json"))):
        m = re.match(r"probe-(\w+?)-(llama-server|tetraeder)-r(\d+)-c([14])\.json", os.path.basename(f))
        if not m:
            continue
        s = json.load(open(f))["summary"]
        rows[(m[1], int(m[3]), m[2], int(m[4]))] = s
    w += ["## Probe (D2a workload: 512 prompt tokens, 128 max tokens, 8 requests; seeds equal across engines per round)", ""]
    for model in sorted({k[0] for k in rows}):
        for c in (1, 4):
            w += [f"### {model} c{c}", "",
                  "round | engine | throughput tps | ttft median s | ttft p90 s | decode tps median | errors | throttle before/after",
                  "--- | --- | --- | --- | --- | --- | --- | ---"]
            by = {}
            for rd in sorted({k[1] for k in rows if k[0] == model}):
                for eng in ("llama-server", "tetraeder"):
                    s = rows.get((model, rd, eng, c))
                    if not s:
                        continue
                    by[(rd, eng)] = s["throughput_tps"]
                    w.append(f"r{rd} | {eng} | {s['throughput_tps']:.1f} | {s['ttft_median_s']:.3f} | {s['ttft_p90_s']:.3f} | "
                             f"{s['decode_tps_median']:.1f} | {s['errors']} | {gpu_hex(out, f'{model}-{eng}-r{rd}')}")
            w += ["", "round | tetraeder / llama-server throughput", "--- | ---"]
            rs = []
            for rd in sorted({k[0] for k in by}):
                r = by.get((rd, "tetraeder")), by.get((rd, "llama-server"))
                w.append(f"r{rd} | {ratio(*r)}")
                if r[0] and r[1]:
                    rs.append(r[0] / r[1])
            ta = [v for (rd, e), v in by.items() if e == "tetraeder"]
            la = [v for (rd, e), v in by.items() if e == "llama-server"]
            if ta and la:
                w.append(f"mean | {ratio(statistics.mean(ta), statistics.mean(la))}")
                js.setdefault("probe_ratio", {})[f"{model}|c{c}"] = {"per_round": rs, "mean_of_means": statistics.mean(ta) / statistics.mean(la)}
            w.append("")


def stats_row(label, xs):
    return (f"{label} | {len(xs)} | {fmt(pct(xs, 50), 2)} | {fmt(pct(xs, 90), 2)} | {fmt(pct(xs, 99), 2)} | "
            f"{fmt(max(xs) if xs else None, 2)}")


def parse_log(path):
    """Steps with the admit lines that preceded them (admits are logged at placement, before the step)."""
    steps, pending = [], []
    for line in read(path).splitlines():
        line = ANSI.sub("", line)
        if " admit seq=" in line:
            pending.append({k: float(v) for k, v in KV.findall(line)})
        elif " step dec_seqs=" in line:
            st = {k: float(v) for k, v in KV.findall(line)}
            st["admits"], pending = pending, []
            steps.append(st)
    return steps, pending


def logs(out, w, js):
    w += ["## tetraeder probe logs (RUST_LOG=info,tetraeder::step=debug)", ""]
    for f in sorted(glob.glob(os.path.join(out, "server-*-tetraeder-r*.log"))):
        run = os.path.basename(f)[len("server-"):-len(".log")]
        steps, trailing = parse_log(f)
        if not steps:
            w += [f"### {run}", "", "no step lines", ""]
            continue
        adm = [s for s in steps if s["admits"]]
        pre = [s for s in steps if s.get("pre_seqs", 0) > 0]
        w += [f"### {run}: {len(steps)} steps, {sum(len(s['admits']) for s in steps)} admits, {len(adm)} admitting steps, "
              f"{len(pre)} prefill steps", "", "Per admitting step (admits logged since the previous step):", "",
              "step | admits | pre_seqs | pre_toks | save_ms (sum) | place_ms (sum) | queued_ms (max) | gap_ms | decode_ms",
              "--- | --- | --- | --- | --- | --- | --- | --- | ---"]
        for i, s in enumerate(steps):
            if not s["admits"]:
                continue
            a = s["admits"]
            w.append(f"{i} | {len(a)} | {int(s.get('pre_seqs', 0))} | {int(s.get('pre_toks', 0))} | "
                     f"{sum(x.get('save_ms', 0) for x in a):.2f} | {sum(x.get('place_ms', 0) for x in a):.2f} | "
                     f"{max(x.get('queued_ms', 0) for x in a):.2f} | {s.get('gap_ms', -1):.2f} | {s.get('decode_ms', 0):.2f}")
        gaps = lambda ss: [s["gap_ms"] for s in ss if s.get("gap_ms", -1) >= 0]
        non = [s for s in steps if not s["admits"]]
        w += ["", "gap_ms distribution (gap -1, no previous step, excluded):", "",
              "steps | n | p50 | p90 | p99 | max", "--- | --- | --- | --- | --- | ---",
              stats_row("all", gaps(steps)), stats_row("admitting", gaps(adm)), stats_row("other", gaps(non)), ""]
        cnt = Counter(int(s["pre_seqs"]) for s in pre)
        w += ["Prompts per prefill step (pre_seqs):", "", "prompts | prefill steps", "--- | ---"]
        w += [f"{k} | {v}" for k, v in sorted(cnt.items())]
        w.append("")
        js.setdefault("logs", {})[run] = {
            "steps": len(steps), "admits": sum(len(s["admits"]) for s in steps), "admitting_steps": len(adm),
            "prefill_prompts_hist": dict(cnt), "save_ms_per_admitting_step": [sum(x.get("save_ms", 0) for x in s["admits"]) for s in adm],
            "gap_ms_all": {"p50": pct(gaps(steps), 50), "p90": pct(gaps(steps), 90), "p99": pct(gaps(steps), 99)},
        }


def main():
    out = sys.argv[1]
    w, js = ["# Pin confirm summary", ""], {}
    env = read(os.path.join(out, "env.json"))
    if env:
        w += ["```", env.strip(), "```", ""]
    tests(out, w)
    shapes(out, w, js)
    probe(out, w, js)
    logs(out, w, js)
    open(os.path.join(out, "summary.md"), "w").write("\n".join(w) + "\n")
    json.dump(js, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print("\n".join(w))


main()
