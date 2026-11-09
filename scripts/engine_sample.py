"""Host sampler for engine runs, the same columns for the builder and KTransformers (ADR 0045, "Diagnosing the
KTransformers decode gap"). Reuses the readers of detector_sample.py.

probe-disk: python3 engine_sample.py probe-disk MODEL.gguf
            Prints MAJ:MIN of the block device the model's direct reads hit.
sample:     python3 engine_sample.py sample --pid PID --disk MAJ:MIN --out s.jsonl [--interval 1]
            Every second until PID exits: disk read bytes and ops, queue depth, major page faults of PID's process tree,
            per-CPU jiffies (/proc/stat), cgroup CPU usage, GPU util and memory, tree RSS, cgroup file and anon memory.
summarize:  python3 engine_sample.py summarize --samples s.jsonl --threads N (--builder-json H | --kt-json R) --out sum.json
            [--merge-into H]
            Splits the samples into a prefill and a decode window and writes {"host": {"prefill": {...}, "decode":
            {...}}, "env": {...}}. The builder windows come from the harness JSON (decode_windows wall clock, prefill
            before them), KTransformers' from the client result (t_start, t_first, t_end). --merge-into sets "host"
            and the new "env" keys in that JSON as well.
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detector_sample as ds  # noqa: E402


# the probe reads 2 GB; a device must show at least about half of it
PROBE_MIN_BYTES = 1_000_000_000


def probe_disk(a):
    """The device whose read counter moves most while reading 256 x 8 MB of the model with O_DIRECT."""

    def snap():
        return {f"{f[0]}:{f[1]}": int(f[5]) for f in (ln.split() for ln in ds.read("/proc/diskstats").splitlines()) if len(f) > 5}

    x = snap()
    r = subprocess.run(["dd", f"if={a.model}", "of=/dev/null", "bs=8M", "count=256", "iflag=direct"], capture_output=True)
    y = snap()
    best = max(y, key=lambda k: y[k] - x.get(k, 0), default=None)
    moved = (y[best] - x.get(best, 0)) * 512 if best else 0
    # a failed read, or no device that moved about the 2 GB read, gives no device ("none": the disk fields stay null)
    if r.returncode != 0 or moved < PROBE_MIN_BYTES:
        print(f"probe-disk: no device (dd rc {r.returncode}, largest read delta {moved} B): disk fields null",
              file=sys.stderr)
        print("none")
        return
    print(best)


def disk_stats(majmin):
    """(reads completed, IOs in flight now, time-weighted ms of in-flight IOs) of the device, from /proc/diskstats."""
    maj, mn = majmin.split(":")
    for line in ds.read("/proc/diskstats").splitlines():
        f = line.split()
        if len(f) > 13 and f[0] == maj and f[1] == mn:
            return int(f[3]), int(f[11]), int(f[13])
    return None, None, None


def cgroup_io():
    """Bytes and ops read by the container's cgroup (v2 io.stat), summed over devices; None without it."""
    rb = ro = 0
    seen = False
    for line in ds.read("/sys/fs/cgroup/io.stat").splitlines():
        for kv in line.split()[1:]:
            k, _, v = kv.partition("=")
            if k == "rbytes":
                rb += int(v)
                seen = True
            elif k == "rios":
                ro += int(v)
    return {"rbytes": rb, "rios": ro} if seen else None


def majflt(pids):
    """Major faults of the processes (the stat field covers all their threads)."""
    n = 0
    for p in pids:
        st = ds.read(f"/proc/{p}/stat")
        if st:
            n += int(st.rsplit(")", 1)[1].split()[9])
    return n


def tree_rss_kb(pids):
    kb = 0
    for p in pids:
        for line in ds.read(f"/proc/{p}/status").splitlines():
            if line.startswith("VmRSS:"):
                kb += int(line.split()[1])
    return kb


def stat_jiffies():
    """[(busy, total)] per CPU from /proc/stat."""
    out = []
    for line in ds.read("/proc/stat").splitlines():
        if line.startswith("cpu") and line[3:4].isdigit():
            v = [int(x) for x in line.split()[1:]]
            idle = v[3] + (v[4] if len(v) > 4 else 0)
            out.append((sum(v[:8]) - idle, sum(v[:8])))
    return out


def cgroup_mem():
    """File (page cache) and anon bytes of the container's cgroup, v2 then v1 memory.stat."""
    d = {}
    for line in ds.read("/sys/fs/cgroup/memory.stat").splitlines() or ds.read("/sys/fs/cgroup/memory/memory.stat").splitlines():
        k, _, v = line.partition(" ")
        d[k] = v
    f = d.get("file", d.get("total_cache"))
    an = d.get("anon", d.get("total_rss"))
    return {"file": int(f) if f else None, "anon": int(an) if an else None}


def gpu_util():
    try:
        out = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        u, m = (x.strip() for x in out.split(","))
        return {"util": int(u), "mem_mib": int(m)}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def sample(a):
    with open(a.out, "w") as out:
        while os.path.exists(f"/proc/{a.pid}"):
            pids = ds.tree(a.pid)
            ops, inflight, weighted_ms = disk_stats(a.disk) if a.disk != "none" else (None, None, None)
            rb = ds.disk_read_bytes(a.disk) if a.disk != "none" else None
            rec = {"t": time.time(), "disk_read_bytes": rb, "disk_read_ops": ops,
                   "disk_inflight": inflight, "disk_weighted_ms": weighted_ms,
                   "cg_io": cgroup_io(), "majflt": majflt(pids), "cpus": stat_jiffies(),
                   "cg_cpu_usec": ds.cgroup_cpu_usec(), "own_cpu_s": ds.proc_cpu_s(pids),
                   "rss_kb": tree_rss_kb(pids), "cg_mem": cgroup_mem(), "gpu": gpu_util()}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            time.sleep(a.interval)


def load_samples(path):
    rows = []
    for line in ds.read(path).splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def delta(rows, key):
    """Last minus first of a counter over the rows (None if missing)."""
    v = [r[key] for r in rows if r.get(key) is not None]
    return v[-1] - v[0] if len(v) >= 2 else None


def phase(rows, t0, t1, tokens, threads):
    """Summary of one window. Counters use the samples bracketing it, so a 1 s sample grid costs at most one interval."""
    inside = [r for r in rows if t0 <= r["t"] <= t1]
    before = [r for r in rows if r["t"] < t0]
    after = [r for r in rows if r["t"] > t1]
    win = ([before[-1]] if before else []) + inside + ([after[0]] if after else [])
    if len(win) < 2:
        return {"seconds": t1 - t0, "tokens": tokens, "samples": len(inside), "error": "too few samples"}
    dur = win[-1]["t"] - win[0]["t"]
    rb, ro, mf = delta(win, "disk_read_bytes"), delta(win, "disk_read_ops"), delta(win, "majflt")
    wms = delta(win, "disk_weighted_ms")
    infl = [r["disk_inflight"] for r in win if r.get("disk_inflight") is not None]
    cg = delta(win, "cg_cpu_usec")
    own = delta(win, "own_cpu_s")
    # busy share of the CPUs the engine may use: cgroup CPU time over wall time times threads, else the tree's own
    busy = (cg / 1e6 if cg is not None else own) / (dur * threads) * 100 if threads and dur else None
    # host view: busiest `threads` CPUs of /proc/stat (the container may see every host core)
    shares = []
    first, last = win[0].get("cpus") or [], win[-1].get("cpus") or []
    for (b0, t0j), (b1, t1j) in zip(first, last):
        if t1j > t0j:
            shares.append((b1 - b0) / (t1j - t0j) * 100)
    top = sorted(shares, reverse=True)[:threads] if threads else shares
    gpu = [r["gpu"]["util"] for r in inside if r.get("gpu")]
    gmem = [r["gpu"]["mem_mib"] for r in inside if r.get("gpu")]
    cgio = [r["cg_io"] for r in (win[0], win[-1])]
    cg_rb = cgio[1]["rbytes"] - cgio[0]["rbytes"] if all(cgio) else None
    n = max(tokens, 1)
    rss = [r["rss_kb"] for r in win]
    fil = [r["cg_mem"]["file"] for r in win if r.get("cg_mem") and r["cg_mem"]["file"] is not None]
    an = [r["cg_mem"]["anon"] for r in win if r.get("cg_mem") and r["cg_mem"]["anon"] is not None]
    return {
        "seconds": dur, "tokens": tokens, "samples": len(inside),
        "read_bytes": rb, "read_ops": ro, "cgroup_read_bytes": cg_rb,
        "read_bytes_per_token": rb / n if rb is not None else None,
        "read_mb_s": rb / dur / 1e6 if rb is not None and dur else None,
        # mean queue depth = weighted IO milliseconds per wall millisecond (all IOs of the device, not only ours)
        "disk_queue_depth_mean": wms / (dur * 1e3) if wms is not None and dur else None,
        "disk_inflight_max": max(infl) if infl else None,
        "majflt": mf, "majflt_per_token": mf / n if mf is not None else None,
        "cpu_busy_pct": busy,
        "cpu_busy_pct_top_cpus": sum(top) / len(top) if top else None,
        "gpu_busy_pct": sum(gpu) / len(gpu) if gpu else None,
        "gpu_mem_max_mib": max(gmem) if gmem else None,
        "rss_max_gb": max(rss) / 1e6 if rss else None,
        "cache_file_gb_max": max(fil) / 1e9 if fil else None,
        "anon_gb_max": max(an) / 1e9 if an else None,
    }


def windows_builder(path):
    """(prefill window, decode window, tokens) of the first run in a harness JSON."""
    run = json.loads(ds.read(path))["runs"][0]
    w = run["decode_windows"]
    d0, d1 = w[0]["start_unix_s"], w[-1]["end_unix_s"]
    # the hot set refresh sits between prefill and decode
    p1 = d0 - run.get("hot_refresh_s", 0.0)
    p0 = p1 - run["prefill_s"]
    return (p0, p1, run["prefill_tokens"]), (d0, d1, run["decode_tokens"])


def windows_kt(path):
    r = json.loads(ds.read(path))
    n = r["decode_tokens"]
    return (r["t_start"], r["t_first"], r["prompt_tokens"]), (r["t_first"], r["t_end"], max(n - 1, 0))


def env_info():
    flags = sorted({w for ln in ds.read("/proc/cpuinfo").splitlines() if ln.startswith("flags") for w in ln.split()
                    if w.startswith("avx512")})
    model = next((ln.split(":", 1)[1].strip() for ln in ds.read("/proc/cpuinfo").splitlines() if ln.startswith("model name")), None)
    return {"cpu_model": model, "cpu_avx512_flags": flags}


def summarize(a):
    rows = load_samples(a.samples)
    pre, dec = windows_builder(a.builder_json) if a.builder_json else windows_kt(a.kt_json)
    out = {"host": {"prefill": phase(rows, pre[0], pre[1], pre[2], a.threads),
                    "decode": phase(rows, dec[0], dec[1], dec[2], a.threads)},
           "env": env_info()}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out["host"]))
    if a.merge_into:
        d = json.loads(ds.read(a.merge_into))
        d["host"] = out["host"]
        d.setdefault("env", {}).update(out["env"])
        json.dump(d, open(a.merge_into, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe-disk")
    p.add_argument("model")
    s = sub.add_parser("sample")
    s.add_argument("--pid", type=int, required=True)
    s.add_argument("--disk", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--interval", type=float, default=1.0)
    m = sub.add_parser("summarize")
    m.add_argument("--samples", required=True)
    m.add_argument("--threads", type=int, required=True)
    g = m.add_mutually_exclusive_group(required=True)
    g.add_argument("--builder-json")
    g.add_argument("--kt-json")
    m.add_argument("--out", required=True)
    m.add_argument("--merge-into")
    a = ap.parse_args()
    {"probe-disk": probe_disk, "sample": sample, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
