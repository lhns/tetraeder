"""Host sampler for engine runs, the same columns for the builder and KTransformers (ADR 0045, "Diagnosing the
KTransformers decode gap"). Reuses the readers of detector_sample.py.

probe-disk: python3 engine_sample.py probe-disk MODEL.gguf
            Prints MAJ:MIN of the block device the model's direct reads hit.
sample:     python3 engine_sample.py sample --pid PID --disk MAJ:MIN --out s.jsonl [--interval 1]
            Every second until PID exits: disk read bytes and ops, queue depth, major page faults of PID's process tree,
            per-CPU jiffies (/proc/stat), cgroup CPU usage, GPU util and memory, tree RSS, cgroup file and anon memory.
summarize:  python3 engine_sample.py summarize --samples s.jsonl --threads N (--builder-json H | --kt-json R) --out sum.json
            [--merge-into H] [--curve-dir D | --curve FILE]
            Splits the samples into a prefill and a decode window and writes {"host": {"prefill": {...}, "decode":
            {...}}, "env": {...}}. The builder windows come from the harness JSON (decode_windows wall clock, prefill
            before them), KTransformers' from the client result (t_start, t_first, t_end). --merge-into sets "host"
            and the new "env" keys in that JSON as well. Each phase also gets io_some_share and io_full_share (I/O
            pressure stall time over wall time), disk_util (read_mb_s over the loaded disk curve's rate at the mean
            queue depth) and io_class (see io_class()). The curve is disk-curve-loaded-*.json in D (default: the
            directory of the merged or output JSON).
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detector_sample as ds  # noqa: E402


# the probe reads 2 GB; a device must show at least about half of it
PROBE_MIN_BYTES = 1_000_000_000

# Provisional thresholds of io_class (ADR 0056 addendum); tune them when more paired runs exist.
# Stall share (I/O pressure, or the builder's read wait over its step) from which a phase counts as I/O bound.
IO_BOUND_SHARE = 0.25
# Achieved read rate over the disk curve's rate at the same queue depth from which the disk itself is the limit.
DISK_UTIL_BOUND = 0.7
# Mean queue depth below which the reads are serial (one request at a time, or page faults), not too few.
SERIAL_QUEUE_DEPTH = 2.0


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


def io_pressure():
    """I/O pressure stall totals in usec of the container's cgroup (v2 io.pressure), else /proc/pressure/io.

    {"src", "some", "full"}; src is None when neither file has a "some" line."""
    for path in ("/sys/fs/cgroup/io.pressure", "/proc/pressure/io"):
        got = {}
        for line in ds.read(path).splitlines():
            kind = line.split(" ", 1)[0]
            if kind in ("some", "full") and "total=" in line:
                got[kind] = int(line.rsplit("total=", 1)[1])
        if "some" in got:
            return {"src": path, "some": got["some"], "full": got.get("full")}
    return {"src": None, "some": None, "full": None}


def cpu_pressure():
    """CPU pressure stall totals in usec (cgroup v2 cpu.pressure, else /proc/pressure/cpu); same shape as io_pressure."""
    for path in ("/sys/fs/cgroup/cpu.pressure", "/proc/pressure/cpu"):
        got = {}
        for line in ds.read(path).splitlines():
            kind = line.split(" ", 1)[0]
            if kind in ("some", "full") and "total=" in line:
                got[kind] = int(line.rsplit("total=", 1)[1])
        if "some" in got:
            return {"src": path, "some": got["some"], "full": got.get("full")}
    return {"src": None, "some": None, "full": None}


def cgroup_throttle():
    """CFS throttling counters of the container's cgroup: {"src", "periods", "throttled", "usec"}.

    v2 cpu.stat (nr_periods, nr_throttled, throttled_usec), else v1 cpu.stat (throttled_time in ns). src is None
    when no file has nr_periods (v2 without a cpu.max quota has none)."""
    for path in ("/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat", "/sys/fs/cgroup/cpu,cpuacct/cpu.stat"):
        d = {}
        for line in ds.read(path).splitlines():
            k, _, v = line.partition(" ")
            if v.strip().isdigit():
                d[k] = int(v)
        if "nr_periods" in d:
            usec = d.get("throttled_usec", d["throttled_time"] // 1000 if "throttled_time" in d else None)
            return {"src": path, "periods": d["nr_periods"], "throttled": d.get("nr_throttled"), "usec": usec}
    return {"src": None, "periods": None, "throttled": None, "usec": None}


def cpu_steal():
    """Host-wide steal and total jiffies from the aggregate cpu line of /proc/stat; None without it."""
    for line in ds.read("/proc/stat").splitlines():
        if line.startswith("cpu "):
            v = [int(x) for x in line.split()[1:]]
            if len(v) >= 8:
                return {"steal": v[7], "total": sum(v[:8])}
    return None


def cpu_mhz():
    """Mean "cpu MHz" over the CPUs of /proc/cpuinfo; None without them."""
    v = []
    for line in ds.read("/proc/cpuinfo").splitlines():
        if line.startswith("cpu MHz"):
            try:
                v.append(float(line.split(":", 1)[1]))
            except ValueError:
                pass
    return sum(v) / len(v) if v else None


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
                   "cg_io": cgroup_io(), "io_psi": io_pressure(), "majflt": majflt(pids), "cpus": stat_jiffies(),
                   "cpu_psi": cpu_pressure(), "cg_throttle": cgroup_throttle(), "cpu_steal": cpu_steal(),
                   "cpu_mhz": cpu_mhz(),
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
    used = cg / 1e6 if cg is not None else own
    busy = used / (dur * threads) * 100 if used is not None and threads and dur else None
    # host view: busiest `threads` CPUs of /proc/stat (the container may see every host core)
    shares = []
    first, last = win[0].get("cpus") or [], win[-1].get("cpus") or []
    for (b0, t0j), (b1, t1j) in zip(first, last):
        if t1j > t0j:
            shares.append((b1 - b0) / (t1j - t0j) * 100)
    top = sorted(shares, reverse=True)[:threads] if threads else shares
    gpu = [r["gpu"]["util"] for r in inside if r.get("gpu")]
    gmem = [r["gpu"]["mem_mib"] for r in inside if r.get("gpu")]
    psi0, psi1 = (r.get("io_psi") or {} for r in (win[0], win[-1]))
    same_src = psi0.get("src") is not None and psi0.get("src") == psi1.get("src")

    def stall(kind):
        if not same_src or psi0.get(kind) is None or psi1.get(kind) is None or not dur:
            return None
        return min(1.0, (psi1[kind] - psi0[kind]) / 1e6 / dur)

    cpsi0, cpsi1 = (r.get("cpu_psi") or {} for r in (win[0], win[-1]))
    cpu_same = cpsi0.get("src") is not None and cpsi0.get("src") == cpsi1.get("src")

    def cpu_stall(kind):
        if not cpu_same or cpsi0.get(kind) is None or cpsi1.get(kind) is None or not dur:
            return None
        return min(1.0, (cpsi1[kind] - cpsi0[kind]) / 1e6 / dur)

    th0, th1 = (r.get("cg_throttle") or {} for r in (win[0], win[-1]))
    th_same = th0.get("src") is not None and th0.get("src") == th1.get("src")

    def th_delta(k):
        return th1[k] - th0[k] if th_same and th0.get(k) is not None and th1.get(k) is not None else None

    periods, throttled, th_usec = th_delta("periods"), th_delta("throttled"), th_delta("usec")
    st0, st1 = (r.get("cpu_steal") for r in (win[0], win[-1]))
    steal = None
    if st0 and st1 and st1["total"] > st0["total"]:
        steal = (st1["steal"] - st0["steal"]) / (st1["total"] - st0["total"])
    mhz = [m for m in (r.get("cpu_mhz") for r in (win[0], win[-1])) if m is not None]

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
        "io_psi_src": psi0.get("src") if same_src else None,
        "io_some_share": stall("some"), "io_full_share": stall("full"),
        "cpu_psi_src": cpsi0.get("src") if cpu_same else None,
        "cpu_some_share": cpu_stall("some"), "cpu_full_share": cpu_stall("full"),
        "throttle_src": th0.get("src") if th_same else None,
        "throttled_share": throttled / periods if throttled is not None and periods else None,
        "throttled_ms_per_s": th_usec / 1e3 / dur if th_usec is not None and dur else None,
        "cpu_steal_share": steal,
        "cpu_mhz_mean": sum(mhz) / len(mhz) if mhz else None,
        "majflt": mf, "majflt_per_token": mf / n if mf is not None else None,
        "cpu_busy_pct": busy,
        "cpu_busy_pct_top_cpus": sum(top) / len(top) if top else None,
        "gpu_busy_pct": sum(gpu) / len(gpu) if gpu else None,
        "gpu_mem_max_mib": max(gmem) if gmem else None,
        "rss_max_gb": max(rss) / 1e6 if rss else None,
        "cache_file_gb_max": max(fil) / 1e9 if fil else None,
        "anon_gb_max": max(an) / 1e9 if an else None,
    }


def load_curve(path):
    """[(depth, bytes_per_s)] of a disk-curve JSON, sorted by depth; [] if unreadable."""
    try:
        pts = json.loads(ds.read(path))["points"]
        return sorted((float(p["depth"]), float(p["bytes_per_s"])) for p in pts)
    except (ValueError, KeyError, TypeError):
        return []


def find_curve(directory):
    """The newest disk-curve-loaded-*.json in the directory, or None."""
    found = sorted(glob.glob(os.path.join(directory, "disk-curve-loaded-*.json")))
    return found[-1] if found else None


def curve_capacity(depth, curve):
    """The curve's bytes/s at `depth` (linear between points, clamped to the curve's range); None without inputs."""
    if depth is None or not curve:
        return None
    d = min(max(depth, curve[0][0]), curve[-1][0])
    cap = curve[-1][1]
    for (d0, b0), (d1, b1) in zip(curve, curve[1:]):
        if d0 <= d <= d1:
            cap = b0 + (b1 - b0) * (d - d0) / (d1 - d0) if d1 > d0 else b1
            break
    return cap


def disk_util(read_mb_s, depth, curve):
    """Achieved read rate over the curve's rate at `depth`."""
    cap = curve_capacity(depth, curve)
    return read_mb_s * 1e6 / cap if read_mb_s is not None and cap else None


def read_wait_share(profile, io=None):
    """(read_wait_ms over total_ms of a builder profile block, unreliable).

    Prefetch-only runs report read_wait_ms 0 although they read (`io` bytes > 0): the share is then None and
    unreliable, not 0."""
    try:
        if profile["read_wait_ms"] == 0 and io and io.get("bytes", 0) > 0:
            return None, True
        return profile["read_wait_ms"] / profile["total_ms"], False
    except (KeyError, TypeError, ZeroDivisionError):
        return None, False


def io_class(h):
    """(class, missing inputs) of one host phase summary, same rule for the builder and the baselines. Report-only.

    disk_util >= DISK_UTIL_BOUND is `disk_bandwidth` whatever the stall share. Else the stall share s (the larger of
    io_some_share and read_wait_share) below IO_BOUND_SHARE is `not_io_bound`; else a queue depth below
    SERIAL_QUEUE_DEPTH is `io_latency` (serial reads or page faults), anything else `io_under_issue`."""
    if h.get("disk_util") is not None and h["disk_util"] >= DISK_UTIL_BOUND:
        return "disk_bandwidth", []
    shares = [h[k] for k in ("io_some_share", "read_wait_share") if h.get(k) is not None]
    if not shares:
        return "unknown", ["io_some_share"]
    if max(shares) < IO_BOUND_SHARE:
        return "not_io_bound", []
    if h.get("disk_util") is None or h.get("disk_queue_depth_mean") is None:
        return "unknown", [k for k in ("disk_util", "disk_queue_depth_mean") if h.get(k) is None]
    return ("io_latency" if h["disk_queue_depth_mean"] < SERIAL_QUEUE_DEPTH else "io_under_issue"), []


def classify(h, curve, rw=(None, False)):
    """Adds the attribution numbers and the report-only io_class to a phase summary: bytes_per_token, disk_util,
    tok_s_ceiling_disk (curve capacity at the observed depth over bytes_per_token), read_wait_share, io_class."""
    if "error" in h:
        return h
    cap = curve_capacity(h.get("disk_queue_depth_mean"), curve)
    bpt = h.get("read_bytes_per_token")
    h["bytes_per_token"] = bpt
    h["disk_util"] = disk_util(h.get("read_mb_s"), h.get("disk_queue_depth_mean"), curve)
    h["tok_s_ceiling_disk"] = cap / bpt if cap and bpt else None
    h["read_wait_share"], unreliable = rw
    if unreliable:
        h["read_wait_share_unreliable"] = True
    h["io_class"], h["io_class_missing"] = io_class(h)
    return h


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


N_DECODE_WINDOWS = 4


def equal_windows(dec, n=N_DECODE_WINDOWS):
    """n equal time slices of a decode window; the token count is split evenly (an estimate: engines that report no
    per-token times cannot say better)."""
    t0, t1, tokens = dec
    step = (t1 - t0) / n
    return [(t0 + i * step, t0 + (i + 1) * step, tokens / n, (None, False)) for i in range(n)]


def host_summary(rows, pre, dec, threads, curve, rw=((None, False), (None, False)), windows=None):
    """{"prefill", "decode", "decode_windows"} summaries with the I/O attribution fields. rw = read_wait_share result
    of prefill and decode; windows = [(t0, t1, tokens, rw)] of the decode (default: 4 equal time slices, tokens
    estimated, flagged "tokens_estimated")."""
    est = windows is None
    out = {"prefill": classify(phase(rows, pre[0], pre[1], pre[2], threads), curve, rw[0]),
           "decode": classify(phase(rows, dec[0], dec[1], dec[2], threads), curve, rw[1]),
           "decode_windows": []}
    for t0, t1, n, wrw in windows or equal_windows(dec):
        w = classify(phase(rows, t0, t1, n, threads), curve, wrw)
        if est:
            w["tokens_estimated"] = True
        out["decode_windows"].append(w)
    return out


def builder_extras(path):
    """(read-wait results of prefill and decode, per-window (t0, t1, tokens, read-wait result)) of a harness JSON."""
    run = json.loads(ds.read(path))["runs"][0]
    rw = (read_wait_share(run.get("prefill_profile_per_ubatch"), run.get("prefill_io")),
          read_wait_share(run.get("decode_profile_per_token"), run.get("decode_io")))
    wins = [(w["start_unix_s"], w["end_unix_s"], w["tokens"], read_wait_share(w.get("profile_per_token"), w.get("io")))
            for w in run["decode_windows"]]
    return rw, wins


def kt_summary(summary_path):
    """The {"host", "env"} of a summarize output; a missing or broken file gives host {"error": ...}, never empty."""
    try:
        d = json.loads(ds.read(summary_path))
        if d.get("host"):
            return {"host": d["host"], "env": d.get("env")}
    except ValueError:
        pass
    return {"host": {"error": f"no host summary at {summary_path}"}, "env": None}


def summarize(a):
    rows = load_samples(a.samples)
    pre, dec = windows_builder(a.builder_json) if a.builder_json else windows_kt(a.kt_json)
    curve_file = a.curve or find_curve(a.curve_dir or os.path.dirname(os.path.abspath(a.merge_into or a.out)))
    curve = load_curve(curve_file) if curve_file else []
    rw, wins = builder_extras(a.builder_json) if a.builder_json else (((None, False),) * 2, None)
    out = {"host": host_summary(rows, pre, dec, a.threads, curve, rw, wins), "env": env_info()}
    out["env"]["disk_curve"] = os.path.basename(curve_file) if curve else None
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
    m.add_argument("--curve-dir")
    m.add_argument("--curve")
    a = ap.parse_args()
    {"probe-disk": probe_disk, "sample": sample, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
