"""Host sampler and observation writer for the detector box runs (ADR 0041).

sample:  python3 detector_sample.py sample --pid PID --disk MAJ:MIN --out s.jsonl
         Every 0.5 s until PID exits: wall time, CPU used by the container cgroup and by PID's process tree,
         container CPU throttling, disk sectors read (whole device and by PID), swap-ins, GPU state.
observe: python3 detector_sample.py observe --harness h.json --samples s.jsonl --threads N --quota Q
         [--calib calib.json] [--io-max-bytes B] [--vram-other-bytes B] [--planned-slot-bytes B] --id ID --out obs.json
         Builds one analysis::Observation (crates/analysis/fixtures/observation_example.json) from the harness
         decode windows and the samples inside each window.
calib:   python3 detector_sample.py calib --obs a.json b.json ... --out calib.json
         Expected per-resource rates and band from no-fault calibration observations (median, band from spread).
The fault label of a run is never written into the observation; the campaign keeps it in a separate manifest.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def read(path, default=""):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def cgroup_cpu_usec():
    for line in read("/sys/fs/cgroup/cpu.stat").splitlines():
        k, _, v = line.partition(" ")
        if k == "usage_usec":
            return int(v)
    v1 = read("/sys/fs/cgroup/cpu,cpuacct/cpuacct.usage") or read("/sys/fs/cgroup/cpuacct/cpuacct.usage")
    return int(v1) // 1000 if v1.strip() else None


def cgroup_throttled_usec():
    for line in read("/sys/fs/cgroup/cpu.stat").splitlines():
        k, _, v = line.partition(" ")
        if k == "throttled_usec":
            return int(v)
    return None


def tree(pid):
    """pid and all its descendants."""
    kids = {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        st = read(f"/proc/{p}/stat")
        if not st:
            continue
        ppid = int(st.rsplit(")", 1)[1].split()[1])
        kids.setdefault(ppid, []).append(int(p))
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo.extend(kids.get(p, []))
    return out


def proc_cpu_s(pids):
    s = 0.0
    for p in pids:
        st = read(f"/proc/{p}/stat")
        if st:
            f = st.rsplit(")", 1)[1].split()
            s += (int(f[11]) + int(f[12])) / CLK
    return s


def proc_read_bytes(pids):
    b = 0
    for p in pids:
        for line in read(f"/proc/{p}/io").splitlines():
            if line.startswith("read_bytes:"):
                b += int(line.split()[1])
    return b


def disk_read_bytes(majmin):
    maj, mn = majmin.split(":")
    for line in read("/proc/diskstats").splitlines():
        f = line.split()
        if f[0] == maj and f[1] == mn:
            return int(f[5]) * 512
    return None


def pswpin():
    for line in read("/proc/vmstat").splitlines():
        if line.startswith("pswpin "):
            return int(line.split()[1])
    return 0


GPU_Q = ("memory.total,memory.used,utilization.gpu,clocks_throttle_reasons.active,"
         "pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max")


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "-i", "0", f"--query-gpu={GPU_Q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None
    f = [x.strip() for x in out.split(",")]
    num = lambda x: int(x) if x.isdigit() else None
    return {"mem_total_mib": num(f[0]), "mem_used_mib": num(f[1]), "util": num(f[2]),
            "throttle": f[3], "gen": num(f[4]), "gen_max": num(f[5]), "width": num(f[6]), "width_max": num(f[7])}


def sample(a):
    n = 0
    with open(a.out, "w") as out:
        while os.path.exists(f"/proc/{a.pid}"):
            pids = tree(a.pid)
            rec = {"t": time.time(), "cg_cpu_usec": cgroup_cpu_usec(), "cg_throttled_usec": cgroup_throttled_usec(),
                   "own_cpu_s": proc_cpu_s(pids), "own_read_bytes": proc_read_bytes(pids),
                   "disk_read_bytes": disk_read_bytes(a.disk), "pswpin": pswpin()}
            if n % 4 == 0:  # nvidia-smi takes about 50 ms; every 2 s is enough
                rec["gpu"] = gpu()
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n += 1
            time.sleep(0.5)


# NVML clock event reasons that mean a fault: 0x8 HW slowdown, 0x20 SW thermal, 0x40 HW thermal, 0x80 power brake.
# Not faults: 0x1 idle, 0x2 applications clocks, 0x4 SW power cap (a 3090 under full load sits at its power limit).
THROTTLE_BITS = {0x8: "clocks", 0x20: "thermal", 0x40: "thermal", 0x80: "power"}


def window_samples(samples, t0, t1):
    inside = [s for s in samples if t0 <= s["t"] <= t1]
    before = [s for s in samples if s["t"] < t0]
    after = [s for s in samples if s["t"] > t1]
    first = before[-1] if before else (inside[0] if inside else None)
    last = after[0] if after else (inside[-1] if inside else None)
    return first, last, inside


def observe(a):
    h = json.load(open(a.harness))
    samples = [json.loads(line) for line in open(a.samples) if line.strip()]
    run = h["runs"][-1]
    windows = []
    env_other, env_swap, env_disk_other, throttles, pcie = [], [], [], set(), None
    for w in run["decode_windows"]:
        first, last, inside = window_samples(samples, w["start_unix_s"], w["end_unix_s"])
        dt = max(last["t"] - first["t"], 1e-3) if first and last else w["wall_s"]
        prof, io = w["profile_per_token"], w["io"]
        n = w["tokens"]
        own_cpu = (last["own_cpu_s"] - first["own_cpu_s"]) if first and last else 0.0
        cg = ((last["cg_cpu_usec"] - first["cg_cpu_usec"]) / 1e6) if first and last and first["cg_cpu_usec"] is not None else own_cpu
        # the sampler itself costs a few percent of one core; count it as ours
        env_other.append(max(0.0, cg - own_cpu - 0.05 * dt) / dt)
        if first and last:
            env_swap.append((last["pswpin"] - first["pswpin"]) * 4096 / dt)
            if first["disk_read_bytes"] is not None:
                dev = last["disk_read_bytes"] - first["disk_read_bytes"]
                own = last["own_read_bytes"] - first["own_read_bytes"]
                env_disk_other.append(max(0, dev - own) / dt)
        gs = [s["gpu"] for s in inside if s.get("gpu")]
        util = statistics.mean([g["util"] for g in gs if g["util"] is not None]) / 100 if gs else 0.0
        for g in gs:
            try:
                bits = int(g["throttle"], 16)
            except (TypeError, ValueError):
                bits = 0
            throttles.update(r for b, r in THROTTLE_BITS.items() if bits & b)
            pcie = g
        ms = lambda k: prof.get(k, 0.0) * n / 1e3
        windows.append({
            "tokens": n,
            "wall_s": w["wall_s"],
            "busy_s": {
                **({"gpu:gpu0": util * w["wall_s"]} if gs else {}),  # no GPU on a CPU runner
                "cpu": own_cpu / max(a.threads, 1),
                "disk:nvme0": io["read_s"],
                "host": ms("inputs_ms") + ms("build_ms") + ms("alloc_ms") + ms("readback_ms"),
            },
            "split_s": {
                "compute": ms("compute_ms"),
                "expert_load_wait": ms("streamer_ms"),
                "host": ms("inputs_ms") + ms("build_ms") + ms("alloc_ms") + ms("readback_ms"),
            },
            "bytes": {"disk:nvme0": float(io["bytes"])},
        })
    g0 = next((s["gpu"] for s in samples if s.get("gpu")), None) or {}
    io_max = a.io_max_bytes  # the cgroup the engine ran in, when it is not the container's own (fault testbed)
    for line in read("/sys/fs/cgroup/io.max").splitlines() if io_max is None else []:
        for kv in line.split()[1:]:
            k, _, v = kv.partition("=")
            if k == "rbps" and v != "max":
                io_max = float(v)
    opts = h.get("options", {})
    slot_bytes = opts.get("expert_slot_bytes")
    per_slot = h.get("slot_bytes_all_layers") or 0
    slots_actual = int(slot_bytes // per_slot) if slot_bytes and per_slot else opts.get("expert_slots")
    slots_planned = int(a.planned_slot_bytes // per_slot) if a.planned_slot_bytes and per_slot else None
    obs = {
        "id": a.id,
        "batch": 1,
        "windows": windows,
        "env": {
            "throttle": [{"device": "gpu0", "reason": r} for r in sorted(throttles)],
            "pcie": [{"device": "gpu0", "gen_current": pcie["gen"], "width_current": pcie["width"],
                      "gen_max": pcie["gen_max"], "width_max": pcie["width_max"]}] if pcie and pcie["gen"] else [],
            "cpu_quota_cores": a.quota,
            "cpu_other_cores": statistics.median(env_other) if env_other else 0.0,
            "swap_in_bytes_per_s": statistics.median(env_swap) if env_swap else 0.0,
            "vram_other_bytes": {"gpu0": int(a.vram_other_bytes)},
            "vram_total_bytes": {"gpu0": int((g0.get("mem_total_mib") or 0) * 1048576)},
            "disk_other_read_bytes_per_s": statistics.median(env_disk_other) if env_disk_other else 0.0,
            "io_max_bytes_per_s": io_max,
            "expert_hit_rate": run["decode_layers"]["full_hit_rate_mean"],
            "slots_planned": slots_planned,
            "slots_actual": slots_actual,
        },
    }
    if a.calib:
        obs["expected"] = json.load(open(a.calib))
    json.dump(obs, open(a.out, "w"), indent=1)


def calib(a):
    """Per resource: median rate (bytes per busy second) and median busy seconds per token over no-fault runs."""
    per_tok, rates = {}, {}
    for p in a.obs:
        for w in json.load(open(p))["windows"]:
            for r, b in w["busy_s"].items():
                per_tok.setdefault(r, []).append(b / w["tokens"])
                if w["bytes"].get(r) and b > 0:
                    rates.setdefault(r, []).append(w["bytes"][r] / b)
    spread = []
    for v in rates.values():
        m = statistics.median(v)
        spread += [abs(x / m - 1) for x in v]
    band = max(0.1, 3 * statistics.median(spread)) if spread else 0.15
    out = {"rates": {r: statistics.median(v) for r, v in rates.items()},
           "busy_s_per_token": {r: statistics.median(v) for r, v in per_tok.items()},
           "band": band}
    json.dump(out, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--pid", type=int, required=True)
    s.add_argument("--disk", required=True)
    s.add_argument("--out", required=True)
    o = sub.add_parser("observe")
    o.add_argument("--harness", required=True)
    o.add_argument("--samples", required=True)
    o.add_argument("--threads", type=int, required=True)
    o.add_argument("--quota", type=float, required=True)
    o.add_argument("--calib")
    o.add_argument("--io-max-bytes", type=float)
    o.add_argument("--vram-other-bytes", type=float, default=0)
    o.add_argument("--planned-slot-bytes", type=float)
    o.add_argument("--id", required=True)
    o.add_argument("--out", required=True)
    c = sub.add_parser("calib")
    c.add_argument("--obs", nargs="+", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    {"sample": sample, "observe": observe, "calib": calib}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
