"""Host samples as OTLP metrics (ADR 0060).

points(rec):     one engine_sample.py sample record as OTLP data points (the sampler's series).
request(...):    an OTLP/HTTP JSON ExportMetricsServiceRequest for those points.
Poster:          posts requests to a collector; never raises, gives up after MAX_FAILS straight failures.
read_points():   the data points of collector file-exporter output (OTLP JSON lines).
rows(points):    sampler-shaped records again, sampler series on their own ticks and the hostmetrics series joined
                 to each tick (disk counters interpolated, the rest as of the tick), so engine_sample.host_summary()
                 derives the same host block from either path.
Stdlib only: the box runs the system python3, often with -I.
"""

import bisect
import json
import os
import subprocess
import urllib.request

SCOPE = "tetraeder.host.sampler"
SERVICE = "tetraeder-host-sampler"
GPU_ID = "gpu0"
# nvidia-smi clocks_throttle_reasons.active bits worth an alert (0x1 idle and 0x2 application clocks are not)
THROTTLE_REASONS = {0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake"}
# hostmetrics series the derivation reads (deploy/otelcol/config-file.yaml); tetraeder.host.cpu.busy is the
# collector's metricstransform of system.cpu.time per CPU
HOSTMETRICS = ("system.disk.io", "system.disk.operations", "system.disk.pending_operations",
               "system.disk.weighted_io_time", "system.cpu.time", "tetraeder.host.cpu.busy")
CPU_STATES = ("user", "nice", "system", "idle", "wait", "interrupt", "softirq", "steal")
# a hostmetrics point older than this at a sampler tick is not used (the collector stopped or fell behind)
MAX_AGE_S = 2.5
MAX_FAILS = 10

THREAD_CTXSW = "tetraeder.host.process.thread.context_switches"
GPU = {"hw.id": GPU_ID, "hw.type": "gpu"}
READ = {"disk.io.direction": "read"}


def points(rec):
    """[(name, kind, unit, attrs, value)] of one sample record; kind is counter, updown or gauge. None is left out."""
    out = []

    def add(name, kind, unit, value, attrs=None):
        if value is not None:
            out.append((name, kind, unit, attrs or {}, value))

    cg = rec.get("cg_cpu_usec")
    add("container.cpu.time", "counter", "s", cg / 1e6 if cg is not None else None)
    io = rec.get("cg_io")
    if io:
        add("container.disk.io", "counter", "By", io.get("rbytes"), READ)
        add("tetraeder.host.cgroup.disk.operations", "counter", "{operation}", io.get("rios"), READ)
    now = rec.get("cg_mem_now") or {}
    add("container.memory.usage", "updown", "By", now.get("current"))
    add("tetraeder.host.cgroup.memory.limit", "updown", "By", now.get("max"))
    mem = rec.get("cg_mem") or {}
    for kind in ("file", "anon"):
        add("tetraeder.host.cgroup.memory.usage", "updown", "By", mem.get(kind), {"tetraeder.memory.kind": kind})
    add("tetraeder.host.cgroup.cpu.quota", "gauge", "{cpu}", rec.get("cpu_quota"))
    th = rec.get("cg_throttle") or {}
    if th.get("src") is not None:
        src = {"tetraeder.source": th["src"]}
        add("tetraeder.host.cgroup.cpu.periods", "counter", "{period}", th.get("periods"), src)
        add("tetraeder.host.cgroup.cpu.throttled_periods", "counter", "{period}", th.get("throttled"), src)
        usec = th.get("usec")
        add("tetraeder.host.cgroup.cpu.throttled_time", "counter", "s", usec / 1e6 if usec is not None else None, src)
    for resource, key in (("io", "io_psi"), ("cpu", "cpu_psi")):
        psi = rec.get(key) or {}
        if psi.get("src") is None:
            continue
        for kind in ("some", "full"):
            v = psi.get(kind)
            add("tetraeder.host.pressure.stall", "counter", "s", v / 1e6 if v is not None else None,
                {"tetraeder.pressure.resource": resource, "tetraeder.pressure.kind": kind, "tetraeder.source": psi["src"]})
    mhz = rec.get("cpu_mhz")
    add("tetraeder.host.cpu.frequency", "gauge", "Hz", mhz * 1e6 if mhz is not None else None)
    add("tetraeder.host.process.cpu.time", "counter", "s", rec.get("own_cpu_s"))
    add("tetraeder.host.process.paging.faults", "counter", "{fault}", rec.get("majflt"), {"type": "major"})
    cs = rec.get("ctxsw")
    if cs:
        add("tetraeder.host.process.context_switches", "counter", "{count}", cs.get("vol"), {"type": "voluntary"})
        add("tetraeder.host.process.context_switches", "counter", "{count}", cs.get("invol"), {"type": "involuntary"})
    # per-thread counters: a thread's counter vanishes when it exits, so the rate needs per-thread deltas (see
    # engine_sample.ctxsw_delta); the sum above falls then
    for tid, (vol, invol) in sorted((rec.get("ctxsw_tasks") or {}).items()):
        add(THREAD_CTXSW, "counter", "{count}", vol, {"tetraeder.thread": tid, "type": "voluntary"})
        add(THREAD_CTXSW, "counter", "{count}", invol, {"tetraeder.thread": tid, "type": "involuntary"})
    rss = rec.get("rss_kb")
    add("tetraeder.host.process.memory.usage", "gauge", "By", rss * 1024 if rss is not None else None)
    ahp = rec.get("anon_huge_kb")
    add("tetraeder.host.process.memory.anon_huge", "gauge", "By", ahp * 1024 if ahp is not None else None)
    g = rec.get("gpu")
    if g:
        add("hw.gpu.utilization", "gauge", "1", g["util"] / 100, {**GPU, "hw.gpu.task": "general"})
        add("hw.gpu.memory.usage", "updown", "By", g["mem_mib"] * 1048576, GPU)
        total = g.get("mem_total_mib")
        add("hw.gpu.memory.limit", "updown", "By", total * 1048576 if total is not None else None, GPU)
        add("hw.power", "gauge", "W", g.get("power_w"), GPU)
        add("hw.temperature", "gauge", "Cel", g.get("temp_c"), GPU)
        for clock, key in (("sm", "sm_mhz"), ("memory", "mem_mhz")):
            v = g.get(key)
            add("tetraeder.host.gpu.clock", "gauge", "Hz", v * 1e6 if v is not None else None,
                {**GPU, "tetraeder.gpu.clock": clock})
        bits = g.get("throttle")
        if bits is not None:
            for bit, reason in THROTTLE_REASONS.items():
                add("tetraeder.host.gpu.throttled", "gauge", "", 1 if bits & bit else 0,
                    {**GPU, "tetraeder.gpu.throttle_reason": reason})
        add("tetraeder.host.gpu.pcie.link.generation", "gauge", "", g.get("pcie_gen"), GPU)
        add("tetraeder.host.gpu.pcie.link.width", "gauge", "", g.get("pcie_width"), GPU)
    return out


def membw_points(m):
    """The membw probe result ({"membw_gb_s", "threads", ...}) as one gauge point."""
    return [("tetraeder.host.membw", "gauge", "By/s", {"tetraeder.membw.threads": m["threads"]}, m["membw_gb_s"] * 1e9)]


def curve_points(curve, kind):
    """[(depth, bytes_per_s)] of a disk curve as gauge points, one per queue depth."""
    return [("tetraeder.host.disk.curve.rate", "gauge", "By/s",
             {"tetraeder.disk.curve": kind, "tetraeder.disk.queue_depth": float(d)}, float(b)) for d, b in curve]


def _value(v):
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


def _attrs(d):
    return [{"key": k, "value": _value(v)} for k, v in d.items() if v is not None]


def request(pts, t, resource, start=None):
    """OTLP JSON ExportMetricsServiceRequest of points taken at unix time t (start: the counters' start time)."""
    ns = str(int(round(t * 1e9)))
    start_ns = str(int(round((start if start is not None else t) * 1e9)))
    metrics = {}
    for name, kind, unit, attrs, value in pts:
        dp = {"attributes": _attrs(attrs), "timeUnixNano": ns}
        if isinstance(value, int) and not isinstance(value, bool):
            dp["asInt"] = str(value)
        else:
            dp["asDouble"] = float(value)
        if kind != "gauge":
            dp["startTimeUnixNano"] = start_ns
        m = metrics.get(name)
        if m is None:
            m = {"name": name, "unit": unit}
            if kind == "gauge":
                m["gauge"] = {"dataPoints": []}
            else:
                m["sum"] = {"dataPoints": [], "aggregationTemporality": 2, "isMonotonic": kind == "counter"}
            metrics[name] = m
        (m["gauge"] if "gauge" in m else m["sum"])["dataPoints"].append(dp)
    return {"resourceMetrics": [{"resource": {"attributes": _attrs(resource)},
                                 "scopeMetrics": [{"scope": {"name": SCOPE}, "metrics": list(metrics.values())}]}]}


def resource(env=None, **extra):
    """Resource attributes of the sampler: service, host and session provenance from the harness environment."""
    env = os.environ if env is None else env
    r = {"service.name": SERVICE,
         "host.id": env.get("CONTAINER_ID") or None,
         "host.name": os.uname().nodename if hasattr(os, "uname") else None,
         "tetraeder.session": env.get("TETRAEDER_SESSION") or None,
         "tetraeder.stage": env.get("TETRAEDER_STAGE") or None,
         "tetraeder.git.commit": env.get("TETRAEDER_GIT_COMMIT") or None}
    r.update(extra)
    return {k: v for k, v in r.items() if v is not None}


def endpoint(arg=None, env=None):
    """The collector base URL: --otlp, else TETRAEDER_HOST_OTLP; None when unset or SAMPLER_OTLP=0."""
    env = os.environ if env is None else env
    if env.get("SAMPLER_OTLP") == "0":
        return None
    return arg or env.get("TETRAEDER_HOST_OTLP") or None


class Poster:
    """POSTs OTLP JSON to <base>/v1/metrics with a short timeout. Errors are counted, never raised; after MAX_FAILS
    straight failures it stops trying, so a dead collector costs the sampler at most MAX_FAILS timeouts."""

    def __init__(self, base, timeout=0.5, opener=None):
        self.url = base.rstrip("/") + "/v1/metrics"
        self.timeout = timeout
        self.open = opener or urllib.request.urlopen
        self.fails = 0
        self.sent = 0
        self.off = False

    def send(self, body):
        if self.off:
            return False
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with self.open(req, timeout=self.timeout) as r:
                r.read()
            self.fails = 0
            self.sent += 1
            return True
        except Exception:  # noqa: BLE001 - host sampling must never fail the run
            self.fails += 1
            if self.fails >= MAX_FAILS:
                self.off = True
            return False


def _plain(v):
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "boolValue" in v:
        return v["boolValue"]
    return v.get("stringValue")


def _kv(lst):
    return {a["key"]: _plain(a.get("value") or {}) for a in lst or []}


def points_of(req):
    """Data points of one OTLP JSON request: dicts with name, scope, resource, attrs, t_ns, value."""
    for rm in req.get("resourceMetrics") or []:
        res = _kv((rm.get("resource") or {}).get("attributes"))
        for sm in rm.get("scopeMetrics") or []:
            scope = (sm.get("scope") or {}).get("name", "")
            for m in sm.get("metrics") or []:
                body = m.get("sum") or m.get("gauge")
                if not body:
                    continue
                for dp in body.get("dataPoints") or []:
                    if "asInt" in dp:
                        value = int(dp["asInt"])
                    elif "asDouble" in dp:
                        value = float(dp["asDouble"])
                    else:
                        continue
                    yield {"name": m["name"], "scope": scope, "resource": res, "attrs": _kv(dp.get("attributes")),
                           "t_ns": int(dp.get("timeUnixNano", 0)), "value": value}


def open_lines(path):
    """Text lines of a collector file, plain or .zst (ADR 0060 addendum). Zstd goes through the stdlib module
    (Python 3.14) or the zstd CLI, which the box image has; raises OSError when neither can read it."""
    if not str(path).endswith(".zst"):
        return open(path, encoding="utf-8")
    try:
        from compression import zstd
        return zstd.open(path, "rt", encoding="utf-8")
    except ImportError:
        pass
    try:
        proc = subprocess.Popen(["zstd", "-dc", "--", str(path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                text=True, encoding="utf-8")
    except FileNotFoundError as e:
        raise OSError("zstd CLI missing") from e
    return _Pipe(proc)


class _Pipe:
    """Context manager over a zstd -dc process's stdout."""

    def __init__(self, proc):
        self.proc = proc

    def __iter__(self):
        return iter(self.proc.stdout)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.proc.stdout.close()
        self.proc.wait()


def read_points(paths, keep=None):
    """Points of OTLP JSON-lines files (the collector's file exporter), plain or .zst, sampler and hostmetrics series only.
    keep(t_s) -> bool limits them to the windows wanted. Unparseable lines (a cut final line) are skipped."""
    out = []
    for path in paths:
        try:
            f = open_lines(path)
        except OSError:
            continue
        with f:
            for line in f:
                try:
                    req = json.loads(line)
                except ValueError:
                    continue
                for p in points_of(req):
                    if p["scope"] != SCOPE and p["name"] not in HOSTMETRICS:
                        continue
                    if keep is not None and not keep(p["t_ns"] / 1e9):
                        continue
                    out.append(p)
    return out


def _cpu_index(name):
    digits = "".join(c for c in str(name) if c.isdigit())
    return int(digits) if digits else -1


def _record(pts):
    """A sampler record from the points of one tick ({(name, frozenset(attrs.items())): (value, attrs)})."""

    def get(name, **attrs):
        for (n, a), (v, _) in pts.items():
            if n == name and all(dict(a).get(k) == x for k, x in attrs.items()):
                return v
        return None

    def src(name, **attrs):
        for (n, a), (_, full) in pts.items():
            if n == name and all(full.get(k) == x for k, x in attrs.items()):
                return full.get("tetraeder.source")
        return None

    def us(v):
        return int(round(v * 1e6)) if v is not None else None

    def kb(v):
        return int(v) // 1024 if v is not None else None

    rec = {}
    rec["cg_cpu_usec"] = us(get("container.cpu.time"))
    rb = get("container.disk.io", **{"disk.io.direction": "read"})
    rec["cg_io"] = None if rb is None else {
        "rbytes": rb, "rios": get("tetraeder.host.cgroup.disk.operations", **{"disk.io.direction": "read"}) or 0}
    for resource, key in (("io", "io_psi"), ("cpu", "cpu_psi")):
        q = {"tetraeder.pressure.resource": resource}
        some = get("tetraeder.host.pressure.stall", **q, **{"tetraeder.pressure.kind": "some"})
        if some is None:
            rec[key] = {"src": None, "some": None, "full": None}
        else:
            rec[key] = {"src": src("tetraeder.host.pressure.stall", **q, **{"tetraeder.pressure.kind": "some"}),
                        "some": us(some),
                        "full": us(get("tetraeder.host.pressure.stall", **q, **{"tetraeder.pressure.kind": "full"}))}
    periods = get("tetraeder.host.cgroup.cpu.periods")
    if periods is None:
        rec["cg_throttle"] = {"src": None, "periods": None, "throttled": None, "usec": None}
    else:
        rec["cg_throttle"] = {"src": src("tetraeder.host.cgroup.cpu.periods"), "periods": periods,
                              "throttled": get("tetraeder.host.cgroup.cpu.throttled_periods"),
                              "usec": us(get("tetraeder.host.cgroup.cpu.throttled_time"))}
    hz = get("tetraeder.host.cpu.frequency")
    rec["cpu_mhz"] = hz / 1e6 if hz is not None else None
    vol = get("tetraeder.host.process.context_switches", type="voluntary")
    invol = get("tetraeder.host.process.context_switches", type="involuntary")
    rec["ctxsw"] = None if vol is None and invol is None else {"vol": vol or 0, "invol": invol or 0}
    tasks = {}
    for (n, _), (v, attrs) in pts.items():
        if n == THREAD_CTXSW and "tetraeder.thread" in attrs:
            tasks.setdefault(attrs["tetraeder.thread"], [0, 0])[0 if attrs.get("type") == "voluntary" else 1] = v
    rec["ctxsw_tasks"] = tasks or None
    rec["anon_huge_kb"] = kb(get("tetraeder.host.process.memory.anon_huge"))
    rec["majflt"] = get("tetraeder.host.process.paging.faults", type="major") or 0
    rec["own_cpu_s"] = get("tetraeder.host.process.cpu.time")
    rec["rss_kb"] = kb(get("tetraeder.host.process.memory.usage")) or 0
    rec["cg_mem"] = {k: get("tetraeder.host.cgroup.memory.usage", **{"tetraeder.memory.kind": k}) for k in ("file", "anon")}
    rec["cg_mem_now"] = {"current": get("container.memory.usage"), "max": get("tetraeder.host.cgroup.memory.limit")}
    rec["cpu_quota"] = get("tetraeder.host.cgroup.cpu.quota")
    util, mem = get("hw.gpu.utilization"), get("hw.gpu.memory.usage")
    rec["gpu"] = None if util is None or mem is None else {"util": int(round(util * 100)), "mem_mib": int(mem) // 1048576}
    return rec


def disk_device(pts):
    """The model disk's device name, from the sampler's resource (tetraeder.host.disk.device); None if absent."""
    for p in pts:
        d = p["resource"].get("tetraeder.host.disk.device")
        if d:
            return d
    return None


def _series(pts, disk):
    """{key: ([t_ns], [value])} of the hostmetrics points the host block reads, sorted by time."""
    out = {}
    for p in pts:
        if p["scope"] == SCOPE or p["name"] not in HOSTMETRICS:
            continue
        a = p["attrs"]
        if p["name"].startswith("system.disk."):
            if disk is None or a.get("device") != disk:
                continue
            key = (p["name"], a.get("direction"))
        elif p["name"] == "system.cpu.time":
            if "cpu" in a:  # the per-CPU form, before the collector's transform
                continue
            key = (p["name"], a.get("state"))
        else:
            key = (p["name"], a.get("cpu"))
        out.setdefault(key, []).append((p["t_ns"], p["value"]))
    return {k: ([t for t, _ in sorted(v)], [x for _, x in sorted(v)]) for k, v in out.items()}


def _asof(series, t):
    """(t_ns, value) of the last point at or before t, unless older than MAX_AGE_S; else None."""
    ts, vs = series
    i = bisect.bisect_right(ts, t) - 1
    if i < 0 or (t - ts[i]) / 1e9 > MAX_AGE_S:
        return None
    return ts[i], vs[i]


def _at(series, t):
    """A cumulative counter at t: linear between the points around it (the sampler and hostmetrics tick on their own
    clocks, so an as-of value would lag up to one interval), else the as-of value."""
    ts, vs = series
    i = bisect.bisect_right(ts, t) - 1
    if i >= 0 and ts[i] == t:
        return vs[i]
    if 0 <= i < len(ts) - 1 and (ts[i + 1] - ts[i]) / 1e9 <= 2 * MAX_AGE_S:
        return vs[i] + (vs[i + 1] - vs[i]) * (t - ts[i]) / (ts[i + 1] - ts[i])
    v = _asof(series, t)
    return v[1] if v else None


def rows(pts, disk=None):
    """Sampler-shaped records of OTLP points: one per sampler tick (all hostmetrics times if no sampler point), the
    sampler fields from that tick's points, the disk and CPU fields from the hostmetrics points: disk counters
    interpolated to the tick, queue length and CPU times as of it."""
    disk = disk or disk_device(pts)
    ticks = {}
    for p in pts:
        if p["scope"] == SCOPE:
            key = (p["name"], frozenset(p["attrs"].items()))
            ticks.setdefault(p["t_ns"], {})[key] = (p["value"], p["attrs"])
    series = _series(pts, disk)
    grid = sorted(ticks) if ticks else sorted({t for ts, _ in series.values() for t in ts})
    empty = ([], [])
    busy = sorted((k for k in series if k[0] == "tetraeder.host.cpu.busy"), key=lambda k: _cpu_index(k[1]))
    out = []
    for t in grid:
        rec = _record(ticks.get(t, {}))
        rec["t"] = t / 1e9
        rec["disk_read_bytes"] = _at(series.get(("system.disk.io", "read"), empty), t)
        rec["disk_read_ops"] = _at(series.get(("system.disk.operations", "read"), empty), t)
        q = _asof(series.get(("system.disk.pending_operations", None), empty), t)
        rec["disk_inflight"] = q[1] if q else None
        w = _at(series.get(("system.disk.weighted_io_time", None), empty), t)
        rec["disk_weighted_ms"] = w * 1e3 if w is not None else None
        cpus = []
        for key in busy:
            v = _asof(series[key], t)
            if v:
                # a CPU accrues one second per second in all states together: busy seconds over the scrape time
                cpus.append((v[1], v[0] / 1e9))
        rec["cpus"] = cpus
        st = [_asof(series.get(("system.cpu.time", s), empty), t) for s in CPU_STATES]
        # steal over all states at one scrape: a ratio of points taken together, so no interpolation
        rec["cpu_steal"] = None if any(x is None for x in st) else {
            "steal": st[CPU_STATES.index("steal")][1], "total": sum(x[1] for x in st)}
        out.append(rec)
    return out
