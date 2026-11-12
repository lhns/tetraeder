"""Host metrics as OTLP (ADR 0060): the host block derived from the collector's OTLP file equals the one derived from
the sampler's JSONL for the same counters, field for field; plus the encoder, the poster and the GPU parser."""

import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("engine_sample", SCRIPTS / "engine_sample.py")
es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(es)
oh = es.oh

# whole seconds: exact as float seconds and as integer nanoseconds
T0 = 1_700_000_000
TICKS = 40
CPUS = 4
DISK = "nvme0n1"
PSI_IO, PSI_CPU = "/sys/fs/cgroup/io.pressure", "/sys/fs/cgroup/cpu.pressure"
STATES = ("user", "nice", "system", "idle", "wait", "interrupt", "softirq", "steal")


def counters(constant=False):
    """Synthetic cumulative counters per tick, in the sampler's units, with rates that change over time (or not)."""
    out = []
    c = {"rb": 10**9, "ops": 5000, "wms": 100, "rbytes": 7 * 10**8, "rios": 300, "some": 10**6, "full": 4 * 10**5,
         "csome": 2 * 10**6, "cfull": 10**5, "periods": 1000, "thr": 50, "thr_us": 70000, "maj": 20, "vol": 900,
         "invol": 80, "cg_us": 5 * 10**9, "own": 4000.25}
    busy = [1000 * (i + 1) for i in range(CPUS)]
    states = {s: 10**5 * (j + 1) for j, s in enumerate(STATES)}
    for k in range(TICKS):
        t = T0 + k
        r = 1 if constant else 1 + (k % 7)
        c["rb"] += r * 3 * 10**8
        c["ops"] += r * 300
        c["wms"] += r * 2100
        c["rbytes"] += r * 2 * 10**8
        c["rios"] += r * 200
        c["some"] += r * 90_000
        c["full"] += r * 40_000
        c["csome"] += r * 30_000
        c["cfull"] += r * 5_000
        c["periods"] += 10
        c["thr"] += r % 3
        c["thr_us"] += r * 1500
        c["maj"] += r * 11
        c["vol"] += r * 1000
        c["invol"] += r * 37
        c["cg_us"] += r * 2_000_000
        c["own"] += r * 1.75
        for i in range(CPUS):
            busy[i] += (i + 1) * 9 + r * 3
        for j, s in enumerate(STATES):
            states[s] += 100 * CPUS if s == "idle" else (j + 1) * r
        out.append({
            "t": float(t), "disk_read_bytes": c["rb"], "disk_read_ops": c["ops"], "disk_inflight": r % 5,
            "disk_weighted_ms": c["wms"],
            "cg_io": {"rbytes": c["rbytes"], "rios": c["rios"]},
            "io_psi": {"src": PSI_IO, "some": c["some"], "full": c["full"]},
            "majflt": c["maj"],
            # per-CPU (busy, total) jiffies: a CPU accrues 100 jiffies per second
            "cpus": [(busy[i], 10**7 + 100 * k) for i in range(CPUS)],
            "cpu_psi": {"src": PSI_CPU, "some": c["csome"], "full": c["cfull"]},
            "cg_throttle": {"src": "/sys/fs/cgroup/cpu.stat", "periods": c["periods"], "throttled": c["thr"],
                            "usec": c["thr_us"]},
            "cpu_steal": {"steal": states["steal"], "total": sum(states.values())},
            "cpu_mhz": 2400.0 + 10 * r,
            "ctxsw": {"vol": c["vol"], "invol": c["invol"]},
            "anon_huge_kb": 2048 * r,
            "cg_cpu_usec": c["cg_us"], "own_cpu_s": c["own"],
            "rss_kb": 70_000_000 + 1000 * r,
            "cg_mem": {"file": 30 * 10**9 + r * 10**6, "anon": 60 * 10**9 + r * 10**6},
            "gpu": {"util": 10 * r, "mem_mib": 20000 + r, "mem_total_mib": 24576, "power_w": 210.5, "temp_c": 61.0,
                    "sm_mhz": 1695.0, "mem_mhz": 9751.0, "throttle": 0x4, "pcie_gen": 4, "pcie_width": 16},
            "cg_mem_now": {"current": 90 * 10**9, "max": 128 * 10**9},
            "cpu_quota": 23.04,
            "_states": dict(states),
        })
    return out


def hostmetrics_line(rec, t_ns):
    """One collector export of what hostmetrics (disk, cpu) and the config's metricstransform write: names, units and
    attributes as in the 0.116 receiver docs. A second disk and a leftover per-CPU point must be ignored."""

    def dp(attrs, value, monotonic=True):
        a = [{"key": k, "value": {"stringValue": v}} for k, v in attrs.items()]
        d = {"attributes": a, "startTimeUnixNano": str(T0 * 10**9 - 10**9), "timeUnixNano": str(t_ns)}
        if isinstance(value, int):
            d["asInt"] = str(value)
        else:
            d["asDouble"] = value
        return d

    def total(name, unit, points, monotonic=True):
        return {"name": name, "unit": unit,
                "sum": {"dataPoints": points, "aggregationTemporality": 2, "isMonotonic": monotonic}}

    disk = [
        total("system.disk.io", "By", [dp({"device": DISK, "direction": "read"}, rec["disk_read_bytes"]),
                                       dp({"device": DISK, "direction": "write"}, 12345),
                                       dp({"device": "sda", "direction": "read"}, 999)]),
        total("system.disk.operations", "{operations}", [dp({"device": DISK, "direction": "read"}, rec["disk_read_ops"]),
                                                         dp({"device": "sda", "direction": "read"}, 7)]),
        total("system.disk.pending_operations", "{operations}", [dp({"device": DISK}, rec["disk_inflight"])], False),
        total("system.disk.weighted_io_time", "s", [dp({"device": DISK}, rec["disk_weighted_ms"] / 1000)]),
    ]
    cpu = [
        total("system.cpu.time", "s", [dp({"state": s}, v / 100) for s, v in rec["_states"].items()]
              + [dp({"cpu": "cpu0", "state": "user"}, 1.0)]),
        total("tetraeder.host.cpu.busy", "s", [dp({"cpu": f"cpu{i}"}, b / 100) for i, (b, _) in enumerate(rec["cpus"])]),
    ]
    res = {"attributes": [{"key": "host.id", "value": {"stringValue": "c-1"}}]}
    scope = "github.com/open-telemetry/opentelemetry-collector-contrib/receiver/hostmetricsreceiver/internal/scraper/"
    return {"resourceMetrics": [{"resource": res, "scopeMetrics": [
        {"scope": {"name": scope + "diskscraper"}, "metrics": disk},
        {"scope": {"name": scope + "cpuscraper"}, "metrics": cpu}]}]}


# per-tick disk increments at rate 1: a constant-rate counter read host_offset_s early
DISK_RATE = {"disk_read_bytes": 3 * 10**8, "disk_read_ops": 300, "disk_weighted_ms": 2100}


def write_inputs(tmp_path, host_offset_s=0.0, host_ticks=TICKS, constant=False):
    """Sampler JSONL, the collector's OTLP file (sampler and hostmetrics lines) and a builder result. With an offset,
    hostmetrics scrapes that much before each tick; with constant rates its disk counters are what they were then."""
    recs = counters(constant)
    rows = [{k: v for k, v in r.items() if k != "_states"} for r in recs]
    samples = tmp_path / "samples.jsonl"
    samples.write_text("".join(json.dumps(r) + "\n" for r in rows))
    otel = tmp_path / "otel" / "120000"
    otel.mkdir(parents=True)
    res = oh.resource(env={"TETRAEDER_STAGE": "main"}, **{"tetraeder.host.disk.device": DISK})
    with open(otel / "metrics.jsonl", "w") as f:
        for k, r in enumerate(recs):
            f.write(json.dumps(oh.request(oh.points(r), r["t"], res, start=T0 - 1)) + "\n")
            if k < host_ticks:
                hm = dict(r)
                if constant:
                    for key, rate in DISK_RATE.items():
                        hm[key] = r[key] - int(round(host_offset_s * rate))
                f.write(json.dumps(hostmetrics_line(hm, int(round((r["t"] - host_offset_s) * 10**9)))) + "\n")
        f.write('{"resourceMetrics": [')  # a final line cut by a stop
    results = tmp_path / "results"
    results.mkdir()
    win = [(T0 + 13.5 + 6 * i, T0 + 19.5 + 6 * i) for i in range(4)]
    run = {"prefill_s": 10.0, "hot_refresh_s": 1.0, "prefill_tokens": 512, "decode_tokens": 64,
           "decode_windows": [{"start_unix_s": a, "end_unix_s": b, "tokens": 16} for a, b in win]}
    builder = results / "exit-main-120000.json"
    builder.write_text(json.dumps({"options": {"threads": 3}, "runs": [run]}))
    return samples, otel.parent, results, builder


def summarize_sampler(samples, builder, tmp_path):
    es.summarize(types.SimpleNamespace(samples=str(samples), builder_json=str(builder), kt_json=None, threads=3,
                                       out=str(tmp_path / "sum.json"), merge_into=str(builder), curve_dir=None,
                                       curve=None, membw=None))


def assert_same(a, b, path=""):
    assert set(a) == set(b), path
    for k in a:
        u, v = a[k], b[k]
        if isinstance(u, dict):
            assert_same(u, v, f"{path}.{k}")
        elif isinstance(u, list) and u and isinstance(u[0], dict):
            assert len(u) == len(v), f"{path}.{k}"
            for i, (x, y) in enumerate(zip(u, v)):
                assert_same(x, y, f"{path}.{k}.{i}")
        elif isinstance(u, (int, float)) and not isinstance(u, bool):
            assert v == pytest.approx(u, rel=1e-9, abs=1e-9), f"{path}.{k}"
        else:
            assert u == v, f"{path}.{k}"


def test_otlp_host_block_equals_the_sampler_block(tmp_path):
    samples, otel, results, builder = write_inputs(tmp_path)
    summarize_sampler(samples, builder, tmp_path)
    es.otlp_merge(types.SimpleNamespace(otel_dir=str(otel), results=str(results)))
    d = json.loads(builder.read_text())
    h, o = d["host"], d["host_otlp"]
    # the sampler path has every field the summary knows, with numbers, so a null on either side would show
    for k in ("read_mb_s", "disk_queue_depth_mean", "cpu_busy_pct_top_cpus", "cpu_steal_share", "io_some_share",
              "throttled_share", "cpu_mhz_mean", "ctxsw_vol_per_s", "anon_huge_gb", "gpu_busy_pct", "rss_max_gb",
              "cache_file_gb_max", "majflt_per_token", "cgroup_read_bytes", "cpu_busy_pct"):
        assert h["decode"][k] is not None, k
    assert len(h["decode_windows"]) == 4 and h["decode_windows"][0]["samples"] == 6
    assert_same(h, o)
    assert d["host_otlp_diff"] == {}


def test_host_block_agrees_when_hostmetrics_scrapes_off_the_tick(tmp_path):
    # the sampler and hostmetrics run on their own clocks: disk counters are interpolated to the tick, CPU shares use
    # the scrape times, so constant rates give the same block although every scrape is 0.4 s early
    samples, otel, results, builder = write_inputs(tmp_path, host_offset_s=0.4, constant=True)
    summarize_sampler(samples, builder, tmp_path)
    es.otlp_merge(types.SimpleNamespace(otel_dir=str(otel), results=str(results)))
    d = json.loads(builder.read_text())
    assert d["host"]["decode"]["read_mb_s"] is not None
    assert_same(d["host"], d["host_otlp"])
    assert d["host_otlp_diff"] == {}


def test_hostmetrics_points_join_as_of_the_sampler_tick(tmp_path):
    # hostmetrics scraped 0.4 s before each sampler tick: every row still gets the disk and CPU fields
    samples, otel, results, builder = write_inputs(tmp_path, host_offset_s=0.4)
    pts = oh.read_points(sorted(otel.rglob("*.jsonl")))
    rows = oh.rows(pts)
    assert len(rows) == TICKS
    assert all(r["disk_read_bytes"] is not None and len(r["cpus"]) == CPUS and r["cpu_steal"] for r in rows)
    # the busiest-CPU totals are the scrape times, not the tick times
    assert rows[5]["cpus"][0][1] == pytest.approx(T0 + 5 - 0.4)


def test_stale_or_foreign_hostmetrics_points_are_not_used(tmp_path):
    # the collector stops after 5 scrapes: from MAX_AGE_S after the last one, rows get no disk or CPU fields
    samples, otel, results, builder = write_inputs(tmp_path, host_ticks=5)
    rows = oh.rows(oh.read_points(sorted(otel.rglob("*.jsonl"))))
    assert rows[6]["disk_read_bytes"] is not None and rows[7]["disk_read_bytes"] is None
    assert rows[7]["cpus"] == [] and rows[7]["cpu_steal"] is None
    # another device's points are not the model disk's
    assert oh.rows(oh.read_points(sorted(otel.rglob("*.jsonl"))), disk="sdb")[0]["disk_read_ops"] is None


def test_kt_style_result_uses_its_host_windows(tmp_path):
    samples, otel, results, builder = write_inputs(tmp_path)
    builder.unlink()
    rows = es.load_samples(str(samples))
    pre, dec = (T0 + 2.5, T0 + 12.5, 512), (T0 + 12.5, T0 + 36.5, 63)
    kt = results / "exit-ktransformers-120000.json"
    kt.write_text(json.dumps({"engine": "ktransformers", "host": es.host_summary(rows, pre, dec, 3, []),
                              "host_windows": es.host_windows(pre, dec, 3)}))
    es.otlp_merge(types.SimpleNamespace(otel_dir=str(otel), results=str(results)))
    d = json.loads(kt.read_text())
    assert_same(d["host"], d["host_otlp"])
    assert d["host_otlp"]["decode_windows"][0]["tokens_estimated"] is True


def test_merge_without_files_or_windows_changes_nothing(tmp_path, capsys):
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "x.json").write_text('{"host": {}}')
    es.otlp_merge(types.SimpleNamespace(otel_dir=str(tmp_path / "none"), results=str(tmp_path / "r")))
    assert "nothing to merge" in capsys.readouterr().out
    assert json.loads((tmp_path / "r" / "x.json").read_text()) == {"host": {}}


def test_host_diff_lists_nulls_and_large_differences():
    a = {"decode": {"x": 1.0, "y": 2.0, "z": None, "s": "a", "l": [1]}, "decode_windows": [{"x": 1.0}]}
    b = {"decode": {"x": 1.04, "y": 3.0, "z": 5.0, "s": "b", "l": [2]}, "decode_windows": [{"x": 2.0}]}
    assert es.host_diff(a, b) == {"decode.y": [2.0, 3.0], "decode.z": [None, 5.0], "decode.s": ["a", "b"],
                                  "decode_windows.0.x": [1.0, 2.0]}


def test_request_shape():
    rec = counters()[0]
    req = oh.request(oh.points(rec), rec["t"], oh.resource(env={"CONTAINER_ID": "c-9"}), start=T0 - 1)
    rm = req["resourceMetrics"][0]
    res = {a["key"]: a["value"] for a in rm["resource"]["attributes"]}
    assert res["service.name"] == {"stringValue": oh.SERVICE} and res["host.id"] == {"stringValue": "c-9"}
    ms = {m["name"]: m for m in rm["scopeMetrics"][0]["metrics"]}
    assert ms["container.cpu.time"]["sum"]["isMonotonic"] is True and ms["container.cpu.time"]["unit"] == "s"
    assert ms["container.memory.usage"]["sum"]["isMonotonic"] is False
    assert "gauge" in ms["hw.gpu.utilization"]
    dp = ms["tetraeder.host.process.paging.faults"]["sum"]["dataPoints"][0]
    assert dp["asInt"] == str(rec["majflt"]) and dp["timeUnixNano"] == str(T0 * 10**9)
    thr = {(a["value"]["stringValue"]): d["asInt"] for d in ms["tetraeder.host.gpu.throttled"]["gauge"]["dataPoints"]
           for a in d["attributes"] if a["key"] == "tetraeder.gpu.throttle_reason"}
    assert thr["sw_power_cap"] == "1" and thr["hw_thermal"] == "0"
    # one source per field: the sampler leaves disk and per-CPU series to hostmetrics
    assert not {"system.disk.io", "system.cpu.time", "tetraeder.host.cpu.busy"} & set(ms)
    assert oh.points({"gpu": None, "io_psi": {"src": None, "some": None, "full": None}}) == []


def test_poster_never_raises_and_gives_up():
    calls = []

    def broken(req, timeout):
        calls.append(timeout)
        raise OSError("refused")

    p = oh.Poster("http://127.0.0.1:4318/", opener=broken)
    for _ in range(oh.MAX_FAILS + 5):
        assert p.send({}) is False
    assert len(calls) == oh.MAX_FAILS and p.off and p.url == "http://127.0.0.1:4318/v1/metrics"


def test_endpoint_switch():
    assert oh.endpoint(env={"TETRAEDER_HOST_OTLP": "http://c:4318"}) == "http://c:4318"
    assert oh.endpoint(env={"TETRAEDER_HOST_OTLP": "http://c:4318", "SAMPLER_OTLP": "0"}) is None
    assert oh.endpoint("http://x:1", env={}) == "http://x:1"
    assert oh.endpoint(env={}) is None


def test_gpu_parser_keeps_util_when_other_fields_are_missing():
    g = es.parse_gpu("37, 20480, 24576, [N/A], 61, 1695, 9751, 0x0000000000000024, 4, 16")
    assert g["util"] == 37 and g["mem_mib"] == 20480 and g["power_w"] is None and g["throttle"] == 0x24
    assert es.parse_gpu("12, 100") == {"util": 12, "mem_mib": 100, "mem_total_mib": None, "power_w": None,
                                       "temp_c": None, "sm_mhz": None, "mem_mhz": None, "throttle": None,
                                       "pcie_gen": None, "pcie_width": None}
    assert es.parse_gpu("[N/A], 100") is None and es.parse_gpu("") is None


def test_cgroup_limits(monkeypatch):
    files = {"/sys/fs/cgroup/cpu.max": "2304000 100000\n", "/sys/fs/cgroup/memory.current": "123\n",
             "/sys/fs/cgroup/memory.max": "max\n"}
    monkeypatch.setattr(es.ds, "read", lambda p, default="": files.get(p, default))
    assert es.cpu_quota() == 23.04 and es.cgroup_mem_now() == {"current": 123, "max": None}
    files = {"/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "-1\n", "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000\n",
             "/sys/fs/cgroup/memory/memory.usage_in_bytes": "5\n",
             "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n"}
    assert es.cpu_quota() is None and es.cgroup_mem_now() == {"current": 5, "max": None}
    files = {"/proc/diskstats": " 259       0 nvme0n1 1 2 3\n 259       1 nvme0n1p1 1 2 3\n"}
    assert es.disk_name("259:0") == "nvme0n1" and es.disk_name("259:1") == "nvme0n1p1" and es.disk_name("8:0") is None


def busiest_disk():
    """MAJ:MIN of the device with the most sectors read (not loop or ram)."""
    best = max((ln.split() for ln in open("/proc/diskstats") if not ln.split()[2].startswith(("loop", "ram"))),
               key=lambda f: int(f[5]))
    return f"{best[0]}:{best[1]}"


@pytest.mark.skipif(not (sys.platform == "linux" and os.environ.get("OTELCOL_BIN") and shutil.which("flock")),
                    reason="needs Linux and OTELCOL_BIN (scripts/otelcol_install.sh); CI's otelcol step sets it")
def test_box_collector_end_to_end(tmp_path):
    # scripts/otelcol_box.sh with the real collector and config, the sampler posting for 6 s: the rows carry the
    # Linux-only hostmetrics fields (8 CPU states, weighted I/O time) and the sampler fields
    env = dict(os.environ, OTELCOL_STATE=str(tmp_path), TETRAEDER_SESSION="")
    otel = tmp_path / "otel"
    box = str(SCRIPTS / "otelcol_box.sh")
    subprocess.run(["bash", box, "start", str(otel)], env=env, check=True, timeout=60)
    try:
        sleeper = subprocess.Popen(["sleep", "6"])
        sampler = subprocess.Popen([sys.executable, str(SCRIPTS / "engine_sample.py"), "sample", "--pid",
                                    str(sleeper.pid), "--disk", busiest_disk(), "--out", str(tmp_path / "s.jsonl")],
                                   env=dict(env, TETRAEDER_HOST_OTLP="http://127.0.0.1:4318", TETRAEDER_STAGE="ci"))
        # reap the sleeper: the sampler runs while /proc/PID exists, and a zombie keeps it
        sleeper.wait()
        assert sampler.wait(timeout=30) == 0
    finally:
        subprocess.run(["bash", box, "stop"], env=env, timeout=60)
    pts = oh.read_points(sorted(otel.rglob("*.jsonl")))
    assert any(p["resource"].get("tetraeder.stage") == "ci" for p in pts)
    assert any(p["resource"].get("tetraeder.session") == "unknown" for p in pts if p["scope"] != oh.SCOPE)
    rows = oh.rows(pts)
    assert len(rows) >= 4, len(rows)
    for r in rows[2:]:
        assert r["cpus"] and r["cpu_steal"] is not None, r
        assert r["disk_read_bytes"] is not None and r["disk_weighted_ms"] is not None, r
        assert r["majflt"] is not None and r["rss_kb"] is not None
