"""scripts/vastctl.py: offer ranking, endpoint resolution, health classification, watchdog policies, destroy guard
and key hygiene. No network: httpx.MockTransport plays the Vast API, subprocess.run is replaced by a fake ssh/scp."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("vastctl", ROOT / "scripts" / "vastctl.py")
vc = importlib.util.module_from_spec(_spec)
sys.modules["vastctl"] = vc
_spec.loader.exec_module(vc)

KEY = "sk-TESTKEY-0123456789abcdef"
INSTANCE_KEY = "INSTKEY-SECRET-987654321"


def make_instance(iid=100, status="running", port=41022, host_id=555, runtype="ssh_direc ssh_proxy"):
    return {"id": iid, "actual_status": status, "host_id": host_id, "machine_id": 9, "gpu_name": "RTX 3090",
            "dph_total": 0.4, "public_ipaddr": "1.2.3.4", "ssh_host": "ssh5.vast.ai", "ssh_port": 19554,
            "image_runtype": runtype, "label": "tetraeder", "start_date": time.time() - 3600,
            "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": str(port)}]} if status == "running" else None,
            "instance": {"diskHour": 0.06}, "extra_env": [["A", "1"]], "jupyter_token": "JUPYTOKEN-abcdef"}


def offer(oid, host, price, score, rel=0.995, verified=None, verification="verified", ports=10, vram=24564):
    return {"id": oid, "host_id": host, "machine_id": host + 1, "gpu_name": "RTX 3090", "gpu_ram": vram,
            "dph_total": price, "dlperf_per_dphtotal": score, "reliability2": rel, "verified": verified,
            "verification": verification, "direct_port_count": ports, "geolocation": "X", "inet_down": 900,
            "cuda_max_good": 12.8}


class FakeVast:
    def __init__(self):
        self.requests: list[tuple[str, str, bytes]] = []
        self.instances: dict[int, dict] = {}
        self.offers: list[dict] = []
        self.credit = 10.0
        self.fail = False

    def handler(self, req: httpx.Request) -> httpx.Response:
        self.requests.append((req.method, req.url.path, req.content))
        assert req.headers["authorization"] == f"Bearer {KEY}"
        if self.fail:  # an API that echoes the credential back must not leak it through our error messages
            return httpx.Response(500, text=f"internal error for {req.headers['authorization']}")
        p = req.url.path
        if p == "/api/v0/users/current/":
            return httpx.Response(200, json={"credit": self.credit, "balance": 0})
        if p == "/api/v1/instances/":
            return httpx.Response(200, json={"instances": list(self.instances.values()), "next_token": None})
        if p == "/api/v0/bundles/":
            return httpx.Response(200, json={"offers": self.offers})
        m = re.fullmatch(r"/api/v0/asks/(\d+)/", p)
        if m and req.method == "PUT":
            self.instances[999] = make_instance(999, host_id=777)
            return httpx.Response(200, json={"success": True, "new_contract": 999, "instance_api_key": INSTANCE_KEY})
        if re.fullmatch(r"/api/v0/instances/reboot/(\d+)/", p):
            return httpx.Response(200, json={"success": True})
        m = re.fullmatch(r"/api/v0/instances/(\d+)/", p)
        if m:
            iid = int(m.group(1))
            if req.method == "GET":
                return httpx.Response(200, json={"instances": self.instances.get(iid)})
            if req.method == "PUT":
                body = json.loads(req.content)
                if body.get("state") == "stopped":
                    self.instances[iid]["actual_status"] = "exited"
                return httpx.Response(200, json={"success": True})
            if req.method == "DELETE":
                self.instances.pop(iid, None)
                return httpx.Response(200, json={"success": True})
        return httpx.Response(404, text="not found")

    def calls(self, method, pattern=""):
        return [r for r in self.requests if r[0] == method and pattern in r[1]]


def tar_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in files.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


class FakeShell:
    """Stands in for subprocess.run: ssh/scp by port; ports in `down` refuse connections."""

    def __init__(self, tmp: Path):
        self.argvs: list[list[str]] = []
        self.down: set[int] = set()
        self.health = "exp_procs=2\nbatch_alive=1\ngpu_util=90\nprogress_age_s=30\nwatchdog=1\n"
        self.archive = self._archive(tmp)
        self.verify = "verify_gpu=1\nverify_tools=1\nverify_ok=1\n"
        self.git_paths = ["crates", "scripts", "Cargo.toml"]
        self.sub_files = {"CMakeLists.txt": b"project(llama)"}
        self.stdins: list[bytes] = []
        self.nightly_rc = 0

    @staticmethod
    def _archive(tmp: Path) -> bytes:
        src = tmp / "box"
        f = src / "bench" / "results" / "20260101" / "DONE"
        f.parent.mkdir(parents=True)
        f.write_text('{"ok": 1}')
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            tf.add(f, arcname="bench/results/20260101/DONE")
        return buf.getvalue()

    def __call__(self, argv, input=None, capture_output=True, timeout=None, check=False, stdin=None, **kw):
        self.argvs.append(list(argv))
        flag = "-P" if argv[0] == "scp" else "-p"
        port = int(argv[argv.index(flag) + 1]) if flag in argv else 0

        def res(rc, out=b"", err=b""):
            return subprocess.CompletedProcess(argv, rc, out, err)

        if argv[0] == "git":
            if "ls-tree" in argv:
                return res(0, "\n".join(self.git_paths).encode())
            if "archive" in argv:
                if "-C" in argv:  # the submodule
                    prefix = next(a.split("=", 1)[1] for a in argv if a.startswith("--prefix="))
                    return res(0, tar_bytes({prefix + k: v for k, v in self.sub_files.items()}))
                paths = argv[argv.index("--") + 1:]
                return res(0, tar_bytes({f"{x}/f.txt" if "." not in x else x: b"x" for x in paths}))
            return res(0, b"")
        if input is not None:
            self.stdins.append(input)
        if port in self.down:
            return res(255, b"", b"ssh: connect to host 1.2.3.4 port %d: Connection refused" % port)
        if argv[0] == "scp":
            Path(argv[-1]).write_bytes(self.archive)
            return res(0)
        cmd = argv[-1]
        banner = b"Welcome to vast.ai. Have fun!\nAI agents: read the docs\n"
        if cmd == "echo up":
            return res(0, banner + b"up\n")
        if "vast_box.sh.tmp" in cmd:
            return res(0, b"helper-ok\n")
        if "vast_box.sh health" in cmd:
            return res(0, banner + self.health.encode())
        if "vast_box.sh pack" in cmd:
            return res(0, b"1 files, 4K -> /root/results.tgz\nmd5=%s\n" % hashlib.md5(self.archive).hexdigest().encode())
        if "vast_box.sh verify" in cmd:
            return res(0, self.verify.encode())
        if "install-push" in cmd:
            return res(0, b"installed 3 files into /root/tetraeder\n")
        if "nightly.log" in cmd:
            return res(0, b"12:00:00\nNIGHTLY_DONE rc=%d\n" % self.nightly_rc)
        if "ensure-watchdog" in cmd:
            return res(0, b"watchdog=1 mode=supervisor\n")
        return res(0, b"ok\n")


@pytest.fixture
def env(tmp_path, monkeypatch):
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "vast_box.sh", tmp_path / "scripts" / "vast_box.sh")
    key = tmp_path / "vast_api_key"
    key.write_text(KEY + "\n")
    fake = FakeVast()
    shell = FakeShell(tmp_path)
    monkeypatch.setattr(vc.subprocess, "run", shell)
    monkeypatch.setattr(vc.subprocess, "call", lambda argv: shell(argv).returncode)
    ctx = vc.Ctx(root=tmp_path, vast_dir=tmp_path / ".vast", key_file=key, ssh_key=tmp_path / "id_test",
                 transport=httpx.MockTransport(fake.handler), sleep=lambda s: None)
    return ctx, fake, shell


def tick(ctx, pol=None):
    """The watchdog reloads state from disk each pass (so manual commands in another shell are seen): persist
    the test setup first, and re-read records afterwards via R(ctx)."""
    ctx.store.save()
    return vc.watchdog_tick(ctx, pol or vc.Policy())


def R(ctx, iid=100):
    return ctx.store.rec(iid)


def manage(ctx, fake, iid=100, **kw):
    fake.instances[iid] = make_instance(iid, **kw)
    rec = ctx.store.rec(iid, create=True)
    rec["host_id"] = fake.instances[iid]["host_id"]
    ctx.store.state["current"] = str(iid)
    return rec


# ------------------------------------------------------------------------------------------------ offers
def test_offer_filtering_and_ranking_with_blocklist():
    offers = [
        offer(1, 483833, 0.20, 400),                       # best score but blocklisted host
        offer(2, 10, 0.30, 250),                           # verified=None, verification=verified -> kept
        offer(3, 11, 0.15, 200),                           # cheapest
        offer(4, 12, 0.25, 300, verification="unverified"),
        offer(5, 13, 0.60, 500),                           # over max price
        offer(6, 14, 0.22, 260, ports=0),                  # no direct ports
        offer(7, 15, 0.21, 280, rel=0.95),                 # unreliable
        offer(8, 16, 0.18, 270, vram=12288),               # too little VRAM
    ]
    ranked = vc.filter_rank_offers(offers, {"483833"}, max_price=0.45, min_rel=0.99, min_vram_gb=24)
    assert [o["id"] for o in ranked] == [2, 3]
    assert {o["id"]: o["price_rank"] for o in ranked} == {3: 1, 2: 2}
    by_price = vc.filter_rank_offers(offers, {"483833"}, max_price=0.45, min_rel=0.99, min_vram_gb=24,
                                     sort="price")
    assert [o["id"] for o in by_price] == [3, 2]
    q = vc.build_offer_query(["RTX 3090"], 24, 0.99, 500, 50, 12.4, False)
    assert q["gpu_ram"]["gte"] < 23034  # an L4 (23034 MiB) still counts as a 24 GB card
    assert q["reliability2"] == {"gte": 0.99} and q["type"] == "on-demand"


def test_offers_command_uses_state_blocklist(env, capsys):
    ctx, fake, _ = env
    fake.offers = [offer(1, 483833, 0.2, 400), offer(2, 10, 0.3, 250)]
    assert vc.main(["offers"], ctx) == 0
    out = capsys.readouterr().out
    assert "offer         2" in out and "host 483833" not in out
    body = json.loads(fake.calls("POST", "/bundles/")[0][2])
    assert body["verified"] == {"eq": True} and body["direct_port_count"] == {"gte": 1}
    assert body["gpu_name"] == {"in": ["RTX 3090"]}  # one GPU class for comparable benchmarks


# ------------------------------------------------------------------------------------------------ endpoints
def test_endpoint_resolution_from_instance_with_changed_ports():
    inst = make_instance(port=41022, runtype="jupyter_direc ssh_direc ssh_proxy")
    eps = vc.endpoints_from_instance(inst)
    assert eps[0] == vc.Endpoint("direct", "1.2.3.4", 41022)
    assert vc.Endpoint("proxy", "ssh5.vast.ai", 19555) in eps  # jupyter runtype: proxy port + 1
    assert vc.endpoints_from_instance(make_instance(status="exited"))[0].kind == "proxy"
    assert vc.endpoints_from_instance(None) == []


def test_remote_run_reresolves_stale_cached_port(env):
    ctx, fake, shell = env
    rec = manage(ctx, fake, port=41022)
    rec["endpoints"] = {"last_ok": {"kind": "direct", "host": "1.2.3.4", "port": 22488}}  # port before a restart
    shell.down = {22488}
    r = vc.Remote(ctx, 100).run("echo up")
    assert r.out == "up"  # banner stripped
    ports = [int(a[a.index("-p") + 1]) for a in shell.argvs if a[0] == "ssh"]
    assert ports[:2] == [22488, 41022]
    assert ctx.store.rec(100)["endpoints"]["last_ok"]["port"] == 41022
    ssh = shell.argvs[0]
    for opt in ("ConnectTimeout=10", "BatchMode=yes", "StrictHostKeyChecking=accept-new", "IdentitiesOnly=yes"):
        assert opt in ssh


def test_remote_run_raises_unreachable_after_rounds(env):
    ctx, fake, shell = env
    manage(ctx, fake, port=41022)
    shell.down = {41022, 19554}
    with pytest.raises(vc.Unreachable):
        vc.Remote(ctx, 100).run("echo up", rounds=2)
    assert len([a for a in shell.argvs if a[0] == "ssh"]) == 6  # 3 rounds x (direct + proxy)


# ------------------------------------------------------------------------------------------------ health
CFG = vc.DEFAULT_CONFIG


def test_classify_states():
    assert vc.classify("running", None, CFG)[0] == "unreachable"
    assert vc.classify("running", {"exp_procs": "3", "gpu_util": "85", "progress_age_s": "60"}, CFG)[0] == "healthy"
    assert vc.classify("running", {"exp_procs": "0", "batch_alive": "0", "gpu_util": "0"}, CFG)[0] == "idle"
    assert vc.classify("running", {"exp_procs": "2", "gpu_util": "70", "progress_age_s": "1500"}, CFG)[0] == "stuck"
    assert vc.classify("running", {"exp_procs": "2", "gpu_util": "0", "progress_age_s": "400"}, CFG)[0] == "stuck"
    assert vc.classify("running", {"exp_procs": "2", "gpu_util": "0", "progress_age_s": "30"}, CFG)[0] == "healthy"
    assert vc.classify("exited", None, CFG)[0] == "exited"


def test_health_records_outage_and_recommends_migration(env):
    ctx, fake, shell = env
    rec = manage(ctx, fake)
    now = time.time()
    rec["outages"] = [{"start": now - 7000, "end": now - 5000, "min": 33}, {"start": now - 3000, "end": now - 1500,
                                                                            "min": 25}]
    shell.down = {41022, 19554}
    h = vc.health_check(ctx, "100")
    assert h["class"] == "unreachable" and rec["unreachable_since"]
    assert any(p["key"] == "migrate-100" for p in ctx.store.state["pending"])  # 3rd outage within 3 h
    shell.down = set()
    rec["unreachable_since"] = time.time() - 1680
    h = vc.health_check(ctx, "100")
    assert h["class"] == "healthy" and "unreachable_since" not in rec
    assert rec["outages"][-1]["min"] == pytest.approx(28, abs=0.5)


# ------------------------------------------------------------------------------------------------ destroy / fetch
def test_destroy_refused_without_fetch(env, capsys):
    ctx, fake, _ = env
    rec = manage(ctx, fake)
    rec["runs"] = {"bench/results": {"ok": False}}
    assert vc.main(["destroy"], ctx) == 1          # no --yes
    assert vc.main(["destroy", "--yes"], ctx) == 1  # runs not fetched
    assert not fake.calls("DELETE")
    assert "not fetched" in capsys.readouterr().out
    assert vc.main(["destroy", "--yes", "--force"], ctx) == 0
    assert fake.calls("DELETE", "/instances/100/")


def test_fetch_verifies_unpacks_and_allows_destroy(env):
    ctx, fake, _ = env
    rec = manage(ctx, fake)
    rec["runs"] = {"bench/results": {"ok": False}}
    assert vc.fetch(ctx, "100")
    assert (ctx.root / "bench/results/20260101/DONE").exists()
    assert rec["runs"]["bench/results"]["ok"] and rec["runs"]["bench/results"]["markers"] == 1
    assert vc.main(["destroy", "--yes"], ctx) == 0
    assert fake.calls("DELETE")


# ------------------------------------------------------------------------------------------------ watchdog
def test_watchdog_idle_fetches_then_stops(env):
    ctx, fake, shell = env
    rec = manage(ctx, fake)
    rec["runs"] = {"bench/results": {"ok": False}}
    rec["idle_since"] = time.time() - 25 * 60
    shell.health = "exp_procs=0\nbatch_alive=0\ngpu_util=0\nwatchdog=1\n"
    tick(ctx)
    assert any(a[0] == "scp" for a in shell.argvs)
    assert json.loads(fake.calls("PUT", "/instances/100/")[0][2]) == {"state": "stopped"}
    assert R(ctx)["runs"]["bench/results"]["ok"]


def test_watchdog_relaunches_dead_onbox_watchdog(env):
    ctx, fake, shell = env
    manage(ctx, fake)
    shell.health = "exp_procs=1\nbatch_alive=1\ngpu_util=80\nprogress_age_s=10\nwatchdog=0\n"
    tick(ctx)
    assert any("ensure-watchdog" in a[-1] for a in shell.argvs if a[0] == "ssh")


@pytest.mark.parametrize("busy,minutes,stops", [(False, 50, True), (True, 50, False), (True, 185, True)])
def test_watchdog_stops_unreachable_and_blocklists_host(env, busy, minutes, stops):
    ctx, fake, shell = env
    rec = manage(ctx, fake)
    rec["runs"] = {"bench/results": {"ok": False}}
    rec["last_contact"] = {"t": time.time() - minutes * 60, "exp_procs": 2 if busy else 0, "batch_alive": 0}
    rec["unreachable_since"] = time.time() - minutes * 60
    shell.down = {41022, 19554}
    tick(ctx)
    assert bool(fake.calls("PUT", "/instances/100/")) == stops
    assert ("555" in ctx.store.state["blocklist"]) == stops
    assert (R(ctx)["role"] == "salvage") == stops
    assert not fake.calls("DELETE")


def test_unreachable_stop_without_runs_keeps_box_for_the_12h_rule(env):
    ctx, fake, shell = env
    rec = manage(ctx, fake)
    rec["unreachable_since"] = time.time() - 60 * 60
    shell.down = {41022, 19554}
    tick(ctx)
    tick(ctx)
    assert fake.calls("PUT", "/instances/100/") and not fake.calls("DELETE")
    assert R(ctx)["role"] == "active"


def test_missing_ssh_binary_is_not_an_outage(env, monkeypatch):
    ctx, fake, _ = env
    rec = manage(ctx, fake)
    rec["unreachable_since"] = time.time() - 5 * 3600

    def no_ssh(argv, **kw):
        raise FileNotFoundError(argv[0])
    monkeypatch.setattr(vc.subprocess, "run", no_ssh)
    ctx.store.save()
    assert vc.watchdog_loop(ctx, vc.Policy(), once=True, interval=1) == 0
    assert not fake.calls("PUT") and not fake.calls("DELETE")
    assert "not found on PATH" in (ctx.store.dir / "events.log").read_text()


def test_watchdog_destroys_long_stopped_only_when_fetched(env):
    ctx, fake, _ = env
    rec = manage(ctx, fake, status="exited")
    rec["stopped_since"] = time.time() - 13 * 3600
    rec["runs"] = {"bench/results": {"ok": False}}
    tick(ctx)
    assert not fake.calls("DELETE") and R(ctx)["role"] == "salvage"
    R(ctx)["runs"]["bench/results"]["ok"] = True
    tick(ctx)
    assert fake.calls("DELETE", "/instances/100/") and R(ctx)["role"] == "gone"


def test_salvage_starts_old_box_then_gives_up_and_logs_losses(env):
    ctx, fake, _ = env
    rec = manage(ctx, fake, status="exited")
    rec.update(role="salvage", salvage_since=time.time() - 3 * 3600, last_salvage_attempt=time.time() - 3 * 3600,
               runs={"bench/results": {"ok": False}})
    tick(ctx)
    assert json.loads(fake.calls("PUT", "/instances/100/")[-1][2]) == {"state": "running"}
    R(ctx)["salvage_since"] = time.time() - 25 * 3600
    tick(ctx)
    assert fake.calls("DELETE")
    log = (ctx.store.dir / "events.log").read_text()
    assert "LOST results" in log and "bench/results" in log


@pytest.fixture
def replace_env(env, monkeypatch):
    ctx, fake, shell = env
    (ctx.vast_dir / "config.json").write_text(json.dumps({"auto_replace": True}))  # off by default
    ctx.store.config = ctx.store._load_config()
    ctx.store.state["batch"] = {"script": "scripts/nightly.sh", "args": [], "runs_dirs": ["bench/results"],
                                  "instance": "100"}
    ctx.store.state["need_replacement"] = {"t": time.time(), "old": "100", "why": "migrate"}
    fake.offers = [offer(1, 555, 0.2, 400), offer(2, 20, 0.3, 250)]
    started, order = [], []
    monkeypatch.setattr(vc, "wait_ready", lambda c, i, m: vc.Endpoint("direct", "1.2.3.4", 41022))
    monkeypatch.setattr(vc, "provision", lambda c, i, *a, **k: order.append(("provision", i)) or True)
    monkeypatch.setattr(vc, "push_runs", lambda c, r, d: order.append(("push_runs", r.iid)) or 0)
    monkeypatch.setattr(vc, "start_batch", lambda c, i, b: order.append(("start_batch", i)) or started.append(i))
    return ctx, fake, shell, started, order


def test_make_before_break_switches_only_after_verify(replace_env):
    ctx, fake, shell, _, order = replace_env
    rec = manage(ctx, fake)  # old box still running and reachable
    rec["runs"] = {"bench/results": {"ok": False}}
    rec["inventory"] = {"models": ["org/model:m.gguf"]}
    tick(ctx)
    asks = fake.calls("PUT", "/asks/")
    assert asks and asks[0][1] == "/api/v0/asks/2/"  # the old host 555 is excluded
    assert any(a[0] == "ssh" and "vast_box.sh verify" in a[-1] for a in shell.argvs), "replacement not verified"
    stops = [r for r in fake.requests if r[0] == "PUT" and r[1] == "/api/v0/instances/100/"]
    assert stops and json.loads(stops[0][2]) == {"state": "stopped"}  # old one stopped only after the switch
    assert order == [("provision", "999"), ("push_runs", "999"), ("start_batch", "999")]
    assert ctx.store.state["current"] == "999" and ctx.store.state["need_replacement"] is None
    assert R(ctx)["runs"]["bench/results"]["ok"]  # final fetch from the old box before retiring it
    assert fake.calls("DELETE", "/instances/100/")  # retired and fully fetched -> destroyed


def test_failed_replacement_is_destroyed_and_old_kept(replace_env):
    ctx, fake, shell, started, _ = replace_env
    manage(ctx, fake)
    shell.verify = "verify_gpu=1\nverify_tools=0\nverify_ok=0\n"
    tick(ctx)
    assert fake.calls("DELETE", "/instances/999/") and not fake.calls("DELETE", "/instances/100/")
    assert ctx.store.state["current"] == "100" and started == []
    assert "20" in ctx.store.state["blocklist"]
    assert not [r for r in fake.requests if r[0] == "PUT" and r[1] == "/api/v0/instances/100/"]


def test_replacement_limits_and_spend_guard(replace_env):
    ctx, fake, _, started, _ = replace_env
    manage(ctx, fake, status="exited")
    manage(ctx, fake, iid=101)  # another running managed box; old one stopped -> steady-state max 1 applies
    ctx.store.state["current"] = "100"
    tick(ctx, vc.Policy(destroy=False))
    assert not fake.calls("PUT", "/asks/")
    ctx.store.state["instances"].pop("101")
    fake.instances.pop(101)
    fake.credit = 0.5  # runway < 3 h with the new box billing too -> no rental
    tick(ctx, vc.Policy(destroy=False))
    assert not fake.calls("PUT", "/asks/")
    ctx.store.state["need_replacement"]["next_try"] = 0
    fake.credit = 50.0
    tick(ctx, vc.Policy(destroy=False))
    assert fake.calls("PUT", "/asks/2/") and started == ["999"]


def test_low_balance_warns(env, capsys):
    ctx, fake, _ = env
    manage(ctx, fake)
    fake.credit = 1.0  # 0.4 $/h -> 2.5 h runway
    tick(ctx)
    assert any(p["key"] == "balance" for p in ctx.store.state["pending"])
    assert "low balance" in capsys.readouterr().out


def test_watchdog_loop_survives_errors(env):
    ctx, fake, _ = env
    fake.fail = True
    assert vc.watchdog_loop(ctx, vc.Policy(), once=True, interval=1) == 0


# ------------------------------------------------------------------------------------------------ secrets
def test_key_never_printed_logged_or_on_command_line(env, capsys):
    ctx, fake, shell = env
    rec = manage(ctx, fake)
    rec["runs"] = {"bench/results": {"ok": False}}
    fake.offers = [offer(2, 10, 0.3, 250)]
    for argv in (["balance"], ["--json", "balance"], ["offers"], ["status"], ["endpoint"], ["health"], ["fetch"],
                 ["logs"], ["ssh", "--", "nvidia-smi"], ["create", "--auto", "--allow-second"]):
        vc.main(argv, ctx)
    fake.fail = True
    for argv in (["balance"], ["offers"], ["status"], ["destroy", "--yes", "--force"], ["stop"], ["--json", "reboot"]):
        vc.main(argv, ctx)
    vc.watchdog_loop(ctx, vc.Policy(), once=True, interval=1)
    out = capsys.readouterr()
    text = out.out + out.err
    text += (ctx.store.dir / "state.json").read_text() + (ctx.store.dir / "events.log").read_text()
    text += json.dumps(shell.argvs)
    for secret in (KEY, INSTANCE_KEY, "JUPYTOKEN-abcdef"):
        assert secret not in text
    assert "***" in text  # the echoed credential was redacted, not silently dropped


# ------------------------------------------------------------------------------------------------ tetraeder additions
def test_defaults_and_seed_blocklist(env):
    ctx, _, _ = env
    cfg = json.loads((ctx.vast_dir / "config.json").read_text())  # written by Store when missing
    assert cfg["auto_replace"] is False and cfg["max_instances"] == 1
    assert cfg["max_price_per_hour"] == 0.45 and cfg["disk_gb"] == 80
    assert ctx.store.config["auto_replace"] is False
    assert {"483833", "366851"} <= set(ctx.store.blocklist())
    assert vc.LABEL == "tetraeder" and vc.GPU_CLASS == ["RTX 3090"] and vc.IMAGE.startswith("ghcr.io/lhns/")


def test_offers_prefer_last_good_host_but_blocklist_wins():
    offers = [offer(1, 10, 0.20, 400), offer(2, 20, 0.30, 250), offer(3, 30, 0.25, 300)]
    base = vc.filter_rank_offers(offers, set(), max_price=0.45, min_rel=0.99, min_vram_gb=24)
    assert [o["id"] for o in base] == [1, 3, 2]
    pref = vc.filter_rank_offers(offers, set(), max_price=0.45, min_rel=0.99, min_vram_gb=24, prefer_host="20")
    assert [o["id"] for o in pref] == [2, 1, 3]  # last-good host first, the rest keep their order
    blocked = vc.filter_rank_offers(offers, {"20"}, max_price=0.45, min_rel=0.99, min_vram_gb=24, prefer_host="20")
    assert [o["id"] for o in blocked] == [1, 3]
    over = vc.filter_rank_offers(offers, set(), max_price=0.25, min_rel=0.99, min_vram_gb=24, prefer_host="20")
    assert [o["id"] for o in over] == [1, 3]  # preference never overrides the price cap


def test_offers_command_ranks_last_good_host_first(env, capsys):
    ctx, fake, _ = env
    fake.offers = [offer(1, 10, 0.20, 400), offer(2, 20, 0.30, 250)]
    ctx.store.state["last_good_host"] = "20"
    assert vc.main(["offers"], ctx) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "offer " in ln and "host" in ln]
    assert "offer         2" in lines[0]


def test_verified_fetch_records_last_good_host(env):
    ctx, fake, _ = env
    rec = manage(ctx, fake)
    rec["runs"] = {"bench/results": {"ok": False}}
    assert vc.fetch(ctx, "100")
    assert ctx.store.state["last_good_host"] == "555"


def git_env(env):
    ctx, fake, shell = env
    (ctx.root / "vendor" / "llama.cpp").mkdir(parents=True)
    (ctx.root / "vendor" / "llama.cpp" / ".git").write_text("gitdir: ../../.git/modules/llama.cpp\n")
    return ctx, shell


def test_git_archive_includes_submodule_in_one_tarball(env):
    ctx, shell = git_env(env)
    data = vc.git_archive(ctx)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        names = tf.getnames()
    assert "vendor/llama.cpp/CMakeLists.txt" in names and "Cargo.toml" in names and "crates/f.txt" in names
    sub_calls = [a for a in shell.argvs if a[0] == "git" and "-C" in a]
    assert sub_calls and "--prefix=vendor/llama.cpp/" in sub_calls[0]
    main_call = next(a for a in shell.argvs if a[:2] == ["git", "archive"])
    assert main_call[main_call.index("--") + 1:] == ["crates", "scripts", "Cargo.toml"]  # missing paths skipped


def test_git_archive_without_submodule_or_missing_paths(env, capsys):
    ctx, fake, shell = env
    shell.git_paths = ["scripts"]
    data = vc.git_archive(ctx)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        assert tf.getnames() == ["scripts/f.txt"]
    out = capsys.readouterr().out
    assert "skipped" in out and "submodule vendor/llama.cpp not checked out" in out


def test_push_code_sends_gzip_to_tetraeder_dirs(env):
    ctx, fake, shell = env
    manage(ctx, fake)
    git_env(env)
    shell.stdins.clear()
    vc.push_code(ctx, vc.Remote(ctx, 100))
    cmd = [a[-1] for a in shell.argvs if a[0] == "ssh" and "install-push" in a[-1]][0]
    assert "/root/.tetraeder_push" in cmd and "/root/tetraeder" in cmd and "tar -xz" in cmd
    assert shell.stdins[-1][:2] == b"\x1f\x8b"


@pytest.fixture
def nightly_env(env, monkeypatch):
    ctx, fake, shell = env
    fake.offers = [offer(1, 555, 0.3, 400)]
    order = []
    monkeypatch.setattr(vc, "wait_ready", lambda c, i, m: order.append("wait_ready") or vc.Endpoint("d", "1.2.3.4", 1))
    monkeypatch.setattr(vc, "provision", lambda c, i, models=None, w=None: order.append(("provision", models)) or True)
    monkeypatch.setattr(vc, "start_batch",
                        lambda c, i, b: (order.append(("start", b["script"])), c.store.rec(i)["runs"].update(
                            {"bench/results": {"ok": False}}), vc.Result(0, "", ""))[2])

    def fake_fetch(c, i, *a, **k):
        order.append("fetch")
        c.store.rec(i)["runs"]["bench/results"]["ok"] = shell.fetch_ok
        return shell.fetch_ok

    shell.fetch_ok = True
    monkeypatch.setattr(vc, "fetch", fake_fetch)
    return ctx, fake, shell, order


def test_nightly_orchestration_order_and_cost(nightly_env, capsys):
    ctx, fake, shell, order = nightly_env
    assert vc.main(["nightly", "--models", "org/m:a.gguf,org/n:b.gguf"], ctx) == 0
    assert order == ["wait_ready", ("provision", ["org/m:a.gguf", "org/n:b.gguf"]),
                     ("start", "scripts/nightly.sh"), "fetch"]
    assert fake.calls("PUT", "/asks/1/") and fake.calls("DELETE", "/instances/999/")  # created, then destroyed last
    assert fake.requests.index(next(r for r in fake.requests if r[0] == "PUT" and "/asks/" in r[1])) < \
        fake.requests.index(next(r for r in fake.requests if r[0] == "DELETE"))
    assert any(a[0] == "ssh" and "nightly.log" in a[-1] and "sleep 5" in a[-1] for a in shell.argvs)  # wait on the box
    log = (ctx.vast_dir / "events.log").read_text()
    assert "nightly OK" in log and "/h" in log and "DESTROYED" in log


def test_nightly_failed_fetch_keeps_box_and_fails(nightly_env):
    ctx, fake, shell, order = nightly_env
    shell.fetch_ok = False
    assert vc.main(["nightly"], ctx) == 1
    assert not fake.calls("DELETE")  # the destroy guard holds the box for the watchdog
    assert "999" in ctx.store.state["instances"] and ctx.store.rec(999)["role"] == "active"


def test_nightly_script_failure_still_fetches_and_destroys_but_exits_nonzero(nightly_env):
    ctx, fake, shell, order = nightly_env
    shell.nightly_rc = 1
    assert vc.main(["nightly"], ctx) == 1
    assert order[-1] == "fetch" and fake.calls("DELETE", "/instances/999/")


def test_nightly_provision_failure_destroys_empty_box(nightly_env, monkeypatch):
    ctx, fake, shell, order = nightly_env
    monkeypatch.setattr(vc, "provision", lambda *a, **k: False)
    assert vc.main(["nightly"], ctx) == 1
    assert fake.calls("DELETE", "/instances/999/") and "fetch" not in order


def test_nightly_refuses_second_box(nightly_env):
    ctx, fake, shell, order = nightly_env
    manage(ctx, fake)
    ctx.store.rec(100)["role"] = "active"
    assert vc.main(["nightly"], ctx) == 1
    assert not fake.calls("PUT", "/asks/") and order == []


def test_create_uses_image_override(env):
    ctx, fake, shell = env
    fake.offers = [offer(1, 10, 0.3, 400)]
    assert vc.main(["create", "--auto", "--image", vc.FALLBACK_IMAGE], ctx) == 0
    body = json.loads(fake.calls("PUT", "/asks/")[0][2])
    assert body["image"] == vc.FALLBACK_IMAGE and body["label"] == "tetraeder"
    assert vc.main(["create", "--auto", "--allow-second"], ctx) == 0
    assert json.loads(fake.calls("PUT", "/asks/")[1][2])["image"] == vc.IMAGE


def test_parse_models():
    assert vc.parse_models("") == [] and vc.parse_models(None) == []
    assert vc.parse_models("a/b:x.gguf, c/d:y.gguf+z.gguf") == ["a/b:x.gguf", "c/d:y.gguf+z.gguf"]


def test_spec_models_reads_nightly_spec():
    models = vc.spec_models(Path(__file__).parent.parent / vc.NIGHTLY_SPEC)
    assert models and all(re.fullmatch(r"[^/:]+/[^/:]+:[^:]+\.gguf", m) for m in models)


def test_git_archive_refuses_when_nothing_is_committed(env):
    ctx, fake, shell = env
    shell.git_paths = []
    with pytest.raises(vc.VastError, match="commit first"):
        vc.git_archive(ctx)


def test_nightly_unreachable_during_provision_destroys_box(nightly_env, monkeypatch):
    ctx, fake, shell, order = nightly_env

    def boom(*a, **k):
        raise vc.Unreachable("no endpoints")
    monkeypatch.setattr(vc, "provision", boom)
    assert vc.main(["nightly"], ctx) == 1
    assert fake.calls("DELETE", "/instances/999/")


def test_nightly_respects_runway_minimum(nightly_env):
    ctx, fake, shell, order = nightly_env
    fake.credit = 0.5  # 0.3 $/h -> 1.7 h < runway_rent_min_h
    assert vc.main(["nightly"], ctx) == 1
    assert not fake.calls("PUT", "/asks/") and order == []
