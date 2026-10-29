#!/usr/bin/env python3
"""Vast.ai lifecycle manager for Tetraeder GPU boxes (REST API via httpx + ssh/scp; no vastai CLI).

    uv run python scripts/vastctl.py balance | offers | status
    uv run python scripts/vastctl.py create --auto [--image IMG]   # rent, wait for SSH, record endpoints
    uv run python scripts/vastctl.py provision [--models LIST]     # repo + toolchain + models + on-box idle watchdog
    uv run python scripts/vastctl.py run scripts/nightly.sh        # push HEAD, start a script detached
    uv run python scripts/vastctl.py health | watch | logs | ssh [-- cmd] | endpoint
    uv run python scripts/vastctl.py fetch                         # pack on the box, scp back, verify, unpack
    uv run python scripts/vastctl.py nightly [--models LIST]       # create, provision, run, wait, fetch, destroy
    uv run python scripts/vastctl.py watchdog [--once]             # local health + cost-control loop
    uv run python scripts/vastctl.py stop | start | reboot | destroy --yes

State lives in .vast/ (git-ignored): state.json (instances, endpoints cache, blocklist, runs, outages, batch),
config.json (policy thresholds; created with defaults), events.log (every action, timestamped).
The account API key is read from ~/.config/vastai/vast_api_key and only ever sent as an HTTP header; every line
this tool prints or logs goes through `redact`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
API_BASE = "https://console.vast.ai"
KEY_FILE = Path.home() / ".config" / "vastai" / "vast_api_key"
SSH_KEY = Path.home() / ".ssh" / "lrn_vast_ed25519"
# Prebuilt box image (CUDA toolchain, rustup, cmake/ninja/sccache, ggml built at the pin); built by a GitHub Actions
# workflow in Step 3 and must be public for Vast to pull it. Until then pass `--image` or use FALLBACK_IMAGE, on which
# `provision` installs the missing tools itself (slow).
IMAGE = "ghcr.io/lhns/tetraeder-box:latest"
FALLBACK_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
LABEL = "tetraeder"
REMOTE_DIR = "/root/tetraeder"
STAGING_DIR = "/root/.tetraeder_push"
BOX_HELPER = "/root/vast_box.sh"
BANNER = ("AI agents", "Welcome", "Have fun")
# One GPU model for every run so benchmark numbers stay comparable (override with --gpu for experiments).
GPU_CLASS = ["RTX 3090"]  # 24 GB, usually under the price cap; RTX 4090 rarely is
SEED_BLOCKLIST = {
    "483833": "2026-09-29: host network down ~28 min of every ~45 min (SSH + DNS outages)",
    "366851": "2026-10: stopped from outside twice, GPU rented away while stopped",
}
# Dirs tracked for fetch verification: a run counts as fetched when its DONE marker arrived (bench/results/*/DONE).
# Logs under RUNS_DIR are packed too but never gate destroy.
DEFAULT_RUNS = ["bench/results"]
RUNS_DIR = "runs/vast"
DONE_MARKER = "DONE"
NIGHTLY_SCRIPT = "scripts/nightly.sh"
NIGHTLY_SPEC = "bench/spec/phase0.toml"  # nightly.sh runs this spec; `nightly` downloads its models
ONSTART = f"[ -f {BOX_HELPER} ] && bash {BOX_HELPER} ensure-watchdog >> /root/autostop.log 2>&1 || true"

DEFAULT_CONFIG: dict[str, Any] = {
    "idle_stop_min": 20,              # running + idle this long -> fetch, then stop
    "idle_gpu_util": 10,              # GPU util below this (%) with no build/bench processes = idle
    "stuck_log_min": 20,              # build/bench processes alive but no log progress this long = stuck
    "stuck_gpu_zero_min": 5,          # ... or GPU util 0 with jobs alive and no log progress this long
    "destroy_after_stopped_h": 12,    # stopped this long (storage still bills) -> destroy once results fetched
    "unreachable_stop_idle_min": 45,  # unreachable while API says running; last contact idle -> stop
    "unreachable_stop_busy_min": 180,  # ... last contact running jobs (they keep running through host outages)
    "reboot_after_unreachable_min": 30,  # with --auto-heal: one reboot (restarts the container!)
    "outage_count": 3,                # >= this many outages ...
    "outage_window_h": 3,             # ... within this window -> recommend migration
    "runway_warn_h": 6,
    "runway_stop_h": 1,               # below: stop idle instances
    "runway_rent_min_h": 3,           # below: never rent
    "auto_replace": False,            # ephemeral boxes: an automatic replacement could start a second paid box
    "max_price_per_hour": 0.45,
    "max_instances": 1,               # instances running at a time in steady state (salvage windows excepted)
    "max_instances_during_migration": 2,  # make-before-break: old + replacement overlap while it is verified
    "salvage_every_h": 2,             # start a stopped old box this often to fetch its unfetched results
    "salvage_window_min": 30,         # ... and stop it again after this long
    "destroy_unfetched_after_h": 24,  # give up salvage, destroy, log what was lost
    "replace_retry_min": 30,
    "watch_interval_s": 60,
    "ssh_retry_s": 20,
    "ssh_retry_rounds": 3,
    "api_backoff_max_s": 900,
    "disk_gb": 80,
    "create_wait_min": 15,
    "provision_wait_min": 40,
    "fetch_wait_min": 10,
    "nightly_wait_min": 90,           # `nightly` gives the on-box run this long, then fetches whatever exists
}


# ----------------------------------------------------------------------------------------------- output / secrets
class Redactor:
    def __init__(self) -> None:
        self.secrets: set[str] = set()

    def add(self, s: str | None) -> None:
        if s and len(s) >= 6:
            self.secrets.add(s)

    def __call__(self, text: Any) -> str:
        text = str(text)
        for s in self.secrets:
            text = text.replace(s, "***")
        return text


redact = Redactor()


def ts(t: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, tz=UTC).astimezone().strftime("%H:%M:%S")


def iso(t: float | None) -> str:
    return datetime.fromtimestamp(t, tz=UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S") if t else "-"


class Out:
    json_mode = False

    @classmethod
    def say(cls, msg: str) -> None:
        stream = sys.stderr if cls.json_mode else sys.stdout
        print(f"[{ts()}] {redact(msg)}", file=stream, flush=True)


say = Out.say


# ----------------------------------------------------------------------------------------------- REST API
class VastError(RuntimeError):
    pass


class Api:
    def __init__(self, key_file: Path = KEY_FILE, transport: httpx.BaseTransport | None = None,
                 base: str = API_BASE) -> None:
        self.key_file, self.transport, self.base = key_file, transport, base
        self._client: httpx.Client | None = None

    def client(self) -> httpx.Client:
        if self._client is None:
            try:
                key = self.key_file.read_text(encoding="utf-8").strip()
            except OSError as e:
                raise VastError(f"cannot read API key file {self.key_file}: {type(e).__name__}") from None
            redact.add(key)
            self._client = httpx.Client(base_url=self.base, headers={"Authorization": f"Bearer {key}"},
                                        timeout=30, transport=self.transport)
        return self._client

    def req(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = self.client().request(method, path, **kw)
        except httpx.HTTPError as e:
            raise VastError(redact(f"{method} {path}: {type(e).__name__}: {e}")) from None
        if r.status_code >= 400:
            raise VastError(redact(f"{method} {path} -> HTTP {r.status_code}: {r.text[:300]}"))
        try:
            data = r.json()
        except ValueError:
            raise VastError(f"{method} {path}: non-JSON response") from None
        if isinstance(data, dict):
            for k in ("instance_api_key", "jupyter_token", "api_key"):
                if isinstance(data.get(k), str):
                    redact.add(data[k])
        return data

    def user(self) -> dict:
        return self.req("GET", "/api/v0/users/current/")

    def instances(self) -> list[dict]:
        rows: list[dict] = []
        params = {"select_filters": "{}", "order_by": json.dumps([{"col": "id", "dir": "asc"}]), "limit": "25"}
        for _ in range(40):
            data = self.req("GET", "/api/v1/instances/", params=params)
            rows += [_norm_instance(r) for r in data.get("instances") or []]
            if not data.get("next_token"):
                break
            params["after_token"] = data["next_token"]
        return rows

    def instance(self, iid: int | str) -> dict | None:
        data = self.req("GET", f"/api/v0/instances/{iid}/", params={"owner": "me"})
        row = data.get("instances") if isinstance(data, dict) else None
        return _norm_instance(row) if row else None

    def offers(self, query: dict) -> list[dict]:
        return self.req("POST", "/api/v0/bundles/", json=query).get("offers", [])

    def create(self, offer_id: int, payload: dict) -> dict:
        return self.req("PUT", f"/api/v0/asks/{offer_id}/", json=payload)

    def set_state(self, iid: int | str, state: str) -> dict:
        return self.req("PUT", f"/api/v0/instances/{iid}/", json={"state": state})

    def reboot(self, iid: int | str) -> dict:
        return self.req("PUT", f"/api/v0/instances/reboot/{iid}/", json={})

    def destroy(self, iid: int | str) -> dict:
        return self.req("DELETE", f"/api/v0/instances/{iid}/", json={})


def _norm_instance(row: dict) -> dict:
    row = dict(row)
    if isinstance(row.get("extra_env"), list):
        row["extra_env"] = {e[0]: e[1] for e in row["extra_env"] if isinstance(e, list | tuple) and len(e) == 2}
    for k in ("jupyter_token", "instance_api_key"):
        if isinstance(row.get(k), str):
            redact.add(row[k])
    return row


# ----------------------------------------------------------------------------------------------- pure helpers
@dataclass(frozen=True)
class Endpoint:
    kind: str
    host: str
    port: int

    def __str__(self) -> str:
        return f"{self.kind} {self.host}:{self.port}"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "host": self.host, "port": self.port}

    @classmethod
    def from_dict(cls, d: dict | None) -> Endpoint | None:
        return cls(d["kind"], d["host"], int(d["port"])) if d and d.get("host") and d.get("port") else None


def endpoints_from_instance(inst: dict | None) -> list[Endpoint]:
    """Current SSH endpoints of an instance: direct (public ip + mapped 22/tcp) first, then the proxy.
    Ports change after every restart, so this is always computed from a fresh API record."""
    if not inst:
        return []
    eps: list[Endpoint] = []
    ip = (inst.get("public_ipaddr") or "").strip()
    p22 = (inst.get("ports") or {}).get("22/tcp") or []
    if ip and p22 and p22[0].get("HostPort"):
        eps.append(Endpoint("direct", ip, int(p22[0]["HostPort"])))
    host, port = inst.get("ssh_host"), inst.get("ssh_port")
    if host and port:
        # the vastai CLI's rule: jupyter runtypes put sshd on the proxy's port + 1
        if "jupyter" in (inst.get("image_runtype") or ""):
            eps.append(Endpoint("proxy", host, int(port) + 1))
        eps.append(Endpoint("proxy", host, int(port)))
    return eps


def build_offer_query(gpus: list[str], min_vram_gb: float, min_rel: float, min_inet: float, min_disk: float,
                      min_cuda: float, interruptible: bool, limit: int = 100) -> dict:
    return {
        "verified": {"eq": True}, "external": {"eq": False}, "rentable": {"eq": True}, "rented": {"eq": False},
        "num_gpus": {"eq": 1}, "gpu_name": {"in": gpus},
        "gpu_ram": {"gte": min_vram_gb * 950},  # MiB; 5 % slack because a "24 GB" L4 reports 23034 MiB
        "reliability2": {"gte": min_rel}, "inet_down": {"gte": min_inet}, "disk_space": {"gte": min_disk},
        "direct_port_count": {"gte": 1}, "cuda_max_good": {"gte": min_cuda},
        "order": [["dph_total", "asc"]], "type": "bid" if interruptible else "on-demand",
        "limit": limit, "allocated_storage": min_disk,
    }


def offer_score(o: dict) -> float:
    if o.get("dlperf_per_dphtotal"):
        return float(o["dlperf_per_dphtotal"])
    if o.get("dlperf") and o.get("dph_total"):
        return float(o["dlperf"]) / float(o["dph_total"])
    return 0.0


def filter_rank_offers(offers: list[dict], blocklist: set[str], max_price: float | None = None,
                       min_rel: float = 0.0, min_vram_gb: float = 0.0, sort: str = "score",
                       prefer_host: str | None = None) -> list[dict]:
    """Drop blocklisted hosts, unverified machines, offers without direct ports, over-price or unreliable
    offers; rank by throughput per dollar (or by price), then move offers of `prefer_host` (the last host that
    delivered a verified run) to the front. The blocklist always wins. The API returns verified=None with the
    truth in `verification`, so both are accepted."""
    keep = []
    for o in offers:
        if str(o.get("host_id")) in blocklist or str(o.get("machine_id")) in blocklist:
            continue
        if not (o.get("verified") is True or o.get("verification") == "verified"):
            continue
        if (o.get("direct_port_count") or 0) < 1:
            continue
        if max_price is not None and float(o.get("dph_total") or 1e9) > max_price:
            continue
        if float(o.get("reliability2") or o.get("reliability") or 0) < min_rel:
            continue
        if float(o.get("gpu_ram") or 0) < min_vram_gb * 950:
            continue
        o = dict(o, score=round(offer_score(o), 1))
        keep.append(o)
    by_price = sorted(keep, key=lambda o: float(o.get("dph_total") or 1e9))
    for i, o in enumerate(by_price, 1):
        o["price_rank"] = i
    ranked = by_price if sort == "price" else sorted(
        keep, key=lambda o: (-o["score"], float(o.get("dph_total") or 1e9)))
    if prefer_host:
        ranked = sorted(ranked, key=lambda o: str(o.get("host_id")) != str(prefer_host))  # stable
    return ranked


def spend_summary(user: dict, instances: list[dict]) -> dict:
    """Credit, $/h across instances (running: full price; stopped: storage still bills) and hours of runway."""
    credit = float(user.get("credit") or 0.0)
    rate = 0.0
    for i in instances:
        if i.get("actual_status") == "running":
            rate += float(i.get("dph_total") or 0.0)
        else:
            rate += float((i.get("instance") or {}).get("diskHour") or 0.0)
    runway = credit / rate if rate > 0 else float("inf")
    return {"credit": round(credit, 4), "spend_per_h": round(rate, 4),
            "runway_h": round(runway, 2) if runway != float("inf") else None,
            "running": sum(1 for i in instances if i.get("actual_status") == "running"), "instances": len(instances)}


def parse_kv(text: str) -> dict[str, str]:
    kv = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            if k and " " not in k:
                kv[k.strip()] = v.strip()
    return kv


def _num(kv: dict, k: str, default: float | None = None) -> float | None:
    try:
        return float(kv[k])
    except (KeyError, ValueError, TypeError):
        return default


def classify(api_status: str | None, probe: dict | None, cfg: dict) -> tuple[str, list[str]]:
    """healthy / idle / stuck / unreachable (or the API status when not running)."""
    if api_status != "running":
        return (api_status or "gone"), []
    if probe is None:
        return "unreachable", ["no SSH on direct or proxy endpoint"]
    procs = int(_num(probe, "exp_procs", 0) or 0)
    batch = int(_num(probe, "batch_alive", 0) or 0)
    util = _num(probe, "gpu_util")
    age = _num(probe, "progress_age_s")
    if age is None:
        age = _num(probe, "batch_log_age_s")
    age_min = (age or 0) / 60.0
    if procs > 0:
        if age is not None and age_min > cfg["stuck_log_min"]:
            return "stuck", [f"{procs} build/bench processes, no log progress for {age_min:.0f} min"]
        if util == 0 and age is not None and age_min > cfg["stuck_gpu_zero_min"]:
            return "stuck", [f"{procs} build/bench processes, GPU util 0, no log progress for {age_min:.0f} min"]
        return "healthy", [f"{procs} build/bench processes, GPU {util:.0f}%" if util is not None else ""]
    if batch > 0:
        return "healthy", ["batch queue alive"]
    if util is not None and util >= cfg["idle_gpu_util"]:
        return "healthy", [f"GPU busy {util:.0f}% without Tetraeder processes"]
    return "idle", ["no build or benchmark processes"]


def outages_in_window(outages: list[dict], now: float, window_h: float) -> int:
    return sum(1 for o in outages if (o.get("end") or now) >= now - window_h * 3600)


def verify_archive(path: Path) -> dict[str, int]:
    """DONE-marker count per member path in a results archive (raises if it does not unpack)."""
    counts: dict[str, int] = {}
    with tarfile.open(path, "r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile() and m.name.rsplit("/", 1)[-1] == DONE_MARKER:
                counts[m.name] = counts.get(m.name, 0) + 1
    return counts


# ----------------------------------------------------------------------------------------------- state / config
class Store:
    def __init__(self, vast_dir: Path) -> None:
        self.dir = vast_dir
        self.path = vast_dir / "state.json"
        self.config_path = vast_dir / "config.json"
        self.events_path = vast_dir / "events.log"
        self.state = self._load()
        self.config = self._load_config()

    def _load(self) -> dict:
        try:
            st = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            st = {}
        st.setdefault("instances", {})
        st.setdefault("current", None)
        st.setdefault("blocklist", {})
        for h, why in SEED_BLOCKLIST.items():
            st["blocklist"].setdefault(h, why)
        st.setdefault("events", [])
        st.setdefault("batch", None)
        st.setdefault("pending", [])
        return st

    def _load_config(self) -> dict:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        try:
            cfg.update(json.loads(self.config_path.read_text(encoding="utf-8")))
        except OSError:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.config_path.write_text(json.dumps(DEFAULT_CONFIG, indent=1) + "\n", encoding="utf-8")
        except ValueError:
            say(f"WARNING: {self.config_path} is not valid JSON; using defaults")
        return cfg

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state["events"] = self.state["events"][-300:]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(redact(json.dumps(self.state, indent=1, sort_keys=True)), encoding="utf-8")
        os.replace(tmp, self.path)

    def event(self, msg: str, iid: Any = None, echo: bool = True) -> None:
        now = time.time()
        line = redact(f"{iso(now)} {'[' + str(iid) + '] ' if iid else ''}{msg}")
        self.state["events"].append({"t": now, "id": iid, "msg": redact(msg)})
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        if echo:
            say(f"{'[' + str(iid) + '] ' if iid else ''}{msg}")
        self.save()

    def rec(self, iid: Any, create: bool = False) -> dict:
        key = str(iid)
        if key not in self.state["instances"]:
            if not create:
                raise VastError(f"instance {iid} is not managed by vastctl (use `adopt {iid}` first)")
            self.state["instances"][key] = {"id": int(iid), "role": "active", "runs": {}, "outages": [],
                                            "endpoints": {}, "added": time.time()}
        return self.state["instances"][key]

    def blocklist(self) -> set[str]:
        return set(self.state["blocklist"])

    def pend(self, key: str, msg: str) -> None:
        self.state["pending"] = [p for p in self.state["pending"] if p["key"] != key]
        self.state["pending"].append({"key": key, "t": time.time(), "msg": redact(msg)})

    def unpend(self, key: str) -> None:
        self.state["pending"] = [p for p in self.state["pending"] if p["key"] != key]


# ----------------------------------------------------------------------------------------------- ssh / scp
class Unreachable(RuntimeError):
    pass


class ToolMissing(RuntimeError):
    """ssh/scp/git missing locally: never to be mistaken for an unreachable box (no cloud action may follow)."""


@dataclass
class Result:
    rc: int
    out: str
    err: str
    endpoint: Endpoint | None = None


def run_proc(argv: list[str], timeout: float, stdin: bytes | None = None) -> tuple[int, str, str]:
    """Run an executable with a hard timeout (no dependency on a `timeout` binary). Bytes in, text out."""
    try:
        p = subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout, check=False,
                           stdin=None if stdin is not None else subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        return 124, _dec(e.stdout), _dec(e.stderr) + (f"\n[timeout after {timeout:.0f}s; for long commands pass "
                                                        f"`ssh --timeout SECONDS` or run them detached with nohup]")
    except FileNotFoundError:
        raise ToolMissing(f"{argv[0]} not found on PATH (Git for Windows / OpenSSH)") from None
    return p.returncode, _dec(p.stdout), _dec(p.stderr)


def _dec(b: bytes | str | None) -> str:
    if b is None:
        return ""
    return b if isinstance(b, str) else b.decode("utf-8", errors="replace")


def strip_banner(text: str) -> str:
    return "\n".join(line for line in text.replace("\r\n", "\n").split("\n")
                     if not any(b in line for b in BANNER)).strip("\n")


CONN_ERRORS = ("Connection refused", "Connection timed out", "timed out", "Connection closed",
               "lost connection", "Could not resolve", "No route to host", "Connection reset",
               "kex_exchange_identification", "Network is unreachable")


class Remote:
    """Every remote command goes through here: cached endpoint first, then a fresh API lookup (direct, proxy),
    retries with backoff; the working endpoint is written back to state."""

    def __init__(self, ctx: Ctx, iid: Any) -> None:
        self.ctx, self.iid = ctx, str(iid)
        self.rec = ctx.store.rec(iid)
        self.known_hosts = ctx.store.dir / f"known_hosts_{self.iid}"
        self.helper_ok = False

    def _common(self) -> list[str]:
        return ["-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={self.known_hosts.as_posix()}", "-i", self.ctx.ssh_key.as_posix(),
                "-o", "IdentitiesOnly=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
                "-o", "LogLevel=ERROR"]

    def ssh_argv(self, ep: Endpoint, cmd: str | None, tty: bool = False) -> list[str]:
        argv = ["ssh", *self._common(), "-p", str(ep.port)] + (["-t"] if tty else []) + [f"root@{ep.host}"]
        return argv + ([cmd] if cmd else [])

    def scp_argv(self, ep: Endpoint, remote_path: str, local: Path) -> list[str]:
        return ["scp", *self._common(), "-P", str(ep.port), f"root@{ep.host}:{remote_path}", local.as_posix()]

    def fresh_endpoints(self) -> list[Endpoint]:
        inst = self.ctx.api.instance(self.iid)
        eps = endpoints_from_instance(inst)
        self.rec["endpoints"] = {"resolved_at": time.time(), "all": [e.to_dict() for e in eps],
                                 "last_ok": self.rec.get("endpoints", {}).get("last_ok")}
        if inst:
            self.rec["host_id"] = inst.get("host_id")
            self.rec["api_status"] = inst.get("actual_status")
        return eps

    def cached(self) -> Endpoint | None:
        return Endpoint.from_dict((self.rec.get("endpoints") or {}).get("last_ok"))

    def _mark_ok(self, ep: Endpoint) -> None:
        self.rec.setdefault("endpoints", {})["last_ok"] = dict(ep.to_dict(), at=time.time())
        self.ctx.store.save()

    def attempt(self, fn: Callable[[Endpoint], tuple[int, str, str]], is_conn_err: Callable[[int, str, str], bool],
                rounds: int | None = None, backoff: float | None = None, label: str = "ssh") -> Result:
        cfg = self.ctx.store.config
        rounds = cfg["ssh_retry_rounds"] if rounds is None else rounds
        backoff = cfg["ssh_retry_s"] if backoff is None else backoff
        last = ""
        for r in range(rounds + 1):
            tried: list[Endpoint] = []
            order: list[Endpoint] = []
            if r == 0 and self.cached():
                order.append(self.cached())  # type: ignore[arg-type]
            for i in range(2):
                for ep in order:
                    if ep in tried:
                        continue
                    tried.append(ep)
                    rc, out, err = fn(ep)
                    if not is_conn_err(rc, out, err):
                        self._mark_ok(ep)
                        return Result(rc, strip_banner(out), strip_banner(err), ep)
                    last = f"{ep}: rc={rc} {strip_banner(err).strip()[-160:]}"
                if i == 0:
                    try:
                        order = self.fresh_endpoints()
                    except VastError as e:
                        last = f"API lookup failed: {e}"
                        order = []
            if r < rounds:
                say(f"{label}: no endpoint reachable ({last}); retry in {backoff:.0f}s ({r + 1}/{rounds})")
                self.ctx.sleep(backoff)
        raise Unreachable(redact(last or "no endpoints"))

    def run(self, cmd: str, timeout: float = 60, stdin: bytes | None = None, rounds: int | None = None,
            backoff: float | None = None) -> Result:
        def conn_err(rc: int, out: str, err: str) -> bool:
            return rc == 255 or (rc == 124 and not out.strip())
        return self.attempt(lambda ep: run_proc(self.ssh_argv(ep, cmd), timeout, stdin), conn_err, rounds, backoff)

    def scp_from(self, remote_path: str, local: Path, timeout: float = 600, rounds: int | None = None,
                 backoff: float | None = None) -> Result:
        def conn_err(rc: int, out: str, err: str) -> bool:
            return rc != 0 and (rc in (124, 255) or any(c in err for c in CONN_ERRORS))
        return self.attempt(lambda ep: run_proc(self.scp_argv(ep, remote_path, local), timeout), conn_err,
                            rounds, backoff, label="scp")

    def probe(self, ep: Endpoint) -> bool:
        rc, out, _ = run_proc(self.ssh_argv(ep, "echo up"), 25)
        return rc == 0 and "up" in strip_banner(out).split()

    def ensure_helper(self, rounds: int | None = None) -> None:
        if self.helper_ok:
            return
        data = (self.ctx.root / "scripts" / "vast_box.sh").read_bytes().replace(b"\r\n", b"\n")
        r = self.run(f"cat > {BOX_HELPER}.tmp && mv -f {BOX_HELPER}.tmp {BOX_HELPER} && echo helper-ok",
                     stdin=data, timeout=60, rounds=rounds)
        if "helper-ok" not in r.out:
            raise VastError(f"could not upload {BOX_HELPER}: {r.err[-200:]}")
        self.helper_ok = True

    def box(self, args: str, timeout: float = 60, rounds: int | None = None, env: dict | None = None) -> Result:
        self.ensure_helper(rounds)
        prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (env or {}).items() if v not in (None, ""))
        return self.run(f"{prefix + ' ' if prefix else ''}bash {BOX_HELPER} {args}", timeout=timeout, rounds=rounds)


# ----------------------------------------------------------------------------------------------- context
@dataclass
class Ctx:
    root: Path = ROOT
    vast_dir: Path | None = None
    key_file: Path = KEY_FILE
    ssh_key: Path = SSH_KEY
    transport: httpx.BaseTransport | None = None
    sleep: Callable[[float], None] = time.sleep
    api: Api = field(init=False)
    store: Store = field(init=False)

    def __post_init__(self) -> None:
        self.vast_dir = self.vast_dir or self.root / ".vast"
        self.api = Api(self.key_file, self.transport)
        self.store = Store(self.vast_dir)

    def iid(self, given: Any = None) -> str:
        iid = given or self.store.state.get("current")
        if not iid:
            raise VastError("no instance: pass --id or create/adopt one")
        return str(iid)


# ----------------------------------------------------------------------------------------------- operations
def search_offers(ctx: Ctx, a: argparse.Namespace | None = None, extra_block: set[str] | None = None,
                  max_price: float | None = None) -> list[dict]:
    g = (lambda k, d: getattr(a, k, None) if a is not None and getattr(a, k, None) is not None else d)
    gpus = [x.strip() for x in g("gpu", ",".join(GPU_CLASS)).split(",") if x.strip()]
    min_vram = float(g("min_vram", 24))
    min_rel = float(g("min_reliability", 0.99))
    q = build_offer_query(gpus, min_vram, min_rel, float(g("min_inet", 500)), float(g("min_disk", 50)),
                          float(g("min_cuda", 12.4)), bool(g("interruptible", False)))
    block = ctx.store.blocklist() | set(extra_block or set())
    block |= {h.strip() for h in (g("exclude_hosts", "") or "").split(",") if h.strip()}
    mp = max_price if max_price is not None else g("max_price", None)
    return filter_rank_offers(ctx.api.offers(q), block, mp, min_rel, min_vram, g("sort", "score"),
                              prefer_host=ctx.store.state.get("last_good_host"))


def offer_line(o: dict) -> str:
    return (f"offer {o.get('id'):>9}  {o.get('gpu_name', '?'):<10} {float(o.get('gpu_ram') or 0) / 1024:4.0f}GB  "
            f"${float(o.get('dph_total') or 0):.3f}/h (#{o.get('price_rank')})  score {o.get('score'):>6}  "
            f"rel {float(o.get('reliability2') or o.get('reliability') or 0):.4f}  "
            f"host {o.get('host_id')}  {str(o.get('geolocation') or '?')[:22]:<22} "
            f"down {float(o.get('inet_down') or 0):.0f}Mbps cuda {o.get('cuda_max_good')}")


def create_instance(ctx: Ctx, offer: dict, interruptible: bool = False, reason: str = "",
                    make_current: bool = True, image: str | None = None) -> str:
    st = ctx.store
    payload = {"client_id": "me", "image": image or IMAGE, "env": {}, "disk": st.config["disk_gb"], "label": LABEL,
               "runtype": "ssh_direc ssh_proxy", "onstart": ONSTART, "image_login": None, "python_utf8": False,
               "lang_utf8": False, "use_jupyter_lab": False, "jupyter_dir": None, "force": False,
               "cancel_unavail": True, "extra": None,
               "price": round(float(offer.get("min_bid") or offer["dph_total"]) * 1.15, 4) if interruptible
               else None}
    res = ctx.api.create(offer["id"], payload)
    if not res.get("success") or not res.get("new_contract"):
        raise VastError(f"create failed: {redact(json.dumps(res)[:300])}")
    iid = str(res["new_contract"])
    rec = st.rec(iid, create=True)
    rec.update({"created": time.time(), "label": LABEL, "host_id": offer.get("host_id"),
                "role": "active" if make_current else "candidate", "api_status": "created",
                "image": image or IMAGE,
                "offer": {k: offer.get(k) for k in ("id", "gpu_name", "dph_total", "geolocation", "host_id")}})
    if make_current:
        st.state["current"] = iid
    st.event(f"created instance on host {offer.get('host_id')} ({offer.get('gpu_name')}, "
             f"${float(offer.get('dph_total') or 0):.3f}/h) {reason}".strip(), iid)
    return iid


def wait_ready(ctx: Ctx, iid: str, minutes: float, every: float = 15) -> Endpoint | None:
    """Poll API status and `echo up` on both endpoints until one answers."""
    remote = Remote(ctx, iid)
    deadline = time.time() + minutes * 60
    while True:
        try:
            inst = ctx.api.instance(iid)
        except VastError as e:
            inst, status = None, f"api error {e}"
        status = (inst or {}).get("actual_status")
        msg = f"status={status}"
        if status == "running":
            eps = endpoints_from_instance(inst)
            ok = {str(ep): remote.probe(ep) for ep in eps}
            remote.rec["endpoints"] = {"resolved_at": time.time(), "all": [e.to_dict() for e in eps],
                                       "probe": ok, "last_ok": remote.rec.get("endpoints", {}).get("last_ok")}
            msg += "  " + "  ".join(f"{k}={'up' if v else 'down'}" for k, v in ok.items())
            good = [ep for ep in eps if ok[str(ep)]]
            if good:
                remote._mark_ok(good[0])
                ctx.store.event(f"ready: {msg}", iid)
                return good[0]
        say(f"[{iid}] waiting: {msg}")
        if time.time() >= deadline:
            ctx.store.event(f"not ready after {minutes:.0f} min ({msg})", iid)
            return None
        ctx.sleep(every)


ARCHIVE_PATHS = ["crates", "patches", "bench", "scripts", "docs/adr", "Cargo.toml", "Cargo.lock",
                 "rust-toolchain.toml", "AGENTS.md"]
SUBMODULE = "vendor/llama.cpp"


def _git(ctx: Ctx, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ctx.root, capture_output=True, check=False)


def git_archive(ctx: Ctx) -> bytes:
    """Committed HEAD (paths that exist in HEAD) plus the vendor/llama.cpp submodule checkout, as one tar.gz.
    `git archive` skips submodules, so they are archived with a second call and merged into the same tarball."""
    p = _git(ctx, "ls-tree", "--name-only", "HEAD", "--", *ARCHIVE_PATHS)
    paths = [ln.strip() for ln in _dec(p.stdout).splitlines() if ln.strip()]
    if p.returncode != 0 or not paths:
        raise VastError("nothing to push: commit first (git ls-tree HEAD found none of the archive paths)")
    skipped = [x for x in ARCHIVE_PATHS if x not in paths]
    if skipped:
        say(f"git archive: not in HEAD, skipped: {', '.join(skipped)}")
    if paths:
        dirty = _dec(_git(ctx, "status", "--porcelain", "--", *paths).stdout).strip()
        if dirty:
            say(f"WARNING: uncommitted changes are NOT pushed (git archive HEAD): {len(dirty.splitlines())} paths")
    tars = []
    if paths:
        a = _git(ctx, "archive", "--format=tar", "HEAD", "--", *paths)
        if a.returncode != 0:
            raise VastError(f"git archive failed: {_dec(a.stderr)[-200:]}")
        tars.append(a.stdout)
    if (ctx.root / SUBMODULE / ".git").exists():
        s = _git(ctx, "-C", SUBMODULE, "archive", "--format=tar", f"--prefix={SUBMODULE}/", "HEAD")
        if s.returncode != 0:
            raise VastError(f"git archive of {SUBMODULE} failed: {_dec(s.stderr)[-200:]}")
        tars.append(s.stdout)
    else:
        say(f"git archive: submodule {SUBMODULE} not checked out; not pushed")
    return _merge_tars(tars)


def _merge_tars(tars: list[bytes]) -> bytes:
    """Concatenated tar streams do not unpack as one, so copy the members into a single gzip tar."""
    out = io.BytesIO()
    seen: set[str] = set()
    with tarfile.open(fileobj=out, mode="w:gz") as dst:
        for raw in tars:
            if not raw:
                continue
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as src:
                for m in src:
                    if m.name in seen or m.name == "pax_global_header":
                        continue
                    seen.add(m.name)
                    dst.addfile(m, src.extractfile(m) if m.isfile() else None)
    return out.getvalue()


def push_code(ctx: Ctx, remote: Remote) -> None:
    remote.ensure_helper()
    data = git_archive(ctx)
    r = remote.run(f"rm -rf {STAGING_DIR} && mkdir -p {STAGING_DIR} {REMOTE_DIR} && tar -xz -C {STAGING_DIR} "
                   f"&& bash {BOX_HELPER} install-push", stdin=data, timeout=600)
    if r.rc != 0 or "installed" not in r.out:
        raise VastError(f"push failed rc={r.rc}: {(r.out + r.err)[-300:]}")
    say(f"[{remote.iid}] pushed HEAD ({len(data) // 1024} KB): {r.out.splitlines()[-1]}")


def push_runs(ctx: Ctx, remote: Remote, dirs: list[str]) -> int:
    """Upload locally fetched results so a resumed batch skips finished jobs."""
    buf = io.BytesIO()
    n = 0
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for d in dirs:
            p = ctx.root / d
            if p.exists():
                for f in p.rglob("*"):
                    if f.is_file() and not f.name.endswith(".gguf"):
                        tf.add(f, arcname=f.relative_to(ctx.root).as_posix())
                        n += 1
    if n:
        r = remote.run(f"mkdir -p {REMOTE_DIR} && tar -xz -C {REMOTE_DIR} && echo pushed-runs",
                       stdin=buf.getvalue(), timeout=300)
        if "pushed-runs" not in r.out:
            raise VastError(f"pushing runs failed: {r.err[-200:]}")
    ctx.store.event(f"pushed {n} local result files back ({', '.join(dirs)})", remote.iid)
    return n


def parse_models(spec: str | None) -> list[str]:
    """`--models a/b:file.gguf,c/d:other.gguf` -> list; each entry is org/name:file[+file...] (see vast_box.sh)."""
    return [m.strip() for m in (spec or "").split(",") if m.strip()]


def spec_models(path: str | Path) -> list[str]:
    """`hf` entries of a benchmark spec's [model.*] tables, in `--models` format."""
    spec = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return [m["hf"] for m in spec.get("model", {}).values() if m.get("hf")]


def provision(ctx: Ctx, iid: str, models: list[str] | None = None, wait_min: float | None = None) -> bool:
    st = ctx.store
    remote = Remote(ctx, iid)
    push_code(ctx, remote)
    models = list(models or [])
    args = " ".join(shlex.quote(m) for m in models)
    remote.run(f"nohup setsid bash {BOX_HELPER} provision {args} > /root/provision.log 2>&1 "
               "< /dev/null & echo launched", timeout=30)
    st.event(f"provision launched (models: {', '.join(models) or 'none'}); log /root/provision.log", iid)
    deadline = time.time() + 60 * (wait_min or st.config["provision_wait_min"])
    last = ""
    while time.time() < deadline:
        ctx.sleep(20)
        try:
            out = remote.run("tail -n 40 /root/provision.log", timeout=30, rounds=1).out
        except Unreachable as e:
            say(f"[{iid}] provision poll: unreachable ({e}); still waiting")
            continue
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        if tail != last:
            say(f"[{iid}] provision: {tail}")
            last = tail
        if "PROVISION_DONE" in out:
            ok = "PROVISION_DONE rc=0" in out
            kv = [ln for ln in out.splitlines() if ln.startswith(("model=", "run=", "watchdog=", "cuda:"))]
            rec = st.rec(iid)
            rec["inventory"] = {"t": time.time(), "models": models, "lines": kv}
            rec["provisioned"] = time.time() if ok else None
            st.event(f"provision {'OK' if ok else 'FAILED'}: {'; '.join(kv)[:300]}", iid)
            if not ok:
                say(out[-1500:])
            return ok
    st.event("provision did not finish in time (check /root/provision.log)", iid)
    return False


def start_batch(ctx: Ctx, iid: str, batch: dict) -> Result:
    """Push HEAD and start `batch["script"]` detached on the box (vast_box.sh run; the script guards itself with
    flock). Marks the tracked result dirs as unfetched."""
    remote = Remote(ctx, iid)
    push_code(ctx, remote)
    args = " ".join(shlex.quote(str(x)) for x in [batch["script"], *batch.get("args", [])])
    r = remote.box(f"run {args}", timeout=90)
    batch = dict(batch, instance=iid, started=time.time(), all_done=False)
    ctx.store.state["batch"] = batch
    rec = ctx.store.rec(iid)
    for d in batch.get("runs_dirs", DEFAULT_RUNS):
        rec["runs"].setdefault(d, {"ok": False})
        rec["runs"][d]["ok"] = False  # new results will appear; needs a fresh fetch
    ctx.store.event(f"job started: {batch['script']} {' '.join(map(str, batch.get('args', [])))}".strip(), iid)
    return r


def fetch(ctx: Ctx, iid: str, dirs: list[str] | None = None, minutes: float | None = None) -> bool:
    """Pack small result files on the box, scp the archive back (both endpoints, every ssh_retry_s), verify it
    unpacks and contains DONE markers (bench/results/*/DONE), unpack into the repo, record per-dir success in
    state. A verified fetch also records the host as last-good for the next `offers` ranking."""
    st = ctx.store
    rec = st.rec(iid)
    dirs = dirs or list(rec.get("runs") or {}) or list(DEFAULT_RUNS)
    minutes = st.config["fetch_wait_min"] if minutes is None else minutes
    rounds = max(1, int(minutes * 60 / st.config["ssh_retry_s"]))
    remote = Remote(ctx, iid)
    r = remote.box("pack " + " ".join(shlex.quote(d) for d in dirs), timeout=300, rounds=rounds)
    if r.rc != 0:
        st.event(f"fetch: pack failed rc={r.rc}: {(r.out + r.err)[-200:]}", iid)
        return False
    remote_md5 = parse_kv(r.out).get("md5")
    local_dir = st.dir / "fetch"
    local_dir.mkdir(parents=True, exist_ok=True)
    archive = local_dir / f"results-{iid}-{datetime.now(tz=UTC).astimezone().strftime('%Y%m%d-%H%M%S')}.tgz"
    remote.scp_from("/root/results.tgz", archive, rounds=rounds)
    if not archive.exists():
        st.event("fetch: archive missing after scp", iid)
        return False
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    if remote_md5 and md5 != remote_md5:
        st.event(f"fetch: md5 mismatch ({md5} != {remote_md5})", iid)
        return False
    try:
        counts = verify_archive(archive)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(ctx.root, filter="data")
    except (tarfile.TarError, OSError, ValueError) as e:
        st.event(f"fetch: archive does not unpack: {e}", iid)
        return False
    all_ok = True
    for d in dirs:
        n = sum(v for k, v in counts.items() if k.startswith(d.rstrip("/") + "/"))
        rec["runs"][d] = {"ok": n > 0, "markers": n, "fetched_at": time.time(), "archive": archive.name,
                          "md5": md5}
        all_ok &= n > 0
    st.event(f"fetch: {archive.name} {archive.stat().st_size // 1024} KB, DONE markers per dir: "
             f"{ {d: rec['runs'][d]['markers'] for d in dirs} } -> {'verified' if all_ok else 'INCOMPLETE'}", iid)
    if all_ok and rec.get("host_id"):
        st.state["last_good_host"] = str(rec["host_id"])
        st.save()
    return all_ok


def all_fetched(rec: dict) -> tuple[bool, list[str]]:
    missing = [d for d, v in (rec.get("runs") or {}).items() if not v.get("ok")]
    return (not missing), missing


def destroy(ctx: Ctx, iid: str, force: bool = False, reason: str = "") -> bool:
    st = ctx.store
    rec = st.state["instances"].get(str(iid))
    if rec is None and not force:
        say(f"refusing: {iid} is not managed by vastctl (pass --force to destroy anyway)")
        return False
    if rec is not None:
        ok, missing = all_fetched(rec)
        if not ok and not force:
            say(f"refusing to destroy {iid}: runs not fetched+verified: {', '.join(missing)} "
                "(run `fetch`, or pass --force to lose them)")
            return False
        rec["destroyed_inventory"] = {"inventory": rec.get("inventory"), "runs": rec.get("runs"),
                                      "last_health": rec.get("last_health")}
    ctx.api.destroy(iid)
    if rec is not None:
        rec["role"] = "gone"
        rec["destroyed"] = time.time()
    if str(st.state.get("current")) == str(iid):
        st.state["current"] = None
    st.event(f"DESTROYED {reason}".strip(), iid)
    return True


def stop(ctx: Ctx, iid: str, reason: str = "") -> None:
    ctx.api.set_state(iid, "stopped")
    rec = ctx.store.state["instances"].get(str(iid))
    if rec is not None:
        rec["stop_requested"] = time.time()
    ctx.store.event(f"stop requested {reason}".strip(), iid)


# ----------------------------------------------------------------------------------------------- health
def health_check(ctx: Ctx, iid: str, inst: dict | None = None, heal_watchdog: bool = True,
                 rounds: int = 0) -> dict:
    """One check: API status, reachable endpoint, on-box probe; outage/idle bookkeeping in state."""
    st, cfg = ctx.store, ctx.store.config
    rec = st.rec(iid)
    now = time.time()
    if inst is None:
        inst = ctx.api.instance(iid)
    status = (inst or {}).get("actual_status") or "gone"
    rec["api_status"] = status
    if inst:
        rec["host_id"] = inst.get("host_id")
    probe = None
    endpoint = None
    if status == "running":
        remote = Remote(ctx, iid)
        try:
            r = remote.box("health", timeout=40, rounds=rounds)
            probe, endpoint = parse_kv(r.out), r.endpoint
            if "exp_procs" not in probe:
                probe = None
                say(f"[{iid}] health probe returned no data: {(r.out + r.err)[-200:]}")
        except (Unreachable, VastError) as e:
            say(f"[{iid}] unreachable: {e}")
        if probe is not None and heal_watchdog and probe.get("watchdog") == "0":
            try:
                w = remote.box("ensure-watchdog", timeout=60, rounds=0)
                st.event(f"on-box watchdog was not running; relaunched: {w.out.strip()[-120:]}", iid)
            except (Unreachable, VastError) as e:
                say(f"[{iid}] could not relaunch watchdog: {e}")
    cls, why = classify(status, probe, cfg)
    # outage bookkeeping
    if cls == "unreachable":
        if not rec.get("unreachable_since"):
            rec["unreachable_since"] = now
            st.event("outage started (neither direct nor proxy SSH answers)", iid)
    elif rec.get("unreachable_since") and status == "running":
        start = rec.pop("unreachable_since")
        rec.setdefault("outages", []).append({"start": start, "end": now, "min": round((now - start) / 60, 1)})
        rec["outages"] = rec["outages"][-50:]
        dns = probe.get("dns_ok") if probe else "?"
        st.event(f"outage ended after {(now - start) / 60:.0f} min (box DNS ok={dns})", iid)
    elif status != "running":
        rec.pop("unreachable_since", None)
    # idle bookkeeping
    if cls == "idle":
        rec.setdefault("idle_since", now)
    else:
        rec.pop("idle_since", None)
    if status in ("stopped", "exited"):
        rec.setdefault("stopped_since", now)
    elif status == "running":
        rec.pop("stopped_since", None)
    if probe is not None:
        rec["last_contact"] = {"t": now, "exp_procs": int(_num(probe, "exp_procs", 0) or 0),
                               "batch_alive": int(_num(probe, "batch_alive", 0) or 0),
                               "batch_done": probe.get("batch_done"), "batch_started": probe.get("batch_started"),
                               "batch_all_done": probe.get("batch_all_done")}
        b = st.state.get("batch")
        if (b and str(b.get("instance")) == str(iid) and not b.get("all_done")
                and probe.get("batch_all_done") == "1" and probe.get("batch_alive") == "0"):
            b["all_done"] = True
            b["done_at"] = now
            st.event(f"batch ALL DONE ({probe.get('batch_done')} jobs, {probe.get('batch_failed')} failed)", iid)
    unreach_min = (now - rec["unreachable_since"]) / 60 if rec.get("unreachable_since") else 0.0
    idle_min = (now - rec["idle_since"]) / 60 if rec.get("idle_since") else 0.0
    n_out = outages_in_window(rec.get("outages", []), now, cfg["outage_window_h"]) + (1 if rec.get("unreachable_since") else 0)
    h = {"t": now, "id": iid, "status": status, "class": cls, "why": [w for w in why if w],
         "endpoint": str(endpoint) if endpoint else None, "unreachable_min": round(unreach_min, 1),
         "idle_min": round(idle_min, 1), "outages_in_window": n_out, "probe": probe}
    rec["last_health"] = h
    if n_out >= cfg["outage_count"] and status == "running":
        rec["migrate_recommended"] = now
        st.pend(f"migrate-{iid}", f"instance {iid} (host {rec.get('host_id')}) had {n_out} outages in "
                f"{cfg['outage_window_h']} h: migrate (auto with auto_replace / --allow-migrate)")
    elif rec.pop("migrate_recommended", None):
        st.unpend(f"migrate-{iid}")
    st.save()
    return h


def health_line(h: dict) -> str:
    p = h.get("probe") or {}
    bits = [f"[{h['id']}] {h['class'].upper():<11} api={h['status']}"]
    if h.get("endpoint"):
        bits.append(f"via {h['endpoint']}")
    if p:
        bits.append(f"gpu {p.get('gpu_util', '?')}% {p.get('gpu_mem_used', '?')}/{p.get('gpu_mem_total', '?')}MiB")
        bits.append(f"exp={p.get('exp_procs')} batch={p.get('batch_alive')}")
        if p.get("batch_started"):
            bits.append(f"jobs {p.get('batch_done')}/{p.get('batch_started')} started"
                        f"{' ALL DONE' if p.get('batch_all_done') == '1' else ''}")
        age = _num(p, "progress_age_s")
        if age is not None:
            bits.append(f"log age {age / 60:.0f}m")
        bits.append(f"watchdog={p.get('watchdog')} disk={p.get('disk_free_gb')}G dns={p.get('dns_ok')}")
        bits.append(f"ram={p.get('ram_gb', '?')}G disk_read={p.get('disk_read_mbs', '?')}MB/s")
    if h.get("unreachable_min"):
        bits.append(f"unreachable {h['unreachable_min']:.0f} min")
    if h.get("idle_min"):
        bits.append(f"idle {h['idle_min']:.0f} min")
    if h.get("outages_in_window"):
        bits.append(f"outages/{h.get('outages_in_window')} in window")
    if h.get("why"):
        bits.append("(" + "; ".join(h["why"]) + ")")
    return "  ".join(bits)


# ----------------------------------------------------------------------------------------------- watchdog
@dataclass
class Policy:
    auto_heal: bool = False
    allow_migrate: bool = False
    stop_idle: bool = True
    destroy: bool = True
    dry_run: bool = False


def pending_work(st: Store) -> bool:
    b = st.state.get("batch")
    return bool(b and not b.get("all_done"))


def watchdog_tick(ctx: Ctx, pol: Policy) -> dict:
    """One pass over every managed instance: health, cost control, unreachability, salvage, replacement.
    Raises VastError on API failure (the loop backs off)."""
    st = ctx.store
    st.state, st.config = st._load(), st._load_config()  # pick up manual fetch/adopt/block and config edits
    cfg = st.config
    now = time.time()
    user = ctx.api.user()
    insts = {str(i["id"]): i for i in ctx.api.instances()}
    bal = spend_summary(user, list(insts.values()))
    st.state["last_balance"] = dict(bal, t=now)
    runway = bal["runway_h"]
    low = runway is not None and runway < cfg["runway_stop_h"]
    if runway is not None and runway < cfg["runway_warn_h"]:
        st.pend("balance", f"LOW BALANCE: ${bal['credit']:.2f}, ${bal['spend_per_h']:.3f}/h, runway {runway:.1f} h")
        say(f"WARNING low balance: runway {runway:.1f} h")
    else:
        st.unpend("balance")
    report: dict[str, Any] = {"balance": bal, "instances": {}}
    for iid, rec in list(st.state["instances"].items()):
        if rec.get("role") == "gone":
            continue
        inst = insts.get(iid)
        if inst is None:
            rec["role"] = "gone"
            st.event("instance no longer exists at Vast; marked gone", iid)
            continue
        try:
            rec["api_status"] = inst.get("actual_status")
            if rec.get("role") == "salvage":
                report["instances"][iid] = _tick_salvage(ctx, pol, iid, rec, inst)
            elif rec.get("role") == "candidate":
                _tick_candidate(ctx, iid, rec)
            else:
                report["instances"][iid] = _tick_active(ctx, pol, iid, rec, inst, low)
        except (Unreachable, VastError, OSError, subprocess.SubprocessError) as e:
            st.event(f"watchdog step failed: {type(e).__name__}: {e}", iid)
    _maybe_replace(ctx, pol, bal)
    st.state["last_watchdog"] = time.time()
    st.save()
    return report


def _tick_active(ctx: Ctx, pol: Policy, iid: str, rec: dict, inst: dict, low_balance: bool) -> dict:
    st, cfg = ctx.store, ctx.store.config
    h = health_check(ctx, iid, inst)
    say(health_line(h))
    cls, status = h["class"], h["status"]
    if status == "running" and cls == "unreachable":
        last = rec.get("last_contact") or {}
        busy = (last.get("exp_procs") or 0) > 0 or (last.get("batch_alive") or 0) > 0
        limit = cfg["unreachable_stop_busy_min"] if busy else cfg["unreachable_stop_idle_min"]
        mins = h["unreachable_min"]
        if mins >= limit:
            why = (f"unreachable {mins:.0f} min >= {limit} min; last contact {iso(last.get('t'))}: "
                   f"{last.get('exp_procs', '?')} build/bench procs, batch alive={last.get('batch_alive', '?')}, "
                   f"jobs {last.get('batch_done', '?')}/{last.get('batch_started', '?')}")
            if not pol.dry_run:
                stop(ctx, iid, f"({why})")
                host = str(rec.get("host_id") or inst.get("host_id"))
                if host and host != "None" and host not in st.state["blocklist"]:
                    st.state["blocklist"][host] = f"{iso(time.time())}: auto-blocked, {why}"
                    st.event(f"host {host} added to blocklist", iid)
                if not all_fetched(rec)[0]:  # else stay active: stopped_since + destroy_after_stopped_h apply
                    rec["role"] = "salvage"
                    rec["salvage_since"] = time.time()
                    rec["last_salvage_attempt"] = time.time()
                if pending_work(st):
                    st.state["need_replacement"] = {"t": time.time(), "old": iid, "why": "unreachable"}
            else:
                say(f"[{iid}] (dry-run) would stop: {why}")
        elif pol.auto_heal and mins >= cfg["reboot_after_unreachable_min"] and not rec.get("rebooted_for_outage"):
            if not pol.dry_run:
                ctx.api.reboot(iid)
                rec["rebooted_for_outage"] = time.time()
                st.event(f"REBOOT (auto-heal) after {mins:.0f} min unreachable; this restarts the container "
                         "and kills running jobs", iid)
        return h
    if status == "running":
        rec.pop("rebooted_for_outage", None)
    if cls == "stuck":
        st.pend(f"stuck-{iid}", f"instance {iid} looks stuck: {'; '.join(h['why'])} (not killing anything)")
    else:
        st.unpend(f"stuck-{iid}")
    migrate = bool(rec.get("migrate_recommended")) and status == "running" and cls != "unreachable"
    if (migrate and (pol.allow_migrate or cfg["auto_replace"]) and pending_work(st) and not pol.dry_run
            and not st.state.get("need_replacement")):
        st.event("migration: fetching results before replacing this host", iid)
        fetch(ctx, iid)
        st.state["need_replacement"] = {"t": time.time(), "old": iid, "why": "migrate"}
        return h
    if status == "running" and cls == "idle":
        idle_for = h["idle_min"]
        if (pol.stop_idle and idle_for >= cfg["idle_stop_min"]) or (low_balance and pol.stop_idle):
            why = "low balance" if low_balance and idle_for < cfg["idle_stop_min"] else f"idle {idle_for:.0f} min"
            if pol.dry_run:
                say(f"[{iid}] (dry-run) would fetch + stop ({why})")
            else:
                if rec.get("runs"):
                    fetch(ctx, iid)
                stop(ctx, iid, f"({why})")
    if status in ("stopped", "exited"):
        stopped_h = (time.time() - rec.get("stopped_since", time.time())) / 3600
        if stopped_h >= cfg["destroy_after_stopped_h"] and pol.destroy:
            ok, missing = all_fetched(rec)
            if ok and not pol.dry_run:
                destroy(ctx, iid, reason=f"(stopped {stopped_h:.1f} h, all results fetched)")
            elif not ok:
                rec["role"] = "salvage"
                rec.setdefault("salvage_since", time.time())
                rec.setdefault("last_salvage_attempt", 0)
                st.event(f"stopped {stopped_h:.1f} h with unfetched runs {missing}: salvage", iid)
    return h


def _tick_salvage(ctx: Ctx, pol: Policy, iid: str, rec: dict, inst: dict) -> dict:
    """Old box with results not yet fetched: periodically start it, fetch in a short window, destroy when
    verified; give up after destroy_unfetched_after_h and log what was lost."""
    st, cfg = ctx.store, ctx.store.config
    now = time.time()
    status = inst.get("actual_status")
    ok, missing = all_fetched(rec)
    info = {"id": iid, "role": "salvage", "status": status, "missing": missing}
    say(f"[{iid}] SALVAGE api={status} unfetched={missing or 'none'}")
    if pol.dry_run:
        return info
    if ok:
        if pol.destroy:
            destroy(ctx, iid, reason="(salvage complete: all results fetched and verified)")
        return info
    age_h = (now - rec.get("salvage_since", now)) / 3600
    if age_h >= cfg["destroy_unfetched_after_h"] and pol.destroy:
        lost = {d: rec["runs"][d] for d in missing}
        st.event(f"salvage gave up after {age_h:.1f} h; LOST results: {json.dumps(lost)}; last contact "
                 f"{json.dumps(rec.get('last_contact'))}; inventory {json.dumps(rec.get('inventory'))}", iid)
        destroy(ctx, iid, force=True, reason="(salvage timeout, results lost as logged)")
        return info
    window = rec.get("salvage_window_start")
    if status == "running":
        h = health_check(ctx, iid, inst, heal_watchdog=False)
        if h["class"] != "unreachable" and fetch(ctx, iid, minutes=2) and pol.destroy:
            destroy(ctx, iid, reason="(salvage: results fetched and verified)")
            return info
        if window and (now - window) / 60 >= cfg["salvage_window_min"]:
            stop(ctx, iid, "(salvage window over)")
            rec.pop("salvage_window_start", None)
        elif not window:
            stop(ctx, iid, "(re-issuing stop: API still reports running)")
    elif status in ("stopped", "exited"):
        rec.pop("salvage_window_start", None)
        if (now - rec.get("last_salvage_attempt", 0)) / 3600 >= cfg["salvage_every_h"]:
            ctx.api.set_state(iid, "running")
            rec["salvage_window_start"] = now
            rec["last_salvage_attempt"] = now
            st.event(f"salvage: started old instance for a {cfg['salvage_window_min']} min fetch window", iid)
    return info


def verify_box(ctx: Ctx, iid: str, models: list[str]) -> bool:
    """Replacement acceptance: SSH (cached or re-resolved endpoint), GPU visible, cargo works, models present,
    CUDA initialises."""
    try:
        r = Remote(ctx, iid).box("verify " + " ".join(models), timeout=240, rounds=2)
    except (Unreachable, VastError) as e:
        ctx.store.event(f"verify: unreachable ({e})", iid)
        return False
    kv = parse_kv(r.out)
    ok = kv.get("verify_ok") == "1"
    ctx.store.event(f"verify {'OK' if ok else 'FAILED'}: gpu={kv.get('verify_gpu')} tools={kv.get('verify_tools')}"
                    + ("" if ok else f" {(r.out + r.err)[-300:]}"), iid)
    return ok


def _running_ids(st: Store) -> list[str]:
    return [i for i, r in st.state["instances"].items()
            if r.get("role") != "gone" and r.get("api_status") in ("running", "loading", "created")]


def _maybe_replace(ctx: Ctx, pol: Policy, bal: dict) -> None:
    """Make-before-break: rent, provision and verify a replacement while the old box keeps running (or stays
    stopped); only then switch state, push fetched results, resume the batch, and fetch + retire the old box.
    A replacement that fails any step is destroyed (it holds no data) and the old box is kept."""
    st, cfg = ctx.store, ctx.store.config
    need = st.state.get("need_replacement")
    if not need:
        return
    old_id = str(need["old"])
    if not pending_work(st):
        st.state["need_replacement"] = None
        return
    if not (cfg["auto_replace"] or pol.allow_migrate):
        st.pend("replace", f"replacement needed ({need.get('why')}) but auto_replace is off")
        return
    if pol.dry_run:
        say("(dry-run) would rent + provision + verify a replacement, then resume the batch there")
        return
    if time.time() < need.get("next_try", 0):
        return
    running = _running_ids(st)
    limit = cfg["max_instances_during_migration"] if old_id in running else cfg["max_instances"]
    if len(running) + 1 > limit:
        st.pend("replace", f"replacement needed but {len(running)} instances running (limit {limit})")
        return
    old = st.rec(old_id)
    offers = search_offers(ctx, None, extra_block={str(old.get("host_id"))}, max_price=cfg["max_price_per_hour"])
    if not offers:
        st.pend("replace", f"replacement needed: no offer under ${cfg['max_price_per_hour']}/h")
        need["next_try"] = time.time() + 60 * cfg["replace_retry_min"]
        return
    best = offers[0]
    runway = bal["credit"] / (bal["spend_per_h"] + float(best["dph_total"]))  # combined rate during the overlap
    if runway < cfg["runway_rent_min_h"]:
        st.pend("replace", f"replacement needed but runway would be {runway:.1f} h < {cfg['runway_rent_min_h']} h "
                "with both boxes billing")
        need["next_try"] = time.time() + 60 * cfg["replace_retry_min"]
        return
    st.unpend("replace")
    models = list((old.get("inventory") or {}).get("models") or [])
    new = create_instance(ctx, best, reason=f"(candidate replacement for {old_id}: {need.get('why')})",
                          make_current=False)
    ok = (wait_ready(ctx, new, cfg["create_wait_min"]) is not None
          and provision(ctx, new, models) and verify_box(ctx, new, models))
    if not ok:
        st.event(f"candidate failed; destroying it (holds no data) and keeping {old_id}", new)
        st.state["blocklist"].setdefault(str(best.get("host_id")), f"{iso(time.time())}: replacement failed")
        st.rec(new)["runs"] = {}
        destroy(ctx, new, reason="(failed replacement)")
        need["next_try"] = time.time() + 60 * cfg["replace_retry_min"]
        return
    # switch
    st.rec(new)["role"] = "active"
    st.state["current"] = new
    batch = st.state["batch"]
    push_runs(ctx, Remote(ctx, new), batch.get("runs_dirs", DEFAULT_RUNS))
    start_batch(ctx, new, batch)
    st.state["need_replacement"] = None
    st.unpend(f"migrate-{old_id}")
    st.event(f"switched: {new} is current and running the batch (old {old_id})", new)
    _retire(ctx, pol, old_id)


def _retire(ctx: Ctx, pol: Policy, iid: str) -> None:
    """Old box after a switch: final fetch if it is up, stop it, destroy when everything is fetched, else salvage."""
    st = ctx.store
    rec = st.rec(iid)
    rec["role"] = "retired"
    if rec.get("api_status") == "running":
        try:
            fetch(ctx, iid, minutes=2)
        except (Unreachable, VastError) as e:
            say(f"[{iid}] final fetch failed: {e}")
        stop(ctx, iid, "(retired after switch)")
    if all_fetched(rec)[0]:
        if pol.destroy:
            destroy(ctx, iid, reason="(retired; all results fetched)")
    else:
        rec.update(role="salvage", salvage_since=time.time(), last_salvage_attempt=time.time())
        st.event(f"retired with unfetched runs {all_fetched(rec)[1]}: salvage", iid)


def _tick_candidate(ctx: Ctx, iid: str, rec: dict) -> None:
    """A candidate left over from an interrupted replacement (it holds no data): destroy once it is stale."""
    cfg = ctx.store.config
    age_min = (time.time() - rec.get("created", time.time())) / 60
    if age_min > cfg["create_wait_min"] + cfg["provision_wait_min"] + 15:
        rec["runs"] = {}
        destroy(ctx, iid, reason=f"(stale candidate, {age_min:.0f} min old, never switched to)")


def watchdog_loop(ctx: Ctx, pol: Policy, once: bool, interval: float) -> int:
    backoff = interval
    while True:
        try:
            watchdog_tick(ctx, pol)
            backoff = interval
        except VastError as e:
            backoff = min(max(backoff * 2, interval), ctx.store.config["api_backoff_max_s"])
            say(f"watchdog: API error ({e}); next try in {backoff:.0f}s")
        except Exception as e:  # noqa: BLE001 - never let one bad iteration kill the loop
            try:
                ctx.store.event(f"watchdog iteration crashed: {type(e).__name__}: {e}")
            except Exception:  # noqa: BLE001
                say(f"watchdog iteration crashed: {type(e).__name__}: {e}")
        if once:
            return 0
        ctx.sleep(backoff)


# ----------------------------------------------------------------------------------------------- commands
def cmd_balance(ctx: Ctx, a: argparse.Namespace) -> dict:
    s = spend_summary(ctx.api.user(), ctx.api.instances())
    rw = f"{s['runway_h']:.1f} h" if s["runway_h"] is not None else "unlimited (nothing billing)"
    say(f"credit ${s['credit']:.2f}  spend ${s['spend_per_h']:.3f}/h across {s['instances']} instances "
        f"({s['running']} running)  runway {rw}")
    return s


def cmd_offers(ctx: Ctx, a: argparse.Namespace) -> dict:
    offers = search_offers(ctx, a)
    say(f"{len(offers)} offers after filters (blocklist: {', '.join(sorted(ctx.store.blocklist()))}); "
        f"top {min(a.top, len(offers))} by {a.sort}:")
    for o in offers[: a.top]:
        say(offer_line(o))
    return {"offers": [{k: o.get(k) for k in ("id", "gpu_name", "gpu_ram", "dph_total", "price_rank", "score",
                                               "reliability2", "host_id", "machine_id", "geolocation", "inet_down",
                                               "cuda_max_good", "disk_space")} for o in offers[: a.top]]}


def cmd_create(ctx: Ctx, a: argparse.Namespace) -> dict:
    st = ctx.store
    cur = st.state.get("current")
    if cur and st.state["instances"].get(str(cur), {}).get("role") == "active" and not a.allow_second:
        inst = ctx.api.instance(cur)
        if inst and inst.get("actual_status") in ("running", "loading", "created"):
            raise VastError(f"instance {cur} is still {inst.get('actual_status')}; pass --allow-second to rent another")
    max_price = a.max_price if a.max_price is not None else st.config["max_price_per_hour"]
    offers = search_offers(ctx, a, max_price=max_price)
    if a.offer:
        chosen = next((o for o in offers if str(o["id"]) == str(a.offer)), None)
        if chosen is None:
            raise VastError(f"offer {a.offer} is not among the filtered offers (blocklisted/over price/gone?)")
    elif a.auto:
        if not offers:
            raise VastError(f"no offers match under ${max_price}/h")
        chosen = offers[0]
    else:
        raise VastError("pass --offer ID or --auto")
    say("renting " + offer_line(chosen))
    iid = create_instance(ctx, chosen, a.interruptible, image=a.image)
    ep = wait_ready(ctx, iid, st.config["create_wait_min"])
    return {"id": iid, "endpoint": str(ep) if ep else None}


NIGHTLY_LOG = f"{RUNS_DIR}/nightly.log"


def wait_nightly(ctx: Ctx, iid: str, minutes: float) -> int | None:
    """Wait on the box (the loop runs there; each ssh call stays under its 540 s timeout) for the nightly log's
    final `NIGHTLY_DONE rc=N` line. Returns rc, or None on timeout."""
    remote = Remote(ctx, iid)
    deadline = time.time() + minutes * 60
    cmd = (f"cd {REMOTE_DIR} && for i in $(seq 100); do grep -q '^NIGHTLY_DONE' {NIGHTLY_LOG} 2>/dev/null && break; "
           f"sleep 5; done; date +%T; tail -n 3 {NIGHTLY_LOG} 2>/dev/null")
    while time.time() < deadline:
        try:
            out = remote.run(cmd, timeout=540).out
        except Unreachable as e:
            say(f"[{iid}] wait: unreachable ({e}); still waiting")
            ctx.sleep(30)
            continue
        say(f"[{iid}] nightly: {(out.strip().splitlines() or [''])[-1]}")
        m = re.search(r"NIGHTLY_DONE rc=(\d+)", out)
        if m:
            return int(m.group(1))
    return None


def nightly(ctx: Ctx, a: argparse.Namespace) -> dict:
    """create --auto, provision, run scripts/nightly.sh detached, wait on the box, fetch, destroy. The destroy guard
    applies: a box whose results are not verified stays up for the watchdog to salvage."""
    st = ctx.store
    t0 = time.time()
    res: dict[str, Any] = {"ok": False, "rc": None, "id": None}
    cur = st.state.get("current")
    if cur and st.state["instances"].get(str(cur), {}).get("role") == "active":
        inst = ctx.api.instance(cur)
        if inst and inst.get("actual_status") in ("running", "loading", "created"):
            raise VastError(f"instance {cur} is still {inst.get('actual_status')}; destroy it first")
    max_price = a.max_price if a.max_price is not None else st.config["max_price_per_hour"]
    offers = search_offers(ctx, a, max_price=max_price)
    if not offers:
        raise VastError(f"no offers match under ${max_price}/h")
    chosen = offers[0]
    bal = spend_summary(ctx.api.user(), ctx.api.instances())
    runway = bal["credit"] / (bal["spend_per_h"] + float(chosen["dph_total"]))
    if runway < st.config["runway_rent_min_h"]:
        raise VastError(f"runway would be {runway:.1f} h < runway_rent_min_h {st.config['runway_rent_min_h']}: "
                        f"not renting (credit ${bal['credit']:.2f})")
    say("nightly: renting " + offer_line(chosen))
    iid = create_instance(ctx, chosen, image=a.image, reason="(nightly)")
    res["id"] = iid
    price = float(chosen.get("dph_total") or 0)

    def finish(ok: bool, why: str = "") -> dict:
        hours = (time.time() - t0) / 3600
        res.update(ok=ok, minutes=round(hours * 60, 1), cost=round(hours * price, 3))
        st.event(f"nightly {'OK' if ok else 'FAILED'}{' (' + why + ')' if why else ''}: {hours * 60:.0f} min, "
                 f"~${hours * price:.2f} at ${price:.3f}/h", iid)
        return res

    def abort(why: str) -> dict:
        st.rec(iid)["runs"] = {}  # nothing was produced on the box: the destroy guard has nothing to protect
        destroy(ctx, iid, reason=f"(nightly aborted: {why})")
        return finish(False, why)

    try:
        if wait_ready(ctx, iid, st.config["create_wait_min"]) is None:
            return abort("box not reachable")
        if not provision(ctx, iid, parse_models(a.models) if a.models else spec_models(NIGHTLY_SPEC)):
            return abort("provision failed")
        start_batch(ctx, iid, {"script": NIGHTLY_SCRIPT, "args": [], "runs_dirs": list(DEFAULT_RUNS)})
    except (Unreachable, VastError) as e:  # nothing was produced yet: do not leave a billed box behind
        return abort(f"{type(e).__name__}: {e}")
    rc = wait_nightly(ctx, iid, a.wait_min or st.config["nightly_wait_min"])
    res["rc"] = rc
    if rc is not None and st.state.get("batch"):
        st.state["batch"]["all_done"] = True
    fetched = fetch(ctx, iid)
    if not fetched:
        say(f"[{iid}] results not verified: box kept for the watchdog (destroy guard)")
        return finish(False, "fetch not verified" if rc is not None else "timeout, fetch not verified")
    destroy(ctx, iid, reason="(nightly: results fetched and verified)")
    return finish(rc == 0, "" if rc == 0 else f"nightly rc={rc}" if rc is not None else "timeout")


def cmd_nightly(ctx: Ctx, a: argparse.Namespace) -> dict:
    return nightly(ctx, a)


def cmd_adopt(ctx: Ctx, a: argparse.Namespace) -> dict:
    inst = ctx.api.instance(a.instance)
    if not inst:
        raise VastError(f"instance {a.instance} not found")
    rec = ctx.store.rec(a.instance, create=True)
    rec.update({"host_id": inst.get("host_id"), "created": inst.get("start_date"), "label": inst.get("label"),
                "role": "active"})
    for d in a.runs if a.runs is not None else DEFAULT_RUNS:
        rec["runs"].setdefault(d, {"ok": False})
    ctx.store.state["current"] = str(a.instance)
    ctx.store.event(f"adopted (host {inst.get('host_id')}, {inst.get('gpu_name')}, status "
                    f"{inst.get('actual_status')})", a.instance)
    return {"id": str(a.instance)}


def cmd_endpoint(ctx: Ctx, a: argparse.Namespace) -> dict:
    iid = ctx.iid(a.id)
    inst = ctx.api.instance(iid)
    if not inst:
        raise VastError(f"instance {iid} not found")
    eps = endpoints_from_instance(inst)
    st = ctx.store
    managed = str(iid) in st.state["instances"]
    remote = Remote(ctx, iid) if managed else None
    res: dict[str, Any] = {"id": iid, "status": inst.get("actual_status"), "endpoints": []}
    tester = remote or _TmpRemote(ctx, iid)
    for ep in eps:
        up = tester.probe(ep) if inst.get("actual_status") == "running" else False
        res["endpoints"].append({**ep.to_dict(), "up": up})
        say(f"{ep}: {'UP' if up else 'down'}   {' '.join(shlex.quote(x) for x in tester.ssh_argv(ep, None))}")
        if up and remote:
            remote._mark_ok(ep)
    if not eps:
        say(f"no SSH endpoints (status {inst.get('actual_status')})")
    return res


class _TmpRemote(Remote):
    def __init__(self, ctx: Ctx, iid: Any) -> None:  # unmanaged instance: no state record
        self.ctx, self.iid, self.rec, self.helper_ok = ctx, str(iid), {}, False
        self.known_hosts = ctx.store.dir / f"known_hosts_{iid}"


def cmd_ssh(ctx: Ctx, a: argparse.Namespace) -> int:
    iid = ctx.iid(a.id)
    remote = Remote(ctx, iid)
    cmd = " ".join(a.cmd).strip()
    if cmd:
        r = remote.run(cmd, timeout=a.timeout)
        if r.out:
            print(redact(r.out))
        if r.err.strip():
            print(redact(r.err), file=sys.stderr)
        return r.rc
    r = remote.run("echo up", timeout=30)
    say(f"connecting via {r.endpoint}")
    return subprocess.call(remote.ssh_argv(r.endpoint, None, tty=True))  # type: ignore[arg-type]


def cmd_put(ctx: Ctx, a: argparse.Namespace) -> dict:
    """Copy local files to the box through ssh stdin (long `ssh -- "echo <base64> | ..."` lines failed silently)."""
    iid = ctx.iid(a.id)
    remote = Remote(ctx, iid)
    out = {}
    for local in a.files:
        data = Path(local).read_bytes()
        if not a.binary:
            data = data.replace(b"\r\n", b"\n")
        dst = a.dest.rstrip("/") + "/" + Path(local).name if a.dest.endswith("/") else a.dest
        q = shlex.quote(dst)
        r = remote.run(f"mkdir -p \"$(dirname {q})\" && cat > {q}.tmp && mv -f {q}.tmp {q} && md5sum {q}",
                       stdin=data, timeout=120)
        want = hashlib.md5(data).hexdigest()
        ok = want in r.out
        say(f"put {local} -> {dst}: {'ok' if ok else 'FAILED'} ({len(data)} bytes)")
        if not ok:
            raise VastError(f"upload of {local} failed: {r.err[-200:]}")
        out[dst] = want
    return out


def cmd_provision(ctx: Ctx, a: argparse.Namespace) -> dict:
    iid = ctx.iid(a.id)
    ok = provision(ctx, iid, parse_models(a.models), a.wait_min)
    return {"id": iid, "ok": ok}


def cmd_run(ctx: Ctx, a: argparse.Namespace) -> dict:
    iid = ctx.iid(a.id)
    batch = {"script": a.script, "args": a.args, "runs_dirs": list(DEFAULT_RUNS)}
    r = start_batch(ctx, iid, batch)
    say(r.out[-1500:])
    return {"id": iid, "rc": r.rc}


def cmd_fetch(ctx: Ctx, a: argparse.Namespace) -> dict:
    iid = ctx.iid(a.id)
    ok = fetch(ctx, iid, a.dirs or None, a.minutes)
    return {"id": iid, "ok": ok, "runs": ctx.store.rec(iid).get("runs")}


def cmd_health(ctx: Ctx, a: argparse.Namespace) -> dict:
    iid = ctx.iid(a.id)
    h = health_check(ctx, iid, heal_watchdog=not a.no_heal)
    say(health_line(h))
    rec, cfg = ctx.store.rec(iid), ctx.store.config
    if h["class"] == "unreachable" and a.auto_heal and h["unreachable_min"] >= a.reboot_after \
            and not rec.get("rebooted_for_outage"):
        say("WARNING: rebooting restarts the container (running jobs die)")
        ctx.api.reboot(iid)
        rec["rebooted_for_outage"] = time.time()
        ctx.store.event(f"REBOOT (auto-heal) after {h['unreachable_min']:.0f} min unreachable", iid)
    if h["class"] == "stuck":
        say("STUCK: reporting only; nothing is killed automatically")
    if h["class"] == "idle" and a.stop_idle and h["idle_min"] >= cfg["idle_stop_min"]:
        if rec.get("runs"):
            fetch(ctx, iid)
        stop(ctx, iid, f"(idle {h['idle_min']:.0f} min)")
    return h


def cmd_watch(ctx: Ctx, a: argparse.Namespace) -> int:
    while True:
        try:
            cmd_health(ctx, a)
        except Exception as e:  # noqa: BLE001 - keep watching
            say(f"health failed: {type(e).__name__}: {e}")
        ctx.sleep(a.interval)


def cmd_watchdog(ctx: Ctx, a: argparse.Namespace) -> int:
    pol = Policy(auto_heal=a.auto_heal, allow_migrate=a.allow_migrate, stop_idle=not a.no_stop_idle,
                 destroy=not a.no_destroy, dry_run=a.dry_run)
    if not a.once:
        ctx.store.event(f"local watchdog started (interval {a.interval}s, {pol})")
    return watchdog_loop(ctx, pol, a.once, a.interval or ctx.store.config["watch_interval_s"])


def cmd_status(ctx: Ctx, a: argparse.Namespace) -> dict:
    st = ctx.store
    res: dict[str, Any] = {}
    try:
        insts = ctx.api.instances()
        bal = spend_summary(ctx.api.user(), insts)
        res["balance"] = bal
        rw = f"{bal['runway_h']:.1f} h" if bal["runway_h"] is not None else "unlimited"
        say(f"credit ${bal['credit']:.2f}  spend ${bal['spend_per_h']:.3f}/h  runway {rw}")
        for i in insts:
            rec = st.state["instances"].get(str(i["id"]))
            say(f"instance {i['id']}  {i.get('actual_status') or '?':<8} {i.get('gpu_name')}  ${float(i.get('dph_total') or 0):.3f}/h"
                f"  host {i.get('host_id')}  label {i.get('label')}  "
                f"{'managed:' + rec.get('role', '?') if rec else 'unmanaged'}")
        res["instances"] = [{k: i.get(k) for k in ("id", "actual_status", "gpu_name", "dph_total", "host_id",
                                                   "label")} for i in insts]
    except VastError as e:
        say(f"API unavailable: {e}")
    say(f"current: {st.state.get('current')}  batch: {json.dumps(st.state.get('batch'))[:200]}")
    for iid, rec in st.state["instances"].items():
        if rec.get("role") == "gone":
            continue
        h = rec.get("last_health")
        if h:
            say(f"last health ({iso(h['t'])}): " + health_line(h))
        outs = rec.get("outages") or []
        if outs or rec.get("unreachable_since"):
            say(f"[{iid}] outages: " + ", ".join(f"{iso(o['start'])[11:16]}+{o['min']:.0f}m" for o in outs[-10:])
                + (f"  OPEN since {iso(rec['unreachable_since'])}" if rec.get("unreachable_since") else ""))
        ok, missing = all_fetched(rec)
        say(f"[{iid}] role={rec.get('role')} runs fetched: {'all' if ok else 'missing ' + ', '.join(missing)}")
    for p in st.state.get("pending", []):
        say(f"PENDING ({iso(p['t'])}): {p['msg']}")
    say(f"blocklist: {json.dumps(st.state['blocklist'])}")
    say(f"last watchdog pass: {iso(st.state.get('last_watchdog'))}")
    for e in st.state["events"][-a.events:]:
        say(f"  event {iso(e['t'])} {'[' + str(e['id']) + '] ' if e.get('id') else ''}{e['msg']}")
    res.update({"current": st.state.get("current"), "pending": st.state.get("pending"),
                "managed": st.state["instances"], "blocklist": st.state["blocklist"]})
    return res


def cmd_simple(ctx: Ctx, a: argparse.Namespace) -> dict:
    iid = ctx.iid(a.id)
    if a.command == "stop":
        stop(ctx, iid, "(manual)")
    elif a.command == "start":
        ctx.api.set_state(iid, "running")
        ctx.store.event("start requested (manual)", iid)
        if a.wait:
            wait_ready(ctx, iid, ctx.store.config["create_wait_min"])
    elif a.command == "reboot":
        say("WARNING: reboot restarts the container; running jobs die")
        ctx.api.reboot(iid)
        ctx.store.event("reboot requested (manual)", iid)
    return {"id": iid, "action": a.command}


def cmd_destroy(ctx: Ctx, a: argparse.Namespace) -> dict:
    iid = ctx.iid(a.id)
    if not a.yes:
        raise VastError("destroy is irreversible: pass --yes")
    ok = destroy(ctx, iid, force=a.force, reason="(manual)")
    return {"id": iid, "destroyed": ok}


def cmd_logs(ctx: Ctx, a: argparse.Namespace) -> dict:
    iid = ctx.iid(a.id)
    r = Remote(ctx, iid).box(f"logs {int(a.n)}", timeout=40)
    print(redact(r.out))
    return {"id": iid}


def cmd_block(ctx: Ctx, a: argparse.Namespace) -> dict:
    if a.remove:
        ctx.store.state["blocklist"].pop(str(a.host), None)
        ctx.store.event(f"host {a.host} removed from blocklist")
    else:
        ctx.store.state["blocklist"][str(a.host)] = a.reason or "manual"
        ctx.store.event(f"host {a.host} blocklisted: {a.reason or 'manual'}")
    return {"blocklist": ctx.store.state["blocklist"]}


# ----------------------------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    # SUPPRESS: the options work before or after the subcommand without the subparser resetting them
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="machine-readable output on stdout")
    common.add_argument("--id", default=argparse.SUPPRESS,
                        help="instance id (default: current instance in .vast/state.json)")
    p = argparse.ArgumentParser(prog="vastctl", description=__doc__.split("\n")[0], parents=[common])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("balance", parents=[common], help="credit, spend rate, runway")
    for name in ("offers", "create"):
        s = sub.add_parser(name, parents=[common], help="search offers" if name == "offers" else "rent + wait")
        s.add_argument("--gpu", help=f"comma list (default {','.join(GPU_CLASS)})")
        s.add_argument("--min-vram", type=float, default=24)
        s.add_argument("--max-price", type=float)
        s.add_argument("--min-reliability", type=float, default=0.99)
        s.add_argument("--min-inet", type=float, default=500)
        s.add_argument("--min-disk", type=float, default=50)
        s.add_argument("--min-cuda", type=float, default=12.4)
        s.add_argument("--exclude-hosts", default="", help="extra host ids to skip (comma list)")
        s.add_argument("--interruptible", action="store_true")
        s.add_argument("--sort", choices=["score", "price"], default="score")
        s.add_argument("--top", type=int, default=10)
        if name == "create":
            s.add_argument("--offer", help="offer id from `offers`")
            s.add_argument("--auto", action="store_true", help="best score within max price")
            s.add_argument("--allow-second", action="store_true")
            s.add_argument("--image", help=f"container image (default {IMAGE}; fallback {FALLBACK_IMAGE})")
    s = sub.add_parser("adopt", parents=[common], help="manage an existing instance")
    s.add_argument("instance")
    s.add_argument("--runs", nargs="*", help="run dirs on the box to track for fetch (default bench/results)")
    sub.add_parser("endpoint", parents=[common], help="resolve + test current SSH endpoints")
    s = sub.add_parser("ssh", parents=[common], help="ssh (interactive) or run `-- cmd`")
    s.add_argument("--timeout", type=float, default=120)
    s.add_argument("cmd", nargs=argparse.REMAINDER)
    s = sub.add_parser("put", parents=[common], help="copy local files to the box (via ssh stdin, md5-checked)")
    s.add_argument("files", nargs="+")
    s.add_argument("dest", help="remote path; ending in / means a directory (one file per local name)")
    s.add_argument("--binary", action="store_true", help="keep CRLF bytes (default: text, CRLF -> LF)")
    s = sub.add_parser("provision", parents=[common], help="repo + toolchain + models + watchdog on the box")
    s.add_argument("--models", default="", help="comma list of org/name:file.gguf (default: none)")
    s.add_argument("--wait-min", type=float)
    s = sub.add_parser("run", parents=[common], help="push HEAD + start a script detached on the box")
    s.add_argument("script", help="repo-relative script, e.g. scripts/nightly.sh")
    s.add_argument("args", nargs="*")
    s = sub.add_parser("nightly", parents=[common], help="create, provision, run nightly.sh, wait, fetch, destroy")
    s.add_argument("--models", default="", help=f"comma list of org/name:file.gguf (default: the models of {NIGHTLY_SPEC})")
    s.add_argument("--image", help=f"container image (default {IMAGE}; fallback {FALLBACK_IMAGE})")
    s.add_argument("--gpu", help=f"comma list (default {','.join(GPU_CLASS)})")
    s.add_argument("--max-price", type=float)
    s.add_argument("--wait-min", type=float, help="how long to wait for the on-box run (default nightly_wait_min)")
    s = sub.add_parser("fetch", parents=[common], help="pack results on the box, scp back, verify, unpack")
    s.add_argument("dirs", nargs="*")
    s.add_argument("--minutes", type=float, help="keep retrying this long")
    for name in ("health", "watch"):
        s = sub.add_parser(name, parents=[common], help="one health check" if name == "health" else "loop")
        s.add_argument("--auto-heal", action="store_true", help="reboot once when unreachable (restarts container)")
        s.add_argument("--reboot-after", type=float, default=DEFAULT_CONFIG["reboot_after_unreachable_min"])
        s.add_argument("--stop-idle", action="store_true", help="fetch + stop when idle >= idle_stop_min")
        s.add_argument("--no-heal", action="store_true", help="do not relaunch a dead on-box watchdog")
        if name == "watch":
            s.add_argument("--interval", type=float, default=60)
    s = sub.add_parser("watchdog", parents=[common], help="local health + cost-control loop")
    s.add_argument("--once", action="store_true", help="one pass (for Task Scheduler)")
    s.add_argument("--interval", type=float)
    s.add_argument("--auto-heal", action="store_true")
    s.add_argument("--allow-migrate", action="store_true")
    s.add_argument("--no-stop-idle", action="store_true")
    s.add_argument("--no-destroy", action="store_true")
    s.add_argument("--dry-run", action="store_true", help="report decisions, change nothing")
    s = sub.add_parser("status", parents=[common], help="instances, balance, health, outages, pending actions")
    s.add_argument("--events", type=int, default=15)
    for name in ("stop", "start", "reboot"):
        s = sub.add_parser(name, parents=[common])
        if name == "start":
            s.add_argument("--wait", action="store_true")
    s = sub.add_parser("destroy", parents=[common], help="irreversible; needs --yes and fetched runs")
    s.add_argument("--yes", action="store_true")
    s.add_argument("--force", action="store_true", help="destroy even with unfetched runs / unmanaged id")
    s = sub.add_parser("logs", parents=[common], help="tail the nightly log and the watchdog log")
    s.add_argument("-n", type=int, default=30)
    s = sub.add_parser("block", parents=[common], help="add/remove a host in the blocklist")
    s.add_argument("host")
    s.add_argument("--reason")
    s.add_argument("--remove", action="store_true")
    return p


COMMANDS: dict[str, Callable[[Ctx, argparse.Namespace], Any]] = {
    "balance": cmd_balance, "offers": cmd_offers, "create": cmd_create, "adopt": cmd_adopt,
    "endpoint": cmd_endpoint, "ssh": cmd_ssh, "put": cmd_put, "provision": cmd_provision, "run": cmd_run, "nightly": cmd_nightly, "fetch": cmd_fetch,
    "health": cmd_health, "watch": cmd_watch, "watchdog": cmd_watchdog, "status": cmd_status,
    "stop": cmd_simple, "start": cmd_simple, "reboot": cmd_simple, "destroy": cmd_destroy, "logs": cmd_logs,
    "block": cmd_block,
}


def main(argv: list[str] | None = None, ctx: Ctx | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252; box output is UTF-8
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    a = build_parser().parse_args(argv)
    a.json, a.id = getattr(a, "json", False), getattr(a, "id", None)
    if a.command == "ssh" and a.cmd and a.cmd[0] == "--":
        a.cmd = a.cmd[1:]
    Out.json_mode = bool(a.json)
    ctx = ctx or Ctx()
    try:
        res = COMMANDS[a.command](ctx, a)
    except (VastError, Unreachable, ToolMissing) as e:
        say(f"ERROR: {e}")
        if a.json:
            print(redact(json.dumps({"error": str(e)})))
        return 1
    except KeyboardInterrupt:
        return 130
    if isinstance(res, int):
        return res
    if a.json:
        print(redact(json.dumps(res, indent=1, default=str)))
    if a.command == "destroy" and isinstance(res, dict) and not res.get("destroyed"):
        return 1
    if a.command == "nightly" and isinstance(res, dict) and not res.get("ok"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
