#!/usr/bin/env python3
"""Case S scout probe (advisor, delegated by the owner 2026-11-27; docs/box-driver.md "Pass list and scouts").

Rents ONE box on a static-passing machine that is on neither the pass list nor the skip list, reads the host properties the
offer does not carry at first SSH, and destroys the box within minutes. It never holds a box for a partner and never
judges anything: a pass puts the machine on the pass list (PASS_MACHINES in scripts/phase6_pair_finder.py, kind "scout"),
a host-property failure puts the host on the skip list, by hand, with the probe's README row.

  uv run python scripts/phase6_scout.py probe OUT_DIR [--offer ID --gpu CLASS --machine M]   pick (finder --scout), create, check, destroy
  uv run python scripts/phase6_scout.py check --id ID --out OUT_DIR [--no-cuda]             the checks on a box that is up (pair_create's lone box)

Checks, in this order (one ssh call, then provision only if the raw checks pass):
  sockets    lscpu Socket(s) == 1
  numa       lscpu NUMA node(s) == 1 (the gate of phase6_box.sh gates)
  power      every GPU's power.limit within 1 W of power.default_limit (scripts/power_gate.sh's rule)
  memory     cgroup memory.max at least 64 GB (decimal, the balloon's unit); an offer property, not a host one
  cuda       `provision --models ,` logs `cuda: ok` or `cuda: ok_preload` (also proves the pull and the push work)
Host properties (skip list on failure): sockets, numa, power, cuda. memory fails only that offer. A box that never opens SSH
within the create's wait is a stall: not a host property after one (docs/box-driver.md), the second puts the host on the list.
Writes OUT_DIR/scout-<host>-<machine>.json and the raw readings beside it. Exit 0 pass, 1 a check failed, 2 stall or error.
The API key never passes through here: taxameter reads it from its file.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
MIN_MEMORY_BYTES = 64_000_000_000
HOST_CHECKS = ("sockets", "numa", "power", "cuda")
RAW = (
    'echo "K memmax $(cat /sys/fs/cgroup/memory.max 2> /dev/null)"; echo "K cpumax $(cat /sys/fs/cgroup/cpu.max 2> /dev/null)"; '
    "lscpu | sed -n -e 's/^Model name: */K model /p' -e 's/^CPU(s): */K cpus /p' -e 's/^Socket(s): */K sockets /p' "
    "-e 's/^NUMA node(s): */K numa /p'; "
    "nvidia-smi --query-gpu=name,power.limit,power.default_limit,power.max_limit --format=csv,noheader,nounits | sed 's/^/K smi /'"
)


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tax(*args, timeout=300, log=None):
    """`uv run taxameter ARGS`; returns (rc, output). Output goes to `log` too when given."""
    try:
        r = subprocess.run(["uv", "run", "taxameter", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        out, rc = (r.stdout or "") + (r.stderr or ""), r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = f"{e.stdout or ''}{e.stderr or ''}\ntimeout after {timeout} s", 124
        out = out if isinstance(out, str) else out.decode(errors="replace")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"$ taxameter {' '.join(args)}\n{out}\n")
    return rc, out


def parse_raw(text):
    """The K lines of the raw ssh call into readings."""
    k = {"smi": []}
    for line in text.replace("\r", "").splitlines():
        m = re.match(r"^K (\w+) ?(.*)$", line.strip())
        if not m:
            continue
        key, val = m.groups()
        if key == "smi":
            k["smi"].append([v.strip() for v in val.split(",")])
        else:
            k[key] = val.strip()
    return k


def judge_raw(k):
    """{check: (ok, detail)} for sockets, numa, power, memory."""

    def intval(key):
        try:
            return int(k.get(key, "").split()[0])
        except (IndexError, ValueError):
            return None

    res = {}
    res["sockets"] = (intval("sockets") == 1, f"Socket(s) {k.get('sockets')}")
    res["numa"] = (intval("numa") == 1, f"NUMA node(s) {k.get('numa')}")
    gpus = []
    ok = bool(k["smi"])
    for row in k["smi"]:
        try:
            lim, dflt = float(row[1]), float(row[2])
            good = abs(lim - dflt) <= 1
        except (IndexError, ValueError):
            good, lim, dflt = False, None, None
        ok = ok and good
        gpus.append(f"{row[0] if row else '?'} limit {lim} W default {dflt} W{'' if good else ' LIMITED'}")
    res["power"] = (ok, "; ".join(gpus) or "no nvidia-smi values")
    mem = intval("memmax")
    res["memory"] = (mem is not None and mem >= MIN_MEMORY_BYTES, f"memory.max {k.get('memmax')}")
    return res


def check(box, outd, cuda=True, meta=None):
    """The checks on box `box` (an id that is up); writes the result JSON, returns it."""
    outd = pathlib.Path(outd)
    outd.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    log = outd / f"scout-{box}.log"
    say(f"raw checks on box {box}")
    rc, out = tax("ssh", "--id", str(box), "--timeout", "90", "--", RAW, timeout=150, log=log)
    k = parse_raw(out)
    checks = {n: {"ok": ok, "detail": d} for n, (ok, d) in judge_raw(k).items()}
    raw_ok = rc == 0 and all(c["ok"] for c in checks.values())
    if cuda and raw_ok:
        say(f"provision --models , on box {box} (cuda)")
        prc, pout = tax("provision", "--id", str(box), "--models", ",", timeout=900, log=log)
        good = re.search(r"cuda: ok(_preload)?\b", pout) is not None
        checks["cuda"] = {"ok": good, "detail": (re.findall(r"cuda: \S+", pout) or [f"provision rc={prc}, no cuda line"])[-1]}
    elif cuda:
        checks["cuda"] = {"ok": None, "detail": "not run: a raw check failed"}
    host_fail = [n for n in HOST_CHECKS if checks.get(n, {}).get("ok") is False]
    passed = rc == 0 and all(c["ok"] is True for c in checks.values())
    res = {
        "box": str(box), **meta, "t": time.time(), "ssh_rc": rc, "raw": {k_: v for k_, v in k.items()}, "checks": checks,
        "pass": passed, "kind": "scout" if passed else None,
        "host_property_failures": host_fail,
        "list": "pass" if passed else ("skip" if host_fail else None),
    }  # fmt: skip
    name = f"scout-{meta.get('host', 'box')}-{meta.get('machine', box)}.json"
    (outd / name).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for n, c in checks.items():
        say(f"  {n:<8} {'ok' if c['ok'] else ('FAIL' if c['ok'] is False else '-')}  {c['detail']}")
    say(f"scout {'PASS' if passed else 'FAIL'} ({outd / name})")
    return res


def created_id(text):
    m = re.search(r"\[(\d+)\] created instance", text)
    return m.group(1) if m else None


def destroy(box, log):
    rc, out = tax("destroy", "--id", str(box), "--yes", "--force", timeout=180, log=log)
    say(f"destroy {box}: rc={rc} {out.strip().splitlines()[-1] if out.strip() else ''}")
    return rc


def pick(outd, finder_args):
    """The first scout candidate of the finder (live market): (offer id, gpu class, machine, host, row)."""
    py = sys.executable
    r = subprocess.run([py, str(ROOT / "scripts/phase6_pair_finder.py"), "--scout", "--out", str(outd / "scouts.json"), *finder_args],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)  # fmt: skip
    print(r.stdout.strip())
    rows = json.loads((outd / "scouts.json").read_text(encoding="utf-8")).get("scouts", []) if (outd / "scouts.json").exists() else []
    return rows[0] if rows else None


def current_offer(machine, fallback, gpu, outd):
    """The listed offer id of the machine now (offer ids rotate; .vast/offers.json, taxameter ADR 0013), else the finder's."""
    subprocess.run(["bash", "scripts/sc_offers.sh", "offers", "--gpu", gpu, "--top", "500"], cwd=ROOT, capture_output=True, timeout=300)
    path = ROOT / ".vast" / "offers.json"
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    hit = next((k for k, v in d.items() if str(v.get("machine_id")) == str(machine)), None)
    if hit:
        return hit
    # one Vast search returns a varying subset of the market (2026-11-27: two searches a second apart shared 17 of 24 machines),
    # so the listing may miss the machine the finder's merged reads saw. Record the finder's id under its machine in taxameter's
    # map (ADR 0013 format): `create --offer` then follows it with a search on that machine id, same filters and caps
    d[str(fallback)] = {"machine_id": str(machine), "num_gpus": 1, "gpu_name": gpu}
    path.write_text(json.dumps(d), encoding="utf-8")
    return fallback


def probe(a):
    outd = pathlib.Path(a.out)
    outd.mkdir(parents=True, exist_ok=True)
    log = outd / "probe.log"
    if a.offer:
        row = {"id": a.offer, "gpu_name": a.gpu, "machine_id": a.machine, "host_id": a.host}
    else:
        row = pick(outd, a.finder_args)
        if not row:
            say("no scout candidate: nothing created")
            return 3
    meta = {"host": str(row.get("host_id")), "machine": str(row.get("machine_id")), "gpu": row.get("gpu_name"),
            "cpu_name": row.get("cpu_name"), "host_threads": row.get("host_threads"), "all_in": row.get("all_in"),
            "two_socket_hint": row.get("two_socket_hint")}  # fmt: skip
    offer = current_offer(row.get("machine_id"), row["id"], row["gpu_name"], outd) if row.get("machine_id") else row["id"]
    say(f"scout: offer {offer} host {meta['host']} machine {meta['machine']} ({meta['gpu']}, {meta['cpu_name']}); create")
    for stale in (outd / f"scout-{meta['host']}-{meta['machine']}.json",):
        stale.unlink(missing_ok=True)
    t0 = time.time()
    # the create waits for SSH longer than the 10 min stall rule (docs/box-driver.md): bound it, the log file keeps the box id
    clog = outd / f"create-{meta['host']}.log"
    with open(clog, "w", encoding="utf-8") as f:
        pr = subprocess.Popen(["bash", "scripts/sc_offers.sh", "create", "--offer", str(offer), "--gpu", row["gpu_name"]],
                              cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True)  # fmt: skip
        try:
            create_rc = pr.wait(timeout=a.create_timeout)
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pr.pid)], capture_output=True)
            else:
                pr.kill()
            pr.wait()
            create_rc = 124
    out = clog.read_text(encoding="utf-8", errors="replace")
    if create_rc == 124:
        with open(clog, "a", encoding="utf-8") as f:
            f.write(f"\ncreate stopped after {a.create_timeout} s (stall rule)\n")
    box = created_id(out)
    if not box:
        say("create rented no box: " + (out.strip().splitlines() or ["?"])[-1])
        (outd / f"scout-{meta['host']}-{meta['machine']}.json").write_text(
            json.dumps({**meta, "box": None, "pass": False, "list": None, "note": "no box rented", "t": t0}, indent=1), encoding="utf-8")
        return 2
    meta.update(box=box, t_created=t0)
    try:
        if create_rc == 0:
            res = check(box, outd, cuda=not a.no_cuda, meta=meta)
            rc = 0 if res["pass"] else 1
        else:
            say(f"box {box} rented but create did not finish (SSH or pull stall): destroy; one stall is not a host property")
            (outd / f"scout-{meta['host']}-{meta['machine']}.json").write_text(
                json.dumps({**meta, "pass": False, "list": None, "note": "stall: create did not finish", "t": time.time()}, indent=1),
                encoding="utf-8")  # fmt: skip
            rc = 2
    finally:
        destroy(box, log)
        say(f"box {box} up {(time.time() - t0) / 60:.1f} min")
    return rc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("out")
    pr.add_argument("--offer")
    pr.add_argument("--gpu", default="RTX 3090")
    pr.add_argument("--machine")
    pr.add_argument("--host")
    pr.add_argument("--no-cuda", action="store_true")
    pr.add_argument("--create-timeout", type=int, default=600, help="seconds; the stall rule is 10 min after create")
    pr.add_argument("finder_args", nargs="*", help="after --: extra finder flags (e.g. --skip-hosts H)")
    ck = sub.add_parser("check")
    ck.add_argument("--id", required=True)
    ck.add_argument("--out", required=True)
    ck.add_argument("--host")
    ck.add_argument("--machine")
    ck.add_argument("--no-cuda", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "check":
        res = check(a.id, a.out, cuda=not a.no_cuda, meta={k: v for k, v in (("host", a.host), ("machine", a.machine)) if v})
        return 0 if res["pass"] else 1
    return probe(a)


if __name__ == "__main__":
    sys.exit(main())
