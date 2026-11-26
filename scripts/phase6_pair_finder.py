#!/usr/bin/env python3
"""Case S pair finder (ADR 0091 answers 9 and 10; docs/notes/phase6-rule-changes.md, 2026-11-26 "Case S host rules").

Reads the market (never creates or destroys a box), applies the static host rules to every offer, pairs the survivors on
different hosts and ranks the pairs by expected bulk link rate, then price. Create both boxes together, and only when this
lists a pair (exit 0); never hold one box for a partner.

  uv run python scripts/phase6_pair_finder.py                       live: one `taxameter offers --json` call per GPU class
  uv run python scripts/phase6_pair_finder.py --offers-json F ...   offline: recorded offers (the tests; `--save-offers F` records)
  TETRAEDER_PLAN_BIN=target/release/tetraeder-plan                  also run `candidates --cluster --covering` per pair

Rules per offer, in this order (the first failing one is the one counted in "excluded"):
  gpu_class    1 GPU of an allowed class: RTX 3090; with --allow-aclass also A40, RTX A6000, L40 (ADR 0091 answer 9). A-class
               is off by default: the cluster profile has no 24 GiB cap or prior for it (node cluster-profile-gpu-cap), so a
               pair with one cannot be planned (phase6-exit-s-20261126, pair 1).
  skip_host    host on the skip list below (docs/box-driver.md)
  cores        usable cores at least --min-cores (12)
  ram          container RAM at least the floor derived below
  disk         at least 230 GB
  inet         download at least 1000 Mbit/s
  price        price plus the download traffic spread over the session at most 0.90 USD/h (AGENTS.md)
Unknown on an offer (flagged, left to the gates on the box): power limit (the offer gives only the maximum), NUMA nodes and
sockets (`lscpu` on the box). Pair rules: different hosts; then the covering plan when a planner binary is given.

RAM floor. The planner's host budget for the heavier node of the covering pick (node b, 47 layers: 32 VRAM + 96 RAM slots
of 128 experts, ADR 0091) is 57.1 GB. The budget under the 64 GB balloon is `memory.max - memory.current + inactive_file - 1 GiB`
with the balloon holding `memory.max - 64 GB` (balloon_guard.py), so it is 64 GB minus what the box itself uses: 4.6 GB at
the 59.4 GB the committed pick was made on, 4.2 and 2.9 GB measured on 2026-11-26 (59.77 and 61.06 GB). A box whose real limit
is under 64 GB runs without the balloon on its own headroom, real limit minus the same use. The floor is the need plus the
worst use: 57.1 + 4.6 = 61.7 GB, the same for both nodes since the pick may put either role on either box. From 64 GB up the
balloon fixes the budget at 59.4 GB (2.3 GB over the need), so more RAM buys nothing.
"""

import argparse
import itertools
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

NODE_B_HOST_GB = 57.1  # planner's host estimate for the heavier node of the covering pick
BOX_USE_GB = 4.6  # 64 GB minus the budget 59.4 GB the committed pick was made on (measured 4.2 and 2.9 on 2026-11-26)
BALLOON_GB = 64.0
RAM_FLOOR_GB = round(NODE_B_HOST_GB + BOX_USE_GB, 1)

MIN_CORES = 12
MIN_DISK_GB = 230
MIN_INET_MBPS = 1000
MAX_PRICE = 0.90
DOWNLOAD_GB = 150
SESSION_HOURS = 3.0  # max_box_hours: the traffic is priced over it as taxameter does
CLASSES_3090 = ("RTX 3090",)
CLASSES_A = ("A40", "RTX A6000", "L40")
LAYOUT = "bench/results/segment-confirm-20261125b/layout.json"
GRAPH_TEMPLATE = "crates/costmodel/fixtures/cluster/cluster_two_3090_wan.toml"

# Hosts that must not be rented for Case S. Every id appears in docs/box-driver.md (a test checks), with its reason.
SKIP_HOSTS = {
    "366851": "taxameter blocklist", "483833": "taxameter blocklist", "269862": "slow downloads (13d, 13e)",
    "36536": "GPU power capped (phase6-exit-s-20261126b)", "44173": "SSH rejected the key; two sockets",
    "568479": "two sockets", "546840": "two sockets; never finished the pull", "281790": "two sockets",
    "87485": "busy co-tenant", "213498": "SSH rejected the key", "109790": "stayed loading twice",
    "39788": "stayed loading", "151822": "stalled twice", "130210": "never finished the pull", "51345": "never finished the pull",
    "551866": "never finished the pull", "87657": "never finished the pull", "70221": "never finished the pull",
    "305845": "never finished the pull", "116365": "never finished the pull", "393491": "never finished the pull",
    "92795": "never finished the pull", "166946": "SSH rejected the key", "450573": "SSH rejected the key",
    "569309": "stayed loading", "142": "stayed loading", "443829": "stayed loading", "451840": "75 KB/s downloads",
    "208874": "stalled twice", "167329": "stayed loading", "608696": "GPUs power capped", "685226": "GPUs power capped",
    "399360": "GPUs power capped", "203531": "never opened SSH", "611393": "GPUs below default power",
    "706255": "GPUs below default power", "397236": "stalled, then power capped",
}  # fmt: skip
RULES = ("gpu_class", "skip_host", "cores", "ram", "disk", "inet", "price")


def num(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def all_in(o, download_gb=DOWNLOAD_GB, hours=SESSION_HOURS):
    """USD/h with the download spread over the session, as taxameter's effective price; None when traffic is unknown."""
    cost = num(o.get("inet_down_cost"))
    p = num(o.get("price_per_h"))
    return None if cost is None or p is None else p + cost * download_gb / hours


def failing(o, a):
    """Every rule the offer fails, in RULES order."""
    allowed = CLASSES_3090 + (CLASSES_A if a.allow_aclass else ())
    price = all_in(o, a.download_gb, a.session_hours)
    bad = {
        "gpu_class": int(num(o.get("num_gpus")) or 1) != 1 or o.get("gpu_name") not in allowed,
        "skip_host": str(o.get("host_id")) in a.skip_hosts,
        "cores": (num(o.get("cpu_cores")) or 0) < a.min_cores,
        "ram": (num(o.get("ram_gb")) or 0) < a.ram_floor_gb,  # unknown RAM cannot be shown to fit
        "disk": (num(o.get("disk_gb")) or 0) < a.min_disk,
        "inet": (num(o.get("inet_down_mbps")) or 0) < a.min_inet,
        "price": price is None or price > a.max_price,
    }
    return [r for r in RULES if bad[r]]


def screen(offers, a):
    """(survivors, counts): `excluded` is first-failure attribution, `fails` how many offers fail each rule on its own."""
    counts = {"offers": len(offers), "excluded": dict.fromkeys(RULES, 0), "fails": dict.fromkeys(RULES, 0)}
    keep = []
    for o in offers:
        bad = failing(o, a)
        for r in bad:
            counts["fails"][r] += 1
        if bad:
            counts["excluded"][bad[0]] += 1
        else:
            keep.append(o)
    counts["survivors"] = len(keep)
    counts["no_provider_row"] = sum(1 for o in offers if o.get("ram_gb") is None)  # gone or rotated between the two reads: RAM unknown
    return keep, counts


def bulk_mbps(x, y):
    """Expected bulk rate between two boxes: the slower of the two directions, each limited by the sender's upload and the
    receiver's download (Mbit/s)."""
    ups = [num(x.get("inet_up_mbps")), num(y.get("inet_up_mbps"))]
    downs = [num(x.get("inet_down_mbps")), num(y.get("inet_down_mbps"))]
    known = [v for v in ups + downs if v is not None]
    return min(known) if known else 0.0


def pair_key(p):
    """Rank: bulk rate (500 Mbit/s buckets, capped at 4000: beyond it the path, not the access link, limits), then datacenter
    hosts (hosting_type 1), then price. RTT is not ranked: the covering pick is the same on every link of the committed fixtures."""
    bucket = int(min(p["bulk_mbps"], 4000) // 500)
    return (-bucket, -p["datacenter"], p["price_per_h"])


def make_pairs(survivors, dl=DOWNLOAD_GB, hrs=SESSION_HOURS):
    pairs = []
    for x, y in itertools.combinations(survivors, 2):
        if str(x["host_id"]) == str(y["host_id"]):
            continue
        dc = int(num(x.get("hosting_type")) == 1) + int(num(y.get("hosting_type")) == 1)
        pairs.append({
            "a": x["id"], "b": y["id"], "hosts": [str(x["host_id"]), str(y["host_id"])],
            "gpus": [x["gpu_name"], y["gpu_name"]], "locations": [x.get("location"), y.get("location")],
            "bulk_mbps": bulk_mbps(x, y), "datacenter": dc,
            "price_per_h": round(all_in(x, dl, hrs) + all_in(y, dl, hrs), 3),
            "boxes": [x, y],
        })  # fmt: skip
    pairs.sort(key=pair_key)
    return pairs


def flags(o, a):
    """What the offer does not say and the gates on the box must (power limit, NUMA nodes, sockets)."""
    f = ["power limit unknown (gate on box)", "NUMA/sockets unknown (lscpu gate on box)"]
    if (num(o.get("ram_gb")) or 0) < BALLOON_GB:
        f.append("RAM under 64 GB: no balloon, measured headroom")
    if o.get("gpu_name") in CLASSES_A:
        f.append("A-class: planned against 24 GiB, recorded as a deviation")
    return f


def ram_budget_gb(ram_gb):
    """The node's planner RAM budget: the balloon fixes it at 64 GB minus the box's own use; below 64 GB, the real limit does."""
    return round(min(ram_gb, BALLOON_GB) - BOX_USE_GB, 1)


def pair_graph(x, y, template=None):
    """Two-node graph from the committed WAN fixture (3090 prior, the measured WAN link) with each node's RAM budget and
    thread count from its offer. The GPU-class priors of the Rust lane are not landed: A-class nodes use the 3090 prior."""
    text = pathlib.Path(template or ROOT / GRAPH_TEMPLATE).read_text(encoding="utf-8")
    for node, o in (("a", x), ("b", y)):
        cap = int(ram_budget_gb(num(o["ram_gb"])) * 1e9)
        pat = re.compile(rf'(id = "{node}/cpu0".*?capacity_bytes = )\d+(.*?threads = )\d+', re.DOTALL)
        text, n = pat.subn(lambda m: f"{m.group(1)}{cap}{m.group(2)}{int(num(o['cpu_cores']))}", text, count=1)
        if n != 1:
            raise ValueError(f"graph template has no cpu0 block for node {node}")
    return text


def run_plan(cmd, hw_text, layout, work):
    """`candidates --cluster --covering` on one pair's graph; returns {covering, n, best, error}."""
    work = pathlib.Path(work)
    hw = work / "hw.toml"
    hw.write_text(hw_text, encoding="utf-8")
    out = work / "candidates.json"
    r = subprocess.run(
        [*cmd, "candidates", "--cluster", "--covering", "--hw", str(hw), "--layout", str(layout), "--plans-dir",
         str(work / "plans"), "--out", str(out), "--prompt", "2048"],
        capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    if r.returncode != 0 or not out.exists():
        return {"covering": False, "n": 0, "best": None, "error": (r.stderr or r.stdout).strip().splitlines()[-1:]}
    cands = json.loads(out.read_text(encoding="utf-8")).get("candidates", [])
    best = cands[0] if cands else None
    return {"covering": bool(cands), "n": len(cands), "best": best and {"id": best["id"], "notes": best.get("notes")},
            "error": None}  # fmt: skip


def fetch_class(cls, a):
    """One `taxameter offers --json` call for a GPU class, enriched with the raw rows (RAM, disk, inet, datacenter). Reads only."""
    cmd = ["uv", "run", "taxameter", "offers", "--json", "--gpu", cls, "--num-gpus", "1", "--min-disk", str(a.min_disk),
           "--min-inet", "1", "--min-cpu-cores", "1", "--max-price", "5", "--download-gb", str(a.download_gb), "--top", "500"]  # fmt: skip
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=300)
    if r.returncode != 0:
        raise SystemExit(f"taxameter offers failed for {cls}: {r.stderr.strip()[-300:]}")
    rows = json.loads(r.stdout)["offers"]
    raws = raw_rows(cls, a)
    for o in rows:
        enrich(o, match_raw(o, raws))
    return rows


def match_raw(o, raws):
    """The provider row of a listed offer. Offer ids rotate between two searches (taxameter ADR 0013), so a row is matched by
    id, else by host and price (the same machine at the same price); no match leaves the fields unknown."""
    for raw, off in raws:
        if str(off.id) == str(o["id"]):
            return raw
    hits = [raw for raw, off in raws if str(off.host_id) == str(o["host_id"]) and abs(off.price_per_h - o["price_per_h"]) < 1e-6]
    return hits[0] if hits else None


def raw_rows(cls, a):
    """The provider's own rows (raw, Offer) pairs, through the taxameter library with the CLI's filters (same read-only search)."""
    from taxameter.providers.base import OfferQuery
    from taxameter.providers.vast import VastProvider

    q = OfferQuery(gpus=[cls], min_vram_gb=24, min_disk_gb=a.min_disk, min_reliability=a.min_reliability, min_inet_mbps=1,
                   min_cuda=12.8, min_cpu_cores=1)  # fmt: skip
    return [(o.raw, o) for o in VastProvider().offers(q)]


def enrich(o, raw):
    """Adds the fields the CLI JSON lacks; absent stays absent (the rules treat it as unknown)."""
    if not raw:
        return
    ram = num(raw.get("cpu_ram"))
    o["ram_gb"] = None if ram is None else ram / 1024  # MB, as taxameter reads it
    for k, rk in (("disk_gb", "disk_space"), ("inet_down_mbps", "inet_down"), ("inet_up_mbps", "inet_up"),
                  ("hosting_type", "hosting_type"), ("cpu_name", "cpu_name"), ("gpu_max_power_w", "gpu_max_power")):  # fmt: skip
        if raw.get(rk) is not None:
            o[k] = raw[rk]


def load_offers(files):
    rows = []
    for f in files:
        doc = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        rows += doc["offers"] if isinstance(doc, dict) else doc
    return rows


def find(a, plan_cmd=None):
    """The whole finder: (result dict). `plan_cmd` is the planner command prefix (a list) or None."""
    if a.offers_json:
        offers = load_offers(a.offers_json)
    else:
        classes = CLASSES_3090 + (CLASSES_A if a.allow_aclass else ())
        offers = [o for c in classes for o in fetch_class(c, a)]
        if a.save_offers:
            pathlib.Path(a.save_offers).write_text(json.dumps({"offers": offers}, indent=1), encoding="utf-8")
    survivors, counts = screen(offers, a)
    pairs = make_pairs(survivors, a.download_gb, a.session_hours)
    counts["pairs_same_host_dropped"] = len(survivors) * (len(survivors) - 1) // 2 - len(pairs)
    counts["pairs_no_covering_plan"] = 0
    listed = []
    for p in pairs:
        x, y = p["boxes"]
        p["flags"] = sorted(set(flags(x, a)) | set(flags(y, a)))
        if plan_cmd:
            with tempfile.TemporaryDirectory() as td:
                p["plan"] = run_plan(plan_cmd, pair_graph(x, y), ROOT / a.layout, td)
            if not p["plan"]["covering"]:
                counts["pairs_no_covering_plan"] += 1
                continue
        else:
            p["plan"] = None
            p["flags"].append("covering plan not run (no TETRAEDER_PLAN_BIN)")
        listed.append(p)
    return {"rules": {"ram_floor_gb": a.ram_floor_gb, "min_cores": a.min_cores, "min_disk_gb": a.min_disk,
                      "min_inet_mbps": a.min_inet, "max_price_per_h": a.max_price, "download_gb": a.download_gb,
                      "session_hours": a.session_hours, "allow_aclass": a.allow_aclass},
            "counts": counts, "pairs": listed}  # fmt: skip


def table(res, top):
    c = res["counts"]
    out = [f"{c['offers']} offers, {c['survivors']} pass the static rules; excluded first by rule / failing the rule on its own:"]
    out += [f"  {r:<10} {c['excluded'][r]:>4} / {c['fails'][r]:>4}" for r in RULES]
    out.append(f"  ({c['no_provider_row']} offers had no provider row on the second read, so RAM, disk and inet are unknown: counted under ram)")
    out.append(f"  pairs dropped for the same host: {c['pairs_same_host_dropped']}; for no covering plan: {c['pairs_no_covering_plan']}")
    if not res["pairs"]:
        out.append("NO PAIR: do not create any box (never hold one for a partner); retry later, never loosen a rule.")
        return "\n".join(out)
    out.append(f"{len(res['pairs'])} pair(s); top {min(top, len(res['pairs']))} by bulk rate, datacenter, price:")
    out.append("  #  offers (a, b)        hosts           gpus                  bulk Mbit/s  dc  USD/h  plan")
    for i, p in enumerate(res["pairs"][:top], 1):
        plan = "not run" if p["plan"] is None else f"{p['plan']['n']} covering"
        out.append(f"  {i:<2} {p['a']:>9} {p['b']:>9}  {'/'.join(p['hosts']):<15} {'/'.join(p['gpus']):<21} "
                   f"{p['bulk_mbps']:>10.0f} {p['datacenter']:>3}  {p['price_per_h']:.3f}  {plan}")  # fmt: skip
    out.append("  unknown on every offer, gated on the box: power limit, NUMA nodes and sockets")
    return "\n".join(out)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--offers-json", action="append", help="recorded offers (offline); repeatable")
    p.add_argument("--save-offers", help="write the live offers, enriched, to this file (a fixture)")
    p.add_argument("--out", help="write the result JSON here")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--allow-aclass", action="store_true", help="also A40, RTX A6000, L40 (the planner cannot place them yet)")
    p.add_argument("--min-cores", type=float, default=MIN_CORES)
    p.add_argument("--min-disk", type=float, default=MIN_DISK_GB)
    p.add_argument("--min-inet", type=float, default=MIN_INET_MBPS)
    p.add_argument("--min-reliability", type=float, default=0.99)
    p.add_argument("--max-price", type=float, default=MAX_PRICE)
    p.add_argument("--download-gb", type=float, default=DOWNLOAD_GB)
    p.add_argument("--session-hours", type=float, default=SESSION_HOURS)
    p.add_argument("--ram-floor-gb", type=float, default=RAM_FLOOR_GB)
    p.add_argument("--skip-hosts", default="", help="extra host ids, comma list")
    p.add_argument("--layout", default=LAYOUT)
    a = p.parse_args(argv)
    a.skip_hosts = set(SKIP_HOSTS) | {h.strip() for h in a.skip_hosts.split(",") if h.strip()}
    return a


def main(argv=None):
    a = parse_args(argv)
    binp = os.environ.get("TETRAEDER_PLAN_BIN")
    plan_cmd = shlex.split(binp, posix=os.name != "nt") if binp else None
    res = find(a, plan_cmd)
    print(table(res, a.top))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0 if res["pairs"] else 1


if __name__ == "__main__":
    sys.exit(main())
