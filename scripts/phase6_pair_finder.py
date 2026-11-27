#!/usr/bin/env python3
"""Case S pair finder (ADR 0091 answers 9 and 10; docs/notes/phase6-rule-changes.md, 2026-11-26 "Case S host rules").

Reads the market (never creates or destroys a box), applies the static host rules to every offer, pairs the survivors on
different hosts and ranks the pairs by hosts that passed the gates before, expected bulk link rate, then price. Create both boxes together, and only when this
lists a plannable pair (exit 0); never hold one box for a partner.

  uv run python scripts/phase6_pair_finder.py                       live: --reads (3) searches per GPU class, merged
  uv run python scripts/phase6_pair_finder.py --offers-json F ...   offline: recorded offers (the tests; `--save-offers F` records)
  TETRAEDER_PLAN_BIN=target/release/tetraeder-plan                  also run `candidates --cluster --covering` per pair

Data source. The brief named `taxameter offers --json`, but its JSON lacks RAM, disk, inet and datacenter, and offer ids
rotate between two searches (the second read of the first version lost 11 of 26 offers). So each read is one search through
taxameter's own provider (`VastProvider.offers`, the same query, blocklist replaced by SKIP_HOSTS below), which carries the
rows; --reads searches are merged by machine and price because the market moves between them.

Rules per offer, in this order (the first failing one is the one counted in "excluded"):
  gpu_class    1 GPU of RTX 3090, A40, RTX A6000, L40 or L40S (ADR 0091 answer 9; the per-class offer counts are printed, so a wrong
               class name shows as 0). A pair with an A-class card is plannable since the cluster profile caps and prices it at
               24 GiB (node cluster-profile-gpu-cap, done 2026-11-26); `--no-plan-aclass` restores the old listing.
  skip_host    host on the skip list below (docs/box-driver.md)
  unknown      RAM, disk, inet or traffic price missing from the row: counted apart from the real rules
  cores        usable cores at least --min-cores (12)
  ram          container RAM at least the floor derived below
  disk         at least 230 GB
  inet         download at least 1000 Mbit/s
  price        price plus the download traffic spread over the session at most 0.90 USD/h (AGENTS.md)
Unknown on an offer (flagged, left to the gates on the box): power limit (the offer gives only the maximum), NUMA nodes and
sockets (`lscpu` on the box). Pair rules: different hosts; then the covering plan when a planner binary is given.

RAM floor. The planner's host budget for the heavier node of the covering pick (node b, 47 layers: 32 VRAM + 96 RAM slots
of 128 experts, ADR 0091) is 57.1 GB. The budget under the 64 GB balloon is `memory.max - memory.current + inactive_file - 1 GiB`
with the balloon holding `memory.max - 64 GB` (balloon_guard.py), so it is 64 GB minus what the box itself uses: at most
4.55 GB over the four measured budgets (59.45, 60.47, 59.77, 61.06 GB), taken as 4.6 GB, the 59.4 GB the committed pick was
made on. The rule log's "host margin 1.7 GB" does not reconcile with this (59.4 - 57.1 = 2.3 GB); no source for it was found.
Below 64 GB the floor box has no margin over the estimate: the real planner test settles it. A box whose real limit
is under 64 GB runs without the balloon on its own headroom, real limit minus the same use. The floor is the need plus the
worst use: 57.1 + 4.6 = 61.7 GB, the same for both nodes since the pick may put either role on either box. From 64 GB up the
balloon fixes the budget at 59.4 GB (2.3 GB over the need), so more RAM buys nothing. Unit: the offer's `cpu_ram` is MB; it
is read as 1e6 bytes (not MiB), the lower reading, so a 61.7 floor never accepts a box a MiB reading would let through.
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
CLASSES_A = ("A40", "RTX A6000", "L40", "L40S")  # "L40" matches no offer on Vast (2026-11-26); L40S is the same Ada die
LAYOUT = "bench/results/segment-confirm-20261125b/layout.json"
GRAPH_TEMPLATE = "crates/costmodel/fixtures/cluster/cluster_two_3090_wan.toml"

# Hosts that must not be rented for Case S. Every id appears in docs/box-driver.md (a test checks), with its reason.
SKIP_HOSTS = {
    "366851": "taxameter blocklist", "483833": "taxameter blocklist", "269862": "slow downloads (13d, 13e)",
    "36536": "GPU power capped (phase6-exit-s-20261126b)", "44173": "SSH rejected the key; two sockets",
    "568479": "two sockets", "546840": "two sockets; never finished the pull", "281790": "two sockets",
    "682584": "A40, two sockets and two NUMA nodes (phase6-exit-s-20261126c)",
    "392144": "GPU power capped, 360 of 370 W (phase6-exit-s-20261127)",
    "195919": "GPU power capped, 300 of 420 W (phase6-exit-s-20261127f)",
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
# Hosts that passed every on-box gate of a Case S pair (power, idle, co-tenant, one NUMA node, cuda): a ranking preference only
# (advisor, delegated by the owner 2026-11-27), no rule changes; the gates still run on the box.
GATED_HOSTS = {
    "79466": "passed the gates on 20261126 pair 1 and 20261127",
    "45726": "passed the gates on 20261126c",
}
RULES = ("gpu_class", "skip_host", "unknown", "cores", "ram", "disk", "inet", "price")


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
    price = all_in(o, a.download_gb, a.session_hours)
    ram, disk, inet = (num(o.get(k)) for k in ("ram_gb", "disk_gb", "inet_down_mbps"))
    unknown = None in (ram, disk, inet, price)
    bad = {
        "gpu_class": int(num(o.get("num_gpus")) or 1) != 1 or o.get("gpu_name") not in CLASSES_3090 + CLASSES_A,
        "skip_host": str(o.get("host_id")) in a.skip_hosts,
        "unknown": unknown,
        "cores": (num(o.get("cpu_cores")) or 0) < a.min_cores,
        "ram": ram is not None and ram < a.ram_floor_gb,
        "disk": disk is not None and disk < a.min_disk,
        "inet": inet is not None and inet < a.min_inet,
        "price": price is not None and price > a.max_price,
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
    counts["per_class"] = dict.fromkeys(CLASSES_3090 + CLASSES_A, 0)
    for o in offers:
        counts["per_class"][o.get("gpu_name")] = counts["per_class"].get(o.get("gpu_name"), 0) + 1
    return keep, counts


def bulk_mbps(x, y):
    """Expected bulk rate between two boxes: the slower of the two directions, each limited by the sender's upload and the
    receiver's download (Mbit/s)."""
    ups = [num(x.get("inet_up_mbps")), num(y.get("inet_up_mbps"))]
    downs = [num(x.get("inet_down_mbps")), num(y.get("inet_down_mbps"))]
    known = [v for v in ups + downs if v is not None]
    return min(known) if known else 0.0


def pair_key(p):
    """Rank: plannable pairs first, then the number of hosts that passed the on-box gates before (GATED_HOSTS), then bulk rate (500 Mbit/s buckets, capped at 4000: beyond it the path, not the access link, limits), then datacenter
    hosts (hosting_type 1), then price. RTT is not ranked: the covering pick is the same on every link of the committed fixtures."""
    bucket = int(min(p["bulk_mbps"], 4000) // 500)
    return (not p["plannable"], -p.get("gated", 0), -bucket, -p["datacenter"], p["price_per_h"])


def make_pairs(survivors, dl=DOWNLOAD_GB, hrs=SESSION_HOURS, plan_aclass=True):
    pairs = []
    for x, y in itertools.combinations(survivors, 2):
        if str(x["host_id"]) == str(y["host_id"]):
            continue
        dc = int(num(x.get("hosting_type")) == 1) + int(num(y.get("hosting_type")) == 1)
        pairs.append({
            "a": x["id"], "b": y["id"], "hosts": [str(x["host_id"]), str(y["host_id"])],
            "gpus": [x["gpu_name"], y["gpu_name"]], "locations": [x.get("location"), y.get("location")],
            "bulk_mbps": bulk_mbps(x, y), "datacenter": dc,
            "gated": sum(str(b["host_id"]) in GATED_HOSTS for b in (x, y)),
            "price_per_h": round(all_in(x, dl, hrs) + all_in(y, dl, hrs), 3),
            "plannable": plan_aclass or not any(b["gpu_name"] in CLASSES_A for b in (x, y)),
            "boxes": [x, y],
        })  # fmt: skip
    pairs.sort(key=pair_key)
    return pairs


def flags(o):
    """What the offer does not say and the gates on the box must (power limit, NUMA nodes, sockets)."""
    f = ["power limit unknown (gate on box)", "NUMA/sockets unknown (lscpu gate on box)"]
    if (num(o.get("ram_gb")) or 0) < BALLOON_GB:
        f.append("RAM under 64 GB: no balloon, measured headroom")
    if o.get("gpu_name") in CLASSES_A:
        f.append("A-class: the finder prices it with the 3090 prior; on the boxes cluster profile and merge cap it at 24 GiB and use its class prior")
    return f


def ram_budget_gb(ram_gb):
    """The node's planner RAM budget: the balloon fixes it at 64 GB minus the box's own use; below 64 GB, the real limit does."""
    return round(min(ram_gb, BALLOON_GB) - BOX_USE_GB, 1)


def pair_graph(x, y, template=None):
    """Two-node graph from the committed WAN fixture (3090 prior, the measured WAN link) with each node's RAM budget and
    thread count from its offer. A-class nodes use the 3090 prior here; on the boxes, cluster profile and merge apply the class prior of crates/planner/priors.toml."""
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
        check=False, capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    if r.returncode != 0 or not out.exists():
        return {"covering": False, "n": 0, "best": None, "error": (r.stderr or r.stdout).strip().splitlines()[-1:]}
    cands = json.loads(out.read_text(encoding="utf-8")).get("candidates", [])
    best = cands[0] if cands else None
    return {"covering": bool(cands), "n": len(cands), "best": best and {"id": best["id"], "notes": best.get("notes")},
            "error": None}  # fmt: skip


def row_from_provider(off):
    """One normalized row from taxameter's Offer (its provider row in `.raw`). `cpu_ram` is MB, read as 1e6 bytes (the lower
    reading, see the module docstring); fields the provider omits stay None and the `unknown` rule catches them."""
    raw = off.raw
    ram = num(raw.get("cpu_ram"))
    return {
        "id": off.id, "gpu_name": off.gpu_name, "num_gpus": off.num_gpus, "host_id": off.host_id, "machine_id": off.machine_id,
        "location": off.location, "cpu_cores": off.cpu_cores, "price_per_h": off.price_per_h,
        "inet_down_cost": off.inet_down_cost, "ram_gb": None if ram is None else ram / 1000,
        "disk_gb": num(raw.get("disk_space")), "inet_down_mbps": num(raw.get("inet_down")),
        "inet_up_mbps": num(raw.get("inet_up")), "hosting_type": raw.get("hosting_type"), "cpu_name": raw.get("cpu_name"),
        "gpu_max_power_w": num(raw.get("gpu_max_power")),
    }  # fmt: skip


def fetch_class(provider, cls, a):
    """`--reads` searches of one GPU class through taxameter's provider (read only), merged by machine, GPU and price: offer
    ids rotate and offers appear and vanish between searches."""
    from taxameter.providers.base import OfferQuery

    q = OfferQuery(gpus=[cls], min_vram_gb=24, min_disk_gb=a.min_disk, min_reliability=a.min_reliability, min_inet_mbps=1,
                   min_cuda=12.8, min_cpu_cores=1, num_gpus=1)  # fmt: skip
    rows = {}
    for _ in range(a.reads):
        for off in provider.offers(q):
            r = row_from_provider(off)
            rows.setdefault((r["machine_id"] or r["host_id"], r["gpu_name"], round(r["price_per_h"], 4)), r)
    return list(rows.values())


def load_offers(files):
    rows = []
    for f in files:
        doc = json.loads(pathlib.Path(f).read_text(encoding="utf-8"))
        rows += doc["offers"] if isinstance(doc, dict) else doc
    return rows


def find(a, plan_cmd=None, provider=None):
    """The whole finder: (result dict). `plan_cmd` is the planner command prefix (a list) or None."""
    if a.offers_json:
        offers = load_offers(a.offers_json)
    else:
        if provider is None:
            from taxameter.providers.vast import VastProvider

            provider = VastProvider()
        offers = [o for c in CLASSES_3090 + CLASSES_A for o in fetch_class(provider, c, a)]
        if a.save_offers:
            pathlib.Path(a.save_offers).write_text(json.dumps({"offers": offers}, indent=1), encoding="utf-8")
    survivors, counts = screen(offers, a)
    pairs = make_pairs(survivors, a.download_gb, a.session_hours, a.plan_aclass)
    counts["pairs_same_host_dropped"] = len(survivors) * (len(survivors) - 1) // 2 - len(pairs)
    counts["pairs_no_covering_plan"] = 0
    listed = []
    for p in pairs:
        x, y = p["boxes"]
        p["flags"] = sorted(set(flags(x)) | set(flags(y)))
        if not p["plannable"]:
            p["flags"].append("not plannable until cluster-profile-gpu-cap lands (A-class card): do not rent")
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
                      "session_hours": a.session_hours, "plan_aclass": a.plan_aclass, "reads": a.reads},
            "counts": counts, "pairs": listed}  # fmt: skip


def table(res, top):
    c = res["counts"]
    out = [f"offers per class: {c['per_class']} (a class at 0 may be a wrong name)"]
    out.append(f"{c['offers']} offers, {c['survivors']} pass the static rules; excluded first by rule / failing the rule on its own:")
    out += [f"  {r:<10} {c['excluded'][r]:>4} / {c['fails'][r]:>4}" for r in RULES]
    out.append(f"  pairs dropped for the same host: {c['pairs_same_host_dropped']}; for no covering plan: {c['pairs_no_covering_plan']}")
    if not any(p["plannable"] for p in res["pairs"]):
        out.append(f"NO PLANNABLE PAIR ({len(res['pairs'])} listed with an A-class card): do not create any box (never hold one "
                   "for a partner); retry later, never loosen a rule.")
    if res["pairs"]:
        out.append(f"{len(res['pairs'])} pair(s); top {min(top, len(res['pairs']))} by plannable, gated hosts, bulk rate, datacenter, price:")
        out.append("  #  offers (a, b)        hosts           gpus                  bulk Mbit/s  dc  USD/h  plannable  plan")
    for i, p in enumerate(res["pairs"][:top], 1):
        plan = "not run" if p["plan"] is None else f"{p['plan']['n']} covering"
        out.append(f"  {i:<2} {p['a']:>9} {p['b']:>9}  {'/'.join(p['hosts']):<15} {'/'.join(p['gpus']):<21} "
                   f"{p['bulk_mbps']:>10.0f} {p['datacenter']:>3}  {p['price_per_h']:.3f}  {p['plannable']!s:<9}  {plan}")  # fmt: skip
    out.append("  unknown on every offer, gated on the box: power limit, NUMA nodes and sockets")
    return "\n".join(out)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--offers-json", action="append", help="recorded offers (offline); repeatable")
    p.add_argument("--save-offers", help="write the live offers, normalized, to this file (a fixture)")
    p.add_argument("--out", help="write the result JSON here")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--reads", type=int, default=3, help="searches per GPU class, merged")
    p.add_argument("--no-plan-aclass", dest="plan_aclass", action="store_false",
                   help="list pairs with an A-class card as not plannable (they are plannable since cluster-profile-gpu-cap)")
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
    return 0 if any(p["plannable"] for p in res["pairs"]) else 1


if __name__ == "__main__":
    sys.exit(main())
