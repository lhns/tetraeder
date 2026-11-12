"""List Vast offers with the host facts a Tetraeder session cares about: CPU model and cores (usable quota where Vast
reports it), host RAM, disk bandwidth, GPU, price, traffic price, host id and reliability.

Run from the main checkout (taxameter.toml and the taxameter install are found there):
    uv run python -I scripts/vast_offers.py --min-ram-gb 128 --min-disk-mbps 1000 --cpu-regex EPYC --max-price 0.45

The API key is read by taxameter's Vast provider from ~/.config/vastai/vast_api_key and sent only as a header; this
script never touches, prints or logs it, and prints no raw offer fields beyond the ones in the table.
"""

from __future__ import annotations

import argparse
import re
import sys

from taxameter import config, core
from taxameter.core import Ctx


def num(v: object) -> float:
    return float(v) if isinstance(v, (int, float)) else 0.0


def row(o) -> dict:
    r = o.raw
    return {
        "id": o.id,
        "host": o.host_id or "",
        "gpu": o.gpu_name,
        "cpu": (r.get("cpu_name") or "?").strip(),
        "cores": o.cpu_cores,
        "cores_total": num(r.get("cpu_cores")),
        "ram_gb": o.ram_gb,
        "disk_mbps": num(r.get("disk_bw")),
        "price": o.price_per_h,
        "down_usd_gb": o.inet_down_cost,
        "rel": o.reliability,
        "loc": o.location,
    }


def keep(x: dict, a: argparse.Namespace, cpu: re.Pattern | None) -> bool:
    if x["ram_gb"] < a.min_ram_gb or x["disk_mbps"] < a.min_disk_mbps:
        return False
    return cpu is None or bool(cpu.search(x["cpu"]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--gpu", help="comma list of GPU names (default: taxameter.toml [offers] gpus)")
    p.add_argument("--min-ram-gb", type=float, default=0)
    p.add_argument("--min-disk-mbps", type=float, default=0, help="disk bandwidth Vast reports for the host")
    p.add_argument("--cpu-regex", help="case-insensitive regex on the CPU model, e.g. 'EPYC (7642|9)'")
    p.add_argument("--max-price", type=float, help="USD per hour")
    p.add_argument("--min-cpu-cores", type=float, default=8, help="usable cores")
    p.add_argument("--min-vram", type=float, default=20)
    p.add_argument("--min-reliability", type=float, default=0.95)
    p.add_argument("--min-inet", type=float, default=200, help="download Mbit/s")
    p.add_argument("--download-gb", type=float, default=0, help="GB fetched, priced into the ranking")
    p.add_argument("--sort", default="score")
    p.add_argument("--limit", type=int, default=40)
    a = p.parse_args()
    cpu = re.compile(a.cpu_regex, re.I) if a.cpu_regex else None
    ctx = Ctx(config.load(None))
    ns = argparse.Namespace(
        gpu=a.gpu, min_cpu_cores=a.min_cpu_cores, download_gb=a.download_gb, min_disk=None, max_price=a.max_price,
        sort=a.sort, min_vram=a.min_vram, min_reliability=a.min_reliability, min_inet=a.min_inet, min_cuda=None,
        interruptible=None, exclude_hosts=None,
    )
    rows = [x for x in map(row, core.search_offers(ctx, ns)) if keep(x, a, cpu)]
    print(f"{'id':>9} {'host':>7} {'gpu':<14} {'cpu':<28} {'cores':>5} {'tot':>4} {'ram':>5} {'disk':>6} {'$/h':>6} {'dl$/GB':>7} {'rel':>5} loc")
    for x in rows[: a.limit]:
        dl = "?" if x["down_usd_gb"] is None else f"{x['down_usd_gb']:.4f}"
        print(
            f"{x['id']:>9} {x['host']:>7} {x['gpu'][:14]:<14} {x['cpu'][:28]:<28} {x['cores']:>5g} {x['cores_total']:>4g} "
            f"{x['ram_gb']:>5.0f} {x['disk_mbps']:>6.0f} {x['price']:>6.3f} {dl:>7} {x['rel']:>5.3f} {x['loc']}"
        )
    print(f"{len(rows)} offers match", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
