#!/usr/bin/env python3
"""Writes one cluster.toml per node for the Phase 6 rehearsal (ADR 0091), run by the driver after `create`.

Every node lists the public address (Vast's public IP and mapped port) of every other node; its own entry is the
loopback address, so a node never dials itself through the provider's NAT (hairpin). Anchors are the nodes'
public certificates, `<id>.pem` next to the file. The files hold no keys; keep them under runs/ (git-ignored).

    python3 scripts/phase6_cluster_toml.py --head a --node a=1.2.3.4:13691 --node b=5.6.7.8:50163 --out-dir runs/phase6
"""

import argparse
import pathlib


def render(cluster, head, nodes, me, listen_port, down_ms, listen_host="0.0.0.0", self_host="127.0.0.1"):
    out = [f'cluster = "{cluster}"', f'head = "{head}"', f"node_down_after_ms = {down_ms}", ""]
    for nid, addr in nodes.items():
        out += ["[[nodes]]", f'id = "{nid}"']
        if nid == me:
            out += [f'addr = "{self_host}:{listen_port}"', f'listen = "{listen_host}:{listen_port}"']
        else:
            out += [f'addr = "{addr}"']
        out += [f'anchor = "{nid}.pem"', ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", default="phase6-rehearsal")
    ap.add_argument("--head", required=True)
    ap.add_argument("--node", action="append", required=True, help="ID=PUBLIC_IP:MAPPED_PORT")
    ap.add_argument("--listen-port", type=int, default=7000)
    ap.add_argument("--down-ms", type=int, default=2000)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--node-listen", action="append", default=[], metavar="ID=PORT",
                    help="the loopback dry run: every node listens on its own port of 127.0.0.1")
    a = ap.parse_args()
    nodes = dict(n.split("=", 1) for n in a.node)
    own = {k: int(v) for k, v in (x.split("=", 1) for x in a.node_listen)}
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for me in nodes:
        text = render(a.cluster, a.head, nodes, me, own.get(me, a.listen_port), a.down_ms, "127.0.0.1" if own else "0.0.0.0")
        (out / f"cluster-{me}.toml").write_text(text)
        print(f"wrote {out / f'cluster-{me}.toml'}")


if __name__ == "__main__":
    main()
