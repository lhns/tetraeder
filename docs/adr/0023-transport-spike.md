# 0023. Transport spike: QUIC (quinn) versus TCP with TLS versus plain TCP

Status: Accepted (owner, 2026-11-23, on the advisor's recommendation, after the WAN, same-host and loopback numbers of `transport-wan-20261123`). Proposed earlier with localhost numbers only.

## Context

ADR 0018 says the cross-node transport is chosen by measurement in Phase 0. Cross-node expert parallel adds one round trip per MoE layer per token (about 60 hops, an activation of about 8 KB each), so per-hop latency and CPU cost matter more than peak bandwidth. 1 MB messages stand for bulk moves (expert weights, KV).

The spike is `crates/transport` (`transport-bench`): one stream, ping-pong echo for round-trip latency (p50, p99 over 2000 hops, 400 for 1 MB), and a one-way stream with a final ack for throughput. Transports: plain TCP (the floor), TCP with TLS 1.3 (rustls, ring), QUIC (quinn 0.11, rustls, ring; stream and connection windows raised to 32/64 MiB). Self-signed certificate generated at startup, the client accepts any certificate (spike only). `server` and `client` modes take an address, so the same binary runs between two Vast boxes.

## Measurements (localhost, Windows VM)

Host: QEMU Virtual CPU 2.5+, 16 vCPU, **no AES-NI and no AVX2**, shared with other jobs (about 85% load during the runs). Loopback, release build, one client, one server, one process. (No session directory yet.)

AES-GCM (rustls default) is software-only here and caps TLS and QUIC at about 40 MiB/s whatever the message size, so those numbers say nothing about the protocols. The table uses ChaCha20-Poly1305 (`--cipher chacha`), which is fast in software; the AES run is listed for reference.

| Transport | Size | p50 | p99 | Throughput |
| --- | --- | --- | --- | --- |
| TCP | 8 KB | 326 us | 780 us | 127 MiB/s |
| TLS (ChaCha) | 8 KB | 457 us | 635 us | 90 MiB/s |
| QUIC (ChaCha) | 8 KB | 874 us | 2191 us | 72 MiB/s |
| TCP | 1 MB | 2.3 ms | 27 ms | 672 MiB/s |
| TLS (ChaCha) | 1 MB | 12.1 ms | 32 ms | 214 MiB/s |
| QUIC (ChaCha) | 1 MB | 24.1 ms | 69 ms | 81 MiB/s |
| TLS (AES-GCM, no AES-NI) | 8 KB | 905 us | 2073 us | 36 MiB/s |
| QUIC (AES-GCM, no AES-NI) | 8 KB | 1160 us | 2078 us | 38 MiB/s |

Reading, with that caveat: on this host TCP is the floor at about 0.3 ms per 8 KB hop (Windows loopback is slow; Linux loopback is typically under 0.1 ms). TLS adds about 0.13 ms per 8 KB hop, QUIC about 0.55 ms, in line with the plan's remark that userspace QUIC costs hundreds of microseconds per hop. At 60 hops per token that is about 8 ms (TLS over TCP) versus 33 ms (QUIC) of overhead on top of the 20 ms TCP floor, before any real network latency. These are indications only: crypto cost depends on the CPU, and the VM is noisy (TCP p99 for 1 MB varies by 10x). (No session directory yet.)

## Decision (owner, 2026-11-23)

Phase 6 builds on TCP with TLS (mTLS) for traffic between nodes, behind a transport trait, so the choice is reversible. QUIC stays a later option for bulk transfers and WAN links. That QUIC's 1 MB lead over WAN comes from untuned TCP buffers is a hypothesis, not measured; tune the TCP buffers before relying on it. RDMA is not available on Vast and is revisited on homelab hardware.

## Earlier lean (before the WAN numbers)

Leaning to TCP with TLS (mTLS in Phase 6) for per-layer expert traffic if the WAN numbers confirm a clear latency gap. QUIC stays a candidate for bulk and lossy links (loss recovery, no head-of-line blocking across streams) and for NAT traversal via iroh. Do not decide before the pending measurements.

Pending, to fill in before this ADR is Accepted:

- WAN numbers between two Vast boxes: measured 2026-11-23, see "WAN, loopback and same-host numbers" below.
- Linux loopback: measured on five boxes. Same datacenter: no second host in the datacenter was on offer, so the "same-DC" row is two containers on one host (RTT 0.23 ms minimum); a switch-level same-DC pair is still open.
- RDMA: not available on Vast; revisit on homelab hardware (Phase 6) or drop from the options.
- iroh (QUIC with relay and hole punching) is not measured here; add it if NAT traversal is a requirement.

## WAN, loopback and same-host numbers (2026-11-23)

Session `bench-transport-wan`, tree node `cluster-transport-wan`. Full tables, boxes, parameters and limits: [`bench/results/transport-wan-20261123/README.md`](../../bench/results/transport-wan-20261123/README.md). Two Vast boxes, Connecticut (US) and Germany, RTT about 90 ms (minimum plain-TCP 8 KB round trip 88.8 ms; ICMP is blocked in the containers). All boxes have AES-NI. Medians over repetitions of each call's p50 and p99, in ms; WAN: 3 repetitions of 300 round trips (60 for 1 MB), so its p99 is a tail estimate from few samples. Same host and Linux loopback: 5 repetitions of 2000 round trips (400 for 1 MB). The p50 range over repetitions and p90 are in the README.

| Path | Size | Cipher | TCP p50 / p99 | TCP+TLS p50 / p99 | QUIC p50 / p99 |
| --- | --- | --- | --- | --- | --- |
| WAN, 90 ms | 8 KB | AES | 93.8 / 94.3 | 91.0 / 92.3 | 93.8 / 94.3 |
| WAN, 90 ms | 8 KB | ChaCha | 89.5 / 89.9 | 99.4 / 99.9 | 93.6 / 93.9 |
| WAN, 90 ms | 1 MB | AES | 286 / 379 | 300 / 304 | 196 / 214 |
| WAN, 90 ms | 1 MB | ChaCha | 274 / 365 | 274 / 305 | 198 / 207 |
| Same host, 0.3 ms | 8 KB | AES | 0.52 / 0.62 | 0.41 / 0.66 | 0.58 / 0.78 |
| Same host, 0.3 ms | 8 KB | ChaCha | 0.53 / 0.63 | 0.46 / 0.62 | 0.62 / 0.81 |
| Same host, 0.3 ms | 1 MB | AES | 6.3 / 10.3 | 5.1 / 10.5 | 22.3 / 43.4 |
| Same host, 0.3 ms | 1 MB | ChaCha | 4.2 / 14.4 | 5.1 / 11.7 | 23.6 / 46.0 |
| Linux loopback (a-348060) | 8 KB | AES | 0.07 / 0.09 | 0.12 / 0.14 | 0.22 / 0.30 |
| Linux loopback (a-348060) | 8 KB | ChaCha | 0.07 / 0.11 | 0.14 / 0.16 | 0.23 / 0.29 |
| Linux loopback (a-348060) | 1 MB | AES | 0.71 / 0.95 | 1.56 / 1.96 | 8.5 / 10.6 |
| Linux loopback (a-348060) | 1 MB | ChaCha | 0.74 / 0.96 | 1.94 / 2.10 | 9.8 / 12.1 |

Throughput (MiB/s, median): WAN 1 MB stream TCP 35 to 36, TCP+TLS 35 to 39, QUIC 48 to 54; loopback TCP about 3600, TCP+TLS 1000 to 1300, QUIC 200 to 220.

What the numbers say. At 8 KB the WAN round trip is the network: TCP+TLS and QUIC sit within 6 ms of each other and of plain TCP (about 90 to 99 ms) with the sign flipping between AES and ChaCha, so the gap is below the path noise (the plain-TCP rows alone move 4 ms between calls) and no TLS-versus-QUIC latency gap can be claimed on WAN. At 1 MB QUIC is faster on the WAN, by 78 to 104 ms at p50 (196 to 198 ms against 274 to 300 ms), because quinn's 32 MiB windows keep the 90 ms path full (48 to 54 MiB/s) while TCP ran on stock buffers (35 to 39 MiB/s); that is an untuned-TCP effect, not a protocol property, and a tuned TCP would need its own run. On the same host, where the network is not the limit, the cost of the protocol shows: TCP+TLS is 0.41 to 0.46 ms and QUIC 0.58 to 0.62 ms at 8 KB (QUIC about 0.15 to 0.2 ms slower per hop, matching Linux loopback: 0.12 to 0.14 against 0.22 to 0.23 ms), and at 1 MB TCP+TLS takes 5.1 ms against 22 to 24 ms for QUIC. The ChaCha-versus-AES choice made no consistent difference on these AES-NI hosts. At 60 hops per token, 0.15 to 0.2 ms per hop is 9 to 12 ms per token in favor of TCP+TLS on a fast link, while a 90 ms WAN hop would cost about 5.4 s per token whatever the protocol, so per-layer expert traffic over a WAN is out of reach for any transport tested; the decision matters for LAN-class links. No same-datacenter pair on different hosts was available; the same-host pair goes through the host's NAT.

## Consequences

- No transport trait is designed yet; `tetraeder-transport` only holds the benchmark. Phase 6 defines the trait on top of the winner.
- The numbers above are not a basis for the Phase 6 latency budget until the WAN run exists.
- Hosts without AES-NI (the dev VM) must use `--cipher chacha` for transport measurements.
