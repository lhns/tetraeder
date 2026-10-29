# 0023. Transport spike: QUIC (quinn) versus TCP with TLS versus plain TCP

Status: Proposed (localhost numbers only; WAN and RDMA pending, no transport is chosen yet)

## Context

ADR 0018 says the cross-node transport is chosen by measurement in Phase 0. Cross-node expert parallel adds one round trip per MoE layer per token (about 60 hops, an activation of about 8 KB each), so per-hop latency and CPU cost matter more than peak bandwidth. 1 MB messages stand for bulk moves (expert weights, KV).

The spike is `crates/transport` (`transport-bench`): one stream, ping-pong echo for round-trip latency (p50, p99 over 2000 hops, 400 for 1 MB), and a one-way stream with a final ack for throughput. Transports: plain TCP (the floor), TCP with TLS 1.3 (rustls, ring), QUIC (quinn 0.11, rustls, ring; stream and connection windows raised to 32/64 MiB). Self-signed certificate generated at startup, the client accepts any certificate (spike only). `server` and `client` modes take an address, so the same binary runs between two Vast boxes.

## Measurements (localhost, Windows VM)

Host: QEMU Virtual CPU 2.5+, 16 vCPU, **no AES-NI and no AVX2**, shared with other jobs (about 85% load during the runs). Loopback, release build, one client, one server, one process.

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

Reading, with that caveat: on this host TCP is the floor at about 0.3 ms per 8 KB hop (Windows loopback is slow; Linux loopback is typically under 0.1 ms). TLS adds about 0.13 ms per 8 KB hop, QUIC about 0.55 ms, in line with the plan's remark that userspace QUIC costs hundreds of microseconds per hop. At 60 hops per token that is about 8 ms (TLS over TCP) versus 33 ms (QUIC) of overhead on top of the 20 ms TCP floor, before any real network latency. These are indications only: crypto cost depends on the CPU, and the VM is noisy (TCP p99 for 1 MB varies by 10x).

## Decision (proposed, not final)

Leaning to TCP with TLS (mTLS in Phase 6) for per-layer expert traffic if the WAN numbers confirm a clear latency gap. QUIC stays a candidate for bulk and lossy links (loss recovery, no head-of-line blocking across streams) and for NAT traversal via iroh. Do not decide before the pending measurements.

Pending, to fill in before this ADR is Accepted:

- WAN numbers between two Vast boxes (`transport-bench server` on one, `client --addr` on the other; needs `max_instances 2` for about an hour; run AES and ChaCha, the boxes have AES-NI).
- Linux loopback and same-datacenter numbers, to separate protocol cost from Windows loopback cost.
- RDMA: not available on Vast; revisit on homelab hardware (Phase 6) or drop from the options.
- iroh (QUIC with relay and hole punching) is not measured here; add it if NAT traversal is a requirement.

## Consequences

- No transport trait is designed yet; `tetraeder-transport` only holds the benchmark. Phase 6 defines the trait on top of the winner.
- The numbers above are not a basis for the Phase 6 latency budget until the WAN run exists.
- Hosts without AES-NI (the dev VM) must use `--cipher chacha` for transport measurements.
