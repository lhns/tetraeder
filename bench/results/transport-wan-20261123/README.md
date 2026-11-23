Status: clean. WAN set is thin (3 repetitions, 300 round trips per size, so p99 is the third slowest sample); loopback and same-host sets follow ADR 0023's size (2000 round trips, 5 repetitions). Judged under ADR 0023's method (ping-pong round trips and one-way throughput, repetitions interleaved by cipher and protocol); no verdict rule, no pass or fail, no baseline engine involved.

Session `bench-transport-wan`, 2026-11-23, tree node `cluster-transport-wan`, ADR 0023. Source commit of the benchmark: `4299776e` (p90 added to `transport-bench` rows; later commits on the branch only change `scripts/`). Tool: `transport-bench`, release build, one client stream; plain TCP, TCP with TLS 1.3 (rustls, ring) and QUIC (quinn 0.11), `--cipher aes|chacha`.

## What was run

| Set | Client box to server box | Parameters |
| --- | --- | --- |
| WAN | a-348060 (Connecticut, US) to b-62819 (Germany) over the public IPs | 3 reps, 300 round trips per size (60 for 1 MB), 50 warmup, 128 MiB per throughput run |
| Loopback | each box to itself (127.0.0.1): a-348060, b-62819, x-348060, y-348060, b-115630 | 5 reps, 2000 round trips (400 for 1 MB), 200 warmup, 256 MiB |
| Same host | x-348060 to y-348060 (two containers on host 348060, one public IP) | 5 reps, same parameters as loopback |

Each call is one (cipher, transport) pair; a repetition runs aes tcp, tls, quic, then chacha tcp, tls, quic, so protocols interleave. Plain TCP does not use the cipher, so its two rows are an extra noise estimate. Every call opens a new connection and runs the two sizes on it one after the other. The tables show the median over repetitions of each call's p50, p90, p99 and throughput, and the range of the p50. Raw rows (with min, mean, max) are the JSON files next to each table's scenario.

## Boxes

The cheapest boxes that met `--min-inet 1000` and at least 8 cores (taxameter offers no CPU-only boxes); the GPUs idle. All have AES-NI (`aes` in /proc/cpuinfo) and kernel 6.8; `eth0` reports 10000 Mbit/s (the container's view; the real uplink is not visible). Ports were opened through the Vast create env, see Deviations.

| Name | Box id | Host | Location (Vast) | CPU, quota | GPU |
| --- | --- | --- | --- | --- | --- |
| a-348060 | 62119317 | 348060 | Connecticut, US | EPYC 7662, 15.4 cores | RTX 3060 |
| b-62819 | 62125139 | 62819 | Germany, DE | EPYC 7663, 13.4 cores | RTX 5060 Ti |
| x-348060 | 62128497 | 348060 | Connecticut, US | EPYC 7662, 15.4 cores | RTX 5060 Ti |
| y-348060 | 62128687 | 348060 | Connecticut, US | EPYC 7662, 15.4 cores | RTX 3060 |
| b-115630 | 62118917 | 115630 | listed Poland, PL (public IP 1.193.139.67, not European) | EPYC 7452, 7.7 cores | RTX 3060 |

`meta.txt` in each box directory has kernel, driver, quota and the mapped ports. Boxes ran the stock `nvidia/cuda:12.8.1-devel-ubuntu24.04` image (taxameter's `[image] fallback`), because the GHCR image stayed in `loading` for more than 10 minutes on three boxes.

## RTT

ICMP is not permitted in the containers (`ping: Operation not permitted`), so RTT is the benchmark's own minimum plain-TCP 8 KB round trip:

- WAN Connecticut to Germany: minimum 88.8 ms, median of the per-call minimum 93.5 ms.
- Same host (x to y through the public IP, NAT hairpin): minimum 0.23 ms, median 0.35 ms.
- Loopback: 0.03 to 0.08 ms p50 for plain TCP at 8 KB.

Vast's geolocation is not reliable: the first "Poland" box (host 115630) had a public IP outside Europe, a round trip of about 260 ms and a path that stalled for messages of 6000 bytes and more (sizes up to 5000 passed with retransmission spikes). That path was dropped and replaced by the Germany box; only the loopback run of that box is kept (`b-115630/loop`).

## WAN (a-348060 to b-62819, RTT about 90 ms)

### a-348060 / wan

| Size | Cipher | Transport | reps | p50 ms | p50 range ms | p90 ms | p99 ms | MiB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 KB | aes | TCP | 3 | 93.8 | 93.8 to 93.9 | 94.0 | 94.3 | 31 |
| 8 KB | aes | TCP+TLS | 3 | 91.0 | 89.3 to 99.0 | 91.6 | 92.3 | 31 |
| 8 KB | aes | QUIC | 3 | 93.8 | 91.2 to 95.2 | 94.0 | 94.3 | 50 |
| 8 KB | chacha | TCP | 3 | 89.5 | 89.1 to 95.0 | 89.6 | 89.9 | 32 |
| 8 KB | chacha | TCP+TLS | 3 | 99.4 | 89.6 to 103.3 | 99.6 | 99.9 | 29 |
| 8 KB | chacha | QUIC | 3 | 93.6 | 89.4 to 95.1 | 93.8 | 93.9 | 51 |
| 1 MB | aes | TCP | 3 | 286.1 | 284.8 to 286.7 | 288.6 | 378.7 | 36 |
| 1 MB | aes | TCP+TLS | 3 | 300.3 | 185.7 to 457.2 | 301.5 | 303.6 | 39 |
| 1 MB | aes | QUIC | 3 | 195.9 | 194.3 to 208.7 | 201.0 | 214.3 | 54 |
| 1 MB | chacha | TCP | 3 | 274.0 | 271.6 to 288.7 | 360.0 | 364.7 | 35 |
| 1 MB | chacha | TCP+TLS | 3 | 273.9 | 214.4 to 302.0 | 303.8 | 305.1 | 35 |
| 1 MB | chacha | QUIC | 3 | 198.3 | 189.7 to 206.6 | 202.8 | 207.0 | 48 |


## Same host (x-348060 to y-348060, RTT about 0.3 ms)

### x-348060 / samehost

| Size | Cipher | Transport | reps | p50 ms | p50 range ms | p90 ms | p99 ms | MiB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 KB | aes | TCP | 5 | 0.52 | 0.33 to 0.58 | 0.57 | 0.62 | 238 |
| 8 KB | aes | TCP+TLS | 5 | 0.41 | 0.38 to 0.55 | 0.60 | 0.66 | 111 |
| 8 KB | aes | QUIC | 5 | 0.58 | 0.52 to 0.68 | 0.63 | 0.78 | 80 |
| 8 KB | chacha | TCP | 5 | 0.53 | 0.39 to 0.56 | 0.57 | 0.63 | 131 |
| 8 KB | chacha | TCP+TLS | 5 | 0.46 | 0.38 to 0.50 | 0.55 | 0.62 | 112 |
| 8 KB | chacha | QUIC | 5 | 0.62 | 0.51 to 0.78 | 0.71 | 0.81 | 81 |
| 1 MB | aes | TCP | 5 | 6.33 | 4.08 to 11.2 | 7.96 | 10.3 | 308 |
| 1 MB | aes | TCP+TLS | 5 | 5.07 | 4.60 to 7.58 | 8.42 | 10.5 | 460 |
| 1 MB | aes | QUIC | 5 | 22.3 | 21.8 to 23.8 | 37.7 | 43.4 | 78 |
| 1 MB | chacha | TCP | 5 | 4.18 | 3.98 to 11.3 | 7.63 | 14.4 | 112 |
| 1 MB | chacha | TCP+TLS | 5 | 5.06 | 4.67 to 7.52 | 8.10 | 11.7 | 437 |
| 1 MB | chacha | QUIC | 5 | 23.6 | 22.8 to 24.8 | 37.4 | 46.0 | 81 |


## Loopback (server and client on one box)

### a-348060 / loop

| Size | Cipher | Transport | reps | p50 ms | p50 range ms | p90 ms | p99 ms | MiB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 KB | aes | TCP | 5 | 0.07 | 0.06 to 0.08 | 0.08 | 0.09 | 670 |
| 8 KB | aes | TCP+TLS | 5 | 0.12 | 0.08 to 0.12 | 0.13 | 0.14 | 497 |
| 8 KB | aes | QUIC | 5 | 0.22 | 0.19 to 0.25 | 0.26 | 0.30 | 211 |
| 8 KB | chacha | TCP | 5 | 0.07 | 0.06 to 0.09 | 0.09 | 0.11 | 616 |
| 8 KB | chacha | TCP+TLS | 5 | 0.14 | 0.10 to 0.14 | 0.15 | 0.16 | 436 |
| 8 KB | chacha | QUIC | 5 | 0.23 | 0.21 to 0.24 | 0.25 | 0.29 | 183 |
| 1 MB | aes | TCP | 5 | 0.71 | 0.69 to 0.93 | 0.79 | 0.95 | 3554 |
| 1 MB | aes | TCP+TLS | 5 | 1.56 | 1.55 to 1.76 | 1.61 | 1.96 | 1301 |
| 1 MB | aes | QUIC | 5 | 8.52 | 8.42 to 8.60 | 9.61 | 10.6 | 221 |
| 1 MB | chacha | TCP | 5 | 0.74 | 0.70 to 0.86 | 0.79 | 0.96 | 3752 |
| 1 MB | chacha | TCP+TLS | 5 | 1.94 | 1.87 to 1.98 | 2.00 | 2.10 | 1066 |
| 1 MB | chacha | QUIC | 5 | 9.84 | 9.75 to 10.1 | 11.1 | 12.1 | 201 |


### b-62819 / loop

| Size | Cipher | Transport | reps | p50 ms | p50 range ms | p90 ms | p99 ms | MiB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 KB | aes | TCP | 5 | 0.04 | 0.03 to 0.04 | 0.04 | 0.05 | 805 |
| 8 KB | aes | TCP+TLS | 5 | 0.06 | 0.05 to 0.06 | 0.06 | 0.07 | 692 |
| 8 KB | aes | QUIC | 5 | 0.10 | 0.09 to 0.11 | 0.12 | 0.14 | 530 |
| 8 KB | chacha | TCP | 5 | 0.04 | 0.03 to 0.04 | 0.04 | 0.05 | 1108 |
| 8 KB | chacha | TCP+TLS | 5 | 0.07 | 0.06 to 0.07 | 0.07 | 0.08 | 577 |
| 8 KB | chacha | QUIC | 5 | 0.13 | 0.13 to 0.13 | 0.14 | 0.19 | 437 |
| 1 MB | aes | TCP | 5 | 0.56 | 0.56 to 0.59 | 0.58 | 0.72 | 5136 |
| 1 MB | aes | TCP+TLS | 5 | 0.95 | 0.93 to 1.23 | 1.01 | 1.31 | 1664 |
| 1 MB | aes | QUIC | 5 | 3.29 | 3.19 to 3.59 | 5.32 | 6.79 | 510 |
| 1 MB | chacha | TCP | 5 | 0.57 | 0.56 to 0.91 | 0.61 | 0.79 | 4982 |
| 1 MB | chacha | TCP+TLS | 5 | 1.57 | 1.55 to 1.83 | 1.90 | 2.37 | 1091 |
| 1 MB | chacha | QUIC | 5 | 4.25 | 4.17 to 4.41 | 5.22 | 8.23 | 421 |


### x-348060 / loop

| Size | Cipher | Transport | reps | p50 ms | p50 range ms | p90 ms | p99 ms | MiB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 KB | aes | TCP | 5 | 0.06 | 0.03 to 0.08 | 0.06 | 0.07 | 947 |
| 8 KB | aes | TCP+TLS | 5 | 0.08 | 0.05 to 0.11 | 0.09 | 0.10 | 475 |
| 8 KB | aes | QUIC | 5 | 0.21 | 0.12 to 0.25 | 0.22 | 0.24 | 313 |
| 8 KB | chacha | TCP | 5 | 0.06 | 0.04 to 0.08 | 0.06 | 0.07 | 623 |
| 8 KB | chacha | TCP+TLS | 5 | 0.06 | 0.06 to 0.09 | 0.07 | 0.10 | 568 |
| 8 KB | chacha | QUIC | 5 | 0.23 | 0.13 to 0.29 | 0.27 | 0.30 | 207 |
| 1 MB | aes | TCP | 5 | 0.50 | 0.48 to 0.55 | 0.57 | 0.93 | 3326 |
| 1 MB | aes | TCP+TLS | 5 | 1.62 | 1.58 to 1.70 | 1.71 | 1.89 | 1307 |
| 1 MB | aes | QUIC | 5 | 7.56 | 6.63 to 7.73 | 9.26 | 10.9 | 276 |
| 1 MB | chacha | TCP | 5 | 0.51 | 0.45 to 0.53 | 0.54 | 0.80 | 3270 |
| 1 MB | chacha | TCP+TLS | 5 | 2.06 | 1.99 to 2.14 | 2.09 | 2.38 | 1023 |
| 1 MB | chacha | QUIC | 5 | 9.17 | 8.98 to 9.67 | 10.9 | 12.0 | 225 |


### y-348060 / loop

| Size | Cipher | Transport | reps | p50 ms | p50 range ms | p90 ms | p99 ms | MiB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 KB | aes | TCP | 5 | 0.08 | 0.07 to 0.08 | 0.08 | 0.09 | 619 |
| 8 KB | aes | TCP+TLS | 5 | 0.12 | 0.08 to 0.13 | 0.13 | 0.14 | 504 |
| 8 KB | aes | QUIC | 5 | 0.24 | 0.21 to 0.25 | 0.26 | 0.27 | 209 |
| 8 KB | chacha | TCP | 5 | 0.08 | 0.06 to 0.08 | 0.08 | 0.09 | 623 |
| 8 KB | chacha | TCP+TLS | 5 | 0.13 | 0.06 to 0.14 | 0.14 | 0.15 | 490 |
| 8 KB | chacha | QUIC | 5 | 0.24 | 0.21 to 0.28 | 0.25 | 0.27 | 180 |
| 1 MB | aes | TCP | 5 | 0.75 | 0.73 to 0.96 | 0.82 | 1.03 | 3682 |
| 1 MB | aes | TCP+TLS | 5 | 1.55 | 1.54 to 1.87 | 1.57 | 1.67 | 1240 |
| 1 MB | aes | QUIC | 5 | 8.48 | 8.35 to 8.64 | 9.64 | 10.8 | 225 |
| 1 MB | chacha | TCP | 5 | 0.72 | 0.52 to 1.02 | 0.79 | 1.08 | 3456 |
| 1 MB | chacha | TCP+TLS | 5 | 1.95 | 1.93 to 1.99 | 1.97 | 2.08 | 1055 |
| 1 MB | chacha | QUIC | 5 | 9.99 | 9.77 to 10.1 | 11.2 | 12.8 | 199 |


### b-115630 / loop

| Size | Cipher | Transport | reps | p50 ms | p50 range ms | p90 ms | p99 ms | MiB/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 KB | aes | TCP | 5 | 0.06 | 0.04 to 0.08 | 0.08 | 0.09 | 680 |
| 8 KB | aes | TCP+TLS | 5 | 0.12 | 0.06 to 0.13 | 0.13 | 0.14 | 552 |
| 8 KB | aes | QUIC | 5 | 0.24 | 0.22 to 0.25 | 0.28 | 0.31 | 199 |
| 8 KB | chacha | TCP | 5 | 0.06 | 0.05 to 0.07 | 0.06 | 0.08 | 1080 |
| 8 KB | chacha | TCP+TLS | 5 | 0.10 | 0.08 to 0.14 | 0.11 | 0.13 | 619 |
| 8 KB | chacha | QUIC | 5 | 0.26 | 0.23 to 0.31 | 0.29 | 0.32 | 173 |
| 1 MB | aes | TCP | 5 | 0.67 | 0.60 to 0.73 | 0.68 | 0.71 | 3626 |
| 1 MB | aes | TCP+TLS | 5 | 1.56 | 1.52 to 1.95 | 1.59 | 1.95 | 1290 |
| 1 MB | aes | QUIC | 5 | 8.90 | 8.60 to 9.01 | 10.1 | 10.9 | 225 |
| 1 MB | chacha | TCP | 5 | 0.71 | 0.59 to 0.77 | 0.74 | 0.78 | 3603 |
| 1 MB | chacha | TCP+TLS | 5 | 1.95 | 1.93 to 2.39 | 2.01 | 2.67 | 1017 |
| 1 MB | chacha | QUIC | 5 | 10.2 | 9.91 to 10.4 | 11.5 | 12.8 | 196 |



## Reading

See ADR 0023, section "WAN, loopback and same-host numbers (2026-11-23)". In short: at 8 KB on a 90 ms path the round trip is the network, and the three protocols differ by less than the spread between repetitions (the WAN plain-TCP rows alone differ by up to 5 ms between the aes and chacha calls). At 1 MB the stock TCP buffers cap the stream (35 to 39 MiB/s) while quinn with its 32 MiB windows reaches 48 to 54 MiB/s, so QUIC finishes a 1 MB round trip about 80 to 100 ms earlier on this path. That is a buffer setting, not a protocol property: no `SO_SNDBUF` or sysctl tuning was applied to TCP. In loopback and on the same host the protocol cost shows: TLS adds about 0.05 ms and QUIC about 0.15 ms at 8 KB over plain TCP, and at 1 MB QUIC is 4 to 5 times slower than TCP+TLS on the same host.

## Deviations and limits

- taxameter's Vast `create` sends `env: {}`, so a box exposes only SSH. A local wrapper (not committed) added six `-p` port mappings (7000, 7001, 7100, 7101 tcp; 7002, 7102 udp) to the create payload and then ran taxameter's own CLI; caps, offer filters and state handling are taxameter's. The mapped ports are in PID 1's environment, not in ssh sessions; `scripts/transport_wan_bench.sh` reads them there. On a-348060 and b-115630 `ports.env` was written by hand or fixed afterwards (an early script version read the wrong environment).
- The first two GHCR-image boxes never came up (loading more than 10 minutes, or SSH down for 15 minutes); later boxes use the fallback CUDA image. Destroyed unused: 62113987, 62115863 (host 79466), 62117879 (host 623333), 62117215 (host 115630). 62119317 and 62118917 were the second boxes on hosts 348060 and 115630.
- No same-datacenter pair on different hosts was on offer (no other Connecticut or Germany offer within the caps). The "same host" set is two containers on one physical machine (348060, the host of the WAN client box), so it passes the host's NAT, not a switch. The loopback sets separate protocol cost from that.
- WAN uses 300 round trips per size (60 for 1 MB) and 3 repetitions to fit the time; p99 there is a tail estimate from few samples.
- Both ends of a call use the same `--cipher`, each cipher with its own server process. The negotiated suite is not reported by the tool.
- Co-tenant load, clocks and throttling were not recorded beyond `meta.txt`.

## Cost

About 1 USD in total (credit 30.64 USD before, 29.82 USD after): five boxes at 0.07 to 0.17 USD/h for 0.3 to 1.2 h each, plus four boxes destroyed unused after 10 to 15 minutes. Longest box: a-348060, 1 h 9 min (cap 3 h).
