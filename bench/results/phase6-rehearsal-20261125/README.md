# Session phase6-rehearsal-20261125

- Status: rehearsal, no verdict. The Phase 6 code ran on two Vast boxes (mTLS, profile, probe, a split of the 30B, Case F once per mode). Split parity holds against a one-node run on the same GPU; the failover detection rules hold in all three modes; automatic rejoin fails in two of three modes (4 defects, below).
- Rule: ADR 0091 (Accepted), Case F pass rules applied to one trial per mode, rehearsal only. Code: branch `bench-phase6-rehearsal` on main 8018394b, plus `scripts/phase6_*.py|sh` (no Rust change).
- Boxes (both RTX 3090, power gate ok at default, idle gate 10/10, `cuda: ok`, co-tenant gate ok):
  - A (head, node `a`): box 62452871, host 205610, Switzerland, driver 580.178.04, AMD Ryzen 7 3700X (16 threads, cgroup quota 15.4), 31 GB RAM, 1 NUMA node, 0.398 USD/h, 04:06 to 05:27 (1 h 21 min). Co-tenant 0.03 cores.
  - B (node `b`): box 62449829, host 698519, Bulgaria, driver 595.91.07, Xeon Platinum 8167M (104 threads, 2 sockets, 2 NUMA nodes, quota 25), 282 GB RAM, 0.364 USD/h, 03:37 to 05:27 (1 h 50 min). Co-tenant 0.26 cores. Two NUMA nodes (the judged sessions need one, ADR 0091 answer 8; a rehearsal does not).
  - Cost: credit 22.81 to 20.98 USD for the whole session (1.83 USD) including three boxes destroyed at gates or stalls (below) and the same-host attempt; the two session boxes are about 0.67 + 0.54 USD of box time, the rest traffic (about 19 GB each) and aborted boxes. `taxameter destroy` did not print `inet_down_billed`; the figures are from the balance.
  - Boxes destroyed after their verified fetch: 62449829 and 62452871 (05:27). Aborted: 62449740 (host 73118: GPU limit 350 W against 420 W default, destroyed at the power gate), 62450524 (host 130779: co-tenant 5.04 and 5.48 against limit 4.8, then 4.02; destroyed per the gate), 62451626 (host 205610: image loading 12 min, re-created once, then ready in 50 s).
- Model: Qwen3-30B-A3B Q4_K_M, sha256 0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5, 18.56 GB, downloaded on each box (taxameter box downloader, a few minutes at both).
- Link A to B (public IPs, Vast NAT with mapped ports): round trip 57 to 68 ms (heartbeat histogram, 265 samples), one-way 32 ms at 4 KiB.
- Files: `a/` and `b/` per box. Logs are `*.zst` (decompress with zstd), `a/head-requests.jsonl.gz` is the head's request log, `*/otel/` the collector's OTLP files (metrics, logs). No key and no API key is in the tree; `cluster.toml` and the certificates stayed under `runs/phase6/` (git-ignored; the files here are the box copies without keys).

## Step 1 and 2: certificates, cluster.toml, nodes, profile

- `tetraeder cluster cert --node a|b` on each box (key `0600` on the box); the public certificates went to both boxes with `taxameter put`; `cluster.toml` was written locally by `scripts/phase6_cluster_toml.py` from the public IPs and mapped ports (`taxameter endpoint`) and uploaded. Deviation: each box has its own file in which its own entry is `127.0.0.1:7000` (no hairpin through the provider NAT); the others are public address and mapped port.
- Both daemons started, mutual TLS between the two public addresses worked at the first try: `cluster profile` merged both hardware graphs (`a/hw.json`: devices `a/gpu0`, `a/cpu0`, `a/disk0..2`, same for `b`), `tetraeder serve --cluster` with an API key (`--api-key-file`, required for a public head) listened on 0.0.0.0:8080. No client connected from outside; the load ran on the head box against 127.0.0.1.
- Log noise: every start logs `bad request frame: read failed: connection reset` and sometimes `cannot decrypt peer's message` on the daemons (peer a); service was not affected there.

## Step 2: link probe (`a/probe-*.json`, `a/sysctl-a.txt`)

`cluster probe --repeats 10`, frames 4 KiB to 64 MiB, one-way time per direction (the head probes through the nodes' own transports, which dial with the requested buffers):

| buffers | requested | effective (getsockopt, from the log) | a to b, 4 KiB / 64 MiB | b to a, 4 KiB / 64 MiB | link beta (recorded) |
|---|---|---|---|---|---|
| autotuned | Auto | send 3.9 MB (grown), recv 87 380 initial | 32.4 ms / 1.342 s | 31.5 ms / 1.148 s | 49.5 MB/s |
| explicit | 4 MiB | 8 MiB send and recv (the kernel doubles) | 30.2 ms / 1.313 s | 29.6 ms / 7.046 s | 9.5 MB/s |
| explicit | 16 MiB | 32 MiB | 32.0 ms / 1.282 s | 28.4 ms / 7.885 s | 8.5 MB/s |

Autotuning wins: equal a to b, 6 to 7 times faster b to a. The explicit setting kept the dialer's sender (B to A) at about 8 to 9 MB/s, where autotuning reaches 50 MB/s (about 400 Mbit/s). `net.core.{r,w}mem_max` cannot be read in the container; `tcp_rmem` and `tcp_wmem` max is 128 MiB on A. Source for the rest of the session: `a/probe-auto.json` (`hw.json` carries it). Not explained: why the explicit buffers cost only one direction.

## Link band check (added with the pre-exit checks)

Question: would a start probe against an end probe have held ADR 0091's Case S link rule (agree within the band 0.08 at 16 KiB and at 32 MiB), given the round trip moved from 57 to 68 ms? The session took no end probe, so this is an estimate from the start probes (`a/probe-auto.json`) and the heartbeat histogram, not a measurement.

- Round trip 57 to 68 ms is +19.3%. The one-way floor moves by half of it, +5.5 ms (28.5 to 34.0 ms).
- 16 KiB (a point of the probe): a to b 46.0 ms, b to a 36.1 ms at the start. With +5.5 ms: 51.5 ms (+12.0%) and 41.6 ms (+15.2%). Both exceed 0.08: the rule would not have held.
- 32 MiB is not a probe point (sizes go 4 KiB to 64 MiB). Interpolating the start curve between 16 MiB and 64 MiB linearly: a to b 0.663 s, b to a 0.576 s, so +5.5 ms is +0.8% and +1.0%. At 64 MiB: 1.342 s and 1.148 s, +0.4% and +0.5%. The bulk point would have held, unless the bandwidth moved as well (unknown).
- Noise of the start probe itself, three probes minutes apart: 16 KiB a to b 46.0, 45.8 and 45.2 ms (1.8% spread); 64 MiB a to b 1.342, 1.282 and 1.313 s (4.6%). The explicit-buffer runs are not comparable in the b to a direction (step 2).
- Reading: the small-frame point is latency, so on a public link whose round trip drifts by a fifth the rule fires on the 16 KiB point alone while bulk throughput may be unchanged. The rule is unchanged (owner decision); a judged session on a link like this one should expect to rerun its rounds after a new probe.

## Step 3: split parity (`a/parity-pipeline.json`, `a/parity-solo-a.json`, `b/parity-solo.json`)

- Plan `a/plan-pipeline.toml` (`scripts/phase6_plan.py`): one replica, stage 0 layers 0..24 and the embedding on `b/gpu0`, stage 1 layers 24..48 and the head on `a/gpu0`; the leader is B (so the head only tokenizes), `serve --cluster-plan --cluster-hw`. Written by hand, not by the planner: there is no CLI that emits a cluster plan, and `tetraeder-plan predict --cluster` refuses it ("a segment is a builder placement or all on the CPU", so the cost model cannot price an all-GPU segment; the head logs the same warning and starts without predictions).
- Greedy (temperature 0, seed 1, 64 tokens, 8 fixed prompts through `/v1/completions`, text compared; the server returns no token ids): pipeline against the one-node run on A (same build, same GPU as stage 1): 8 of 8 equal. The one-node run on B differs from both on prompt 0 at character 85 ("Germany is Berlin" against "Portugal is Lisbon", a near tie); B alone against A alone is 7 of 8. So the difference is between the two boxes (driver 595 against 580, other CPU), not the split. No logit comparison was made (rule 2 asks for logits per backend and split point: the pipeline tests cover CPU only; GPU logit parity is open).
- Speed is not judged. One observation: the first start stalled (below).

## Step 4: Case F, one trial per mode (`a/caseF/`)

Two replicas of the 30B, R_A in process on the head, R_B on node B (`replicas.toml`, cap 4 each, `--ctx 4096 --parallel 4`, default KV), closed loop of 8 streaming chat requests through the head for 180 s (`scripts/phase6_load.py`, prompt 395 tokens (the filler text came out at 395 tokens, not 512), 256 output tokens), fault on B about 66 s in (`scripts/phase6_fault.sh`), results by `scripts/phase6_casef.py` (client side). Clocks of the boxes agreed within the ssh latency (about 1 s); the fault epoch is B's.

| mode | in flight at fault | in-flight errors, delay after fault | failures elsewhere | ok after bound (fail) | rejoin |
|---|---|---|---|---|---|
| 1 `kill -9` of B's replica engine | 8 | 4, all at 2.786 s ("the connection to replica 1 was lost") | 0 | 134 (0) | none: the daemon restarted the engine at +2.3 s, loaded at +11 s, but the head never started serving it; no request reached B for the rest of the trial (114 s) and the 2.5 minutes after |
| 2 `kill -9` of B's daemon and engine | 8 | 4, all at 1.864 s (node `down` at +1.86 s) | 0 | 137 (0) | none: daemon restarted at +40 s, node `up` 1.6 s later, no replica start on B for the remaining 78 s |
| 3 SIGSTOP of B's daemon and engine, SIGCONT after 60 s | 8 | 4, all at 2.078 s (node `down` at +2.07 s) | 0 | 189 (0) | 0.78 s after SIGCONT (first request admitted on B), first ok request submitted after the restore ended 4.4 s later |

ADR 0091 Case F rules, per mode (client side, `a/caseF/result-*.json`):
- every request in flight on R_B ended with one clean error within 3 s: holds in all three (2.79, 1.86, 2.08 s; no token after an error);
- no request on R_A failed: holds (no failure outside the in-flight set; which replica a request ran on is read from the admit lines of the two logs, 4 of 8 in flight were on B in each mode);
- every request submitted from 3 s after the fault succeeded: holds (0 of 134, 137 and 189);
- requests waiting at the router move and succeed: holds as far as visible (after the fault the four clients that had been on B waited for R_A's four slots and all succeeded); not separately counted;
- each request ends exactly once: holds client side (no request with two outcomes, none without); the client and gateway counts were not compared;
- B rejoins and serves within 60 s of its restore, trial 1 without manual action: fails in modes 1 and 2 (holds in mode 3). Between trials the head had to be restarted by hand to get B back (the replica then joined in under 25 s).
- Mode 1's error delay (2.79 s) is close to the 3 s bound although a killed process should reset at once.
No speed is judged. Throughput by the 10 s buckets is in the result files (about 24 to 28 requests per 10 s with both replicas, about half on A alone).

## Step 5: prefix fetch and planned drain (`a/caseF/prefix*.json`, `a/caseF/load-drain.jsonl`)

- Prefix fetch: no fetch happened in either direction. With the declared prefill rate lowered to 100 tok/s (`replicas-prefix.toml`, so the fetch is priced to win: about 222 MB of KV at 50 MB/s is 4.4 s against 22.6 s), three runs (X1 on a replica, four fillers sharing the prefix fill it, X2 with the same prompt goes to the other replica): X2 ran on the other replica with `common=13` (recomputed) in all three, and the first run had X1 on A, the second and third on B (admit lines in `a/serve*.log.zst` and `b/node*.log.zst`). Tokens equal: X1 and X2 text identical in every run. Timings: 2259-token prompt recomputed on the other replica 0.67 to 0.81 s (about 3 000 to 3 400 tok/s measured prefill), cold prompt of the same size 0.58 to 0.61 s; a fetch at the measured link rate would have cost about 4.4 s, so at the real prefill rate the fetch rule rightly says recompute, and the declared 100 tok/s was the only reason to expect one. Why it still did not fetch is not found; reading the code, the head's in-process replica never has `set_kv_format` and `set_kv_row_bytes` set (only remote replicas do, at join), so the policy has no row size on that side; not confirmed. With the head leading no replica the rule would be reachable, but a two-node cluster has the head on a node that leads one.
- Planned drain: with `--replicas`, `tetraeder node --drain --id b` (head key) turned B to `draining` in 0.7 s and nothing else: the router kept sending to R_B (B admitted requests for the 40 s that followed), 135 requests, 0 errors. A drain that moves load needs `--cluster-plan` and the replanner; not run with the two-replica layout.

## Step 6: plan source on GPU nodes (`a/otel/`, `a/serve*.log.zst`)

With the pipeline plan and `--cluster-plan`, `kill -9` of B's daemon: node `down` after 2.1 s, two `node_change` triggers (b leaving, b joining) counted by `tetraeder.replan.total`, both with outcome `rejected`; no replica started on A, `no replica is up` for the 70 s watched. So `ClusterPlanSource` made no usable proposal for the two real boxes (the rejection reason is not in the metric; "no proposal" is the likely one: the single-node references and the candidates need priced segments, and the cost model cannot price an all-GPU segment, see step 3). B restored by restarting its daemon: node `up`, but the head did not start the replica again (the same missing rejoin as in Case F).

## Step 7: same-host pair

Not run. Two offers on host id 3497 were rented (62462212 and 62462265, same public IP 185.62.108.226) but are two different machines (128 against 96 threads, EPYC 7452 against 7642, 2 against 4 NUMA nodes), and one had its GPU at 250 W against 350 W (power gate failed). Both were destroyed at the gate (about 0.1 USD, no data).

## Defects found

1. The head does not start a replica again after a node's engine or daemon is restored (Case F modes 1 and 2, pipeline restore): the join loop made no attempt that reached the node (no `Assign`, no start on B), and no "join attempt failed" was logged. A restart of the head fixed it every time. Related: the very first start of the cluster stalled for more than 3 minutes with `no replica is up` (engine loaded on both nodes, nothing logged), and the next start 4 minutes later joined in 6 s. (`cluster-rejoin-after-restore`)
2. No prefix fetch between replicas although it was priced to win and the target had room (step 5). (`cluster-prefix-fetch-not-triggered`)
3. `ClusterPlanSource` proposes nothing for two real GPU boxes, and the cost model refuses an all-GPU segment (steps 3 and 6). (`cluster-plan-source-gpu-segments`)
4. `node --drain` without a cluster plan marks the node but never drains its replica (step 5). May be intended; the ADR's planned drain is the replanner's. (`cluster-drain-without-plan`)
Smaller: the explicit-buffer probe is 6 to 7 times slower in one direction (the autotuned setting is the one to keep); the mode 1 error delay of 2.79 s; cargo's parallel crate downloads stalled on a box whose model download ran at the same time (rerun with `CARGO_HTTP_MULTIPLEXING=false`).

## Deviations

- Per-box `cluster.toml` (own entry on loopback); head leads R_A in process; pipeline plan written by hand with leader B; prefill rate declared 100 tok/s for the fetch trial; `RUST_LOG` debug for `tetraeder` and `tetraeder_control` on all processes (the admit lines are the attribution of requests to replicas); an OTLP collector on each box for the metrics.
- Not done: GPU logit parity, gateway-versus-client request count comparison, rejoin after the head restart in numbers other than "under 25 s", same-host pair, any speed comparison, the NUMA rule.
