# 0091. Phase 6: nodes, the transport trait, plans across nodes, replicas and failover

Status: Accepted (owner, 2026-11-23, on the advisor's recommendation; answers under "Owner answers"). Proposed 2026-11-23. Phase 6, milestone M3. Builds on ADR 0023 (TCP+TLS chosen), 0069 (worker process per box, `RemoteBackend`), 0070 (shapes, measured curves), 0077 (pipelined prefill), 0008 (Linux-only expert parallel across nodes), 0042 (drain-and-restart replanning), 0049 and 0087 (prefix cache). Written before any Phase 6 code (rule 1), as ADR 0068 did for Phase 5. Interfaces: [`docs/notes/phase6-interfaces.md`](../notes/phase6-interfaces.md). Tree nodes: track `cluster`.

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

## Context

Phase 6 (`docs/plan.md`): transport, discovery, mTLS, pipeline and experts across nodes, remote KV, replicas, failover. Exit test: a model larger than any single node runs within 15% of prediction; killing a node with a replica keeps new requests flowing, and in-flight requests on it fail cleanly.

Constraints that shape the design:
- **C1 Transport.** TCP with TLS 1.3, behind a trait (ADR 0023). Per hop on one host: 0.41 to 0.46 ms at 8 KB. WAN (90 ms): a 1 MB echo takes 274 to 300 ms against about 145 ms from `RTT + bytes / bandwidth`, so alpha and beta alone miss by 2x at bulk sizes (untuned TCP buffers, a hypothesis).
- **C2 Vast networking.** Boxes sit behind NAT with mapped ports (`PUBLIC_IPADDR`, `VAST_TCP_PORT_*`, `scripts/transport_wan_bench.sh`); addresses are known only after `create`; no two hosts on one LAN were on offer (ADR 0023); containers lack `NET_ADMIN`. Multicast does not cross that.
- **C3 What one node can already execute.** A stage of one GPU with experts in RAM runs on the streaming builders (`Sched`); stages of more than one GPU must be VRAM resident (ADR 0069, `multi_gpu_cpu_compute` false in phases 5 and 6). Tensor parallel never crosses nodes (ADR 0008).
- **C4 Fault isolation exists per box.** `RemoteBackend` already fails in-flight requests on a dead or hung worker, marks lost sequences, restarts and readmits (ADR 0069 "Failure handling"); the scheduler's `Failed` path fails exactly the affected requests.
- **C5 Snapshots are plan independent.** `seq_snapshot` returns the one-rank layout whatever the shape (ADR 0069), so KV bytes can move between any two backends of the same model and KV format. A long prompt's KV is gigabytes, so a move must beat recompute.
- **C6 Bytes per token.** Logits are `n_vocab x 4` = 594 KiB per output row on Qwen3, 4.7 MB per step at concurrency 8; a hidden row is `n_embd x 4` = 16 KiB on Qwen3-235B; a sampled token is 4 bytes.
- **C7** Single-node plans, fixtures and predictions stay bit-identical (no flag day). **C8** Rules 4 to 6: simulated fixtures and faults first, property tests on plans, metrics, panel and alert per component. **C9** Every node is yours or explicitly trusted (plan.md non-goals); llama.cpp's RPC is plaintext and is not used.

## Decision

Phase 6 adds one concept, the **node**, and expresses everything else with what exists: a replica is today's engine (scheduler, sampling, prefix cache) on its leader node, reached by the gateway through the engine interface; a replica across nodes is a chain of single-node sub-plans (segments), each run by today's worker; network hops are priced from curves measured through the real transport.

### Processes

- **Node** = one box running `tetraeder node`: a long-lived daemon that listens on one TLS port, answers heartbeats, reports its hardware graph, and starts children: a replica engine (`tetraeder replica`, the `serve` engine without the HTTP front end) where the node leads a replica, and `tetraeder worker` children for its segments over stdio, as `serve --worker` does today (ADR 0069). It relays frames between network connections and the children's pipes. A sticky CUDA error kills only a child, and the daemon restarts it (C4). The daemon is forced: something must outlive the worker and accept remote starts.
- **Head** = the node whose `tetraeder serve --cluster cluster.toml` runs the HTTP gateway (templates, tokenization, API formats) and the router. It holds one `Engine` per replica: in process for a replica it leads itself, a `RemoteEngine` for the others.
- **Replica leader** = the first node of a replica. Its engine runs the scheduler and sampling and owns that replica's prefix cache, as `serve` does today. Its backend is a `RemoteBackend` to the local worker, or a `PipelineBackend` over the replica's segments.
- So per token a remote replica sends back only the sampled token and its event (`GenEvent`, a few bytes; log-probabilities only when asked), never logits (C6). Grammars and JSON schemas travel as their source spec and are compiled on the leader, because `TokenConstraint` is a trait object.

### Nodes and the transport trait

- **Transport trait** (`crates/transport`): `connect(peer, purpose)` and `accept()` return an authenticated peer id, a `Read` half, a `Write` half and a close handle usable from another thread. Today's frame format (`u32` length, `u32` header length, JSON header, raw payload; `remote_backend`) moves into `transport` byte for byte and runs over those halves, so `RemoteBackend` keeps its threads, deadlines and supervision and only gains a second way to open its pipes.
- Two implementations: `TlsTransport` (TCP+TLS 1.3, rustls, `TCP_NODELAY`) and `MemTransport` (in-process pipes with a fault script, for simulated nodes in CI). QUIC stays a later implementation of the same trait.
- **Socket buffers are measured, not fixed.** The 2x WAN gap at 1 MB is blamed on untuned buffers, a hypothesis (ADR 0023). Linux caps an explicit `SO_RCVBUF` at `net.core.rmem_max` and turns autotuning off for that socket. So the transport reads the effective sizes back (`getsockopt`) and logs them, and the link probe runs once with kernel autotuning and once with explicit buffers sized to the probed bandwidth-delay product. The faster setting is recorded per link with its measured size, and the transport uses it.
- **One connection per purpose** (`control`, `replica`, `segment <n>`): a 32 MiB prefill hop must not delay a heartbeat behind it on one TCP stream.
- **Trace context** crosses nodes as a `traceparent` field in the frame header.

### Identity and mTLS

- Each node has a key pair and a certificate whose subject alternative name is `tetraeder-node:<id>`. Identity is the node id in the certificate, never an IP or DNS name, because Vast addresses are NAT mappings known only after `create` (C2).
- Both sides verify the peer against the **trust anchors** listed in `cluster.toml`, then require the id in the certificate to be a configured node. An anchor can be a node's self-signed certificate (pinning) or one cluster CA certificate; rustls' verifier treats both alike, so the provisioning question (Open for the owner) changes tooling, not code.
- A failed handshake, a certificate without a configured id, or a second connection claiming a connected id is refused and counted.
- **Keys** are generated on each box (`tetraeder cluster cert --id <id>`) and never leave it: never in the repo, a pushed HEAD (taxameter pushes committed HEAD only) or a log. The session driver builds `cluster.toml` after `create` from each box's public address, mapped port and public certificate, keeps it under `runs/` (git-ignored), and uploads it to every box over taxameter's SSH.

### Discovery and membership: static first

- `cluster.toml` lists the cluster name, the head, `node_down_after`, and per node its id, the address to dial, an optional listen address, and its trust anchor. Static is forced on Vast (C2: no shared L2, addresses known only after `create`, mapped ports), and the list is needed anyway as the trust list (C9). mDNS would only fill in addresses on a homelab LAN; whether it stays in Phase 6 is open for the owner.
- Membership is a pure state machine in `crates/control` (clock injected, as `replan.rs`): a node is `Up` after a successful hello and heartbeat, `Down` after a failed call (connection reset, malformed frame) or after `node_down_after` without any frame from it, `Draining` on `tetraeder node --drain`. Heartbeats run every `node_down_after / 4` on the control connection, answered by the daemon, not a worker, so a long step never looks like a dead node; a hung worker is caught by the existing step deadline (10x the predicted step, at least 5 s, now including the predicted hops).
- `node_down_after` = 2 s, one named value: above Linux's minimum TCP retransmission timeout (200 ms) with several retries and above the WAN p99 round trip (94 ms), below the 5 s step deadline floor, so a dead node is found before any request deadline fires.
- **Join and leave.** A node that becomes `Up` and whose replica loads and answers one probe step joins the router. A replica whose node goes `Down` leaves it. A change to a replica across nodes is a drain and restart of that replica with a new plan (ADR 0042); live resharding stays Phase 7.

### What crosses the wire

| Traffic | Between | When | Bytes | Message |
|---|---|---|---|---|
| Control | head and every node | always | small | `Hello`, `Heartbeat` and `Alive`, `Profile`, `StartReplica` and `Assign` (plan TOML, model `sha256`), `Release`, `Probe`, `Drain` |
| Requests | head and replica leader | per request | prompt tokens, sampling parameters, constraint source | `Generate` |
| Events | replica leader to head | per generated token | a token id and its event; log-probabilities when asked | today's `GenEvent` |
| Backend calls | leader and its segments | per scheduler call | header only | today's `Request` and `Reply`, unchanged |
| Hidden state | leader and segments | per step and segment boundary | 16 KiB per token, every token of the step | `decode` whose payload is hidden rows instead of tokens; its reply is hidden rows or logits |
| Logits | last segment to leader | per output row, only when the head unit is not on the leader | 594 KiB | today's `decode` reply |
| KV | replica leader to replica leader | a cross-replica prefix hit that beats recompute | snapshot bytes | `PrefixFetch` |

Payloads are little-endian f32 (ids i32), as the logits payload is today. Hidden states stay F32 for parity (rule 2); F16 hops are a later lever behind a benchmark (rule 3).

### Plans across nodes

- **Hardware graph.** `Device` gains `node: Option<String>`; `None` is the local node, so every existing fixture and plan is unchanged (C7). `tetraeder cluster profile` merges the node graphs (ids prefixed `<node>/` for uniqueness; code reads the field), adds one `Network` link per node pair and the probe curves. `Tier::Remote` is then derivable from `node` and becomes a deletion candidate once old fixtures are migrated.
- **Plan IR: no new field.** A replica's stages may sit on devices of different nodes. Validator rules: every stage's devices are on one node (C3: no group spans nodes). Today a unit's compute device must be in its stage's group (`Issue::NotInGroup`); that rule gains exactly two allowed cases, both on another node: the head unit may compute on the leader's node, and an expert may compute where it lives on another node (remote experts, parked below; Linux only, ADR 0008). `Forbid` gains `node`, the node trust rule (plan.md hint `forbid node laptop`).
- **Segments** (derived, never stored): a maximal run of consecutive stages on one node, plus a head-only segment when the head unit is on another node than the last stage. `plan::segments` derives them and `plan::segment_plan` cuts the single-node sub-plan each worker runs. A plan on one node has one segment equal to the plan. A segment without the embedding takes hidden rows in; one without the head returns hidden rows. The builders (dense and MoE, executor and `Sched` paths) gain that input and output; nothing else in a worker knows about nodes.
- **Head placement** is a planner choice priced like any other: on the last segment, logits cross the link back (594 KiB per output row); in a trailing segment on the leader, 16 KiB do and the leader computes the output projection. On the measured WAN the difference matters (8 KB echo about 90 ms, 1 MB echo 274 to 300 ms, C1).
- **Pipeline issue.** The leader's `PipelineBackend: BatchBackend` calls segments in order (a star from the leader, not a chain between workers): one hop per boundary per step, so two nodes with the leader on the first pay one round trip per decode token, as plan.md "Parallel shapes across slow links" says. Pipelined prefill (ADR 0077) keeps several micro-batches in flight across segments. When a later segment fails, the leader rolls the earlier ones back with `seq_rm(seq, old_len)` (ADR 0065 rollback); a segment that cannot roll back marks its sequences lost.

### Cost model: network links

- **Measured curves through the real transport.** `tetraeder cluster probe` sends frames of 4 KiB to 64 MiB (powers of 4, 20 repeats, median) between every node pair, both directions, over `TlsTransport` with the worker's frame code, each answered by an empty reply, and records one-way time as that time minus half the empty round trip. Stored as `CollectiveCurve { op: Copy, ranks: [<a>/host, <b>/host], comm: "tls" }`, interpolated as ADR 0070's curves. Without a curve the prior is the link's `alpha + bytes / beta`. Curves are measured, never fitted (ADR 0070), because C1 shows the formula off by 2x at 1 MB.
- **Prediction composes.** `predict_cluster` prices each segment with today's single-node `predict` on its node's calibrated graph and sub-plan, then adds hops: decode step `T = sum of segment times + sum of hop_time(boundary bytes) + hop_time(return bytes)`; prefill follows ADR 0077's pipelined schedule with the hop as the copy curve, serialized per boundary. A one-node plan predicts bit-identically to `predict` (C7). A remote replica adds `hop_time` of its events, a few bytes per token. The simulator (`crates/costmodel/src/sim.rs`) covers routing under load.
- **Planner.** `enumerate_cluster` lists, per replica, the cut points that fit each node's memory (the dynamic program over layers, ADR 0070, with node capacity per stage) and the head placement; it never forms a group across nodes. Deterministic, before calibration, as ADR 0040.

### Replicas and failover

- **Replicas** are `Plan::replicas` (exists), each led by its first node.
- **Routing** (in this order): (1) a request whose prompt extends a prefix that a replica holds, live or in its prefix cache, goes to that replica, so a conversation's next turn keeps the Phase 4 prefix gain (ADR 0049); the head keeps a prefix index per replica, fed by the replicas' save and evict events; (2) else the `Up` replica with the fewest outstanding requests. A replica is sent at most `n_seq_max` requests at a time; the rest wait at the router, so a request not yet started can still go to another replica when its replica leaves.
- **Failing cleanly.** A request is in flight once its replica admitted it. When the replica fails (its leader node goes `Down`, a segment call fails, the event stream breaks), the existing path runs (C4): exactly its in-flight requests end with `GenEvent::Error`; the gateway ends a stream with one error event and closes it, or answers 503 before headers; no token follows the error and no request ends as a normal `stop` or `length`. Each request ends exactly once. A `RemoteEngine` whose connection drops ends its open streams with the same error.
- **Detection bound.** A killed process resets its connection at once; a silent node is found within `node_down_after`; a hung worker within its step deadline. A `Down` transition closes every connection to that node, so a call blocked on it returns at once instead of waiting for the 5 s deadline floor. So an in-flight request on a killed or silent node ends within `node_down_after + 1 s` (the slack is one heartbeat interval plus scheduling).
- **Who restarts what.** The daemon restarts dead children on its node, with `RemoteConfig`'s backoff and cap (ADR 0069). The head never restarts remote processes: when the node is `Up` again it reconnects, starts the replica or assigns the segment again, waits for one probe step, and the router takes the replica back. For a node endpoint, reaching the restart cap takes the replica out of the router until the node next becomes `Up`; it never exits the head (today's default `on_give_up` exits, which stays for `serve --worker`).
- **Gateway** failure is not covered: the head is a single point of failure in M3.

### Remote KV, Phase 6 scope

Each replica keeps its own prefix cache on its leader (RAM and disk tier, ADR 0049, 0087), as today. Prefix bytes cross the network only on demand: when routing puts a request on a replica that lacks a prefix another replica holds, the target fetches the snapshot (`PrefixFetch`) only if the predicted `hop_time(snapshot bytes) + restore time` beats recomputing the prefix at the replica's predicted prefill rate, with ADR 0049's `RESTORE_MIN_GAIN` margin; otherwise it recomputes. Restore across replicas works because snapshots are plan independent (C5). Affinity routing makes such fetches rare. Not in Phase 6: KV tiers on a remote node during attention, and KV moves between prefill and decode hardware (Phase 7, `multi-split-hardware`).

### Experts across nodes: parked until after the exit

Neither exit case needs them, and over the measured WAN one round trip per MoE layer costs about 5.4 s per token (ADR 0023). The design, for later: the stage's MoE layer computes its local experts, and a ggml custom op on the CPU backend sends the tokens routed to the other node's experts as one `ExpertCall` and adds the returned partial sum; ids not held are -1 (patch 0004 semantics); the callback trampoline lives in `ggml-sys` (rule 7). It gets its own ADR addendum when it is picked up; the tree node `cluster-experts-nodes` is parked after the exit.

### Metrics, panel, alerts (rule 6)

One dashboard, `deploy/grafana/dashboards/cluster.json`, and one rules file, `deploy/alerts/cluster.yml`; each build step adds the rows and rules of the component it ships.

| Step | Metrics | Panel row | Alert |
|---|---|---|---|
| 1 transport | `tetraeder.link.bytes` (counter) and `tetraeder.link.latency` (histogram, per frame round trip) from the catalog, attributes `source` and `destination` node, `transport` (`tls`, `mem`), `tetraeder.purpose` (`control`, `request`, `event`, `hidden`, `logits`, `kv`); `tetraeder.cluster.tls.rejected` (counter, `reason`) | "Links: bytes and latency per purpose" | `TetraederTlsRejected`: any rejected handshake in 10 minutes (an unknown or misconfigured peer) |
| 2 planner | none at run time (offline); `tetraeder.plan.residual` gains a `segment` attribute in step 4 | | |
| 3 membership | `tetraeder.cluster.node.up` (gauge, node), `tetraeder.cluster.heartbeat.rtt` (histogram, node), `tetraeder.cluster.node.transitions` (counter, node, `to`) | "Nodes: up, heartbeat RTT, transitions" | `TetraederNodeDown`: a configured node not up for 1 minute; `TetraederLinkSlow`: heartbeat RTT median above 2x the probed small-frame time for 10 minutes (predictions no longer hold) |
| 4 pipeline | `tetraeder.plan.residual` with `segment`; hop time per boundary as `tetraeder.link.latency` with purpose `hidden` or `logits` | "Pipeline: predicted and measured time per segment and hop" | `TetraederSegmentDrift`: a segment's median residual beyond 0.15 (the exit-test line) for 30 minutes |
| 5 replicas | `tetraeder.router.requests` (counter, replica, `result` = ok, failed_in_flight, moved, affinity), `tetraeder.router.replicas.up` (gauge), `tetraeder.kv.prefix.fetch` (counter, `result` = fetched, recomputed) | "Router: requests per replica and result, replicas up, prefix fetches" | `TetraederReplicaOut`: fewer replicas up than the plan has for 5 minutes; failed requests also feed the existing `TetraederRequestsFailing` and SLO burn alerts |

### Security boundaries

- Only configured node ids with a certificate from a configured trust anchor connect; everything between nodes is mutually authenticated TLS 1.3.
- A node sees the activations, KV and therefore the content of the requests its replicas and segments serve. `forbid node` keeps units, and with them that content, off a node; that is the trust rule.
- The wire carries no file paths, code or commands: `Assign` and `StartReplica` name a model by `sha256`, and the node maps it to its own local file or refuses. API clients authenticate at the gateway (Phase 9), never on the node port.
- Private keys stay on the box that generated them (above).

### Out of scope for M3

Tensor parallel across nodes (ADR 0008); NCCL or RDMA between nodes; experts across nodes (parked above); QUIC and iroh (NAT traversal); live resharding and replica scaling (Phase 7); prefill and decode on different machines (Phase 7); remote KV tiers during attention; a highly available head; worker-to-worker pipeline chains; certificate rotation and revocation beyond editing `cluster.toml`. mDNS is open for the owner.

## Exit test (fixed before any measurement)

Two cases, each on one pair of boxes in one session; they may run in separate sessions on different pairs (Case F first, it needs no 142 GB download), so neither session runs into `max_box_hours` (3 h). Box spec: two Linux boxes, one RTX 3090 each (24 GB; class `cuda-default`, sm_86), CPU quota 8 or more, 230 GB disk, within 0.90 USD/h all-in each, both `cuda: ok`. Pair choice, in this order: two hosts with a small-frame TCP round trip under 5 ms; else any two hosts (WAN). Each box records ADR 0068's `env.json` plus its node id, public address, port mapping, the cgroup memory limit and the effective socket buffer sizes.

### Case S: a model larger than any single node

- **Model.** Qwen3-235B-A22B Q4_K_M (142 GB; 94 layers, 128 experts, top 8). **Node size.** Each node has 24 GB VRAM and a RAM budget of 64 GB (`EMULATE_RAM_GB=64`, the balloon of `scripts/exit_run.sh` that the Phase 2 exit runs of the 235B used, `docs/sessions.md`; a run whose balloon died does not count), so a node holds 88 GB < 142 GB and two hold 176 GB. "Larger than any single node" means: weight bytes exceed every node's VRAM plus RAM budget, so no one node holds the model without its disk tier.
- **Candidates.** `tetraeder-plan candidates --cluster` on the merged hardware graph before calibration, restricted to plans that compute on both nodes (at least one segment per node). A single-node plan, which each node could run by streaming from its own disk (as M1 ran the 235B on one box), is never a judged candidate. The judged plan is the planner's analytic pick among the spanning ones. Each segment is one GPU plus RAM experts on the streaming builder (C3).
- **Calibration, separate from the judged run** (ADR 0068: no anchor equals a candidate). Per node, anchors of its segment truncated to the first quarter and the first half of its layer span (hidden input from a fixed-seed tensor, no head), pp2048 and tg128, two rounds; they fit the node's single-node parameters as ADR 0040 and 0056 fit them for the streaming builder. Then `tetraeder cluster probe` both directions. Then `tetraeder-plan predict --cluster` writes `predict-cluster.json` (predicted `pp_tps`, `tg_tps`, `T`, git commit, `sha256` of the merged graph and the probe file) before the first judged run; `verdict.json` records its `sha256`, and a prediction written after a judged run makes the case invalid.
- **Named risk.** ADR 0056 found single-GPU streaming candidates well above 15% on its fixtures (`20261103`), and both segments run on that path. Tree node `bench-235b-stream-predict` measures that error on one box now, before any cluster code; if it is above 0.15, a cost-model fix with its own ADR comes before the exit session, and these rules do not change.
- **Workload and rounds.** ADR 0068's: one sequence, pp2048 then tg128, `-r 1` per call, three rounds. Each round also runs the single-node reference on the leader (the same model streamed from its disk under the same 88 GB; report only), interleaved, order reversed in round 2.
- **Within 15%.** `e_pp = pp_pred / pp_meas - 1`, `e_tg = tg_pred / tg_meas - 1` on the medians of clean rounds. Pass when `|e_pp| <= 0.15` and `|e_tg| <= 0.15`. A pick predicted feasible that fails on the boxes fails the case.
- **Clean.** ADR 0068's rules on both boxes (throttle sampler on both GPUs, idle gate, co-tenant guard, power limit at default), A/A of the judged plan within band `b` on `T`, and the link: the probe repeated at session end must agree with the start probe within `b` at 16 KiB and at 32 MiB; otherwise the rounds are rerun after a new probe and prediction. `b` = the spec band 0.08 (`bench/spec/phase0.toml`), since no two-node noise class exists; recorded as `band_source`.
- **Paired ratio** for `docs/features.md`: `T_single_node / T_two_nodes` on the same boxes and rounds (report only, not gated). The llama.cpp RPC backend is not a baseline (plaintext, C9).

**Expected numbers (hand estimates, not the judged prediction).** No cluster code exists, so these come from measured inputs, not from `predict_cluster`; the planner's frozen prediction replaces them. Inputs: the builder's measured decode of the 235B on one 3090 box with a 91 GB budget, 2.74 to 3.02 tok/s, of which about 33 ms per token was disk reads (`20261113b`), and its measured 32K prefill, 31.5 tok/s, disk bound at 566 MB/s (same session). With two nodes each holds about 71 GB, about 27% of it in VRAM against about 15% on one box, and nothing comes from disk. That leaves about 250 ms per token of expert compute from RAM, split over the two nodes, which run one after the other in decode.

| Configuration | tg128 (tok/s) | pp2048 (tok/s) | `T` (s) |
|---|---|---|---|
| One node, streamed from disk, 64 GB budget (report only) | 2.0 to 2.7 (less RAM than the measured 91 GB, so more disk reads) | about 30 (disk bound) | 115 to 132 |
| Two nodes, link under 1 ms (ADR 0023 same-host numbers) | about 4 | about 100 (experts read from RAM instead of disk, micro-batches over two segments) | about 52 |
| Two nodes, measured WAN pair (90 ms) | about 2.9 (one round trip, about 92 ms, per token on top of 250 ms; the round trip alone caps decode near 11 tok/s) | about 95 (the 32 MiB of hidden state takes about 1 s at 35 MiB/s) | about 66 |

Read plainly: on a fast link two nodes should beat one streaming node at both rates, decode by about 1.5x. On the WAN pair decode is about a tie with one node: 2.9 against 2.0 to 2.7. It loses if the single node at 64 GB keeps its 91 GB decode rate (3.0) or if expert compute from RAM is slower than assumed. Prefill wins by about 3x on either link because nothing is read from disk. Case S judges the prediction, not a speedup, so it can pass on the WAN even where decode loses.

### Case F: failover with replicas

- **Model and plan.** Qwen3-30B-A3B Q4_K_M (18.6 GB) as two replicas, R_A led by node A (also the head) and R_B led by node B, F16 KV, context 4096.
- **Load.** Closed loop through the gateway, streaming chat, concurrency 8, prompt 512 tokens, output 256, for 180 s per trial. Per generated token R_B sends the head one event of a few bytes (C6).
- **Faults**, on node B at t = 60 s, three trials each, B restored between trials: (1) `kill -9` of B's worker (B's engine fails its in-flight requests, the daemon restarts the worker); (2) `kill -9` of B's daemon and its children (connection reset); (3) `SIGSTOP` of B's daemon and children, `SIGCONT` after 60 s (a silent node, the stand-in for a partition, since containers lack `NET_ADMIN`).
- **Pass, per trial:** every request in flight on R_B at the fault ends with one clean error (above) within `node_down_after + 1 s` = 3 s; no request on R_A fails; every request submitted from 3 s after the fault on succeeds; requests waiting at the router for R_B move and succeed; each request ends exactly once (client and gateway counts agree); B rejoins and serves requests within 60 s of being restored (trial 1 without any manual action). No speed is judged.

`scripts/phase6_summary.py` writes `verdict.json` with both cases; Phase 6 passes when both do.

## Simulated first (rules 4 and 5)

The first implementation steps run only in CI, on simulated nodes (`MemTransport` plus fake backends, and loopback `TlsTransport`), before any box.

**Cluster fixtures** (`crates/costmodel/fixtures/`, devices with `node`):
- `cluster_two_3090_lan.toml`: two nodes, each one RTX 3090, CPU, 64 GB RAM, NVMe; 1 Gbit/s, 0.2 ms.
- `cluster_two_3090_wan.toml`: the same nodes, with ADR 0023's measured WAN points as prior curves (90 ms, 35 MiB/s, 1 MB at 274 ms).
- `cluster_two_3090_samehost.toml`: ADR 0023's same-host points (0.41 ms at 8 KB, 5.1 ms at 1 MB).
- `cluster_three_mixed.toml`: a 2x 3090 Linux node, a 1x 3090 Windows node, a CPU-only Linux node; 10 Gbit/s.
- `wan_two_node.toml` gains `node` on its two devices; its existing tests keep their numbers.

**Faults** (`MemTransport` fault script, per link and direction): node crash (connection closed), silent node (no bytes either way, nothing closed), hung worker (heartbeats answered, step never replies), link slowdown (alpha x10, bandwidth /10), one-way partition, flapping (down and up every 3 s), slow node (fake backend step x2), wrong certificate and unknown node id, malformed and truncated frames, node restart with lost KV.

**Property tests:**
- Plans on every cluster fixture: no device or node budget exceeded (RAM budget included); every unit computed exactly once per replica; no stage or group spans nodes; segments cover layers `[0, L)` in order, embed in the first, head in exactly one; `forbid node` respected; with the spanning filter, every candidate computes on every node.
- Segments: one-node plans give one segment equal to the plan; concatenating `segment_plan`s gives back the replica's units.
- Prediction: a one-node plan predicts bit-identically through `predict_cluster` and `predict`; with zero-latency, infinite-bandwidth links it equals the sum of segment times; more latency or less bandwidth on any link never raises predicted tok/s.
- Membership and router, over random sequences of requests, faults and recoveries in simulated time: every request ends exactly once; none is sent to a replica on a `Down` node; every in-flight request on a faulted node ends with an error within the detection bound; no request on a healthy replica fails; a request whose prefix a healthy replica holds goes to it; a prefix fetch happens only when predicted cheaper than recompute; a restored node rejoins.
- Frames: the moved frame code writes byte-identical frames to today's `remote_backend` (golden bytes); any split or delay of the byte stream yields the same frames; a cut stream is an error, never a frame.
- Replanning after node loss (ADR 0042 drain and restart): the new plan is valid on the remaining nodes or is the safe fallback; its migration ends in the target plan.

**Parity (rule 2)** on CPU-only simulated nodes in CI: a two-node split against the one-node run of the same builder, top-5 logits within `tol_batch`, at the first, middle and last layer boundary, with the head on either node; greedy tokens equal outside ADR 0043's near-tie bound.

## Build order

Phase 6 code stays on branches until the regression-gate box (`bench-regression-gate`) has run on main, so that gate's A/B stays attributable to the code it judged.

| # | Tree node | Delivers | Checked by | Runs |
|---|---|---|---|---|
| 0 | `bench-235b-stream-predict` | The 235B streamed on one 3090 box under the 64 GB balloon, predicted against measured (the named risk) | Session README with `e_pp`, `e_tg` | One box; no cluster code; any time |
| 1 | `cluster-transport` | Transport trait, `MemTransport` with faults, `TlsTransport` with trust anchors and measured socket buffers, frames moved into `transport`, `RemoteBackend` over a transport; its metrics, panel row and alert | CI: golden-byte frame test, frame property tests, unknown and wrong certificates refused, pinned self-signed and CA anchors both accepted, effective buffer sizes read back, loopback TLS round trip, `RemoteBackend` E9 tests over `MemTransport`; Phase 5 parity and golden tests green | CI only; crates `transport`, `tetraeder` |
| 2 | `cluster-planner` | `Device.node`, cluster fixtures, validator rules, `segments`, `forbid node`, probe curves, `predict_cluster`, `enumerate_cluster` with the spanning filter | CI: plan and prediction property tests on every cluster fixture | CI only; crates `profiler`, `plan`, `costmodel`, `planner`; parallel with 1 |
| 3 | `cluster-discovery` | `cluster.toml`, `tetraeder node` daemon, membership state machine, heartbeats, `cluster cert`, `cluster profile` and `cluster probe`; its metrics, panel row and alerts | CI: membership property tests under every fault; two daemons on loopback TLS profile and probe each other | CI only; crates `control`, `tetraeder`; after 1 |
| 4 | `cluster-pipeline-nodes` | Hidden-state input and output in the builders, `segment_plan` workers, `PipelineBackend`, pipelined prefill across segments; its metrics, panel row and alert | CI: logit parity two-node against one-node (rule 2) on CPU nodes | CI only; crates `model`, `scheduler`, `tetraeder`; after 2 and 3 |
| 5 | `cluster-replicas` | `tetraeder replica`, `RemoteEngine`, router with prefix affinity, prefix fetch priced against recompute; its metrics, panel row and alert; with it `cluster-failover`, `cluster-join-leave`, `cluster-remote-kv` | CI: Case F on simulated nodes, router property tests, a prefix computed on one replica is fetched or recomputed on the other as priced | CI only; crates `gateway`, `control`, `scheduler`, `tetraeder`; after 3, parallel with 4 (shared `serve` wiring lands last) |
| 6 | `bench-phase6-rehearsal` | First two-box session: certificates, `cluster.toml` upload, profile, probe (autotuned and explicit buffers), parity on the 30B split over two nodes, Case F once | Session README; no verdict | Two boxes, about 1.5 h; after 0, 4, 5, `cluster-mtls` and the owner's answers |
| 7 | `bench-phase6-exit` | The exit test above | `verdict.json` | Two boxes, one or two sessions under 3 h each; after 6 |

`cluster-experts-nodes` is parked after step 7.

## Consequences

- One new concept (node) plus a trait with two implementations; replicas, segments, hops and remote KV reuse the engine interface, the plan IR, the worker, the frame protocol, the snapshot and the curve machinery. The worker and the builders know nothing of nodes beyond hidden rows in and out.
- A remote replica costs a few bytes per token on the wire, because sampling stays with its scheduler on its leader.
- Single-node plans, fixtures and predictions are unchanged; `Tier::Remote` becomes redundant and is a deletion candidate.
- Predictions over a WAN are only as good as the probe curves; a link that drifts during a session makes it unclean rather than wrong.
- On the WAN, two nodes are expected to tie one streaming node on decode and win on prefill (estimates above); the value of two nodes there is fitting the model in RAM, not decode speed.
- The head is a single point of failure for the gateway; acceptable for two to three nodes and revisited with Phase 7.

## Owner answers (2026-11-23)

1. Exit hardware: a Vast WAN pair (option a); a same-host pair runs in the rehearsal as an unjudged point; the homelab later.
2. Certificates: pinned self-signed per node (option a).
3. Node size: the 64 GB budget is emulated with `EMULATE_RAM_GB`, recorded as a caveat as M1 did.
4. mDNS: out of M3; `docs/plan.md` feature row 17 reads "static list in Phase 6, mDNS later".
5. Scope cuts confirmed: no highly available gateway in M3; experts across nodes parked until after the exit.

## Questions as asked

1. **Exit hardware.** (a) Two Vast boxes as specified, a low-RTT pair if offered, else WAN (recommended: available now, within caps, and the 15% rule judges the prediction, which prices the link); (b) your Linux host and Windows desktop on your LAN, the real homelab (needs a 24 GB-class GPU in each and the 235B on both disks; the Windows node then runs pipeline segments only, ADR 0008); (c) two containers on one Vast host (0.23 ms, but shared host memory and PCIe make it less like two machines). Recommendation: (a) for the verdict, (b) as a later confirmation.
2. **Certificates.** (a) Each node generates a self-signed certificate and `cluster.toml` pins them (recommended: no CA to run, adding a node is one line on every node); (b) an own small CA from `tetraeder cluster ca` (rcgen), one anchor for all; (c) step-ca, as plan.md names it (renewal and revocation, one more service). The code is the same for all three (trust anchors); only tooling differs.
3. **Node size by RAM budget.** The 64 GB budget emulates a homelab node on boxes that may have more RAM (as the Phase 2 exit runs did). The alternative, boxes whose real cgroup limit is between 64 and 110 GB, narrows the offer. Recommendation: emulate, and record the real limit.
4. **mDNS.** Static lists are forced on Vast; mDNS only helps a homelab LAN and cannot be tested on Vast. Recommendation: out of M3. If accepted, `docs/plan.md` feature row 17 changes to "static list in Phase 6, mDNS later" in the commit that records the answer.
5. **Other scope cuts to confirm:** no highly available gateway in M3; experts across nodes parked until after the exit.

## History

One bullet per step: its date and title, what changed against the text above, why.

- 2026-11-23, "Step 1, transport (`cluster-transport`)": as built. The decisions above stood; details the interfaces note left open:
  - **Trait.** `Transport` gained `local()` (this node's id, the `source` of the link metrics) and `shutdown()` (ends a blocked `accept`; a test or a daemon needs a way out). `Purpose` also derives `Hash` (key of the duplicate check). `Switchboard::new()` is the same as `MemTransport::switchboard()`. `TlsTransport` adds `local_addr()` (port 0 in tests) and `link_buffers()`; `Anchors`, `node_id_of` and `generate_self_signed` are public for the daemon of step 3.
  - **Handshake.** TLS 1.3 reports a refused client certificate to the dialer only with its first read, so `connect` also sends a preface frame (the `Purpose`, which the acceptor needs anyway) and waits for the acceptor's answer (`ok` or a reason). Identity is the URI SAN `tetraeder-node:<id>` (ids of letters, digits, `.`, `_`, `-`); the server name is not used. Reasons counted in `tetraeder.cluster.tls.rejected`: `untrusted`, `unknown_node`, `no_identity`, `wrong_peer`, `duplicate`, `bad_hello`, `handshake`. Pinned self-signed and CA anchors go through one verifier (rustls-webpki path building against the anchors, no name check). Node certificates are ECDSA P-256, valid 2024 to 2044.
  - **Duplicates.** "A second connection claiming a connected id" means the same `(peer id, purpose)` while the first is open; it ends when either half fails or is dropped, or `close` runs. A crashed dialer whose connection the acceptor has not noticed yet is refused until it does (the next read or write on it fails, or membership calls `close`).
  - **Sockets.** `SocketBuffers::Auto` sets nothing; `Explicit` sets both buffers before `connect` and `listen`. Every connection's effective sizes (`getsockopt`; Linux reports twice the request and caps it) are logged at `info` and kept (`link_buffers`). Nothing is hard-coded; the probe of step 3 picks the setting per link.
  - **Blocking halves.** The transport owns a two-thread tokio runtime and the halves block on it, so they work from plain threads; they must not be used inside another runtime's async code. A peer that vanishes without `close_notify` reads as end of input; a frame cut in the middle is still caught by `read_frame` as truncated.
  - **`MemTransport` faults.** `Close` resets every open connection between the two nodes on both sides and refuses new connections from `from` to `to` until the link is set to `None`; `Blackhole` drops what `from` writes toward `to` for good; `Slow` delays each write by `alpha + len / bytes_per_s` after the previous one. Dropping a half is a FIN. Duplicate connections are refused as in TLS.
  - **Frames and `RemoteBackend`.** `MAX_FRAME`, `Frame`, `encode`, `encode_raw`, `write_frame`, `read_frame` and `parse_header` moved to `transport::frame` unchanged and are re-exported from `remote_backend`; golden bytes (`crates/tetraeder/tests/frame_golden.rs`) were recorded from the code before the move. `RemoteConfig.program` and `.args` became `endpoint: Endpoint` (`Child` or `Node`); `RemoteConfig::new` is unchanged and `RemoteConfig::node` is new. `Request::Decode` gained `trace`, written as the header field `traceparent` and left out when `None`, so old headers are byte-identical. The worker loop moved from the binary to `tetraeder::worker_loop` so tests can serve a backend over a transport; test faults return `Flow::Exit` and the binary exits. The `RemoteBackend` cases of `tests/worker.rs` run unchanged over a `MemTransport` worker in `tests/worker_mem.rs`.
  - **Metrics.** `tetraeder.link.bytes` counts plaintext bytes written by the sender (a sum over nodes counts each byte once), with the purpose of the connection (`control`, `request` for a replica connection, `hidden` for a segment, `kv` for prefix); `tetraeder.link.latency` is recorded per call by `RemoteBackend` over a node endpoint (`hidden` for `decode`, `kv` for snapshot and restore, `request` otherwise). The labels `event` and `logits` are set by the callers of steps 4 and 5 (`LinkMetrics::with_purpose`). Dashboard `deploy/grafana/dashboards/cluster.json`, rule `TetraederTlsRejected` in `deploy/alerts/cluster.yml`.
  - **CLI.** `tetraeder cluster cert --node <id> --out <dir> [--force]` (the note says `--id`; the brief and this step say `--node`) writes `key.pem` (mode 0600 on Unix, owner-only ACL through `icacls` on Windows, removed if that fails) and `cert.pem`, and prints paths only. Step 3 lists it too; it is done here.
  - **New dependencies** (transport): `rustls-webpki` (chain check against the anchors; already in the lock through rustls), `x509-parser` (the id in the SAN), `socket2` (read the buffers back, bind with explicit buffers), `opentelemetry` and `tracing`; `rcgen` gains its `pem` feature. (tetraeder): `tetraeder-transport` and `tracing-opentelemetry` (the `traceparent` of the current span).
