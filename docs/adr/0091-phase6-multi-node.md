# 0091. Phase 6: nodes, the transport trait, plans across nodes, replicas and failover

Status: Proposed (2026-11-23). Phase 6, milestone M3. Builds on ADR 0023 (TCP+TLS chosen), 0069 (worker process per box, `RemoteBackend`), 0070 (shapes, measured curves), 0077 (pipelined prefill), 0008 (Linux-only expert parallel across nodes), 0042 (drain-and-restart replanning). Written before any Phase 6 code (rule 1), as ADR 0068 did for Phase 5. Interfaces: [`docs/notes/phase6-interfaces.md`](../notes/phase6-interfaces.md). Tree nodes: track `cluster`.

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

## Context

Phase 6 (`docs/plan.md`): transport, discovery, mTLS, pipeline and experts across nodes, remote KV, replicas, failover. Exit test: a model larger than any single node runs within 15% of prediction; killing a node with a replica keeps new requests flowing, and in-flight requests on it fail cleanly.

Constraints that shape the design:
- **C1 Transport.** TCP with TLS 1.3, behind a trait (ADR 0023). Per hop on one host: 0.41 to 0.46 ms at 8 KB. WAN (90 ms): a 1 MB echo takes 274 to 300 ms against about 145 ms from `RTT + bytes / bandwidth`, so alpha and beta alone miss by 2x at bulk sizes (untuned TCP buffers).
- **C2 Vast networking.** Boxes sit behind NAT with mapped ports (`PUBLIC_IPADDR`, `VAST_TCP_PORT_*`, `scripts/transport_wan_bench.sh`); addresses are known only after `create`; no two hosts on one LAN were on offer (ADR 0023); containers lack `NET_ADMIN`. Multicast does not cross that.
- **C3 What one node can already execute.** A stage of one GPU with experts in RAM runs on the streaming builders (`Sched`); stages of more than one GPU must be VRAM resident (ADR 0069, `multi_gpu_cpu_compute` false in phases 5 and 6). Tensor parallel never crosses nodes (ADR 0008).
- **C4 Fault isolation exists per box.** `RemoteBackend` already fails in-flight requests on a dead or hung worker, marks lost sequences, restarts and readmits (ADR 0069 "Failure handling"); the scheduler's `Failed` path fails exactly the affected requests.
- **C5 Snapshots are plan independent.** `seq_snapshot` returns the one-rank layout whatever the shape (ADR 0069), so KV bytes can move between any two backends of the same model and KV format.
- **C6** Single-node plans, fixtures and predictions stay bit-identical (no flag day). **C7** Rules 4 to 6: simulated fixtures and faults first, property tests on plans, metrics, panel and alert per component. **C8** Every node is yours or explicitly trusted (plan.md non-goals); llama.cpp's RPC is plaintext and is not used.

## Decision

Phase 6 adds one concept, the **node**, and expresses everything else with what exists: a cross-node plan is a chain of single-node sub-plans (segments), each run by today's worker on its node, joined by network hops that the cost model prices from curves measured through the real transport.

### Nodes and the transport trait

- **Node** = one box running `tetraeder node`: a long-lived daemon that listens on one TLS port, answers heartbeats, reports its hardware graph and starts `tetraeder worker` children for its assigned segments over stdio, as `serve --worker` does today (ADR 0069). It relays worker frames between the network connection and the child's pipes, so a sticky CUDA error kills only the child and the daemon restarts it (C4). One process supervises, one computes: the daemon is forced because something must outlive the worker and accept remote starts.
- **Head** = the node whose `tetraeder serve --cluster cluster.toml` runs the gateway, the router, one scheduler per replica and the prefix cache. It reaches every segment, its own node's included, through the transport (loopback costs 0.12 ms per hop, ADR 0023), so there is one code path.
- **Transport trait** (`crates/transport`): `connect(peer, purpose)` and `accept()` return an authenticated peer id and a `Read` half plus a `Write` half. Today's frame format (`u32` length, `u32` header length, JSON header, raw payload; `remote_backend`) moves into `transport` unchanged and runs over those halves, so `RemoteBackend` keeps its threads, deadlines and supervision and only gains a second way to open its pipes. Two implementations: `TlsTransport` (TCP+TLS 1.3, rustls, `TCP_NODELAY`, send and receive buffers 8 MiB, which covers the bandwidth-delay product of the measured WAN path, 90 ms x 50 MiB/s = 4.5 MB) and `MemTransport` (in-process pipes with a fault script, for simulated nodes in CI). QUIC stays a later implementation of the same trait.
- **One connection per purpose** (`control`, `segment <n>`, `experts <layers>`): a 32 MiB prefill hop must not delay a heartbeat or an expert call behind it on one TCP stream.
- **Trace context** crosses nodes as a `traceparent` field in the frame header.

### Identity and mTLS

- Each node has a key pair and a certificate whose subject alternative name is `tetraeder-node:<id>`. Identity is the node id in the certificate, never an IP or DNS name, because Vast addresses are NAT mappings known only after `create` (C2).
- Both sides verify the peer against the **trust anchors** listed in `cluster.toml`, then require the id from the certificate to be a configured node. The anchors can be each node's self-signed certificate (pinning) or one cluster CA certificate; rustls' verifier treats both alike, so the provisioning question below changes tooling, not code.
- A handshake that fails, a certificate without a configured id, or a node id already connected is refused and counted.

### Discovery and membership: static first

- `cluster.toml` lists the cluster name, the head, `node_down_after`, and per node its id, the address to dial, optional listen address, and its trust anchor. Static is forced on Vast (C2: no shared L2, addresses known only after `create`, mapped ports) and the list is needed anyway as the trust list (C8). mDNS could only fill in addresses on a homelab LAN; it is out of M3 and the plan.md feature map row 17 is amended to "static list in Phase 6, mDNS later".
- Membership is a pure state machine in `crates/control` (clock injected, as `replan.rs`): a node is `Up` after a successful hello and heartbeat, `Down` after a failed call (connection reset, malformed frame) or after `node_down_after` without any frame from it, `Draining` on `tetraeder node --drain`. Heartbeats run every `node_down_after / 4` on the control connection, answered by the daemon, not the worker, so a long step never looks like a dead node; a hung worker is caught by the existing step deadline (10x the predicted step, at least 5 s, now including the predicted hops).
- `node_down_after` = 2 s, one named value: above Linux's minimum TCP retransmission timeout (200 ms) with several retries and above the WAN p99 round trip (94 ms), below the 5 s step deadline floor, so a dead node is found before any request deadline fires.
- **Join and leave.** A node that becomes `Up` and whose replica loads and answers one probe step joins the router. A replica whose node goes `Down` leaves it. A change to a segmented replica (a pipeline over nodes) is a drain and restart of that replica with a new plan (ADR 0042); live resharding stays Phase 7.

### What crosses the wire

| Traffic | When | Bytes | Message |
|---|---|---|---|
| Control | always | small | `Hello`, `Heartbeat` and `Alive`, `Profile`, `Assign` (segment plan TOML, model `sha256`), `Release`, `Probe`, `Drain` |
| Backend calls | per scheduler call | header only | today's `Request` and `Reply` (`seq_reserve`, `seq_rm`, `seq_cp`, `decode`, `seq_snapshot`, `seq_restore`), unchanged |
| Pipeline hidden state | per step and segment boundary | `n_embd x 4` per token, every token of the step (16 KiB on Qwen3-235B) | `decode` whose payload is hidden rows instead of tokens; its reply payload is hidden rows or logits |
| Logits | per output row, only when the head is on a node other than the head process | `n_vocab x 4` (594 KiB on Qwen3) | today's `decode` reply |
| Expert call | per MoE layer with remote experts | normed hidden rows, top-k ids and weights out; partial sums back | `ExpertCall { layer }`, reply payload `n x n_embd` f32 |
| KV | prefix restore or move | snapshot bytes | today's `seq_snapshot` and `seq_restore` |

Every boundary is an ordinary frame; the payload is little-endian f32 (ids i32), as the logits payload is today. Hidden states stay F32 for parity (rule 2); F16 hops are a later lever behind a benchmark (rule 3).

### Plans across nodes

- **Hardware graph.** `Device` gains `node: Option<String>`; `None` is the local node, so every existing fixture and plan is unchanged (C6). `tetraeder cluster profile` merges the node graphs (ids prefixed `<node>/` for uniqueness; code reads the field), adds one `Network` link per node pair and the probe curves. `Tier::Remote` is then derivable from `node` and becomes a deletion candidate once old fixtures are migrated.
- **Plan IR: no new field.** A replica's stages may sit on devices of different nodes. Validator rules: every stage's devices are on one node (C3: no group spans nodes). Today a unit's compute device must be in its stage's group (`Issue::NotInGroup`); that rule gains exactly two allowed cases, both on another node: an expert whose home and compute are on another node is a remote expert, allowed only between Linux nodes (ADR 0008), and the head unit may compute on another node than the last stage. `Forbid` gains `node`, the node trust rule (plan.md hint `forbid node laptop`).
- **Segments** (derived, never stored): a maximal run of consecutive stages on one node, plus a head-only segment when the head is on another node. `plan::segments` derives them and `plan::segment_plan` cuts the single-node sub-plan each worker runs. A plan on one node has one segment equal to the plan. A segment without the embedding takes hidden rows in; one without the head returns hidden rows. The builders (dense and MoE, executor and `Sched` paths) gain that input and output; nothing else in a worker knows about nodes.
- **Head placement** is a planner choice priced like any other: with the head on the remote last segment, logits (594 KiB per output row) cross the link back; with the head in a trailing segment on the head node, 16 KiB do and the head node computes the output projection. On the measured WAN that matters: an 8 KB echo takes about 90 ms, a 1 MB echo 274 to 300 ms (C1).
- **Pipeline issue.** The head's `PipelineBackend: BatchBackend` calls segments in order (a star from the head, not a chain between workers): one hop per boundary per step, so two nodes with the head on the first pay one round trip per token, as plan.md "Parallel shapes across slow links" says. Pipelined prefill (ADR 0077) keeps several micro-batches in flight across segments, one connection each. A chain between workers saves hops only for three or more remote segments; later.
- **Experts across nodes** are remote experts: the stage's MoE layer computes its local experts, and a ggml custom op on the CPU backend sends the tokens routed to the remote node's experts as one `ExpertCall` and adds the returned partial sum. No executor change; the custom op's callback trampoline lives in `ggml-sys` (rule 7), and the expert server is a worker role on the remote node. Ids the server does not hold are -1 (patch 0004 semantics). One round trip per MoE layer, so it is a correctness and LAN feature: on the measured WAN it would cost about 5.4 s per token (ADR 0023) and the planner prices it out.

### Cost model: network links

- **Measured curves through the real transport.** `tetraeder cluster probe` sends frames of 4 KiB to 64 MiB (powers of 4, 20 repeats, median) between every node pair, both directions, over `TlsTransport` with the worker's frame code, each answered by an empty reply, and records one-way time as that time minus half the empty round trip. Stored as `CollectiveCurve { op: Copy, ranks: [<a>/host, <b>/host], comm: "tls" }`, interpolated as ADR 0070's curves. Without a curve the prior is the link's `alpha + bytes / beta`. Curves are measured, never fitted (ADR 0070), because C1 shows the formula off by 2x at 1 MB.
- **Prediction composes.** `predict_cluster` prices each segment with today's single-node `predict` on its node's calibrated graph and sub-plan, then adds hops: decode step `T = sum of segment times + sum of hop_time(boundary bytes) + hop_time(return bytes)`; prefill follows ADR 0077's pipelined schedule with the hop as the copy curve, serialized per boundary; a remote-expert layer adds `hop_time(dispatch) + hop_time(combine)` and the remote node's expert term. A one-node plan predicts bit-identically to `predict` (C6). Replicas add throughput; the simulator (`crates/costmodel/src/sim.rs`) covers routing under load.
- **Planner.** `enumerate_cluster` lists, per replica, the cut points that fit each node's memory (the dynamic program over layers, ADR 0070, with node capacity per stage), the head placement, and remote-expert options between Linux nodes; it never forms a group across nodes. Deterministic, before calibration, as ADR 0040.

### Replicas and failover

- **Replicas** are `Plan::replicas` (exists). The head runs one scheduler per replica; a router sends each new request to the `Up` replica with the fewest outstanding requests. Requests queued for a replica but not admitted move to another replica when it leaves.
- **Failing cleanly.** A request is in flight on a replica once it was admitted there. When the replica's node goes `Down` (or a segment call fails), the existing path runs (C4): the scheduler fails exactly its in-flight requests with `GenEvent::Error`; the gateway ends a stream with one error event and closes it, or answers 503 before headers; no token follows the error and no request ends as a normal `stop` or `length`. Each request ends exactly once. The replica's KV rows are released on the head's books, and its idle prefixes are dropped (`seq_lost`).
- **Detection bound.** A killed process resets its connection at once; a silent node is found within `node_down_after`; a hung worker within its step deadline. A `Down` transition closes every connection to that node (each `Conn` has a close handle usable from another thread), so a scheduler blocked in a `RemoteBackend` call to it gets `Failed` at once instead of waiting for the 5 s deadline floor. So an in-flight request on a killed or silent node ends within `node_down_after + 1 s` (the slack is one heartbeat interval plus scheduling).
- **Who restarts what.** On its node the daemon restarts a dead worker, with `RemoteConfig`'s backoff and cap (ADR 0069). The head never restarts remote processes: when the node is `Up` again it reconnects, sends `Assign`, waits for `Info` and one probe step, and the router takes the replica back. For a node endpoint, reaching the restart cap takes the replica out of the router until the node next becomes `Up`; it never exits the head (today's default `on_give_up` exits, which stays for `serve --worker`).
- **Gateway** failure is not covered: the head is a single point of failure in M3.

### Remote KV, Phase 6 scope

The prefix cache stays in the head's RAM (and disk tier, ADR 0087) and becomes one store shared by every replica scheduler of the same KV format. A prefix computed on one replica restores into another through `seq_snapshot` and `seq_restore` over the transport (C5), so features 3 and 4's cross-node halves come from composition with no new part. Not in Phase 6: KV tiers on a remote node's RAM during attention, and KV moves between prefill and decode hardware (Phase 7, `multi-split-hardware`).

### Metrics, panel, alerts (rule 6)

One dashboard, `deploy/grafana/dashboards/cluster.json`, and one rules file, `deploy/alerts/cluster.yml`; each build step adds the rows and rules of the component it ships.

| Step | Metrics | Panel row | Alert |
|---|---|---|---|
| 1 transport | `tetraeder.link.bytes` (counter) and `tetraeder.link.latency` (histogram, per frame round trip) from the catalog, attributes `source` and `destination` node, `transport` (`tls`, `mem`), `tetraeder.purpose` (`control`, `hidden`, `logits`, `expert`, `kv`); `tetraeder.cluster.tls.rejected` (counter, `reason`) | "Links: bytes and latency per purpose" | `TetraederTlsRejected`: any rejected handshake in 10 minutes (an unknown or misconfigured peer) |
| 2 planner | none at run time (offline); `tetraeder.plan.residual` gains a `segment` attribute in step 4 | | |
| 3 membership | `tetraeder.cluster.node.up` (gauge, node), `tetraeder.cluster.heartbeat.rtt` (histogram, node), `tetraeder.cluster.node.transitions` (counter, node, `to`) | "Nodes: up, heartbeat RTT, transitions" | `TetraederNodeDown`: a configured node not up for 1 minute; `TetraederLinkSlow`: heartbeat RTT median above 2x the probed small-frame time for 10 minutes (predictions no longer hold) |
| 4 pipeline | `tetraeder.plan.residual` with `segment`; hop time per boundary as `tetraeder.link.latency` with purpose `hidden` or `logits` | "Pipeline: predicted and measured time per segment and hop" | `TetraederSegmentDrift`: a segment's median residual beyond 0.15 (the exit-test line) for 30 minutes |
| 5 replicas | `tetraeder.router.requests` (counter, replica, `result` = ok, failed_in_flight, moved), `tetraeder.router.replicas.up` (gauge) | "Router: requests per replica and result, replicas up" | `TetraederReplicaOut`: fewer replicas up than the plan has for 5 minutes; failed requests also feed the existing `TetraederRequestsFailing` and SLO burn alerts |
| 6 remote experts | `tetraeder.link.latency` with purpose `expert`, `tetraeder.expert.remote.calls` (counter, layer) | "Remote experts: calls and round trip per layer" | `TetraederLinkSlow` covers the link; no new rule |

### Security boundaries

- Only configured node ids with a certificate from a configured trust anchor connect; everything between nodes is mutually authenticated TLS 1.3.
- A node sees the activations, KV and therefore the content of the requests its segments serve. `forbid node` keeps units, and with them that content, off a node; that is the trust rule.
- The wire carries no file paths, code or commands: `Assign` names a model by `sha256`, and the node maps it to its own local file or refuses. API clients authenticate at the gateway (Phase 9), never on the node port.
- Keys never leave the node that generated them; `cluster.toml` holds only certificates.

### Out of scope for M3

Tensor parallel across nodes (ADR 0008); NCCL or RDMA between nodes; QUIC and iroh (NAT traversal); mDNS; live resharding and replica scaling (Phase 7); prefill and decode on different machines (Phase 7); remote KV tiers during attention; a highly available head; worker-to-worker pipeline chains; certificate rotation and revocation beyond editing `cluster.toml`.

## Exit test (fixed before any measurement)

Two cases, each on one pair of boxes in one session; they may run in separate sessions on different pairs (Case F first, it needs no 142 GB download), so neither session runs into `max_box_hours` (3 h). Box spec: two Linux boxes, one RTX 3090 each (24 GB; class `cuda-default`, sm_86), CPU quota 8 or more, 230 GB disk, within 0.90 USD/h all-in each, both `cuda: ok`. Pair choice, in this order: two hosts with a small-frame TCP round trip under 5 ms; else any two hosts (WAN). Each box records ADR 0068's `env.json` plus its node id, public address, port mapping and the cgroup memory limit.

### Case S: a model larger than any single node

- **Model.** Qwen3-235B-A22B Q4_K_M (142 GB; 94 layers, 128 experts, top 8). **Node size.** Each node has 24 GB VRAM and a RAM budget of 64 GB (`EMULATE_RAM_GB=64`, the balloon of `scripts/exit_run.sh` that the Phase 2 exit runs of the 235B used, `docs/sessions.md`; a run whose balloon died does not count), so a node holds 88 GB < 142 GB and two hold 176 GB. "Larger than any single node" means: weight bytes exceed every node's VRAM plus RAM budget, so no one node holds the model without its disk tier.
- **Candidates.** `tetraeder-plan candidates --cluster` on the merged hardware graph before calibration; the judged plan is the planner's analytic pick. Each segment is one GPU plus RAM experts on the streaming builder (C3).
- **Calibration, separate from the judged run** (ADR 0068: no anchor equals a candidate). Per node, anchors of its segment truncated to the first quarter and the first half of its layer span (hidden input from a fixed-seed tensor, no head), pp2048 and tg128, two rounds; they fit the node's single-node parameters as ADR 0040 and 0056 fit them for the streaming builder. Then `tetraeder cluster probe` both directions. Named risk: ADR 0056 found single-GPU streaming candidates well above 15% on its fixtures (`20261103`), and both segments run on that path. The rehearsal (build order 7) measures the segment error first; if it is above 0.15, a cost-model fix with its own ADR comes before the exit session, and these rules do not change. Then `tetraeder-plan predict --cluster` writes `predict-cluster.json` (predicted `pp_tps`, `tg_tps`, `T`, git commit, `sha256` of the merged graph and the probe file) before the first judged run; `verdict.json` records its `sha256`, and a prediction written after a judged run makes the case invalid.
- **Workload and rounds.** ADR 0068's: one sequence, pp2048 then tg128, `-r 1` per call, three rounds. Each round also runs the single-node reference on the head node (the same model streamed from its disk under the same 88 GB, report only), interleaved, order reversed in round 2.
- **Within 15%.** `e_pp = pp_pred / pp_meas - 1`, `e_tg = tg_pred / tg_meas - 1` on the medians of clean rounds. Pass when `|e_pp| <= 0.15` and `|e_tg| <= 0.15`. A pick predicted feasible that fails on the boxes fails the case.
- **Clean.** ADR 0068's rules on both boxes (throttle sampler on both GPUs, idle gate, co-tenant guard, power limit at default), A/A of the judged plan within band `b` on `T`, and the link: the probe repeated at session end must agree with the start probe within `b` at 16 KiB and at 32 MiB; otherwise the rounds are rerun after a new probe and prediction. `b` = the spec band 0.08 (`bench/spec/phase0.toml`), since no two-node noise class exists; recorded as `band_source`.
- **Paired ratio** for `docs/features.md`: `T_single_node / T_two_nodes` on the same boxes and rounds (report only, not gated). The llama.cpp RPC backend is not a baseline (plaintext, C8).

### Case F: failover with replicas

- **Model and plan.** Qwen3-30B-A3B Q4_K_M (18.6 GB) as two replicas, R_A on node A (also the head) and R_B on node B, F16 KV, context 4096.
- **Load.** Closed loop through the gateway, streaming chat, concurrency 8, prompt 512 tokens, output 256, for 180 s per trial.
- **Faults**, on node B at t = 60 s, three trials each, B restored between trials: (1) `kill -9` of B's worker (the daemon restarts it), (2) `kill -9` of B's daemon and worker (connection reset), (3) `SIGSTOP` of B's daemon and worker, `SIGCONT` after 60 s (silent node; the stand-in for a partition, since containers lack `NET_ADMIN`).
- **Pass, per trial:** every request in flight on R_B at the fault ends with one clean error (above) within `node_down_after + 1 s` = 3 s; no request on R_A fails; every request submitted from 3 s after the fault on succeeds; requests queued for R_B move and succeed; each request ends exactly once (client and gateway counts agree); B rejoins and serves requests within 60 s of being restored (trial 1 without any manual action). No speed is judged.

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
- Plans on every cluster fixture: no device or node budget exceeded (RAM budget included); every unit computed exactly once per replica; no stage or group spans nodes; segments cover layers `[0, L)` in order, embed in the first, head in exactly one; remote experts only between Linux nodes; `forbid node` respected.
- Segments: one-node plans give one segment equal to the plan; concatenating `segment_plan`s gives back the replica's units.
- Prediction: a one-node plan predicts bit-identically through `predict_cluster` and `predict`; with zero-latency, infinite-bandwidth links it equals the sum of segment times; more latency or less bandwidth on any link never raises predicted tok/s.
- Membership and router, over random sequences of requests, faults and recoveries in simulated time: every request ends exactly once; none is admitted to a replica on a `Down` node; every in-flight request on a faulted node ends with an error within the detection bound; no request on a healthy replica fails; KV accounting returns to zero; a restored node rejoins.
- Frames: any split or delay of the byte stream yields the same frames; a cut stream is an error, never a frame.
- Replanning after node loss (ADR 0042 drain and restart): the new plan is valid on the remaining nodes or is the safe fallback; its migration ends in the target plan.

**Parity (rule 2)** on CPU-only simulated nodes in CI: a two-node split against the one-node run of the same builder, top-5 logits within `tol_batch`, at the first, middle and last layer boundary, with the head on either node; remote experts against local, same bound; greedy tokens equal outside ADR 0043's near-tie bound.

## Build order

| # | Tree node | Delivers | Checked by | Runs |
|---|---|---|---|---|
| 1 | `cluster-transport` | Transport trait, `MemTransport` with faults, `TlsTransport` with mTLS trust anchors, frames moved into `transport`, `RemoteBackend` over a transport; its metrics, panel row and alert (table above) | CI: frame property tests, unknown and wrong certificates refused, both a pinned self-signed anchor and a CA anchor accepted, loopback TLS round trip, `RemoteBackend` E9 tests over `MemTransport` | CI only; crates `transport`, `tetraeder` |
| 2 | `cluster-planner` | `Device.node`, cluster fixtures, validator rules, `segments`, `forbid node`, probe curves, `predict_cluster`, `enumerate_cluster` | CI: plan and prediction property tests on every cluster fixture | CI only; crates `profiler`, `plan`, `costmodel`, `planner`; parallel with 1 |
| 3 | `cluster-discovery` | `cluster.toml`, `tetraeder node` daemon, membership state machine, heartbeats, `cluster profile` and `cluster probe`; its metrics, panel row and alerts | CI: membership property tests under every fault; two daemons on loopback TLS profile and probe each other | CI only; crates `control`, `tetraeder`; after 1 |
| 4 | `cluster-pipeline-nodes` | Hidden-state input and output in the builders, `segment_plan` workers, `PipelineBackend`, pipelined prefill across segments; its metrics, panel row and alert | CI: logit parity two-node against one-node (rule 2) on CPU nodes | CI only; crates `model`, `scheduler`, `tetraeder`; after 2 and 3 |
| 5 | `cluster-replicas` | Router, one scheduler per replica, shared prefix store, its metrics, panel row and alert; with it `cluster-failover`, `cluster-join-leave`, `cluster-remote-kv` | CI: Case F on simulated nodes, router property tests, a prefix computed on one replica restores on the other | CI only; crates `gateway`, `control`, `scheduler`, `tetraeder`; after 3, parallel with 4 (shared `serve` wiring lands last) |
| 6 | `cluster-experts-nodes` | Remote expert custom op, expert server role, its cost term and panel row | CI: remote-expert parity on CPU nodes; the cost term's property tests | CI only; crates `ggml-sys`, `model`, `costmodel`; after 4 |
| 7 | `bench-phase6-rehearsal` | First two-box session: profile, probe, parity on the 30B split over two nodes, Case F once, remote expert round trip against the probe curve | Session README; no verdict | Two boxes, about 1.5 h; after 5 and 6 and the owner's answers |
| 8 | `bench-phase6-exit` | The exit test above | `verdict.json` | Two boxes, one or two sessions under 3 h each; after 7 |

## Consequences

- One new concept (node) plus a trait with two implementations; segments, hops, remote experts and remote KV reuse the plan IR, the worker, the frame protocol, the snapshot and the curve machinery. The worker and the builders know nothing of nodes beyond hidden rows in and out.
- Single-node plans, fixtures and predictions are unchanged; `Tier::Remote` becomes redundant and is a deletion candidate.
- Predictions over a WAN are only as good as the probe curves; a link that drifts during a session makes it unclean rather than wrong.
- The head is a single point of failure, and every hop goes through it; both are acceptable for two to three nodes and revisited with Phase 7.
- Cross-node experts are built and parity-tested but not judged; on Vast's WAN they cannot pay.

## Open for the owner

1. **Exit hardware.** (a) Two Vast boxes as specified, a low-RTT pair if offered, else WAN (recommended: available now, within caps, and the 15% rule judges the prediction, which prices the link); (b) your Linux host and Windows desktop on your LAN, the real homelab (needs a 24 GB-class GPU in each and the 235B on both disks; the Windows node then runs pipeline segments only, ADR 0008); (c) two containers on one Vast host (0.23 ms, but shared host memory and PCIe make it less like two machines). Recommendation: (a) for the verdict, (b) as a later confirmation.
2. **Certificates.** (a) Each node generates a self-signed certificate and `cluster.toml` pins them (recommended: no CA to run, adding a node is one line on every node); (b) an own small CA from `tetraeder cluster ca` (rcgen), one anchor for all; (c) step-ca, as plan.md names it (renewal and revocation, one more service). The code is the same for all three (trust anchors); only tooling differs.
3. **Node size by RAM budget.** The 64 GB budget emulates a homelab node on boxes that may have more RAM (precedent ADR 0045). The alternative, boxes whose real cgroup limit is between 64 and 110 GB, narrows the offer. Recommendation: emulate, and record the real limit.
4. **Scope amendments.** Confirm: mDNS out of M3 (feature 17), gateway high availability out of M3, cross-node experts parity-tested but not judged.
