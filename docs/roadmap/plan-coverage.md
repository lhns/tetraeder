# Plan coverage of the tech tree (2026-11-21)

Every planned item from `docs/plan.md`, `docs/m2-plan.md`, `docs/m1-open.md` and the Proposed ADRs, with its home in `docs/roadmap/nodes.yaml`: an existing node, "fold into <id>" (an item inside a node's `folded` list or notes), or a new node. Built work maps to history; only unbuilt work gets a forward node. Written once to bring the tree up to the plan; after this, the tree is the record.

Items marked "checked" were confirmed done or not done in the code or an ADR, not from the phase number.

## plan.md: feature map (52 rows)

| # | Feature (phase) | Home |
|---|---|---|
| 1 | Tiered weights across VRAM, RAM, NVMe (2) | existing `memory-tiers` |
| 2 | MoE expert placement and streaming (2) | existing `memory-tiers`, `memory-prefetch`, `memory-io-service` |
| 3 | KV cache tiers (4, remote in 6) | VRAM: existing `serving-kv-layout`; RAM snapshots: existing `serving-prefix-cache`; live KV in RAM or on NVMe (checked: `crates/kvcache` has no RAM or disk tier): new node `memory-kv-tiers`; remote: new node `cluster-remote-kv` |
| 4 | Prefix caching shared across nodes (4, cross-node in 6) | 4: existing `serving-prefix-cache`; cross-node: new node `cluster-shared-prefix` |
| 5 | Weight and KV quantization per unit (1, 2) | existing `planner-quality`, `memory-conversion`, `memory-low-bit-kv` |
| 6 | Fast model loading (2) | existing `memory-tiers` (aligned direct reads), `memory-io-service` |
| 7 | Tensor parallel (5) | existing `multi-sharded`, `planner-parallel-shapes`; verdict in `bench-v2-verdict`, `bench-v4a-verdict` |
| 8 | Pipeline parallel (1 in a node, 6 across nodes) | 1: existing `engine-libllama`, `multi-sharded`; across nodes: new node `cluster-pipeline-nodes` |
| 9 | Expert parallel (5, across nodes in 6) | 5: existing `multi-sharded`; across nodes: new node `cluster-experts-nodes` |
| 10 | Context parallel (9) | new node `multi-context-parallel` (with the log-sum-exp attention patch) |
| 11 | Data parallel replicas (6) | new node `cluster-replicas` |
| 12 | CPU as a compute device (1, free in 5) | existing `memory-pair-groups` (CPU experts), `multi-executor` (experts only, ADR 0069) |
| 13 | Prefill and decode on different hardware (7) | new node `multi-split-hardware` |
| 14 | Mixed GPU sizes and vendors (sizes 1, vendors in 8) | sizes: existing `engine-libllama`; vendors: fold into `engine-backend-parity` |
| 15 | Interconnect-aware placement, PCIe topology (3a) | existing `planner-core` (checked: `crates/profiler/src/pcie.rs`, commit 883befe) |
| 16 | Multiple nodes (6) | new node `cluster-transport` |
| 17 | Automatic node discovery (6) | new node `cluster-discovery` |
| 18 | Live node join and leave (6) | new node `cluster-join-leave` |
| 19 | Fault tolerance (6) | new node `cluster-failover` |
| 20 | Security, mTLS and trust rules (6) | new node `cluster-mtls` |
| 21 | Automatic hardware discovery (0) | existing `ops-foundations` (checked: `crates/profiler/src/discover.rs`, commit 68c6b4b) |
| 22 | Measured-cost planner (3a) | existing `planner-core`, `planner-exit` |
| 23 | Objective as limits plus a front position (3a) | existing `planner-core` (ADR 0025) |
| 24 | Live replanning (3a drain-and-restart, 7 live) | 3a: existing `planner-replan`; live: new node `planner-live-replan` |
| 25 | Continuous batching (4) | existing `serving-batching` |
| 26 | Chunked prefill (4) | existing `serving-batching` (token budget, ADR 0065) |
| 27 | Speculative decoding: n-gram, draft, MTP (4) | n-gram and draft: existing `serving-speculative`, `serving-spec-batching`; MTP (checked: no MTP code in `crates/`; parked 2026-11-15): new node `serving-mtp-drafts` |
| 28 | Optimized kernels, CUDA graphs (1, 2, 4) | existing `engine-cuda-graphs`, `engine-iqk-kernels`, `engine-isa-dispatch` |
| 29 | Several models at once, on demand (9) | new node `serving-multi-model` |
| 30 | LoRA hot swap, batched multi-LoRA (9) | new node `serving-multi-lora` (with the batched multi-LoRA patch) |
| 31 | Priorities, quotas, auth (9) | new node `serving-auth-quotas` |
| 32 | OpenAI and Anthropic API (3b) | existing `serving-gateway` |
| 33 | Tool calling, structured output (3b) | existing `serving-gateway`, `serving-tool-calls` |
| 34 | Multimodal, embeddings (9) | new node `serving-multimodal` |
| 35 | Metrics, traces, logs (0, every phase) | existing `ops-telemetry` |
| 36 | No forced telemetry (0) | existing `ops-foundations` (design rule, ADR 0012) |
| 37 | Windows and Linux, container image (1, packages in 9) | 1: existing `ops-ci`; packages and image: new node `ops-packages` |
| 38 | Live change of parallel layout (7) | new node `multi-live-reshard` |
| 39 | Live replica scaling (7) | new node `cluster-replica-scaling` |
| 40 | Separate layouts for prefill and decode (7) | new node `planner-split-layouts` |
| 41 | Bottleneck detection (3a) | existing `planner-detector` |
| 42 | Improvement advisor (3a, auto-apply in 7) | 3a report only: existing `planner-detector`, `planner-core`; auto-apply: new node `planner-advisor-auto` |
| 43 | Detailed profiling (1, 3a) | step timing and per-op calibration: existing `planner-core`; profiling mode with CUPTI or rocprof (checked: no CUPTI or rocprof code in `crates/`): new node `ops-profiling-mode` |
| 44 | Per-request cost accounting (3b) | existing `serving-gateway` (`crates/gateway/src/gateway/cost.rs`) |
| 45 | SLO burn-rate alerts (3a) | existing `planner-replan` (ADR 0042) |
| 46 | Process isolation per device (5) | existing `multi-executor` (worker process per box, ADR 0069) |
| 47 | Request cancellation and backpressure (3b) | existing `serving-gateway` |
| 48 | Quality profile per model (2, 3a) | existing `planner-quality` |
| 49 | Load-time conversion (2, safetensors with own loaders) | conversion: existing `memory-conversion`; safetensors (checked: no safetensors loader in `crates/`): new node `memory-safetensors` |
| 50 | 1 to 2 bit formats as planner choices (2) | checked: `crates/plan/src/format.rs` lists IQ1, IQ2, TQ and Q1_0, but conversion rejects formats that need an importance matrix (ADR 0047) and profile v1 measures none below IQ4_XS (ADR 0048); parked to Phase 9 (m2-plan): new node `planner-low-bit-formats` |
| 51 | Natively ternary families, BitNet (1 via libllama, 9 own builder) | 1: existing `engine-libllama`; own builder: new node `engine-bitnet-builder` |
| 52 | Self-speculative decoding (4) | checked: no low-bit draft copy in `crates/scheduler`; parked 2026-11-15: new node `serving-self-spec` |

## plan.md: phases 0 to 5, items outside the feature map

| Item | Home |
|---|---|
| Phase 0 baselines: llama.cpp (with its expert cache), ik_llama.cpp, KTransformers, Colibri | existing `bench-harness` |
| Phase 0 baseline: vLLM | checked: built as a report-only stage of `scripts/phase5_box.sh` (ADR 0068) but it never produced a number (skipped in every session: no AWQ or GPTQ checkpoint on the box): existing `bench-phase5-harness`; not counted as a measured baseline |
| Phase 0 baseline: Strata | checked: no Strata code or run; parked in m2-plan (no model both engines run): new node `bench-strata-pairing` |
| Phase 0 spike: two ggml backends in one process | moot for M1 (plan, Open decisions): fold into `engine-backend-parity` |
| Phase 0 spike: transport latency | localhost run: existing `ops-foundations` (ADR 0023); WAN run: new node `cluster-transport-wan` |
| Phase 0 spike: exemplars | existing `ops-foundations` (ADR 0022) |
| Patch queue rows "attention with log-sum-exp", "batched multi-LoRA" | fold into `multi-context-parallel`, `serving-multi-lora` |

## plan.md: phases 6 to 9

| Item | Home |
|---|---|
| Phase 6 transport chosen in Phase 0 | new node `cluster-owner-transport` (ADR 0023), after `cluster-transport-wan` |
| Phase 6 deliverables | the `cluster-*` nodes above (rows 3, 4, 8, 9, 11, 16 to 20) |
| Phase 6 exit test | new node `bench-phase6-exit` |
| M3: homelab cluster | new node `bench-m3-report` |
| Phase 7 deliverables | `planner-advisor-auto`, `cluster-replica-scaling`, `multi-live-reshard`, `multi-split-hardware`, `planner-split-layouts`, `planner-live-replan` |
| Phase 7 exit test | new node `bench-phase7-exit` |
| M4: self-tuning | new node `bench-m4-report` |
| Phase 8: own builders on HIP, Vulkan, Metal | new node `engine-backend-parity` (one node for the three) |
| Phase 8: GLM builder | new node `engine-glm-builder` |
| Phase 8: DeepSeek builder with MLA-aware KV | new node `engine-deepseek-builder` |
| Phase 8 exit test | new node `bench-phase8-exit` |
| Phase 9 deliverables | rows 10, 29, 30, 31, 34, 37, 50, 51 above |
| Phase 9 exit test | new node `bench-phase9-exit` |

## plan.md: risks and open decisions

| Item | Home |
|---|---|
| Risks: calendar time, cost model accuracy, ggml drift, libllama internals, CUDA graphs versus streaming, own executor, telemetry overhead, detector false positives, quality estimate | mitigations built: existing `ops-ci` (weekly rebase, nightly), `planner-core`, `planner-detector`, `planner-quality`, `engine-cuda-graphs`, `multi-executor` |
| Risk: parity across backends | fold into `engine-backend-parity` |
| Risk: hardware access for phases 5 to 7 | existing `ops-taxameter`; multi-node rentals: lane `cluster` |
| Risk: advisor harm | fold into `planner-advisor-auto` (done criterion: verify and roll back) |
| Risk: live reshard bugs | fold into `multi-live-reshard` (migration DAG property tests) |
| Open: KV layout | existing `serving-kv-layout` (decided: chunked) |
| Open: mixed vendors | fold into `engine-backend-parity` |
| Open: transport | new node `cluster-owner-transport` |
| Open: quality profile granularity | existing `planner-quality` (ADR 0048) |
| Open: trace backend (Jaeger or Tempo) | no node: an open decision that gates nothing; open call for the owner |
| Open: name check | existing `ops-foundations` (ADR 0010) |

## m2-plan.md

| Item | Home |
|---|---|
| Step 0 safeguards (trace copy, point-3 session profile) | existing `ops-box-driver` |
| C0 to C6, C1.5 (Phase 4 track) | existing `serving-batching`, `engine-dense-builder`, `serving-kv-layout`, `serving-moe-batch`, `serving-prefix-save`, `bench-phase4-exit` |
| Phase 4 deliverable: speculation in `serve` | existing `serving-spec-batching` |
| Phase 4 deliverable: self-speculative and MTP, parked | new nodes `serving-self-spec`, `serving-mtp-drafts`, after `serving-owner-drafters` |
| Review fixes from the C2b and C5 review (7 items) | existing `serving-batching` (checked: commit 8004383 in its `folded`) |
| c4 gap on CPU, unified pool against streams on GPU | existing `serving-c4-gap`; streams question settled by step shapes on GPU (`20261116pc`) |
| Dense builder CPU prompt kernels (repack or AMX path) | built: existing `engine-dense-builder` (ADR 0072, `DenseOptions::extra_bufts` default true); the pending AMX speed check: new node `engine-dense-amx` (experiment, CI) |
| `SAVE_MIN_TOKENS` for short prompts | fold into `serving-prefix-first` |
| Point-1 box result: FP8 KV baseline parked | fold into `bench-d1-session` |
| Third review follow-ups (5 items) | existing `serving-moe-batch` (checked: commit b76e299 in its `folded`) |
| Housekeeping: old remote branches (owner's call) | no node (a cleanup, no verdict); open call for the owner |
| Proposed ADRs waiting on a box: 0059 async stages, 0060 sampler against OTel | new node `bench-open-records` |
| E9 follow-ups: Executor panel, lost sequences free | existing `multi-executor` (checked: commit 67a01b3 in its `folded`) |
| E9 follow-up: sticky CUDA errors detected by message text | fold into `cluster-failover` |
| E1 note: no `ggml_backend_tensor_copy_async` binding | fold into `cluster-pipeline-nodes` (pipeline peer copies) |
| Engine pin b12426, D2 split, D2a reruns, pin A/B, CUDA confirmation | existing `engine-pin-b12426`, `bench-d2a-rerun`, `bench-pin-ab`, `bench-pin-confirm`, `bench-phase4-exit` |
| Speculative rerun (Qwen3.8-27B draft pair) | new node `bench-spec-rerun` (parked) |
| Phase 5 open 1: 4-GPU rehearsal, tp2-pp2, capture fix, cost model 5, D4 host | existing `bench-s3-rehearsal`, `multi-defect-capture`, `multi-prefill-no-capture`, `bench-s3f2`, `planner-cost-model-v5`, `ops-owner-v4a-cap` |
| Phase 5 open 1: one issuing thread per stage | superseded by ADR 0081 (prefill without capture): existing `multi-prefill-no-capture` |
| Phase 5 open 2: expert-parallel prefill over-predicted | new node `planner-ep-prefill-recheck` (next; likely closed by the capture fix) |
| Phase 5 open 3: thermal 0x20, size `RERUNS` and `BUDGET_MIN` | existing `bench-v2-verdict`, `bench-v4a-verdict` (session sizing) |
| Phase 5 open 4: V2, V4a, V4b | existing `bench-v2-verdict`, `bench-v4a-verdict`, `bench-v4b-verdict` |
| Phase 5 open 5: D2a probe, rule-change list | existing `bench-d2a-probe`, `bench-m2-report` |
| TTFT bar (ttft3 to ttft7) | existing `bench-ttft` |
| TTFT open: in-step saves in `save_ms`, not yet replayed | fold into `bench-open-records` (a stage of that one session; a metrics repair, no verdict of its own) |
| Idea: admit the longest prefix hit first | new node `serving-prefix-first` (parked) |
| Parked: D1 (48 GB box) | new node `bench-d1-session` |
| Track B first item: stream prefill misses | new node `memory-prefill-misses` (parked) |
| Track B study: Kinetic Flywheel miss choice; InferSSD as a baseline | fold into `memory-prefill-misses` |
| B0 replay | existing `memory-ram-pool` (abandoned) |
| B1 global RAM tier | existing `memory-ram-pool` (not built) |
| B2 VRAM fill | existing `planner-fill-tier`; its box runs against KT: fold into `bench-d1-session` |
| B3 polling rule | existing `bench-m1-point1` (closed, no rule change, `20261115p1`) |
| B4: ADR 0064 step 2, one graph per token | new node `engine-one-graph-per-token` (parked) |
| B4: tool-call repair | existing `serving-tool-calls` |
| B4: planner slot-count campaign | new node `planner-slot-campaign` (parked) |
| B4: `serve --plan` replanning part B | new node `planner-replan-part-b` (parked) |
| Track A and E housekeeping (m1-open edits, branches, worktrees, otel size cap, ADR statuses) | existing `ops-docs`, `ops-owner-accepts-adrs`; branches: open call for the owner |
| Order step 4, box D1 contents | fold into `bench-d1-session` |
| Order step 6, box D2 contents | existing `bench-phase4-exit`, `bench-ttft`; spec rerun: `bench-spec-rerun`; part B: `planner-replan-part-b` |

## m1-open.md

| Item | Home |
|---|---|
| 1 Speculative rerun | new node `bench-spec-rerun` |
| 2 Read wait under overlap: `-uf-w4`, `-w8`, `-poll0` arms, then delete `Reader` | fold into `bench-d1-session` |
| 3 Noise bands | existing `bench-noise-bands` |
| 4, 5 exit_run rc, proxy tool outcomes | existing `bench-phase3b-proxy` |
| 6 Builder slot-count candidates | new node `planner-slot-campaign` |
| 7 ik_llama quant types | existing `planner-quality` |
| 8 Low-bit KV (b) | existing `memory-low-bit-kv` |
| 8 Miss reads under the CPU hit group (a) | existing `memory-io-service` (ADR 0054) |
| 8 Frequency tiers (c) | existing `memory-ram-freq`, `memory-ram-pool` |
| 8 Message-boundary checkpoints | existing `serving-tool-calls` (skipped, ADR 0051) |
| 8 Tool-call repair | existing `serving-tool-calls` |
| 8 Recheck llama-server and planner baselines with the expert cache | existing `bench-phase4-exit` (tuned llama-server from the post-#30966 build) |
| 8 Strata pairing | new node `bench-strata-pairing` |
| 9 FP8 KV llama.cpp baseline | fold into `bench-d1-session` |
| 10 imatrix and IQ formats below 4 bits | new node `planner-low-bit-formats` |
| 10 Per-expert formats (typed slots) | new node `memory-typed-slots` (parked) |
| 10 `serve --plan` part B | new node `planner-replan-part-b` |
| 10 `DRAIN_S` and `RELOAD_S` guesses | fold into `planner-replan-part-b` |
| 10 Small-RAM fusion-off drift check | fold into `engine-one-graph-per-token` (a CI task with no verdict of its own, next to the other builder-side Track B item) |

## Proposed ADRs

| ADR | Gates | Home |
|---|---|---|
| 0023 Transport spike | Phase 6 transport | new nodes `cluster-transport-wan`, `cluster-owner-transport` |
| 0043 Speculative decoding | `serving-mtp-drafts`, `serving-self-spec` | new owner node `serving-owner-drafters`, after `bench-spec-rerun` |
| 0050 Decode levers | gates nothing open ((a) stage 2 and the pool are not scheduled) | existing `memory-io-service`, `memory-low-bit-kv`, `memory-ram-freq`, `memory-ram-pool`; Track B follow-up `memory-prefill-misses` |
| 0059 Adaptive compute tier | gates nothing open (`admit_share` stays off) | its pending box result: `bench-open-records` |
| 0060 Box metrics as OTel | gates only removing the sampler path, a small repair | its pending box result: `bench-open-records` |
| 0064 Decode dispatch | step 2 | new node `engine-one-graph-per-token` |
| 0068 Phase 5 verdict rules | the M2 report | existing `bench-m2-report` (one-line note added: the owner accepts 0068, 0069 and 0082 with the verdicts) |
| 0069 Own executor | the M2 report | existing `bench-m2-report` (same note) |
| 0072 Dense builder extra buffer types | gates nothing open; built, AMX speed pending | new node `engine-dense-amx` |
| 0081 Prefill without capture | acceptance | existing `multi-accept-adr-0081` |
| 0082 Cost model 5 | V4a; its added anchor was decided 2026-11-21 (ADR 0068 History) | existing `planner-cost-model-v5`; acceptance with `bench-m2-report` (same note) |
