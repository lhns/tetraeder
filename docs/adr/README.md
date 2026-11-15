# Architecture decision records

One file per decision: Status, Context, Decision, Consequences. Do not change an accepted decision; supersede it with a new ADR.

| ADR | Status | Title |
| --- | --- | --- |
| [0001](0001-own-rust-engine-on-ggml.md) | Accepted | Own Rust engine on ggml, reached by a strangler path from libllama |
| [0002](0002-rust-for-host-code.md) | Accepted | Rust for all host code |
| [0003](0003-no-interaction-combinators.md) | Accepted | No interaction combinators, HVM, Bend2 or e-graph search |
| [0004](0004-no-tornadovm-no-mojo.md) | Accepted | No Java with TornadoVM, no Mojo for now |
| [0005](0005-ggml-only-kernel-base.md) | Accepted | Inside the engine, ggml is the only kernel base |
| [0006](0006-one-placement-plan-ir.md) | Accepted | One placement plan IR |
| [0007](0007-measured-cost-model.md) | Accepted | Cost model from measured per-op curves, c0 plus f(M), with c0 per graph mode |
| [0008](0008-tensor-and-cross-node-expert-parallel-linux-only.md) | Accepted | Tensor parallel and cross-node expert parallel are Linux only |
| [0009](0009-drain-and-restart-replanning.md) | Accepted | Replanning is drain-and-restart until Phase 7, then live |
| [0010](0010-name-and-license.md) | Accepted | Name Tetraeder, license Apache-2.0 |
| [0011](0011-private-ggml-patch-queue.md) | Accepted | Ggml is modified only through a private ordered patch queue |
| [0012](0012-opentelemetry-from-first-commit.md) | Accepted | OpenTelemetry for metrics, traces and logs from the first commit |
| [0013](0013-bottleneck-detection-and-advisor.md) | Accepted | Bottleneck detection checks the environment first, picks the regime (load or batch 1), and outputs a ranked list with confidence or "balanced" |
| [0014](0014-own-graph-builder-for-expert-streaming.md) | Accepted | Expert streaming needs an own graph builder, so the first one (Qwen3 MoE, CUDA and CPU) comes in Phase 2 |
| [0015](0015-own-graph-builders-in-rust.md) | Accepted | Own graph builders in Rust per family |
| [0016](0016-own-executor-in-phase-5.md) | Accepted | An own executor replaces ggml's backend scheduler in Phase 5 |
| [0017](0017-worker-process-per-device.md) | Accepted | One worker process per device |
| [0018](0018-transport-chosen-by-measurement.md) | Accepted | Transport is chosen by measurement in Phase 0 (quinn, iroh, TCP with TLS, RDMA where present) |
| [0019](0019-m1-scope.md) | Accepted | M1 scope: dense models via libllama plus one MoE family with an own builder, CUDA and CPU only |
| [0020](0020-no-timing-patch.md) | Accepted | No timing patch: whole-step timing with CUDA graphs in production, per-op timing in calibration runs |
| [0021](0021-gpu-work-on-ephemeral-vast-boxes.md) | Accepted (tooling superseded by 0029) | All GPU work runs on ephemeral Vast boxes |
| [0022](0022-otel-exemplars.md) | Accepted | No OpenTelemetry exemplars for now |
| [0023](0023-transport-spike.md) | Proposed | Transport spike: QUIC versus TCP with TLS versus plain TCP |
| [0024](0024-quantization-per-unit-and-quality-profile.md) | Accepted | Quantization is a planner choice per unit; quality is measured per model |
| [0025](0025-objective-is-limits-plus-pareto-position.md) | Accepted | The objective is hard limits plus a position on the speed and quality front |
| [0026](0026-phase1-engine-on-libllama.md) | Accepted | Phase 1 engine interface: safe libllama wrapper in ggml-sys, engine loop in a new engine crate |
| [0027](0027-placement-plan-ir-shape.md) | Accepted | Placement plan IR: assignment rules over a model description, one expanded form for checks, diff and migration |
| [0028](0028-gateway.md) | Accepted | Gateway: axum, a narrow token-level engine trait, minijinja templates, llguidance constraints |
| [0029](0029-gpu-tooling-from-taxameter.md) | Accepted | GPU box tooling comes from the shared package taxameter |
| [0030](0030-own-moe-graph-builder.md) | Accepted (in part superseded by 0036) | Own Qwen3 MoE graph builder: safe ggml surface, builder in model, one segment per MoE layer |
| [0031](0031-arenas-tiers-slot-cache-direct-io.md) | Accepted | Memory: per-device arenas with prefill and decode budgets, expert slot cache over three tiers, aligned direct I/O |
| [0032](0032-memory-crate-dependencies-and-budget-semantics.md) | Accepted | Memory crate: reserve as accounting, phase budget semantics, I/O dependencies |
| [0033](0033-expert-slots-in-the-builder.md) | Accepted (in part superseded by 0034 and 0035) | Expert slots in the builder: lookup order, overflow passes, RAM tier, failure semantics |
| [0034](0034-expert-slots-at-least-top-k.md) | Superseded by 0035 | Expert slots: at least the experts per token, distinct slots per token in overflow passes |
| [0035](0035-expert-pair-groups-and-cpu-experts.md) | Accepted (supersedes 0034) | Expert pairs computed in compact groups per backend; non-resident experts on the CPU |
| [0036](0036-cpu-threadpool-and-whole-step-graph.md) | Accepted | The CPU backend keeps one threadpool; a step with every expert resident is one graph |
| [0037](0037-split-gguf-and-slot-tiers-from-byte-budgets.md) | Accepted | Split GGUF files; expert slot tiers sized from byte budgets and reported |
| [0038](0038-predicted-next-layer-expert-prefetch.md) | Accepted | Predicted next-layer expert prefetch into the RAM tier |
| [0039](0039-no-cpu-repack-of-expert-weights-for-now.md) | Accepted | No AVX2 repack of the builder's CPU weights for now |
| [0040](0040-planner-candidates-calibration-and-scoring.md) | Accepted | Planner: executable candidates, a cost model calibrated from anchor runs, scored by regret and rank |
| [0041](0041-detector-observation-faults-and-limits.md) | Accepted | Bottleneck detector: one observation type, faults separate from the limiting resource, advisor report only |
| [0042](0042-slo-burn-rate-alerts-and-drain-restart-replanning.md) | Accepted | SLO burn-rate alerts and drain-and-restart replanning |
| [0043](0043-speculative-decoding-in-the-scheduler.md) | Proposed | Speculative decoding in the scheduler: n-gram and draft-model drafts, batched verification |
| [0044](0044-op-offload-by-step-tokens.md) | Accepted | Op offload is gated by the step's token count, not by pair rows |
| [0045](0045-phase2-exit-verdict.md) | Accepted | Phase 2 exit verdict: Qwen3-235B-A22B larger than RAM, all baselines on the same box |
| [0046](0046-serving-real-engines-behind-the-gateway.md) | Accepted (partly supersedes 0028) | Serving: one-sequence scheduler over logits, Rust sampling, prefix reuse, request log |
| [0047](0047-formats-per-role-and-load-time-conversion-cache.md) | Accepted | Formats per tensor role in the plan; load-time conversion into a per-tensor disk cache |
| [0048](0048-quality-profile-file-and-calibration-job.md) | Accepted | Quality profile file and calibration job |
| [0049](0049-multi-entry-prefix-cache-and-lenient-anthropic-input.md) | Accepted | Multi-entry prefix cache in host RAM; lenient Anthropic input |
| [0050](0050-decode-levers-overlap-low-bit-kv-frequency-tiers.md) | Proposed | Decode levers: miss reads under CPU expert work, low-bit KV per attention unit, RAM tier by routing frequency |
| [0051](0051-no-prompt-checkpoints-tool-call-outcomes.md) | Accepted | No prompt checkpoints; tool-call outcomes counted, empty calls dropped |
| [0052](0052-runtime-cpu-isa-dispatch.md) | Proposed | Runtime CPU ISA dispatch: ggml backends as loadable modules, one build for every x86 CPU |
| [0053](0053-iqk-avx512-vnni-q4k-decode-kernel.md) | Proposed | iqk AVX-512 VNNI kernels for Q4_K/Q6_K expert decode, copied as patch 0002 |
| [0054](0054-expert-reads-io-service-and-arrival-order-compute.md) | Accepted | Decode expert reads: persistent I/O service, experts computed in arrival order, RAM tier sized to the container |
| [0055](0055-replanning-wired-into-serve.md) | Accepted (part A built, part B open) | Replanning wired into serve: plan from file or the executable enumerators, lowering from the plan, decode-step residual from the step ring |
| [0056](0056-tier-and-disk-cost-term-virtual-clock-calibration-ledger.md) | Proposed | Cost model: RAM-tier and disk term from replayed routing, virtual clock for I/O tests, calibration ledger in CI |
| [0057](0057-qwen-xml-arguments-typed-from-python-literals.md) | Accepted | Qwen XML tool arguments: Python literals typed by the schema |
| [0058](0058-decode-io-attribution.md) | Accepted | Decode I/O attribution: capacity and demand per window before any bottleneck label |
| [0059](0059-adaptive-compute-tier-admission.md) | Proposed | Adaptive compute tier: decode admits a share of CPU misses into VRAM slots |
| [0060](0060-box-host-metrics-as-otel.md) | Proposed | Box and host metrics as OTel metrics: one collector per box run, the result's host block derived from its file |
| [0061](0061-cpu-threadpool-polling-from-the-cgroup-quota.md) | Accepted | CPU threadpool polling chosen from the cgroup CPU quota: 0 when the quota binds, explicit level wins |
| [0062](0062-cpu-quota-throttle-fault.md) | Accepted | Detector fault class cpu_quota_throttle: the cgroup CPU quota throttling the engine's own threads |
| [0063](0063-genai-request-metrics-and-tier-transfer-family.md) | Accepted | GenAI semconv request metrics from the gateway for every engine; one tier transfer metric family |
| [0064](0064-decode-dispatch-segment-scheduler-and-one-graph-per-token.md) | Proposed | Decode dispatch: a scheduler sized for one segment, then one graph per decode token with a host callback per layer |
| [0065](0065-phase4-verdict-multi-sequence-backend-continuous-batching.md) | Proposed | Phase 4: verdict rules, a multi-sequence backend and continuous batching |
| [0066](0066-own-qwen3-dense-builder.md) | Proposed | Own Qwen3 dense builder on the shared attention module, multi-sequence from the start |
| [0067](0067-kv-layout-experiment-chunked-vs-paged.md) | Proposed | KV layout: chunked contiguous first, paged only if a staged experiment clears thresholds fixed now |
