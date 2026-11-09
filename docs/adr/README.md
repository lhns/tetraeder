# Architecture decision records

One file per decision: Status, Context, Decision, Consequences. Do not change an accepted decision; supersede it with a new ADR.

| ADR | Title |
| --- | --- |
| [0001](0001-own-rust-engine-on-ggml.md) | Own Rust engine on ggml, reached by a strangler path from libllama |
| [0002](0002-rust-for-host-code.md) | Rust for all host code |
| [0003](0003-no-interaction-combinators.md) | No interaction combinators, HVM, Bend2 or e-graph search |
| [0004](0004-no-tornadovm-no-mojo.md) | No Java with TornadoVM, no Mojo for now |
| [0005](0005-ggml-only-kernel-base.md) | Inside the engine, ggml is the only kernel base |
| [0006](0006-one-placement-plan-ir.md) | One placement plan IR |
| [0007](0007-measured-cost-model.md) | Cost model from measured per-op curves, c0 plus f(M), with c0 per graph mode |
| [0008](0008-tensor-and-cross-node-expert-parallel-linux-only.md) | Tensor parallel and cross-node expert parallel are Linux only |
| [0009](0009-drain-and-restart-replanning.md) | Replanning is drain-and-restart until Phase 7, then live |
| [0010](0010-name-and-license.md) | Name Tetraeder, license Apache-2.0 |
| [0011](0011-private-ggml-patch-queue.md) | Ggml is modified only through a private ordered patch queue |
| [0012](0012-opentelemetry-from-first-commit.md) | OpenTelemetry for metrics, traces and logs from the first commit |
| [0013](0013-bottleneck-detection-and-advisor.md) | Bottleneck detection checks the environment first, picks the regime (load or batch 1), and outputs a ranked list with confidence or "balanced" |
| [0014](0014-own-graph-builder-for-expert-streaming.md) | Expert streaming needs an own graph builder, so the first one (Qwen3 MoE, CUDA and CPU) comes in Phase 2 |
| [0015](0015-own-graph-builders-in-rust.md) | Own graph builders in Rust per family |
| [0016](0016-own-executor-in-phase-5.md) | An own executor replaces ggml's backend scheduler in Phase 5 |
| [0017](0017-worker-process-per-device.md) | One worker process per device |
| [0018](0018-transport-chosen-by-measurement.md) | Transport is chosen by measurement in Phase 0 (quinn, iroh, TCP with TLS, RDMA where present) |
| [0019](0019-m1-scope.md) | M1 scope: dense models via libllama plus one MoE family with an own builder, CUDA and CPU only |
| [0020](0020-no-timing-patch.md) | No timing patch: whole-step timing with CUDA graphs in production, per-op timing in calibration runs |
| [0021](0021-gpu-work-on-ephemeral-vast-boxes.md) | All GPU work runs on ephemeral Vast boxes |
| [0022](0022-otel-exemplars.md) | No OpenTelemetry exemplars for now |
| [0023](0023-transport-spike.md) | Transport spike: QUIC versus TCP with TLS versus plain TCP (Proposed) |
| [0024](0024-quantization-per-unit-and-quality-profile.md) | Quantization is a planner choice per unit; quality is measured per model (Proposed) |
| [0025](0025-objective-is-limits-plus-pareto-position.md) | The objective is hard limits plus a position on the speed and quality front (Proposed) |
| [0026](0026-phase1-engine-on-libllama.md) | Phase 1 engine interface: safe libllama wrapper in ggml-sys, engine loop in a new engine crate |
| [0027](0027-placement-plan-ir-shape.md) | Placement plan IR: assignment rules over a model description, one expanded form for checks, diff and migration |
| [0028](0028-gateway.md) | Gateway: axum, a narrow token-level engine trait, minijinja templates, llguidance constraints |
| [0029](0029-gpu-tooling-from-taxameter.md) | GPU box tooling comes from the shared package taxameter |
| [0030](0030-own-moe-graph-builder.md) | Own Qwen3 MoE graph builder: safe ggml surface, builder in model, one segment per MoE layer |
| [0031](0031-arenas-tiers-slot-cache-direct-io.md) | Memory: per-device arenas with prefill and decode budgets, expert slot cache over three tiers, aligned direct I/O |
| [0032](0032-memory-crate-dependencies-and-budget-semantics.md) | Memory crate: reserve as accounting, phase budget semantics, I/O dependencies |
| [0033](0033-expert-slots-in-the-builder.md) | Expert slots in the builder: lookup order, overflow passes, RAM tier, failure semantics |
| [0034](0034-expert-slots-at-least-top-k.md) | Expert slots: at least the experts per token, distinct slots per token in overflow passes |
| [0035](0035-expert-pair-groups-and-cpu-experts.md) | Expert pairs computed in compact groups per backend; non-resident experts on the CPU |
| [0036](0036-cpu-threadpool-and-whole-step-graph.md) | The CPU backend keeps one threadpool; a step with every expert resident is one graph |
| [0037](0037-split-gguf-and-slot-tiers-from-byte-budgets.md) | Split GGUF files; expert slot tiers sized from byte budgets and reported |
| [0038](0038-predicted-next-layer-expert-prefetch.md) | Predicted next-layer expert prefetch into the RAM tier |
| [0039](0039-no-cpu-repack-of-expert-weights-for-now.md) | No AVX2 repack of the builder's CPU weights for now |
| [0043](0043-speculative-decoding-in-the-scheduler.md) | Speculative decoding in the scheduler: n-gram and draft-model drafts, batched verification |
| [0040](0040-planner-candidates-calibration-and-scoring.md) | Planner: executable candidates, a cost model calibrated from anchor runs, scored by regret and rank (Proposed) |
| [0041](0041-detector-observation-faults-and-limits.md) | Bottleneck detector: one observation type, faults separate from the limiting resource, advisor report only (Proposed) |
| [0042](0042-slo-burn-rate-alerts-and-drain-restart-replanning.md) | SLO burn-rate alerts and drain-and-restart replanning (Proposed) |
| [0046](0046-serving-real-engines-behind-the-gateway.md) | Serving: one-sequence scheduler over logits, Rust sampling, prefix reuse, request log |
| [0047](0047-formats-per-role-and-load-time-conversion-cache.md) | Formats per tensor role in the plan; load-time conversion into a per-tensor disk cache |
| [0048](0048-quality-profile-file-and-calibration-job.md) | Quality profile file and calibration job |
| [0049](0049-multi-entry-prefix-cache-and-lenient-anthropic-input.md) | Multi-entry prefix cache in host RAM; lenient Anthropic input |
| [0050](0050-decode-levers-overlap-low-bit-kv-frequency-tiers.md) | Decode levers: miss reads under CPU expert work, low-bit KV per attention unit, RAM tier by routing frequency (Proposed) |
| [0051](0051-no-prompt-checkpoints-tool-call-outcomes.md) | No prompt checkpoints; tool-call outcomes counted, empty calls dropped (Proposed) |
| [0052](0052-runtime-cpu-isa-dispatch.md) | Runtime CPU ISA dispatch: ggml backends as loadable modules, one build for every x86 CPU (Proposed) |
