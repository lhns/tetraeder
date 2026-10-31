# 0033. Expert slots in the builder: lookup order, overflow passes, RAM tier, failure semantics

Status: Accepted (Phase 2)

## Context

ADR 0030 splits a step into one segment per MoE layer and leaves the expert-to-slot map as a hook; ADR 0031 and 0032 give the slot cache, direct I/O and metrics. Step 2 connects them: a layer's expert tensors become slot tensors that hold fewer experts than the layer has, filled from the GGUF on demand. Open: what a lookup round does exactly, what happens when a layer step needs more distinct experts than there are slots (the fixture routes top-2, so 1 slot cannot hold one token), and what a failed load leaves behind.

## Decision

- **Options.** `LoadOptions.expert_slots` (compute tier, per layer; default: all experts) and `ram_slots` (host tier between file and compute tier; default: none). Slots at or above the expert count mean the old path: every expert loaded at load time, identity map, no cache. Below it the expert tensors have `slots` entries in dimension 2, are zero-filled and are not read at load.
- **Lookup round per layer step.** The host reads the top-k ids, takes the distinct experts in order of first appearance and puts those already resident first. They go through `LayerSlots::lookup` in that order, so hits are touched before any miss can evict them. The misses of one round are read with one `DirectFile::read_batch` (three slabs per expert), staged in host memory and uploaded into their slots. With a RAM tier the same round runs on it first: compute-tier misses look up the RAM slots (hits first), RAM misses are read from the file in one batch into their RAM slots, then every missing expert is uploaded from its RAM slot.
- **Overflow: passes.** If the distinct experts exceed the active slots, the round is split into chunks of at most `slots` experts (residents first, so only the first chunk has hits; later chunks evict the earlier ones). The expert FFN then runs once per chunk as its own small graph over all tokens: ids of pairs (token, choice) whose expert is not in the chunk point at a slot of the chunk and get router weight 0, so they add exactly 0. The passes accumulate into a state tensor and the next segment reads that instead of computing the FFN inline. A single chunk (the usual case) stays inline in the segment and costs nothing extra. Chosen over splitting tokens into groups because it also covers `slots < n_expert_used` with static shapes and one mechanism, and over a CPU fallback because that needs the whole expert in host memory and a second compute path. Cost: each pass computes the FFN for every pair, so a prefill that needs P chunks does P times the FFN work; acceptable for correctness first, the phase budgets (ADR 0032) and prefetch are what keep P small. A pair-compacted pass (gather the chunk's pairs, scatter back) is a later optimization behind a benchmark.
- **Failure.** A load error aborts the step with `ModelError::Io`; the layer's slot tables (both tiers) are cleared, since slots were assigned before their bytes arrived. The KV position is not advanced. No step ever computes with slots whose load did not finish.
- **Telemetry.** Cache requests are counted per distinct expert per layer step: tier `vram` is the compute tier (also on the CPU backend, where it is host memory), tier `ram` the RAM tier. `expert.layer_hit` is `full` when the round had no compute-tier miss. Load duration is the batch time divided by its experts, source `nvme` (file) or `ram`; tier read bytes count file bytes under `nvme` and RAM-tier uploads under `ram`. `tetraeder.step.duration` gets a sibling `tetraeder.step.segments` histogram (segments per step; `n_layer + 1` for the builder, 1 for libllama).

## Consequences

- Behavior with all experts resident is unchanged and bitwise identical.
- Pinned hot sets and prefetch (next step) plug into the same `LayerSlots` and the same round; `Bypass` (all slots pinned) is an error until then.
- Overflow results differ from the single pass only by float summation order (tests use the CPU tolerance of the parity test).
- The compute tier's `vram` label is wrong for CPU runs; a device-tier label can come with the plan IR's tier names.
