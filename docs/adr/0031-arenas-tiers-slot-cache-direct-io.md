# 0031. Memory: per-device arenas with prefill and decode budgets, expert slot cache over VRAM, RAM and NVMe, aligned direct I/O

Status: Accepted (Phase 2)

## Context

The plan IR (ADR 0027) gives per-device KV, workspace and slot-cache bytes, and the validator checks them against capacity minus a measured reserve. Phase 2 must allocate them and stream experts from disk. A small card gains most when VRAM moves between uses by phase: prefill wants a large ubatch workspace and touches nearly every expert, decode wants many expert slots. Vast containers report all host RAM, and "NVMe" is often shared.

## Decision

- **Arenas** in `tetraeder-memory`: one ggml backend buffer per (device, purpose), purposes KV, expert slots, workspace and reserve, sized from the expanded plan. The reserve is measured at startup (free memory before and after backend init plus a test allocation) and reported, not assumed.
- **Per-phase budgets:** the IR gains `prefill` and `decode` byte budgets per device for slots and workspace (ADR 0027 allows fields as phases need them; full separate prefill and decode layouts stay Phase 7). Prefill borrows slot bytes beyond the pinned hot set for workspace; decode gets them back and refills from the hot set and routing profile. KV is never borrowed.
- **Slot cache:** per MoE layer a VRAM slot tensor and a RAM slot tensor per weight (gate, up, down), a host table from expert to slot per tier, a pinned hot set from routing counts, LRU for the rest. Experts not in VRAM are computed on the CPU from RAM slots or uploaded, whichever the cost model says (feature in plan.md "Expert streaming", item 6).
- **Direct I/O:** experts read by file offset. Linux uses `O_DIRECT` with page-aligned bounce buffers covering the aligned enclosing range, io_uring when the container allows it, a thread pool otherwise. Windows uses `FILE_FLAG_NO_BUFFERING` with overlapped I/O, sector-aligned. Both paths are tested on their OS in CI with a temp file.
- **Metrics** (rule 6, names from the plan.md catalog): `tetraeder.memory.used` and `.capacity` by device, tier and kind (weights, kv, workspace, slots, reserve); `tetraeder.expert.cache.requests` by layer, tier and result; `tetraeder.expert.load.duration` by tier; `tetraeder.tier.read.bytes`; a new `tetraeder.expert.layer_hit` counter by result (full, partial) that decides ADR 0030's whole-step graphs. Dashboard panel "Experts and cache". Alert: none new; hard out-of-memory and I/O errors go through the error rate SLO.
- **First failing test:** an arena sized from an expanded plan never exceeds its device budget, and a direct read of an unaligned GGUF range returns the same bytes as a buffered read, on both OSes.

## Consequences

- The Phase 2 exit test ("MoE larger than RAM") runs Qwen3-235B-A22B Q4_K_M (about 142 GB) on a box with at most 128 GB RAM and at least 250 GB disk, so the page cache cannot hold the model for any engine. taxameter offers need a max-RAM filter for that job.
- Exit numbers include the per-layer full-hit rate and decode tok/s right after a 32K prefill (the cost of refilling slots).
- Residency-biased routing (steering top-k toward resident experts) changes outputs; it is not part of Phase 2 and comes later only as an opt-in knob with a measured quality cost (ADR 0024, 0025).
