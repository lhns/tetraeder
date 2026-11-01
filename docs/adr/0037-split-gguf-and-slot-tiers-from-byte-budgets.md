# 0037. Split GGUF files; expert slot tiers sized from byte budgets and reported

Status: Accepted (Phase 2)

## Context

The Phase 2 exit test runs Qwen3-235B-A22B Q4_K_M (about 140 GB) on a box with at most 128 GB RAM and one 24 GB GPU (ADR 0031). That GGUF ships as several files written by llama.cpp's `gguf-split`, and the builder read one file. Slot counts were constants per layer (`expert_slots`, `ram_slots`), so a run could neither be sized from the memory it has nor show that it stayed inside it. Q4_K_M gives layers different expert sizes (wider `ffn_down` in some layers), so "bytes per slot" is not one number.

## Decision

- **Split files as llama.cpp writes them.** The first shard (`<prefix>-00001-of-0000N.gguf`) holds all metadata plus `split.no` (u16), `split.count` (u16) and `split.tensors.count` (i32); the others hold only the three split keys and their tensors, offsets relative to their own data section. `Gguf::open` on the first shard opens all shards by llama.cpp's naming rule, checks numbering and the total tensor count, and merges the tensor index; each `TensorInfo` carries its shard. Opening a later shard is an error. A tensor never spans shards, but a layer's gate, up and down tensors may sit in different shards, so an expert slab carries its shard per part. The streamer keeps one direct file per shard and issues one batch per shard per round.
- **Slot tiers from byte budgets.** `LoadOptions` gains `expert_slot_bytes` and `ram_slot_bytes`, each exclusive with its count. A budget buys `floor(bytes / sum over layers of one expert's bytes in that layer)` slots per layer, the same count in every layer, capped at the expert count. A budget that buys no slot is an error. The budget is the tier's slot bytes only, the quantity the plan IR calls a slot-cache buffer; weights, KV and compute buffers are reported next to it, not carved from it.
- **Reporting.** `Qwen3Moe::memory()` lists every buffer the builder owns by device, tier and kind (`weights`, `slots`, `kv`, `io` for the bounce buffers of the expert reads, `workspace`) with capacity and used bytes. The RAM tier's capacity is its buffer size and its used bytes are the resident slots: it is committed lazily (ADR 0035), so used bytes, not capacity, are what the process touches. The same numbers go to `tetraeder.memory.used` and `.capacity`. Peak resident set size of the process comes from the memory crate, so a run can show that it stayed within the box.

## Consequences

- One code path for single and split files; a single file is one shard.
- Uniform slots per layer waste up to one expert of the widest layer per layer; per-layer counts wait until routing profiles show layers differ in need.
- The slot budgets are what the planner will fill from `phase_budgets` (ADR 0032) once the builder runs from a plan; until then the exit run passes them directly.
- Miss reads go through one bounce buffer per shard, reused across steps (at most 256 MiB, reported as `io`), and one copy into the slot. Local exit regime (`tests/exit_regime.rs`, synthetic 8-layer top-4 MoE, 4 of 16 experts in RAM, Windows VM, `bench/results/20261101-local/exit-regime-{before,after}.json`): decode streamer time 73 to 154 ms per token before, 33 to 35 after; read rate 2.4 to 2.6 GB/s after (device time only; the before figure of 0.4 to 0.9 also counted allocation, zeroing and two copies, so the streamer time is the comparable number); greedy tokens unchanged. Local prefill is CPU-compute bound (reads 1 to 5 % of it); on the box `op_offload` runs large CPU-group matmuls on the GPU, as llama.cpp does.
