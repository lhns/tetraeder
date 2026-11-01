# 0035. Expert pairs computed in compact groups per backend; non-resident experts on the CPU

Status: Proposed (Phase 2)

## Context

The run of 2026-11-01 (bench/results/20261101, RTX 3090, Qwen3-30B-A3B Q4_K_M, the builder as of that run) measured:

| Setup | Prefill tok/s | Decode tok/s |
|---|---|---|
| llama.cpp, all on GPU | 2800 (ubatch 512) to 4300 (4096) | 190 |
| llama.cpp, all experts on CPU (`-ot`) | 460 to 2080 | 40 |
| llama.cpp, layers 24 to 47 experts on CPU | 880 to 2900 | 67 |
| Tetraeder, 32 VRAM slots, upload on miss | 9 | 6 |
| Tetraeder, 32 slots + 64 RAM slots | 6 | 18 |
| Tetraeder, 120 slots | 7 | 46 |

Uploading every missed expert loses to computing it where it already is: a decode step misses dozens of experts at a few MB each, and the read and upload path ran at 0.8 to 1.5 GB/s. The ADR 0033 overflow passes also compute every pair in every pass (P times the FFN work) and need `n_used` parking entries per layer to stay exact (ADR 0034).

## Decision

- **Pair groups.** A layer step's (token, choice) pairs are split into groups, each computed by one backend from one expert tensor: the VRAM slot tensor (pairs whose expert is resident there) or the RAM slot tensor (the rest). A group gathers its pairs' inputs (`get_rows` on the normed hidden state by token index), runs `mul_mat_id` with one expert per row, and scatters the unweighted outputs into the per-pair accumulator `[n_embd, n_used, n_tokens]` (`set_rows` by pair index). The next segment weights and sums the accumulator with the same ops as the resident graph, so CUDA's fused weighted sum rounds the same (ADR 0034). No pair is computed twice or masked; work is exactly the routed pairs. This replaces ADR 0033's masked overflow passes: a step with more distinct experts than slots becomes several GPU groups or, by default, GPU hits plus a CPU group.
- **CPU experts come from the RAM slot tier** (ADR 0031, 0033): host slot tensors filled by direct I/O, LRU, as today. With `ram_slots` equal to the expert count it is "every expert in RAM" (Qwen3-30B-A3B on a large box); for a model larger than RAM (the exit test) it is the same mechanism with misses read in one batch per layer step. No memory map: a page-cache fault is a synchronous small random read, the failure mode that makes `-ot` collapse once the model exceeds RAM.
- **Placement v1: hot set in VRAM, everything else on the CPU.** VRAM slots hold a per-layer hot set from routing counts (a routing profile when one exists, else counts from the first prefill), pinned; decode never uploads. A VRAM miss is computed on the CPU from its RAM slot (read from disk first on a RAM miss), never uploaded on the critical path. Hot-set refresh runs between requests. Uploading on miss stays available as a policy for the cost model (feature "Expert streaming", item 6) to choose later.
- **One graph per segment, v1.** The GPU and CPU groups of a layer step go into one segment graph; ggml's scheduler splits it by backend and copies the inputs. Running the groups concurrently is a measured follow-up, not part of v1.
- **First failing tests.** The CPU group, the GPU group and both together are bitwise equal to the resident builder on CPU for the top-8 Q4_K_M fixture of `tests/multipass.rs`; decode with a hot set issues no uploads. The CPU's K-quant `mul_mat_id` picks its kernel per expert by row count (ADR 0034), so a group must give each expert exactly its routed rows in the order the resident `mul_mat_id` visits them (token-major, choice-minor). Whether the tiled kernel's result depends on the input layout (`[n_embd, n_used, n_tokens]` with broadcast against gathered `[n_embd, 1, n_pairs]`) was settled by these tests: it does not (see Implementation notes). The tests only discriminate on AVX machines (CI).

## Consequences

- Exit numbers to beat: llama.cpp `-ot` at the same VRAM use (decode 40 to 67 tok/s, prefill 460 to 2900 by ubatch). A large prefill ubatch matters: llama.cpp's `-ot` prefill gains 4.5x from ubatch 512 to 4096.
- The builder needs a per-step CPU and GPU threading split (CPU quota, ADR 0021) and the `tetraeder.expert.cache.requests` result `cpu` alongside hit and miss.
- Supersedes ADR 0033's overflow passes as the way to run more distinct experts than slots, and with them ADR 0034's mask, parking and distinct-slot rules, once pair groups pass the tests above. The reason is cost, not exactness: with parking the passes are exact on CPU. A CUDA prefill from a slot tensor stays inexact either way (MMQ's stream-k split depends on the expert count, ADR 0034).
- Prefetch and phase budgets (ADR 0031) move after this: with misses computed on the CPU they tune the hot set, they do not gate correctness.

## Implementation notes

- **Stage 1 (one backend).** A layer step whose distinct experts fit the compute slots runs in place as before (slot ids in `ids_map`, no gather). Otherwise the experts are chunked by the compute slot count (residents first, ADR 0033), one group per chunk. Groups are formed per expert, so each expert gets all its routed rows, in token-major order. All but the last group run as small graphs of their own; the last is built into the next segment, which then reads the accumulator. Each pair is written exactly once per layer step, so the accumulator needs no clearing. Counters: `expert_groups()` and `group_graphs()` (standalone graphs, added to `tetraeder.step.segments`).
- **Slots below top-k are allowed again** (`expert_slots >= 1`). One expert per row cannot repeat a slot within a token, so ADR 0034's distinct-slot rule, its mask and its parking entries are gone.
- **Input layout settled.** Both CPU paths address a src1 row as `(choice % ne11) + token * ne11` and build each expert's row list in token-major, choice-minor order; `from_float` quantizes per row. The tiled kernel's windows (and its narrow path) depend only on that row list. So a gathered `[n_embd, 1, n_pairs]` input gives every expert the same row bytes, count and order as the resident layout, and the result is bitwise equal. Evidence: `tests/multipass.rs` (Q4_K_M at 4, 8, 12 and 16 slots, prefill and decode) passes on the AVX2 CI runners, and a throwaway branch that splits groups by pair count instead of by expert fails it there on Linux and Windows (logit differences 0.18 to 0.36 at every slot count, Q8_0 still exact; 2026-11-01). Reversing the row order within each expert passes too, so the row count is what matters, not the order; the builder keeps token-major order anyway.
- **Stage 2 (mixed groups).** The RAM slot tier becomes ggml tensors (per layer gate, up and down, `ram_slots` deep) in a CPU backend buffer marked as weights. ggml's scheduler runs an op on the backend of a weights-usage source buffer, so a CPU group's matmuls run on the CPU from the RAM slots and only its gathered inputs and outputs cross to the GPU; this is llama.cpp's `-ot` mechanism. The compute-tier weights are marked the same way. RAM slots are filled by direct I/O into a staging buffer and one copy into the tensor (one memcpy per disk read, a later optimization). The RAM tier is not zero-cleared at load: a CPU group reads only slots staged for it, and a clear would commit the whole tier up front. Locally the CPU backend stands in for both tiers, which tests the groups but not the backend split; the split is checked by `gpu_*` tests.
- **Miss policy.** `LoadOptions::miss` chooses `Upload` (stage 1 behavior: compute-tier misses are loaded, more experts than slots become several compute groups) or `Cpu` (placement v1: compute-tier misses go to a CPU group from the RAM tier, which `Cpu` requires). Under `Cpu` a layer step is at most one compute group plus CPU groups chunked by the RAM slot count; the last CPU group and the compute group share the segment graph. Compute-tier misses are not looked up (a lookup assigns a slot); they count as `tetraeder.expert.cache.requests` result `cpu`.
- **Hot set.** The streamer counts routed pairs per layer and expert. Under `Cpu` the compute slots are filled only by `refresh_hot_set`: the top experts by count, pinned, uploaded from the RAM tier or the file. It runs at load from a routing profile when one is given, else once after the first multi-token step; callers may run it again between requests. Decode never uploads, checked by an upload counter.
