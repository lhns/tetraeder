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

- **Pair groups.** A layer step's (token, choice) pairs are split into groups, each computed by one backend from one expert tensor: the VRAM slot tensor (pairs whose expert is resident there) or the RAM slot tensor (the rest). A group gathers its pairs' inputs (`get_rows` on the normed hidden state by token index), runs `mul_mat_id` with one expert per row, and scatters the weighted outputs into the per-pair accumulator `[n_embd, n_used, n_tokens]` (`set_rows` by pair index). The next segment sums the accumulator over experts as today. No pair is computed twice or masked; work is exactly the routed pairs. This replaces ADR 0033's masked overflow passes: a step with more distinct experts than slots becomes several GPU groups or, by default, GPU hits plus a CPU group.
- **CPU experts come from the RAM slot tier** (ADR 0031, 0033): host slot tensors filled by direct I/O, LRU, as today. With `ram_slots` equal to the expert count it is "every expert in RAM" (Qwen3-30B-A3B on a large box); for a model larger than RAM (the exit test) it is the same mechanism with misses read in one batch per layer step. No memory map: a page-cache fault is a synchronous small random read, the failure mode that makes `-ot` collapse once the model exceeds RAM.
- **Placement v1: hot set in VRAM, everything else on the CPU.** VRAM slots hold a per-layer hot set from routing counts (a routing profile when one exists, else counts from the first prefill), pinned; decode never uploads. A VRAM miss is computed on the CPU from its RAM slot (read from disk first on a RAM miss), never uploaded on the critical path. Hot-set refresh runs between requests. Uploading on miss stays available as a policy for the cost model (feature "Expert streaming", item 6) to choose later.
- **One graph per segment, v1.** The GPU and CPU groups of a layer step go into one segment graph; ggml's scheduler splits it by backend and copies the inputs. Running the groups concurrently is a measured follow-up, not part of v1.
- **First failing tests.** The CPU group, the GPU group and both together are bitwise equal to the resident builder on CPU for the top-8 Q4_K_M fixture of `tests/multipass.rs`; decode with a hot set issues no uploads. The CPU's K-quant `mul_mat_id` picks its kernel per expert by row count (ADR 0034), so a group must give each expert exactly its routed rows in the order the resident `mul_mat_id` visits them (token-major, choice-minor). Not yet verified, and settled by these tests rather than assumed: that the tiled kernel's result does not depend on the input layout (`[n_embd, n_used, n_tokens]` with broadcast against gathered `[n_embd, 1, n_pairs]`). The tests only discriminate on AVX machines (CI).

## Consequences

- Exit numbers to beat: llama.cpp `-ot` at the same VRAM use (decode 40 to 67 tok/s, prefill 460 to 2900 by ubatch). A large prefill ubatch matters: llama.cpp's `-ot` prefill gains 4.5x from ubatch 512 to 4096.
- The builder needs a per-step CPU and GPU threading split (CPU quota, ADR 0021) and the `tetraeder.expert.cache.requests` result `cpu` alongside hit and miss.
- Supersedes ADR 0033's overflow passes as the way to run more distinct experts than slots, and with them ADR 0034's mask, parking and distinct-slot rules, once pair groups pass the tests above. The reason is cost, not exactness: with parking the passes are exact on CPU. A CUDA prefill from a slot tensor stays inexact either way (MMQ's stream-k split depends on the expert count, ADR 0034).
- Prefetch and phase budgets (ADR 0031) move after this: with misses computed on the CPU they tune the hot set, they do not gate correctness.
