# 0034. Expert slots: at least the experts per token, distinct slots per token in overflow passes

Status: Accepted (Phase 2). Supersedes the part of ADR 0033 that allowed fewer slots than experts per token.

## Context

ADR 0033's overflow passes pointed every pair whose expert is outside the pass at the pass's first slot, with router weight 0, and allowed `expert_slots` below `n_expert_used`. The first CUDA run aborted in ggml's sorted `mul_mat_id` (`ids_to_sorted_host.size() == ne_get_rows`): for batches above the MMVQ path it counts each (token, slot) once, so a token naming the same slot twice is invalid input. On the CPU backend the repeats were harmless, which is why the CPU parity tests passed.

## Decision

- A masked pair gets a slot no other pair of the same token uses in that pass. Any slot works: every slot holds finite weights (loaded, or zero from the load-time clear), and the weight is 0.
- `expert_slots` must be at least `n_expert_used`; smaller values are a load error. Fewer distinct slots than picks per token cannot be expressed to `mul_mat_id`, and no real configuration needs it (Qwen3 MoE routes 8 of 128).

## Consequences

- The overflow passes stay the mechanism for more distinct experts than slots in a step; only the slot below top-k case is gone.
- The load-time zero clear of slot tensors is now required, not just defensive.
- Tests: the slot matrices start at 2 slots for the top-2 fixture; a unit test checks distinct slots per token.
