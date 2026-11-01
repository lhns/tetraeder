# 0034. Expert slots: at least the experts per token, distinct slots per token in overflow passes

Status: Accepted (Phase 2). Supersedes the part of ADR 0033 that allowed fewer slots than experts per token.

## Context

ADR 0033's overflow passes pointed every pair whose expert is outside the pass at the pass's first slot, with router weight 0, and allowed `expert_slots` below `n_expert_used`. The first CUDA run aborted in ggml's sorted `mul_mat_id` (`ids_to_sorted_host.size() == ne_get_rows`): for batches above the MMVQ path it counts each (token, slot) once, so a token naming the same slot twice is invalid input. On the CPU backend the repeats were harmless, which is why the CPU parity tests passed.

## Decision

- A masked pair gets a slot no other pair of the same token uses in that pass. Any slot works: every slot holds finite weights (loaded, or zero from the load-time clear), and the weight is 0.
- The passes multiply by a 0/1 mask instead of the router weights and accumulate the unweighted expert outputs per (token, choice); the next segment applies the router weights and sums over experts with the same ops as a single pass. Each pair is nonzero in exactly one pass, and `x * 1` and `x + 0` are exact, so the result is bitwise equal to a single pass and streaming output does not depend on the slot count. The first real-model run (Qwen3-30B-A3B Q4_K_M, 32 slots) showed why exactness matters: CUDA fuses weighting and sum into one kernel (`moe_weighted_reduction`, which contracts to fused multiply-adds), passes that weighted and summed by themselves rounded differently in the last bit, a near-tied routing choice flipped at layer 2 and the difference cascaded through attention to logit differences above 1. The accumulator is `n_expert_used` times larger than the FFN output (about 270 MB at ubatch 4096 for Qwen3-30B-A3B, inside the phase workspace budget).
- `expert_slots` must be at least `n_expert_used`; smaller values are a load error. Fewer distinct slots than picks per token cannot be expressed to `mul_mat_id`, and no real configuration needs it (Qwen3 MoE routes 8 of 128).

## Consequences

- Bitwise equality to the resident builder holds on CPU and for single-token steps on CUDA (the vector kernel's result per row does not depend on the number of experts in the tensor). A CUDA prefill that needs overflow passes runs MMQ, whose stream-k split of the inner dimension depends on the number of output tiles, which includes the expert count of the slot tensor (32 slots against 128 experts). The last bits then differ, near-tied routing choices can flip and the difference cascades. Open: the same multi-pass prefill also differs on the CPU backend, whose expert matmul is per-element deterministic, with the first routing difference at layer 3 as on CUDA. So the stream-k split is not the whole cause; the investigation continues with a bitwise comparison of router weights per layer. Until it is settled the real-model test checks single-token streaming bit for bit and a multi-pass prefill only for gross errors (identical layer-0 routing, top-1 agreement of at least 75% of positions), and keeps the CPU multi-pass comparison as a failing exact check.

- The overflow passes stay the mechanism for more distinct experts than slots in a step; only the slot below top-k case is gone.
- The load-time zero clear of slot tensors is now required, not just defensive.
- Tests: the slot matrices start at 2 slots for the top-2 fixture; a unit test checks distinct slots per token.
