# 0014. Expert streaming needs an own graph builder, so the first one (Qwen3 MoE, CUDA and CPU) comes in Phase 2

Status: Accepted

## Context

libllama allocates full expert tensors and runs top-k in its graph with no public hook; a sync per layer would break CUDA graphs.

## Decision

Expert streaming needs an own graph builder, so the first one (Qwen3 MoE, CUDA and CPU) comes in Phase 2; sync only on a miss, with predictive prefetch.

## Consequences

Hit rate is a first-class metric; Colibri is the reference design.
