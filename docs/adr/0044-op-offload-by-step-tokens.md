# 0044. Op offload is gated by the step's token count, not by pair rows

Status: Accepted (Phase 2)

## Context

ggml's scheduler offloads an op whose weights live in host memory to the GPU when its batch size is at least `op_offload_min_batch_size` (32 on CUDA). For `MUL_MAT_ID` the batch size is `ne[2]` (`get_op_batch_size`, ggml-cuda.cu). In libllama that is the token count. The builder's CPU pair groups (ADR 0035) put one row per (token, expert) pair into `ne[2]`, so a 7-token step at top-8 has 56 rows and every routed RAM expert of the step crosses PCIe. Track E measured it on the 235B at 1K depth (bench/results/20261103/spec-diag-*.json on track-e-spec): a 7-token step 5560 ms with op offload against 1087 ms without, a 4-token step 879 against 765. Single-token steps are unaffected.

## Decision

- The builder keeps two schedulers when `LoadOptions::op_offload` is on: one with op offload, one without (same backends, separate compute buffers). A step uses the offloading one only when its token count is at least `LoadOptions::op_offload_min_tokens` (default 32, the CUDA threshold applied to tokens as libllama does). The kept whole-step graph (resident experts, ADR 0036) has no host weights and stays on the first scheduler.
- Pair-group row caps (`group_cap`) follow the scheduler the step uses: a RAM group is capped for CUDA only when that step offloads.

## Consequences

- Speculative verify steps (2 to 31 tokens) compute RAM experts on the CPU, like libllama. Prefill ubatches of 2048 are unchanged (32K exit prefill is above the threshold by construction; checked on the box).
- The second scheduler's compute buffers are sized by small steps (at most 31 tokens), a few MB of VRAM, reported under `workspace`.
- Numerics of steps below the threshold change from GPU to CPU expert matmuls; greedy equality with the resident builder is checked by the existing CPU tests, and on the GPU by the prefetch/exit pairs.
