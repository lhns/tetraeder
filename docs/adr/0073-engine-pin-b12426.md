# 0073. Engine pin moves to llama.cpp b12426 before the Phase 4 verdict

Status: Proposed (2026-11-16). Supersedes the engine-pin part of ADR 0065's baseline note ("The engine pin stays at b11976").

## Context

- The engine (ggml-sys, libllama, the builders) was pinned at b11976 (2e0d86beb); the Phase 4 baseline llama-server is b12426 (fc504f91d, ADR 0065).
- On one RTX 3090 (`bench/results/20261116pinab/pin-ab/summary.md`) llama-server built at b11976 reaches 0.930 (Qwen3-8B) and 0.834 (Qwen3-30B-A3B) of llama-server b12426 at concurrency 4, and libllama's 4-sequence decode step takes 83.4 ms per 8 steps at b11976 against 71.8 at b12426; one-sequence steps are level. Part of the c4 gap is the pin, and the verdict compares against b12426.
- `track-pin-b12426` showed the move is cheap on the patch side: patches 0001 to 0003 apply unchanged, 0004 is rebased onto upstream's stream-k `tile_spec`.

## Decision

- The engine pin moves to b12426 (`vendor/PIN` `tag`, `commit`, `image_tag` `b12426-ikb25c3fd2-blfc504f91`), so engine and baseline run the same llama.cpp.
- CPU numerics change upstream in the range, mainly #31854 (F16 matmuls take F32 src1; flash attention keeps Q in F32 for an F16 K) and #29810 (flash attention accumulates an F16 V in F32). The goldens are re-recorded per OS by the `ci` workflow's `record_golden` jobs (`golden-record`, one per OS), and each shift is checked against llama-simple at the new pin (same greedy text) and against the dense builder (within its 0.05).
- Patch 0004 on b12426 (GPU-unverified until the confirmation box runs `mul_mat_id_skip`): `mm_ids_helper` still counts id -1 below every expert, so compact rows 0 to n_neg-1 hold no expert's columns and nothing writes their forward-map entries nor the -1 slots' inverse-map entries. `ggml_cuda_mm_ids_fill` sets the whole map to 0 before the helper: the forward map then reads src1 row 0, the inverse map scatters into the unused row 0. The rebase's first version filled the inverse map with the last compact row, which a matched slot uses, so a -1 slot could overwrite another token's activations.

## Open: batch drift on AVX2 at the new pin

`cpu_batch_invariance_*` (Q4_K_M MoE fixture, 7-row step against one-row steps) fails on the Windows runner's AVX2 tier: rows 0 to 4 within 3.3e-7 of the largest logit, row 5 at 1.0e-2 (0.315 of 31.48), row 6 at 2.2e-2. Measured so far (`track-pin-adopt` diagnostics, runs 43882015665 and 43882124451):
- The diagnostic reproduces 0.3152809 exactly with 1 and 2 threads, so it is not a threading split.
- No expert choice differs on any row; the router weights of row 5 differ by about 2e-7 at layers 0 and 1, 1.2e-3 at layer 2 and 5.6e-3 at layer 3: a jump inside layer 1's FFN or layer 2's attention, not a gradual growth.
- Single ops (F32 and F16 matmul, flash attention over an F16 KV) differ between 7 rows and 1 row by at most 3e-6 relative, the same at both pins and on both OSes; flash attention is bit-equal.
- The F16 and F32 fixtures (no activation quantization) stay at or below 3e-5 on every row at the new pin, against up to 1.9e-3 at the old pin for F16 (src1 then rounded to F16).
- At the old pin, Windows and Linux single-row logits were bit-equal; at the new pin they differ by about 3e-5 already on rows 0 to 4 (likely libm `expf` in flash attention, no longer absorbed by the F16 V accumulator). On Linux the same AVX2 variant passes batch invariance but the speculative tests leave plain greedy at a margin of 0.0078.
- Working hypothesis, not yet shown: the Q8_K quantization of K-quant activations turns a last-bit difference into a one-step int8 change now and then. Diagnostics round 2 (`track-pin-diag2`) measures the jump rate over 84 rows per OS and that amplification on single Q4_K and Q6_K matmuls. Until the cause is shown, no tolerance is widened.

## Consequences

- The CUDA parity validated on b11976 (ADR 0071 and its addenda) has to be confirmed on the new pin on one box before the verdict: the gpu-smoke5/6 suites, `gpu_real_batched` on both real models, and patch 0004's GPU tests.
- Old ADRs keep their b11976 readings of ggml; they describe the code they were written against.
- The box image is rebuilt for the new tag by `box-image.yml` on the merge to main.
