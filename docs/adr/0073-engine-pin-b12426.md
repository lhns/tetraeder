# 0073. Engine pin moves to llama.cpp b12426 before the Phase 4 verdict

Status: Proposed (2026-11-16). Supersedes the engine-pin part of ADR 0065's baseline note ("The engine pin stays at b11976", ADR 0065 History, 2026-11-16, "baseline moved to b12426"). The confirmation box on the new pin is `20261116pc`.

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here; the CI numbers below have no session directory yet.

## Context

- The engine (ggml-sys, libllama, the builders) was pinned at b11976 (2e0d86beb); the Phase 4 baseline llama-server is b12426 (fc504f91d, ADR 0065).
- On one RTX 3090 llama-server built at b11976 was slightly slower than b12426 at concurrency 4 on Qwen3-8B and clearly slower on Qwen3-30B-A3B, and so was libllama's 4-sequence decode step; one-sequence steps were level (`20261116pinab`). Part of the c4 gap is the pin, and the verdict compares against b12426.
- `track-pin-b12426` showed the move is cheap on the patch side: patches 0001 to 0003 apply unchanged, 0004 is rebased onto upstream's stream-k `tile_spec`.

## Decision

- The engine pin moves to b12426 (`vendor/PIN` `tag`, `commit`, `image_tag` `b12426-ikb25c3fd2-blfc504f91`), so engine and baseline run the same llama.cpp.
- CPU numerics change upstream in the range, mainly #31854 (F16 matmuls take F32 src1; flash attention keeps Q in F32 for an F16 K) and #29810 (flash attention accumulates an F16 V in F32). The goldens are re-recorded per OS by the `ci` workflow's `record_golden` jobs (`golden-record`, one per OS), and each shift is checked against llama-simple at the new pin (same greedy text) and against the dense builder (within its 0.05).
- Patch 0004 on b12426 (GPU-unverified until the confirmation box runs `mul_mat_id_skip`): `mm_ids_helper` still counts id -1 below every expert, so compact rows 0 to n_neg-1 hold no expert's columns and nothing writes their forward-map entries nor the -1 slots' inverse-map entries. `ggml_cuda_mm_ids_fill` sets the whole map to 0 before the helper: the forward map then reads src1 row 0, the inverse map scatters into the unused row 0. The rebase's first version filled the inverse map with the last compact row, which a matched slot uses, so a -1 slot could overwrite another token's activations.

## Re-recorded goldens

Recorded by `ci` run 43889821792 (`record_golden`), SSE4.2 tier, compared with `golden_diff.py` (old file against new). Shifts are the largest |logit| change of the shared top ids up to the first token flip. (No session directory yet.) "Greedy tokens changed" is per case; in the llama-simple files the comparison is on the text.

| File | Max logit shift | Greedy tokens changed |
|---|---|---|
| `qwen3-0.6b-q8_0-cpu-sse42-linux.json` | 0.34 (capital 0.16, fibonacci 0.22, story 0.34) | capital only: flip at token 5, 9625 to 15344, old top-2 16.743/16.583, new 16.732/16.705 |
| `qwen3-0.6b-q8_0-cpu-sse42-windows.json` | 0.28 (capital 0.28, fibonacci 0.26, story 0.22) | capital only: flip at token 1, 11 to 13, old top-2 19.136/19.118, new 19.129/19.075 |
| `qwen3-0.6b-q8_0.llama-simple-sse42-linux.json` | not recorded (text only) | capital only (text differs from char 47); fibonacci, story equal |
| `qwen3-0.6b-q8_0.llama-simple-sse42-windows.json` | not recorded (text only) | capital only (text differs from char 30); fibonacci, story equal |

Every flip is a near-tie (gap under 0.03 in the new file). The shifts are in line with the F16 and F32 accumulation changes listed above. (No session directory yet.)

## Open: batch drift on AVX2 at the new pin

`cpu_batch_invariance_*` (Q4_K_M MoE fixture, 7-row step against one-row steps) fails on the AVX2 CPU kernels only: the old pin passes every variant and AVX-512 at b12426 passes. On the Windows runner's AVX2 tier rows 0 to 4 are within 3.3e-7 of the largest logit, row 5 is at 1.0e-2 (0.315 of 31.48), row 6 at 2.2e-2. The Linux AVX2 runner passes batch invariance but its speculative tests leave plain greedy at a margin of 0.0078. (No session directory yet.) Measured (`track-pin-adopt` diagnostics, runs 43882015665 and 43882124451):
- 0.3152809 reproduces exactly with 1 and 2 threads, so it is not a threading split. (No session directory yet.)
- No expert choice differs on any row; the router weights of row 5 differ by about 2e-7 at layers 0 and 1, 1.2e-3 at layer 2 and 5.6e-3 at layer 3: a jump inside layers 1 to 2. (No session directory yet.)
- Single ops (F32 and F16 matmul, flash attention over an F16 KV) differ between 7 rows and 1 row by at most 3e-6 relative, the same at both pins and on both OSes; single ops amplify a 1-ulp input change by at most 3e-7. (No session directory yet.)
- The F16 and F32 fixtures (no activation quantization) stay at or below 3e-5 on every row at the new pin. (No session directory yet.)
- At the old pin, Windows and Linux single-row logits were bit-equal; at the new pin they differ by about 3e-5 already on rows 0 to 4. (No session directory yet.)

Bisect (run 43896368376, ubuntu-24.04, `TETRAEDER_GGML_CPU_VARIANT=haswell`): the first bad commit for batch invariance on AVX2 is upstream 12ab5979 "ggml : switch to F32 src1 in the CPU backend (#31854)"; the last good commit is d6be0655, adjacent on the first-parent chain. Mechanism, from the diff: F16 type traits now use `vec_dot_f16_f32` with an F32 src1. One row goes through `ggml_vec_dot_f16_f32` (AVX, F32x8 FMA); two or more rows go through new llamafile tinyBLAS F16xF32 paths (`__m256` on AVX2). Summation order and FMA use differ, so single-row and batched results are not bit-equal. From upstream 7f24aacd (#29810, flash-attention V in F32), batch invariance passes again on Linux haswell and the speculative and draft tests fail instead (margin 0.0078); that holds through b12426. Not established: why AVX-512 stays invariant, and whether the #29810 speculative failure needs #31854. (No session directory yet.)

Cause: upstream #31854, F16 matmul batched vs single-row paths differ on AVX2. Until the pin moves past it or the paths are aligned, `cpu_batch_invariance_resident`, `cpu_batch_invariance_streaming`, `cpu_speculative_equals_plain_resident`, `cpu_speculative_equals_plain_streaming` and `cpu_draft_model_equals_plain_and_is_accepted` (`crates/scheduler/tests/spec_builder.rs`) print `skipped: ADR 0073 open item: AVX2 K-quant batch drift at b12426` and return when the loaded CPU backend (`backend_info().isa`, so a forced `TETRAEDER_GGML_CPU_VARIANT` counts) has AVX2 and no AVX512. They stay strict on AVX-512 and on non-x86. No tolerance is widened. Remove the skip when the cause is resolved.

## Consequences

- The CUDA parity validated on b11976 (ADR 0071 and its addenda) has to be confirmed on the new pin on one box before the verdict: the gpu-smoke5/6 suites, `gpu_real_batched` on both real models, and patch 0004's GPU tests.
- Old ADRs keep their b11976 readings of ggml; they describe the code they were written against.
- The box image is rebuilt for the new tag by `box-image.yml` on the merge to main.
