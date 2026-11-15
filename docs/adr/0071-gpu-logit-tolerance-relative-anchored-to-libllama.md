# 0071. GPU logit tolerance relative to the largest logit, anchored to libllama on CUDA

Status: Proposed (2026-11-16). Provisional until the gpu-smoke3 session (`scripts/gpu_smoke3.sh`) reads the full per-row tables.

## Context

- The first CUDA run of the own builders (`bench/results/20261115gpu/gpu-smoke2/`, RTX 3090, cc 8.6) failed every own-builder parity test on the tiny fixtures against `TOL_GPU = 5e-3`, an absolute bound written for logits of 1 to 5. The fixtures' logits are 11 to 15. The diffs were 4e-4 (dense, one sequence, top 5) and 1.4e-3 to 2.4e-3 (batched, `seq_cp`) of the largest logit. The Qwen3-8B dense parity test against libllama passed.
- At the pin, ggml-cuda runs F16 matmuls on mmvf for 1 row and on mmf (F16 inputs, F32 accumulation) for 2 to 16 rows that fit its shape rules; otherwise on cuBLAS `COMPUTE_16F` (half accumulation and output): above 16 rows, the 260-row output head of the fixtures, and the 32-wide MoE down experts at any row count. libllama sets no F32 precision on Qwen3 matmuls.
- On the CPU the builder equals libllama bitwise, batched and split; libllama's own batched-against-split gap reaches 1.07e-3 of the largest logit there (`cpu_libllama_batched_vs_split_anchor`).
- The MoE fixture's head size 16 has no CUDA flash-attention kernel, so only the dense fixture (head 64) exercises CUDA flash attention with mixed sequences.

## Decision

- GPU parity checks of the F16 fixtures use, per row, `max(TOL_GPU, 5e-3 * largest |logit|)` (`GPU_TOL` in `crates/model/tests/common/mod.rs`). Top-1 and greedy tokens (teacher-forced) are equal or within ADR 0043's near-tie rule (twice the bound).
- Each GPU suite runs an anchor: libllama on CUDA, batched and split. The builder is asserted against libllama at the same step shapes within the bound; builder batched against split within the bound widened to 1.5x libllama's own worst gap. libllama's own gaps are recorded, not asserted.
- Every comparison is printed before any assertion, and independent parts run to completion.
- A mixing bug, not kernel noise, is declared if the same-shape builder-against-libllama gap approaches libllama's own batched-against-split gap, if the per-row gap grows with position or across sequences, or if a greedy flip is not a near-tie.

## Consequences

- The GPU bound is about twice the observed noise, and the anchor ties it to llama.cpp's behaviour on the same card.
- `TOL_GPU` stays the absolute floor for its other users (`parity`, `slots`, `admit`, `cpu_experts`, `kv_snapshot_decode`); they may need the relative rule once they run on CUDA.
- Forcing F32 matmul precision in the builders would remove the cuBLAS 16F noise but diverge from libllama and cost speed; not taken.
- Open until smoke3: the dense three-sequence failure runs on mmf (F32 accumulation) and CUDA flash attention with mixed sequences, so cuBLAS 16F does not explain it; the dense anchor decides it.
