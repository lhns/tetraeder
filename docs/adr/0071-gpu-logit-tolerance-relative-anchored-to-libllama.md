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

## Addendum: gpu-smoke3 (2026-11-16, `bench/results/20261116gpu3/`, RTX 3090)
- The rule holds on the dense builder: every dense GPU test passes, including the three-sequence one that was open (CUDA flash attention over mixed sequences). On CUDA libllama's own batched against split gap is 0.0895 (5.7e-3 of the largest logit), the builder's 0.0523; builder against libllama at the same shapes 0.0582 batched and 0.0312 split. No growth with position or sequence; no greedy flip.
- MoE on CUDA: the builder batched is bit-equal to libllama batched; batched against split 0.0377 for both.
- Not closed: `gpu_seq_cp_snapshot_and_move` fails in its move subcases (F16 KV jumps to 0.171 at step 19; q8_0 KV is off by 2.4e-2 of the largest logit from step 0, with a greedy flip at step 62) while copy, prefix copy and source continuation pass. This looks like a real fault in the device-side row move, not noise; under diagnosis (`track-gpu-move`). The verdict's gpu-parity gate stays strict until it is fixed.

## Addendum: the move failures were the reference's (2026-11-16, `track-gpu-move`)
- `move_mid_decode` compared a run whose sequence starts at the moved row: before the move that reference has another `n_kv` and row offset, and attention sums in an order that depends on both (split-KV CPU flash attention from `n_kv` 512 with several threads and F16 V accumulation, and CUDA's KV split); on near-tied MoE fixtures this reaches 2e-2. The F16 spike at step 19 comes before the move (step 56), and after the move the gap falls to 4e-4 to 1e-3. The test now checks that the moved bytes are equal and compares only steps with the same rows and `n_kv`; the tolerance is unchanged. `copy_rows` was checked by reading (whole rows, correct stride and offset, synchronized before the next step, no overlap, rotation matrices fixed).
- CPU references (`n_gpu_layers` 0) set `op_offload` off: in a CUDA build libllama otherwise runs host ops of 32 or more rows on the GPU, the likely cause of the box-only CPU anchor gap (1.09e-3 relative). gpu-smoke5 confirms it or not.


## Addendum: gpu-smoke5 closes it (2026-11-16, `bench/results/20261116gpu5/`, RTX 3090)
Every GPU parity test passes: the move is bit-exact after the move (F16 and q8_0 KV); the CPU anchor (builder against libllama with `op_offload` off) is 0 on all 38 rows, confirming the offload cause; the dense and MoE GPU suites stay within the rule (worst builder batched against split 3.35e-3 and 2.40e-3 of the largest logit, libllama's own 5.73e-3 and 3.06e-3). Status stays Proposed until the Phase 4 verdict session runs the parity gate on its own box.

## Addendum: real models need their own anchor (2026-11-16, `bench/results/20261116d2a/phase4/`, RTX 3090)
- `gpu_real_batched` failed on Qwen3-8B (worst 3.0e-2 of the largest logit, no argmax flip) and Qwen3-30B-A3B (worst 8.3e-2, five flips) at the fixtures' bound, without measuring libllama's own gap on those models.
- On Qwen3-8B the one row whose KV cells and step shapes coincide in both engines (sequence 0, first step) was bit-equal; every other row differed by a flat 1.3e-2 to 3e-2, with no growth by position or sequence. The builder reserves rows per sequence where libllama's unified cache allocates cells in step order, so CUDA flash attention sums in another order; Q4_K's q8_1 activations plausibly grow that difference over the layers. The masks are correct.
- Decision: real-model GPU checks anchor on libllama's own gap on the same model and card (batched against split, and against the same steps with the cells shifted by 7 and 256 cells); the bound is max(GPU_TOL, 1.5x the largest own gap); the builder is asserted within it, batched against split and against libllama at the same shapes. One sequence alone, where cells and shapes coincide, is asserted at GPU_TOL with no widening: the strict check against divergences on real weights. Argmax flips print their margins.
- Open until the re-run (`scripts/gpu_real_batched.sh`): MoE row (0,19) differs by 7.7e-3 although its cells coincide, possibly cross-sequence coupling through `mul_mat_id`'s per-expert tiling; the alone control decides between that and a divergence specific to the MoE builder.

## Addendum: the anchored re-run (2026-11-16, `bench/results/20261116rb/`, RTX 3090)
- Qwen3-8B passes: libllama's own gap reaches 4.1e-2 (cells shifted by 7) and 2.3e-2 (batched against split), the builder's stays inside the bound (3.0e-2 against libllama at the same shapes), and sequence 0 alone is exactly 0 on every row. The dense difference is the KV placement order.
- Qwen3-30B-A3B: every anchored comparison is inside the bound (libllama's own gap 1.1e-1; its argmax flips against itself at near ties), but sequence 0 alone fails at one row: the last prompt row (0,19) differs by 0.174 (7.74e-3 of the largest logit) while every decode row after it is exactly 0. The same 0.174 appears in the D2a batched run on another box, so it is deterministic and independent of other sequences: not noise and not `mul_mat_id` coupling. With the KV cache identical, the gap sits after the last layer's attention in steps with more tokens than outputs; the likely cause is that the MoE builder runs the last layer's experts on every prompt row and selects the output rows only at the head, where libllama selects them before that FFN (under fix, `track-moe-outrows`). The strict control stays at the plain bound.
- Fixed (2026-11-16, `track-moe-outrows`): the MoE whole-step graph now selects the output rows after the last layer's attention and before its router and expert FFN, as libllama's `inp_out_ids` does (`vendor/llama.cpp/src/models/qwen3moe.cpp`); selecting only at the head made CUDA pick another `mul_mat_id` shape for the last layer. With routing recorded, a log-only router on all rows keeps the log complete. The streamed segment path still computes the last layer on all rows (its slot buffers are sized by the step's tokens); the verdict's 30B runs resident. To confirm on the GPU: `gpu_qwen3_30b_batched_matches_libllama`, sequence 0 alone exactly 0 at (0,19).
