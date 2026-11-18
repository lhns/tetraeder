# 0071. GPU logit tolerance relative to the largest logit, anchored to libllama on CUDA

Status: Proposed (2026-11-16). The rule holds on every one-GPU parity suite of the fixtures (`20261116gpu5`) and, anchored, on both real models (`20261116rb`, `20261116d2a2`); the anchored tensor-parallel bound for sharded shapes held on the S2d box (`s2d-20261118-step0`). Status stays Proposed until the Phase 4 verdict session runs the parity gate on its own box.

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

## Context

- The first CUDA run of the own builders failed every own-builder parity test on the tiny fixtures against `TOL_GPU = 5e-3`, an absolute bound written for logits of 1 to 5; the fixtures' logits are larger, and the diffs were small relative to the largest logit. The Qwen3-8B dense parity test against libllama passed (`20261115gpu`).
- At the pin, ggml-cuda runs F16 matmuls on mmvf for 1 row and on mmf (F16 inputs, F32 accumulation) for 2 to 16 rows that fit its shape rules; otherwise on cuBLAS `COMPUTE_16F` (half accumulation and output): above 16 rows, the 260-row output head of the fixtures, and the 32-wide MoE down experts at any row count. libllama sets no F32 precision on Qwen3 matmuls.
- On the CPU the builder equals libllama bitwise, batched and split; libllama's own batched-against-split gap reaches 1.07e-3 of the largest logit there (`cpu_libllama_batched_vs_split_anchor`). (No session directory yet.)
- The MoE fixture's head size 16 has no CUDA flash-attention kernel (still none at b12426), so only the dense fixture (head 64) exercises CUDA flash attention with mixed sequences.
- On real models the builder and libllama place KV differently (rows reserved per sequence against a unified cache allocated in step order), so CUDA flash attention sums in another order, and Q4_K's q8_1 activations carry that over the layers: a fixture bound does not fit them (`20261116d2a`, `20261116rb`). Sharded shapes change the reduction again (ADR 0069, 0070).

## Decision

- **Fixtures.** GPU parity checks of the F16 fixtures use, per row, `max(TOL_GPU, 5e-3 * largest |logit|)` (`GPU_TOL` in `crates/model/tests/common/mod.rs`). Top-1 and greedy tokens (teacher-forced) are equal or within ADR 0043's near-tie rule (twice the bound). `parity` and `slots` (`gpu_matches_libllama`, `gpu_slot_counts_match_libllama`) use `GPU_TOL` too; the CPU paths keep `TOL_CPU`.
- **Fixture anchor.** Each GPU suite runs libllama on CUDA, batched and split. The builder is asserted against libllama at the same step shapes within the bound; builder batched against split within the bound widened to 1.5x libllama's own worst gap. libllama's own gaps are recorded, not asserted.
- **Real models** (addendum "real models need their own anchor", 2026-11-16). GPU checks anchor on libllama's own gap on the same model and card: batched against split, and against the same steps with the cells shifted by 7 and 256 cells. The bound is max(`GPU_TOL`, 1.5x the largest own gap); the builder is asserted within it, batched against split and against libllama at the same shapes. One sequence alone, where cells and shapes coincide, is asserted at `GPU_TOL` with no widening: the strict check against divergences on real weights. Argmax flips print their margins.
- **CUDA against the CPU builder** (2026-11-17). libllama on CUDA and on the CPU run the test's own tokens and steps. The builder is asserted against libllama on the same backend and shapes (strict: `GPU_TOL` on CUDA, `TOL_CPU` on the CPU); its CUDA-against-CPU gap (batched, and against each sequence alone on the CPU) within max(`GPU_TOL`, 1.5x libllama's own worst CUDA-against-CPU gap). The anchor needs the same tokens: another script's anchor does not bound these rows. `gpu_libllama_batched_vs_split_anchor` stays as a separate check.
- **Sharded shapes** (2026-11-18). tp and ep shapes reduce partial sums across ranks (tp: `wo` and `ffn_down`; ep: the expert outputs), so they are bounded by max(`GPU_TOL`, 1.5x libllama's own tensor gap) measured on the same model, prompt and teacher-forced tokens: libllama with `-sm tensor` over every visible GPU against libllama on one GPU, same top-5 metric as a share of the largest logit, worst over the prompt's last row and the 64 steps (`common::tp_anchor`, dense and MoE). Layer shapes compute what `single` computes and keep `GPU_TOL`. libllama has no expert-parallel split mode, so the tp anchor also bounds ep-2, ep-4 and tp2-pp2: looser than a true ep anchor, a known limit; on a four-GPU box the anchor splits four ways.
- **References.** CPU references (`n_gpu_layers` 0) set `op_offload` off: in a CUDA build libllama otherwise runs host ops of 32 or more rows on the GPU. A test of moved KV rows checks that the moved bytes are equal and compares only steps with the same rows and `n_kv` on both sides, because attention sums in an order that depends on both.
- Every comparison is printed before any assertion, and independent parts and shapes run to completion (the sharded MoE tests run every shape, print every step and assert once at the end).
- A mixing bug, not kernel noise, is declared if the same-shape builder-against-libllama gap approaches libllama's own batched-against-split gap, if the per-row gap grows with position or across sequences, or if a greedy flip is not a near-tie.

## Consequences

- The GPU bound is about twice the observed noise, and the anchor ties it to llama.cpp's behaviour on the same card.
- `TOL_GPU` stays the absolute floor for its other users (`admit`, `cpu_experts`, `kv_snapshot_decode`); they may need the relative rule once they run on CUDA.
- Forcing F32 matmul precision in the builders would remove the cuBLAS 16F noise but diverge from libllama and cost speed; not taken.
- The MoE builder selects the output rows after the last layer's attention and before its router and expert FFN, as libllama's `inp_out_ids` does (`track-moe-outrows`, streamed path in `track-moe-seg-outrows`); selecting only at the head made CUDA pick another `mul_mat_id` shape for the last layer (`20261116rb`). With routing recorded, a log-only router on all rows keeps the log complete. Tests comparing replay counts against engine stats prefill with `eval`; exit_regime traces taken with `eval_last` overstate last-layer prefill traffic slightly.
- A sharded load refuses `record_routing` (the router runs inside the rank graphs), so a near-tie router flip there is told from a defect only by the top-2 margin of `single` and by the anchor. If a sharded shape exceeds its anchored bound, the next step is routing capture in the sharded path.

## History

- 2026-11-16, addendum "gpu-smoke3" (`20261116gpu3`): the rule holds on the dense builder, including the open three-sequence case; the move subcases of `gpu_seq_cp_snapshot_and_move` failed and kept the gpu-parity gate strict.
- 2026-11-16, addendum "the move failures were the reference's" (`track-gpu-move`, data from `20261116gpu3`): the reference had another `n_kv` and row offset; the test compares only steps with equal rows and `n_kv`, tolerance unchanged; CPU references set `op_offload` off.
- 2026-11-16, addendum "gpu-smoke5 closes it" (`20261116gpu5`): every GPU parity test passes, the CPU anchor with `op_offload` off is exact.
- 2026-11-16, addendum "real models need their own anchor" (`20261116d2a`): `gpu_real_batched` failed at the fixtures' bound on both real models; the real-model anchor and the strict alone control above.
- 2026-11-16, addendum "the anchored re-run" (`20261116rb`, GPU-confirmed in `20261116d2a2`): Qwen3-8B passes; the 30B alone control's one failing row came from selecting output rows only at the head, fixed in the MoE builder (Consequences).
- 2026-11-17, addendum "CUDA against the CPU builder at b12426 is libllama's own gap" (`track-gpu-multiseq`, `20261116pc`): after the pin move the CPU side moved and CUDA did not; the CUDA-against-CPU test follows the anchored rule on its own tokens.
- 2026-11-17, addendum "the b12426 gpu_multi_seq shift is one router near-tie" (`20261117c`): one tied router choice explains the shifted sequence, confirming the anchor over a fixed bound.
- 2026-11-18, addendum "parity and slots use the relative rule" (`track-s2b-fixes`, `s2b-20261118-step0`): both failed at the absolute bound with top-1 equal; they now use `GPU_TOL`.
- 2026-11-18, addendum "MoE tensor and expert shapes use libllama's tensor gap as the anchor" (`track-moe-tp-anchor`, `s2c-20261118-step0`): tp-2 on the 30B failed the fixtures' bound with greedy tokens equal, the mistake the real-model addendum rules out; the sharded-shapes rule above (the dense test had the same fix, ADR 0069 History).
