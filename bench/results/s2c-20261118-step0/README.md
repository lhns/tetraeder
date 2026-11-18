# Session s2c-20261118-step0

- Status: diagnosis, no verdict: the gpu-parity gate failed on `gpu2_moe_shapes_match_single` at the fixtures' bound (ADR 0071); step 0 not run.
- Code and box: box 61174537, host 34742, 2x RTX 3090, driver 615.71.09.
- Raw files: this directory.

## From ADR 0071 (addendum "MoE tensor and expert shapes use libllama's tensor gap as the anchor", `track-moe-tp-anchor`, moved verbatim)

- `gpu2_moe_shapes_match_single` failed on tp-2 against `single` on Qwen3-30B-A3B Q4_K_M: greedy tokens equal at every step, but 60 of 64 steps exceeded `GPU_TOL` (step 0: 0.47 against 0.104; worst 1.877; up to 9.9x the bound). The test stopped at the first failing shape, so ep-2 and layer-2 never ran. This is the mistake the "real models need their own anchor" addendum (2026-11-16) rules out (a fixture bound on a real model at a changed reduction), which `gpu_sharded` had for the dense model too (ADR 0069, S2 addendum).
- Rule, following that addendum: tp and ep shapes reduce partial sums (tp: `wo` and `ffn_down`; ep: the expert outputs) across ranks, so they are bounded by max(`GPU_TOL`, 1.5x libllama's own gap), measured on the same model, prompt and teacher-forced tokens: libllama with `-sm tensor` over every visible GPU against libllama on one GPU, same top-5 metric as a share of the largest logit, worst over the prompt's last row and the 64 steps (`common::tp_anchor`, shared with the dense test). Layer shapes compute what `single` computes and keep `GPU_TOL`.
- libllama has no expert-parallel split mode (none, layer, row, tensor), so the tp anchor also bounds ep-2, ep-4 and tp2-pp2. That is looser than a true ep anchor and a known limit; on a four-GPU box the anchor splits four ways.
- The MoE tests now run every shape to completion, print every step and assert once at the end, so one box run shows tp-2, ep-2 and layer-2 (tp2-pp2, ep-4 and layer-4 on four GPUs).
- Routing is not printed: a sharded load refuses `record_routing` (the router runs inside the rank graphs), so a near-tie router flip is told from a defect only by the printed top-2 margin of `single` and by the anchor, whose own MoE gap is dominated by near-tie flips (1.1e-1 in the 2026-11-16 rerun). If tp-2 still exceeds the anchored bound, the next step is routing capture in the sharded path.
- Open until the next box: the anchored bound is predicted, not observed.
