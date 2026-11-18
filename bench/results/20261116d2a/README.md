# Session 20261116d2a

- Status: unclean, no verdict: Phase 4 D2a stopped at the gpu-parity gate (`gpu_real_batched` failed at the fixtures' bound on both real models, ADR 0071).
- Code and box: results commit be418ea; box 60877696, RTX 3090.
- Raw files: this directory.

## From ADR 0071 (addendum "real models need their own anchor", moved verbatim)

- `gpu_real_batched` failed on Qwen3-8B (worst 3.0e-2 of the largest logit, no argmax flip) and Qwen3-30B-A3B (worst 8.3e-2, five flips) at the fixtures' bound, without measuring libllama's own gap on those models.
- On Qwen3-8B the one row whose KV cells and step shapes coincide in both engines (sequence 0, first step) was bit-equal; every other row differed by a flat 1.3e-2 to 3e-2, with no growth by position or sequence. The builder reserves rows per sequence where libllama's unified cache allocates cells in step order, so CUDA flash attention sums in another order; Q4_K's q8_1 activations plausibly grow that difference over the layers. The masks are correct.
- Decision: real-model GPU checks anchor on libllama's own gap on the same model and card (batched against split, and against the same steps with the cells shifted by 7 and 256 cells); the bound is max(GPU_TOL, 1.5x the largest own gap); the builder is asserted within it, batched against split and against libllama at the same shapes. One sequence alone, where cells and shapes coincide, is asserted at GPU_TOL with no widening: the strict check against divergences on real weights. Argmax flips print their margins.
- Open until the re-run (`scripts/gpu_real_batched.sh`): MoE row (0,19) differs by 7.7e-3 although its cells coincide, possibly cross-sequence coupling through `mul_mat_id`'s per-expert tiling; the alone control decides between that and a divergence specific to the MoE builder.
