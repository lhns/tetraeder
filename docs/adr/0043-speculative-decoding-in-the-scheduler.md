# 0043. Speculative decoding in the scheduler: n-gram and draft-model drafts, batched verification

Status: Proposed (pulled forward from Phase 4 for the Phase 2 exit regime). Measured 2026-11-03: not a gain yet in the exit regime, see Results.

## Context

The exit regime (Qwen3-235B-A22B Q4_K_M, RTX 3090, model larger than RAM, ADR 0035, 0037, 0038) decodes at 2.2 to 2.4 tok/s after a 32K prefill (bench/results/20261102/exit-main*-203803.json): per token about 450 ms, compute 190 (attention on the GPU, routed experts on the CPU) and streamer 225 to 250 (about 600 MB of missed experts read per token, plus copies), RAM tier hit rate about 0.92. Almost all of that is per expert, not per token: an expert read from disk or streamed through the CPU's memory bus costs the same for one row as for seven. A step that verifies k drafted tokens reads each needed expert once for all k+1 rows, so its cost grows with the union of routed experts, not with k. Consecutive tokens share experts, so the union grows slower than 8(k+1) per layer.

The builder already runs multi-token steps (`Qwen3Moe::eval` returns the logits of every position) and its KV cache needs no clearing after a rejection: `run()` writes K and V at positions `n_past..n_past+n` and masks every column above `n_past + i` for row i, so rows left by rejected drafts are invisible to later steps and overwritten by them. Rolling back is setting `n_past`.

Plan rows 27 and 52 put speculative decoding (n-gram, draft model, MTP where the GGUF has heads, self-speculative low-bit copy) in the scheduler crate, Phase 4, with the metric `tetraeder.spec.acceptance` per method.

## Decision

- **Where.** The loop lives in `crates/scheduler` (module `spec`), generic over two traits; the builder gains only `Qwen3Moe::rewind(n_past)`.
  - `Target`: `n_vocab()`, `n_past()`, `eval(&[i32]) -> logits of every position`, `rewind(n_past)`. Implemented for `Qwen3Moe` in the scheduler.
  - `Drafter`: `draft(history, max) -> Vec<i32>` where `history` is every token so far (prompt and output, the last one not yet in the target's KV); it may return fewer than `max`, also none. `method()` names it for metrics.
  - `Speculator::step(target, drafter, history)` and `generate(target, drafter, history, n)`: append the target's own tokens to `history` (whose last token is pending, not yet in the target's KV), as plain decoding would produce them; `generate` may overshoot `n` by up to k.
- **Verification.** Each step evaluates `[g, d1..dk]`, where `g` is the last token chosen and `d` the drafts. Position i's choice is compared with `d(i+1)`; the first mismatch's choice (or, when all match, the last position's) is the step's extra token. Emitted per step: accepted drafts plus one. The target is then rewound to `n_past_before + 1 + accepted`. A step with no drafts is a plain decode step.
- **Choice rule.** Greedy uses one `argmax` (first maximum wins) for both paths. Temperature above 0 is sample-and-match: the token at absolute position p is sampled from the target's softmax with an RNG seeded by `(seed, p)`, and a draft is accepted when it equals that sample. Plain sampling with the same rule gives the same tokens, so speculation is testable bit for bit at any temperature. For deterministic drafters (n-gram, greedy draft model) this accepts with probability p(d) and resamples from p without d, the same as rejection sampling with a one-hot proposal; a stochastic draft distribution would accept more with the min(1, p/q) rule, which stays out of scope until there is one.
- **Drafters.**
  - `NgramDrafter` (prompt lookup): the most recent earlier occurrence of the last m tokens of the history, m from `max_ngram` down to `min_ngram` (default 4 to 2), proposes the tokens that followed it. Free; strong on repetitive or quoted text and code.
  - `DraftModel`: a small model of the same tokenizer (Qwen3-0.6B or 1.7B GGUF) through libllama (`ggml-sys` `llama::Context`, needs a new `seq_rm` binding). It keeps its own KV of the history, rolls it back to the common prefix with the new history, decodes the rest, then drafts greedily and stops early when its top probability is below `p_min` (default 0, as the pinned llama.cpp, whose draft default is also 3 tokens). Vocab size must equal the target's.
  - Self-speculative (a low-bit copy of the same model, plan row 52) and MTP fit the same trait; later.
- **Draft length.** `DraftLen::Fixed(k)` or `DraftLen::Adaptive { max }`. Adaptive keeps an acceptance estimate a (accepted over accepted plus rejections, exponential moving average) and a step time per batch size (moving average, measured by the speculator; unseen sizes interpolated, optimistic before the first sample) and drafts the k that maximizes `(1 - a^(k+1)) / (1 - a) / T(k+1)`. It costs nothing on a resident model, where T is nearly flat, and keeps k small in the streaming regime when the expert union makes T steep.
- **Verify batch cap: 7 tokens (k at most 6).** This keeps every CPU K-quant matmul of the step on `vec_dot`, the kernel single-token steps use: the vendored tiled K-quant kernel takes over at 8 rows (`ggml_tiled_min_batch`, for `mul_mat` and per expert in `mul_mat_id`, ADR 0034). Above 7 the rows of one expert or of a dense weight round differently from decode.
- **Exactness, precisely.** Speculation emits exactly what plain decoding emits if the verify step's logits equal the single-token steps' logits. ggml does not promise that, and where it fails a near-tied argmax can flip and the outputs diverge, as between any two kernel paths (ADR 0034). Known batch-dependent kernels in the pinned ggml:
  - CPU flash attention: a single query row with at least 512 KV entries takes the split-KV path (KV chunked across threads, partials merged); more rows take the per-row path. So CPU decode at a KV of 512 or more is not bitwise equal to a verify step.
  - CPU F32, F16, Q8_0, Q4_0 matmuls on AVX: `llamafile_sgemm` from 2 rows, `vec_dot` for 1 (the F32 router of every Qwen3 MoE layer).
  - CUDA: MMVQ picks its warp count and rows per block by column count (`calc_nwarps`, `calc_rows_per_block`; 1 column may use twice the warps), and the flash-attention kernel configuration depends on the query count.
  So the guarantee is: bit for bit on the CPU with K-quant weights, a KV below 512 and no AVX2 (checked in tests). On the AVX2 CI runners a 7-row step's logits differ from the one-row steps in the last bits (2026-11-03, row 1 onward), so there the tests require agreement to a relative 1e-4 and allow speculative output to leave plain greedy only at a near-tie of the plain run (top-2 margin below 1e-3 of the largest logit); elsewhere the greedy tokens are checked against plain greedy on the real model and every divergence is reported with the plain run's top-1 margin at that position. A batch-invariant mode (patches that keep small batches on the single-row kernels) is the follow-up if divergences show up on real text.
- **Metrics (Rule 6).** `tetraeder.spec.acceptance` (histogram of accepted over drafted per verify step with drafts), `tetraeder.spec.drafted` and `tetraeder.spec.accepted` (counters), all with attribute `method` (`ngram`, `draft`). Dashboard panel: acceptance rate and drafted/accepted rates per method. Alert `TetraederSpecAcceptanceLow`: acceptance below 0.3 for 15 minutes while drafts are issued (speculation then likely costs more than it saves).

## Consequences

- Failing tests first (`crates/scheduler`): a toy target checks the loop (fixed and adaptive k, every drafter, rollback, temperature) against plain decoding; on the synthetic Q4_K_M MoE fixture, resident and with streamed experts (`MissPolicy::Cpu`), speculative output and the logits at every emitted position equal plain greedy bit for bit with an oracle drafter (everything accepted, most rows per expert), a random drafter (rejection and rollback every step), the n-gram drafter and the draft model (libllama on the same fixture).
- Measurement (`crates/scheduler/tests/spec_regime.rs`, the exit regime's `TETRAEDER_EXIT_*` knobs plus `TETRAEDER_EXIT_SPEC_*`, driven by `scripts/spec_run.sh`): one prefill, then one long greedy decode whose 32-token blocks rotate between plain, n-gram and draft model, so every method sees the same cache history and none decodes tokens another already pulled into the RAM tier; then a plain replay of the whole continuation from the prefill checks the tokens. llama.cpp runs `llama-server` with the same `-ot` placement, once without and once with `-md` and the same draft (per-request speculative settings are compiled out of the pinned server, so these are two runs, each with its own prefill, which also resets the page cache to a post-prefill state). Reported per method: tok/s, acceptance, tokens per step, per-token profile and disk bytes per emitted token.
- Speculation trades extra expert reads per step for fewer steps. If the expert union grows nearly linearly with k at the 235B's routing, the gain is small and adaptive k should fall back towards plain decoding; the measurement decides the default.
- A draft model costs VRAM: Qwen3-0.6B Q8_0 is 0.6 GB plus 3.8 GB of F16 KV at 32K (28 layers, 8 KV heads of 128), inside the about 7 GB the exit placement leaves free on a 24 GB card.

## Results (2026-11-03, box 58895206, host 34031, RTX 3090, 91.7 GB container, 5 threads)

Files in bench/results/20261103/. Draft: Qwen3-0.6B Q8_0, fully on the GPU, greedy, p_min 0. Builder: `spec_regime.rs` with op offload and prefetch on (the exit configuration), 32K prefill of the exit prompt, then 3 rounds of 32-token blocks rotating plain, n-gram (adaptive), draft (adaptive, max 6) and draft (fixed 3, llama.cpp's default). llama.cpp: `llama-server` with the same `-ot` (layers 0 to 3 experts on the GPU), 128 tokens, once without and once with `-md ... --spec-type draft-simple` (n_max 3).

| 235B, 32K natural text | tok/s | tokens per step | acceptance | verify step ms (rows) |
|---|---|---|---|---|
| builder plain | 2.67 | 1 | | 375 (1) |
| builder draft, k=3 | 1.76 | 3.88 | 0.96 | 2178 (4) |
| builder draft, adaptive | 1.13 | 6.87 | 0.98 | 6032 (7) |
| builder n-gram, adaptive | 1.12 | 6.69 | 0.95 | 5943 (7) |
| llama.cpp, no draft | 2.53 | 1 | | |
| llama.cpp, draft n_max 3 | 2.47 | | 0.97 | |
| llama.cpp code text 4K, no draft / draft | 3.13 / 2.64 | | 0.55 | |

- **Exact on the box.** The plain replay of all 403 continuation tokens (every method mixed) equals the speculative output: on this text and this build, the CUDA and CPU kernel differences listed above did not flip a token.
- **Verification costs more than it saves, in both engines.** Acceptance is very high (the continuation of the ADR text is predictable), yet a 4-row step costs 5.8x a 1-row step in the builder and llama.cpp gains nothing (0.98x natural, 0.84x code).
- **Cause in the builder: op offload.** A CPU pair group with 4 or 7 tokens has 32 or 56 rows, which crosses ggml's op-offload batch threshold (32), so the step uploads every CPU-routed expert to the GPU over PCIe. A 1K-depth diagnostic (`spec-diag-tiled0` vs `spec-diag-nooffload`): 7-row steps 5560 ms with offload, 1087 ms without; 1-row steps unchanged. Without offload a verify step costs about 1.5x (4 rows) and 2.2x (7 rows) a 1-row step. With the main run's acceptance that projects to about 2.5x (k=3) to 3x (k=6) the plain decode rate, not measured yet.
- **Forcing the tiled K-quant kernel for every row count** (`GGML_CPU_TILED_MM_FORCE=1`, `spec-diag-tiled1`) makes 1-row steps 2x slower; no help.
- The adaptive policy behaves as designed: it picked k=6 on the high-acceptance main text (where its cost model was misled by the offload pathology only after the first steps) and mostly k=0 on the 1K diagnostic text with 0.14 n-gram acceptance.

Decisions from this: (1) op offload must not apply to decode or verify steps: the threshold should count tokens, not pair rows, or the builder should offload only prefill ubatches (a change in `qwen3moe.rs`, owned by the Phase 2 track); (2) rerun the main pair with that fix before choosing defaults; speculation stays off by default until then.
