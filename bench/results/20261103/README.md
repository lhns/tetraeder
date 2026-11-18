# Session 20261103

- Status: several boxes of 2026-11-03. Phase 2 exit runs on boxes A, B and C (clean; first verdict pass on A and B, the KTransformers decode tie not held on box C at a 26-core quota, ADR 0045), the first Claude Code proxy run (missed the Phase 3b cache and 400 clauses, ADR 0046), the gateway bench, planner fixtures (MoE clean, dense partial), the speculation runs (clean, negative, ADR 0043) and quality calibration. Pre-build estimates sized on box C are labelled as such below.
- Code and box: commits of 2026-11-03 (results commits e740d50, 8b8264a, cf18cf9, 2123524 and others, see git log of this directory); box A 58900330 (RTX 3090, Ryzen 9 5950X), box B 58930011 (RTX 3090, Xeon E5-2683 v4), box C 58958565 (RTX 3090, Xeon Gold 6330N, 26-core quota, 91.4 GB), proxy box 58947208 (RTX A6000), planner box 58943840 (RTX 3090), speculation box 58895206 (host 34031, RTX 3090); gateway container driver 610.57.04, others unknown.
- Raw files: this directory.

## From ADR 0038 (section "Correction and default on (2026-11-03)", moved verbatim)

## Correction and default on (2026-11-03)

- **No first-run divergence.** Only one run ever produced `[4370, 69415, ...]`: box 58813464 (`exit-main-120520`). Every 32K run on box 58872072, its first included, gave `[7294, 59549, ...]`. The two boxes ran different commits, and `exit_run.sh` built the prompt from the checkout (`cat AGENTS.md docs/adr/*.md`): 87,924 against 98,881 bytes, first difference at byte 78,551 (about token 20K). 8K prefills shared their text and agreed across boxes. The prompt is now frozen in `bench/prompts/exit_prompt.txt` (the 0956ccf text) and its token hash is in every exit JSON. On box 58900330 the frozen prompt gives `[7294, 59549, ...]` again, and libllama with the same token ids gives the same 64 greedy tokens (top-1 margin 12.9, no near-tie).
- **Default on.** Box 58900330 (RTX 3090, 15 threads, 91.6 GB, bench/results/20261103/exit-main-*-021753.json), 32K, same expert VRAM as llama.cpp's best `-ot` (10.97 GB): decode 2.83 tok/s off, 3.14 on (+11%); prefill 72.2 and 75.8; read time hidden 59% (29% on the 5-thread box). Prediction now runs only where it prefetches (Levers above), so `LoadOptions::prefetch` defaults to true.

## From ADR 0038 (section "Reads that hide (2026-11-03, box 58958565)", moved verbatim)

## Reads that hide (2026-11-03, box 58958565)

At a 26-core quota the decode step is 257 ms: compute 112, streamer 128. The streamer reads 414 MB per token by the prefetch reader at 3.2 GB/s, 28% hidden, and 97 MB synchronously; synchronous reads of the same layer steps reach 5.2 GB/s. Per layer the next layer's read (about 4.4 MB) has one segment (about 1.2 ms) to land. Levers, in order of expected gain:

1. **Depth 2, actually on** (implemented above; never measured because the harness ran it with prefetch off). It doubles the window. Expected: the 95 ms wait drops to about 40.
2. **Reader queue depth** (implemented): the reader's device reads are 1 MiB at queue depth 16 (`IoOptions::chunk_bytes`, new) instead of whole slabs, three requests per expert. Expected: 3.2 to about 5 GB/s.
3. **Reads start before the layer's own misses** (implemented): `prefetch_ahead` runs right after the layer is planned, not after its synchronous reads. The reader has its own direct files, so both proceed at once. That adds the synchronous read time to the window.
4. **RAM tier by frequency** (not implemented): the LRU RAM tier hits 93.2% with 77 of 128 slots per layer. Keeping the most-routed experts, as the VRAM hot set does, should cut the misses; measure first from recorded routing, offline. Recorder, replay and threshold: ADR 0050 (c).
5. **RAM reserve** 8 to 5 GB (not changed): peak RSS was 84.9 of 91.4 GB, so about 3.6% more slots, a small gain.

Together, 1 to 3 should bring the streamer near the synchronous misses alone: about 150 to 170 ms per token, or 6 to 6.7 tok/s, against KTransformers' 5.04 on the same box. They are measured when boxes are allowed again.

## From ADR 0040 Decision ("Quality profile and mix", moved verbatim)

- **Quality profile and mix** (`planner::profile_quality`). `ProfileQuality` implements `QualityModel` from the ADR 0048 profile: a group's KL delta at a format is spread over the units that hold it by their share of the group's params; a format a covering group did not measure, or a unit no group covers, is unknown. `mix_at_budget(profile, bytes)` picks one format per profile group with the search's greedy order (start at the smallest format, take the hull step with the largest KL drop per byte while it fits, then spend the rest on single upgrades), the same procedure as the reference knapsack of `scripts/quality/compare_mix.py`. `tetraeder-plan mix --profile p.json --budget-bytes n [--tensors names.json] --out mix.json` writes `{tensor: format}` for `compare_mix.py --mix` (group names from llama.cpp tensor names, as `qlib.py` builds them), or `{group id: format}` without a tensor list. **Correction:** the summed estimate is mapped to the measured scale by `measured = exp(a) * estimate^b`, least squares on `(ln estimate, ln measured)` over the profile's `checks` (closed form, deterministic, `b > 0` so it never reorders mixes; identity below two usable checks). Band = largest `|measured - predicted| / measured` over the checks. Qwen3-8B (6 checks): raw 66%, corrected 20% (leave-one-out 60%), a = 0.175, b = 1.083. Qwen3-30B-A3B (4 checks): raw 477%, corrected 31% (leave-one-out 117%), a = 0.433, b = 1.589. The band is in-sample and the checks are few; on the greedy mixes of `compare_mix.py` (held-out text, light and structured mixes) the correction is worse than the raw sum: Q4_K_M/Q3_K_M mixes measured 0.0311/0.0840 on the 8B (raw sum -19%/-17%, corrected -29%/-21%) and 0.0216/0.0531 on the 30B (raw +22%/+13%, corrected -78%/-67%). The checks are uniform and random mixes, so the correction does not transfer to knapsack mixes; the band is a statement about the checks only and the planner's claim rests on the measured rows, not on the corrected number. A better fit needs checks that look like knapsack mixes (a calibration-job change). Selection uses the ordering, which the correction does not change. Exit test (`tests/quality_exit.rs`, no hardware): at the bytes of the Q4_K_M preset the planner's mix equals the measured `greedy-at-Q4_K_M` mix on both models and its measured KL is below the preset's.

## From ADR 0040 Consequences (MoE fixture and partial dense fixture, moved verbatim)

- **MoE fixture (2026-11-03, box 58943840, RTX 3090 with 14 GiB left free by the balloon, 9 threads, Qwen3-30B-A3B Q4_K_M, pp2048 + tg128, `bench/results/20261103/planner-moe/`).** 24 candidates measured twice, 7 anchors: regret 0.000 (analytic and final pick both `c05`: experts of layers 31 to 47 on the CPU, ubatch 2048, op offload, 9 threads), Spearman 0.997, anchor error band 9.2%. Request time of the pick 3.14 s against llama.cpp's own fit (`-fitt 1024`) 4.52 s, hand-tuned `-ncmoe 13` 4.67 s and all experts on the CPU 14.0 s, same box and threads. The candidates span 3.1 to 59 s, so the rank correlation is easy here; regret is the sharper number.
- **Dense fixture, partial (same box, Qwen3-32B Q4_K_M, 18 GiB free, `bench/results/20261103/partial-planner-dense/`).** Cut by the 3 h box cap after 11 of 24 candidates (the CPU-only anchors at pp256 took 25 minutes; they are now pp64 tg16 at the candidates' thread counts only). Over those 11: regret 0.000, Spearman 0.991, band 5.8%; baselines not yet measured. Not an exit-test fixture until complete.

## From ADR 0043 Results (moved verbatim)

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

## From ADR 0044 Context (moved verbatim)

ggml's scheduler offloads an op whose weights live in host memory to the GPU when its batch size is at least `op_offload_min_batch_size` (32 on CUDA). For `MUL_MAT_ID` the batch size is `ne[2]` (`get_op_batch_size`, ggml-cuda.cu). In libllama that is the token count. The builder's CPU pair groups (ADR 0035) put one row per (token, expert) pair into `ne[2]`, so a 7-token step at top-8 has 56 rows and every routed RAM expert of the step crosses PCIe. Track E measured it on the 235B at 1K depth (bench/results/20261103/spec-diag-*.json on track-e-spec): a 7-token step 5560 ms with op offload against 1087 ms without, a 4-token step 879 against 765. Single-token steps are unaffected.

## From docs/m2-plan.md, section "Context" (moved verbatim)

Today the gateway reaches 0.41 to 0.61 of llama-server's throughput with 4 parallel slots (`-np 4`, ADR 0046). The scheduler and both engines run one sequence at a time.
