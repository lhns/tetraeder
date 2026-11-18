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

## From ADR 0045 Setup (moved verbatim)

- Two boxes, every engine paired on the same box, same threads (CPU quota 15), same frozen prompt (`bench/prompts/exit_prompt.txt`, token hash `c743c72b8ffceadd`), 32768 prompt tokens, 64 greedy tokens, `scripts/exit_run.sh`:
  - **A**, box 58900330: RTX 3090, Ryzen 9 5950X, 91.6 GB container RAM, NVMe (5 to 8 GB/s direct reads). `bench/results/20261103/*-013730|021753|030753|032306.json`.
  - **B**, box 58930011: RTX 3090, Xeon E5-2683 v4, 45.7 GB container RAM (model about 3x RAM), slower disk (2 GB/s). `*-070656|080404|085333|090438.json`.
- Expert VRAM is equal: `scripts/llama_fit.sh` finds the most layers' experts llama.cpp fits on the 24 GB card at llama-bench's 32K context and ubatch 2048 (7 layers, `blk.[0-6].ffn_.*_exps=CUDA0;exps=CPU`; 8 fit llama-server but not llama-bench), and the builder gets the same expert bytes as VRAM slots (10.97 GB, 10 slots per layer). The builder's RAM tier is the container RAM minus 8 GB.

## From ADR 0045 Results (moved verbatim)

## Results

Decode and prefill in tok/s after a 32K prefill unless noted:

| Box | Engine | prefill | decode | builder decode ratio |
|---|---|---|---|---|
| A | **builder** (prefetch on) | **75.8** | **3.14** | |
| A | builder, prefetch off | 72.2 | 2.83 | |
| A | llama.cpp `-ot`, same tokens | 46.3 | 2.55 | 1.23 |
| A | KTransformers | 35.7 | 3.15 | 1.00 (0.998) |
| A | Colibri (2K depth) | 3.2 | 0.93 | |
| A | builder, random ids | 93.8 | 4.82 | |
| A | llama-bench, random ids (pp2048 / tg32 at depth 32K) | 49.0 | 3.53 | 1.37 |
| B | **builder** (prefetch on) | **26.4** | **0.69** | |
| B | llama.cpp `-ot`, same tokens | 21.9 | 0.43 | 1.60 |
| B | KTransformers, 2K depth | 8.7 | 0.19 | builder at 2K: 13.4 / 0.56 |

- KTransformers' 32K prefill on B was still in its first 4096-token chunk after 11 minutes and was stopped to stay within the 3 h box cap, so B pairs it at 2K (its 2K prefill rate, 8.7 tok/s, puts 32K near an hour).
- Random token ids route very differently from text (builder RAM hit 97.6% against 93.2%, 129 against 511 MB read per token), so the random-id pair is reported for comparison with llama-bench only; the text pairs are the verdict.
- Builder decode on A: 314 ms per token, compute 225, streamer 77; prefetch accuracy 0.85, 59% of the read time hidden. Per-layer full-hit rate 0.0 in every layer on both boxes (10 of 128 experts per layer in VRAM: every layer step routes at least one expert to the CPU); RAM-tier hit rate 0.93 (A) and 0.80 (B).
- **Parity.** The builder's 64 greedy tokens equal libllama's on both boxes (same token ids, 32K). The first token's top-2 ids agree; top-1 logit 39.49 (A) and 36.23 (B) against libllama's 38.72; top-1 margin 13.2 and 14.1 against 12.9, no near-tie. The 235B top-1 logit is off by 2% (A) and 6% (B), and the offset differs between boxes; the cause is open.

## From ADR 0045 Decision (moved verbatim)

- **Pass, with decode parity against KTransformers on box A.** The builder is faster than every baseline on prefill (1.6x to 3x) on both boxes and on decode against llama.cpp with tuned expert overrides (1.23x and 1.60x), Colibri and KTransformers on box B. On box A its decode equals KTransformers' (3.14 against 3.15, one pair each, inside run-to-run noise). ik_llama.cpp and FreeToken cannot run the model within 128 GB.

## From ADR 0045 Consequences (moved verbatim)

- Open, in order of expected gain: prefetch depth 2 (ADR 0038 Levers; implemented, not yet measured on the 235B: box B's disk made each 32K run 25 min and the window ran out), `cpu_poll`/`prefetch_threads`, a second paired KTransformers run on box A's class for a noise band.
- Not taken (the coordinator's FreeToken levers): splitting decode misses between CPU and PCIe gains little here because the CPU expert path already reads about 8 GB of weights per token in 225 ms, near the boxes' DRAM bandwidth, and PCIe DMA reads the same DRAM; prefill double buffering is not needed while prefill leads every baseline by 1.6x or more.
- Box cost of the exit work on 2026-11-03: about 2.9 USD (58894536 0.14, 58897870 and 58899229 about 0.05, 58900330 0.97 plus 0.92 traffic, 58930011 0.83, no traffic).

## From ADR 0045 (addendum "the decode tie does not hold at a 26-core quota", box C, moved verbatim)

## Addendum (2026-11-03, box 58958565): the decode tie does not hold at a 26-core quota

Box C, 58958565: RTX 3090, Xeon Gold 6330N, 26-core quota, 91.4 GB container RAM, direct reads at about 5.2 GB/s. Same frozen prompt, 32K, 7-layer `-ot` and 10.97 GB of builder expert slots (`bench/results/20261103/*-120800.json`):

| Engine | prefill | decode |
|---|---|---|
| builder, prefetch on (two runs) | 72.4, 71.8 | 3.77, 3.73 |
| builder, prefetch off | 69.8 | 3.66 |
| KTransformers 0.7.2 | 50.1 | 5.04 |
| ik_llama.cpp | killed after 22 s (out of memory) | |

- **Decode is 1.34x behind KTransformers on this box.** Builder decode per token: 257 ms, of which compute 112 and the streamer 128. The streamer reads 511 MB per token: 414 MB by the prefetch reader at 3.2 GB/s, with only 28% of that time hidden; the rest are synchronous misses. KTransformers reads the same experts through the page cache with 26 faulting threads. With 15 threads (boxes A and B) the builder's compute dominated and the reads hid better; at 26 threads compute halves and the reads become the step.
- **Verdict, revised:** Phase 2 passes on prefill on every box (1.4x to 3x), and on decode against llama.cpp, Colibri and ik_llama.cpp. Against KTransformers it passes only on 15-thread boxes: a tie on A, ahead on B. At a 26-core quota KTransformers decodes 1.34x faster, so the decode clause is not met there. The prefetch levers below are the work to close it; they are measured when boxes are allowed again.
- The lever stages of this session (`d2`, `d2poll0`, `d2fewer`) ran with prefetch off: the harness defaulted `TETRAEDER_EXIT_PREFETCH` to off while `LoadOptions` defaults to on (fixed). From those runs: prefetch on is +3% here (3.77 against 3.66), and `cpu_poll = 0` costs 90 ms of compute per token (2.72 tok/s), so ggml's default spin stays.
- The speculative pair (Track E's `spec_run.sh main-builder`) did not start: the draft's context did not fit beside 10.97 GB of builder slots (fixed in the script).

### Logit offset against libllama (Rule 2)

2K prompt, same token ids (`exit-drift-*-120800.json`), top-1 logit:

| run | top-1 |
|---|---|
| libllama, ubatch 2048 / 512 | 27.710 / 27.778 |
| builder, prefetch off / on | 27.638 / 27.638 |
| builder, 30 GB RAM tier | 27.389 |
| builder, RAM groups on the CPU (no op offload), prefetch off / on | 28.468 / 28.468 |
| CUDA fusion off: builder / libllama | 27.915 / 27.904 |

- With CUDA fusion off the builder equals libllama to 4e-4. With fusion on, both engines move with how ops are batched: libllama by 0.07 between ubatches, the builder by 0.25 between RAM slot counts, which change how its pair groups split.
- The CPU paths are exact for any RAM slot count, with prefetch off and at depths 1 and 2 (`cpu_long_prefill_is_exact_for_every_ram_slot_count`). On a GPU the synthetic fixture is bit-identical over RAM slots and prefetch (`gpu_cpu_policy_logits_do_not_depend_on_ram_slots_or_prefetch`, `gpu-tests-120800.log`). Expert bytes do not depend on the configuration.
- At 32K the offsets grow over 16 ubatches but are deterministic per configuration across boxes: prefetch off gives 37.892 on boxes A and C, prefetch on 39.486 on both. Prefetch changes which experts are RAM hits, hence the hits-first chunk composition and the batches the fused CUDA kernels see.
- Greedy output equals libllama's in every pair. Conclusion: rounding of fused GPU kernels over a different op batching, not a defect. Open: the small-RAM run with fusion off was cancelled with the session hold; it should equal 27.915.

## From ADR 0045 (addendum "Diagnosing the KTransformers decode gap", 2026-11-09, measured lead-in and estimates on box C data, moved verbatim)

On the 26-core box KTransformers decodes 5.04 tok/s (198 ms per token) against our 3.77 (257 ms: compute 112, streamer 128). We only measured KTransformers end to end. From its source (v0.7.2): experts run in place from an `np.memmap` of the GGUF (no repack, one `MADV_WILLNEED` at load, no prefetch); 26 pinned, spinning workers take 32-row tasks (384 gate/up and 128 down per layer), so misses page-fault in parallel inside the compute; the kernels are iqk Q4_K x Q8_K, VNNI on AVX-512, chosen at runtime from `/proc/cpuinfo` (`KT_KERNEL_CPU_VARIANT` overrides). Our run used `--kt-num-gpu-experts 0` and no deferral, so GPU experts and deferred experts are ruled out. Our box builds use `TETRAEDER_GGML_NATIVE=OFF`, so our CPU experts run AVX2 kernels on an AVX-512 Xeon.
2. **Page cache against a fixed RAM tier.** KTransformers' cache grows to the container limit (peak RSS 100 GB against our 83 GB budget), so more experts are resident and fewer bytes are read per token.
Compare only pairs from one session. The gap is 59 ms per token.
Order: `main-native` against `main` first (pre-box estimate: the microbench's native over portable ratio times the CPU-expert share of the 112 ms compute), then bytes per token and queue depth to separate 1 from 2. The effects can add up, so a partial result from each is still a result.

## From ADR 0046 (section "Measured", moved verbatim)

## Measured (bench/results/20261103/gateway, RTX 3090, 11 threads, 2 paired repetitions)

- SDK conformance (openai and anthropic Python SDKs, `tests/conformance`) passes against the fake in CI and against Qwen3-8B (libllama) and Qwen3-30B-A3B (builder) on the box. Claude Code's first request (`/v1/messages?beta=true`, 14 tools, `max_tokens` 32000, streamed) is answered.
- Against llama-server at concurrency 1, as ratios to llama-server (for TTFT, below 1 is better): decode 1.00 to 1.02, TTFT 0.63 to 0.77 (512 and 3K-token prompts), throughput 1.04 to 1.20; later turns of a growing conversation TTFT 0.89 to 0.94 (both reuse the prefix).
- Concurrency 4: throughput 0.41 (8B) and 0.59 to 0.61 (30B) of llama-server `-np 4`, TTFT 4.9 to 6.6 times longer (queueing). Phase 4-bound.
- API overhead: `serve` decode is 0.94 to 0.98 of `tetraeder bench` tg128; llama-server's is 0.93 to 0.97 of llama-bench.

- Claude Code proxy for the week of use (bench/results/20261103/claude-code-proxy, RTX A6000, Qwen3-Coder-30B-A3B Q4_K_M, 128K context, 10 tasks x 2 reps per engine, ABBA): tasks solved 15 of 20 on both engines; decode 0.88 of llama-server; prompt tokens served from cache 85% against 94%, because Claude Code's side requests evict the single cached sequence (llama-server keeps several prompts in host RAM). 12 requests (4 bursts of 3) got a 400 that llama-server accepted: `tool_result` content with `tool_reference` blocks (ADR 0049).

## From ADR 0046 (addendum "the week of use became a proxy run", 2026-11-08, the bar with the first run's share, moved verbatim)

On 2026-11-03 the user replaced the exit test's week of Claude Code use with a proxy run of a few hours: the same tasks against Tetraeder and llama-server on one box, ABBA. The bar: tasks solved at least llama-server's, prompt tokens served from cache at least its share (94% in the first run), and no 400 for a request llama-server accepts. The plan's Phase 3b exit test says so now. The first run (above) missed the cache and 400 parts; ADR 0049 fixed both, a rerun is open.

## From ADR 0046 (addendum "the A6000 decode gap", 2026-11-09, the measured paragraph, moved verbatim)

The proxy run's `tetraeder serve` had no `--engine`, so `auto` served the Qwen3 MoE builder (all experts resident), not libllama; llama-server ran libllama at the same pin. Settings otherwise match: flash attention on both (the builder always uses `flash_attn_ext`, llama-server `auto`), F16 KV, KV length padded to 256, 128K context, 5 threads, one slot, full-vocab logits to the host per token on both. The gap does not grow with depth: 0.90, 0.91, 0.89 and 0.91 at 0-10K, 10-20K, 20-30K and 30-40K tokens (summary.json `decode_by_depth`), about 1 ms per token. The 3090 gateway bench (1.00 to 1.02) sampled greedily; Claude Code's requests take the temperature path.

## From ADR 0048 Consequences (bullet "Measured 2026-11-03", moved verbatim)

- Measured 2026-11-03 (bench/quality/, bench/results/20261103/quality-*.json): against a BF16 reference the summed estimate is within 14% of the measured KL on Qwen3-8B (random mixes) and on Qwen3-30B-A3B (uniform Q6\_K to Q3\_K); Q2\_K is underestimated (30B: 0.94 against 1.25). Against a Q8\_0 reference the 30B samples showed a floor of about 0.002 and the sum overestimated light mixes about 6 times, so profiles use the most precise reference (`qwen3-30b-bf16.quality.json` supersedes `qwen3-30b.quality.json`). A greedy knapsack over the profile beats Q4\_K\_M, Q3\_K\_M and Q5\_K\_M at equal bytes on held-out text on both models.

## From ADR 0048 (update "format scale and knapsack checks", 2026-11-09, the fitted scale and its errors, moved verbatim)

- **`format_scale`** (optional, `{format: factor >= 1}`): a factor on that format's KL delta in the estimate, for formats the sum underestimates. The job fits it from the `uniform-<format>` check, `(measured - base) / (estimate - base)`, when the sum is more than 15% under (`UNDER_BAND` in `calibrate.py`). Old files lack it and mean 1. Applied by `QualityProfile::estimate`, `estimate_mix_scaled` and `ProfileQuality::loss`; the plain sum (`estimate_mix`), the recorded `checks[].estimate` and the knapsack order stay the ADR sum, so `compare_mix.py` rows still match. Qwen3-30B-A3B: Q2_K 1.3412, giving 0% error on uniform Q2_K (from 25%), at most 13.7% over the five checks (raw 25.4%), 9.2% after the power-law correction (ADR 0040). One measured point per format, so the scale is in-sample; pairs at other mixes are not measured.

## From ADR 0049 Context (moved verbatim)

The Claude Code proxy run (bench/results/20261103/claude-code-proxy, ADR 0046 "Measured") found two gaps against llama-server on the same model and tasks:

- Prompt tokens served from cache: 85% against 94%. ADR 0046 keeps one sequence; Claude Code interleaves side requests with a different prefix, which evict the main conversation (some requests reused 22 tokens). llama-server keeps evicted prompts in host RAM: its logs show `selected slot by LCP similarity` before every task and `making room for prompt cache entry, removing oldest entry (size = 1261 to 6760 MiB)` (server-llama-server-r1.log), the prompt cache of `--cache-ram` (default 8192 MiB, llama.cpp PR 16391).
- 12 requests (4 bursts of 3, Claude Code's retries) got a 400 that llama-server accepted: `tool_result content: block type 'tool_reference' is not supported here` (the proxy logs' `error_message`). Claude Code's tool search answers with `tool_reference` blocks for tools sent with `defer_loading: true`. ADR 0028 also rejects images, other unknown blocks and assistant prefill; llama-server accepts more.
