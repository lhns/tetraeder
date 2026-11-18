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

## From ADR 0050 Context (exit regime numbers, moved verbatim)

Exit regime (ADR 0045 addendum, box C, 26-core quota): Qwen3-235B-A22B Q4_K_M, RTX 3090, 91.4 GB RAM, 32K prefill, 10 VRAM slots per layer (10.97 GB), 77 RAM slots per layer. Decode 3.77 tok/s against KTransformers' 5.04. Per token 257 ms: compute 112, streamer 128. The streamer reads 511 MB per token: 414 MB by the prefetch reader (3.2 GB/s, 28% hidden) and 97 MB synchronously (5.2 GB/s, about 19 ms). RAM tier hit rate 93.2%, so one point of hit rate is about 75 MB, or 14 ms per token when the read is not hidden. Levers 1 to 3 of ADR 0038 are expected to bring the step to 150 to 170 ms. This ADR designs the next three, each sized against those numbers. The bar moves too: llama.cpp's expert cache with a CPU share of misses (ggml-org/llama.cpp#30966) is the new baseline (ADR 0045 addendum).

## From ADR 0050 Context ("(a) Is the layer's work overlapped today?", sizing paragraph moved verbatim)

What overlap can buy is small on the GPU side: with 10 of 128 experts in VRAM, the VRAM group is one to three experts per token (tens of microseconds per layer), and attention L+1 depends on the CPU group of L. The independent work inside a layer window is the synchronous miss read (about 19 ms per token) against the CPU compute of the experts that are RAM hits (about 100 ms per token, 1.1 ms per layer). So this lever is mostly "reads under CPU work", not "GPU under CPU work", and it is not the main part of the KTransformers gap (that gap is read bytes and how much of them hide, ADR 0045 addendum).

## From ADR 0050 Decision (expected gains of (a) and (b) and the size of the (c) threshold, estimates sized on this session, moved verbatim)

- **Expected gain.** The 97 MB synchronous misses (about 19 ms per token on box C, plus their bounce copy) hide behind about 100 ms of hit compute, minus the extra graphs: about 1.4 more per layer (the hit graph always, the miss graph in the 38% of layer steps that miss, `1 - 0.932^6.5`), at 20 to 40 us each on a CPU-only scheduler, 3 to 5 ms per token. Net about 12 to 16 ms per token: 5 to 6% of 257 ms, 8 to 10% of a 160 ms step after the ADR 0038 levers. Stage 2: at most 2 to 4 ms.
- **Expected gain.** Decode: GPU attention reads 6.36 GB of KV per token at 32K, about 7 to 10 ms on the RTX 3090; Q8_0 halves that (3 to 5 ms saved), Q8_0/Q4_0 saves 4 to 6 ms. Two or three more VRAM slots move about 2 to 3% of the routed pairs from the CPU to the GPU, about 3 ms of CPU expert compute. Together about 6 to 9 ms per token, 3% of 257 ms, 4 to 5% of a 160 ms step. Prefill may lose a little (F16 conversion per layer); acceptance: prefill no more than 5% slower, paired. The larger value is outside decode speed: room for the draft model of ADR 0043 and contexts beyond 32K. The replay of (c) prices the slot part offline (VRAM 10 against 12 and 13) before the box run.
- **Decision threshold.** Build the tier policy only if, on a 235B text trace (frozen prompt, 32K, 256 decode tokens), with prefetch replayed, at VRAM 10 and RAM 77 and at the RAM count of a 128 GB box (about 100), the best `freq:P` cuts disk bytes per decode token by at least 10% against `lru` without raising synchronous miss bytes. 10% is about 51 MB, about 10 ms per token when not hidden, 4% of the step, above the noise band. If `belady` is within 10% of `lru`, stop: no online policy can pass.

## From ADR 0050 Consequences (gap arithmetic, moved verbatim)

- None of the three closes the 1.34x gap alone (257 to 191 ms needs 66 ms). Their sum, 25 to 35 ms, matters if ADR 0038's levers 1 to 3 land near 170 ms rather than 150; they are ranked (a), (c) if the replay passes, (b) for decode speed, and (b) first for draft-model room.

## From ADR 0051 Context (cache share by class, tool calls; moved verbatim)

**Cache share by class** (`scripts/claude_code_cache_classes.py`; conversations inferred from token counts, 64-token slack):

| Uncached prompt tokens, share of all prompt tokens | Tetraeder | llama-server |
|---|---|---|
| First request of a conversation (Claude Code's prompts share 21 to 23 tokens) | 2.60% | 2.70% |
| Turn suffix (new turns, re-rendered output) | 1.79% | 2.59% |
| Mid-prompt divergence, shared part already cached | 0.91% | 0.58% |
| Evicted by another conversation | 9.25% | 0.51% |
| **Served from cache, measured** | **85.4%** | **93.6%** |
| Served from cache with ADR 0049, estimated | 94.3% | 94.1% |
| Ceiling of any prefix cache (first and suffix are new tokens) | 95.6% | 94.7% |

The eviction loss is three sessions (r1 feature-late-fee, r1 and r2 rename-compute-total) where Claude Code alternated its main conversation (11K tokens and growing) with a subagent (5K and growing), each request re-prefilled from token 22. ADR 0049 restores the other conversation from host RAM; 8.8 of the 9.25 points come back.

**Tool calls.** No HTTP error beyond the 12 `tool_reference` 400s that ADR 0049 fixed (`protocol_errors` 15 = those 12 plus 3 sessions at Claude Code's turn limit; llama-server has 3 such sessions too). Stop reason `tool_use` on 460 of 483 Tetraeder and 586 of 624 llama-server responses. One parser artifact: the last response of tetraeder-r1-write-shipping-tests (209 tokens, `end_turn`) ended with a bare `<tool_call>` before end of generation; `step_tagged` found an empty block and emitted it as text, which ended up in Claude Code's result. The task passed. No llama-server result shows a marker. Arguments Claude Code rejects (schema validation, unknown tool names) come back as `is_error` tool results in the next request body, so they are unmeasured for both engines.

## From ADR 0051 Decision (mixed lines moved whole; the ADR keeps the decisions)

**No prompt checkpoints.** The KV rows of a prefix do not depend on later tokens in a full-attention model, so ADR 0046 (truncate the live sequence to the longest common prefix) and ADR 0049 (restore a saved sequence, then truncate) already reuse the longest common prefix of any live or saved sequence at token granularity: editing or branching a long history keeps everything before the edit. Checkpoints only pay where state cannot be cut at an arbitrary token: recurrent or linear-attention layers, sliding-window ring buffers, caches kept in blocks. No served model has such state (qwen3moe builder, libllama dense Qwen3). On this traffic they could touch only the divergence class (0.9%), where the shared part is already cached (3,486 of 18,332 tokens after the `tool_reference` retries, 17,226 of 22,424 in r2 write-shipping-tests). Revisit with the first hybrid or sliding-window model, in Phase 4's `kvcache` prefix tree. If restore time shows up in `tetraeder.kv.prefix.restore.duration`, copy only the rows up to the common prefix; that needs no checkpoints.

The cache bar of the Phase 3b exit is llama-server's share in the same run (93.6% here, quoted as 94%), not a fixed number: the estimate beats it by 0.7 points, within what one run moves.

## From ADR 0051 Consequences (moved verbatim)

- Nothing to build for the prefix cache; the rerun shows whether ADR 0049 reaches llama-server's share. Risk: the 8 GiB default holds about 87K tokens of the 30B's f16 KV (98,304 bytes per token); the run's largest prompt was 70.8K, so a long main conversation plus a subagent can evict. `PrefixCacheThrash` fires then; raise `--cache-ram` if the box has the RAM.

## From ADR 0052 Context ("What dispatch buys for the builder", mixed bullet moved whole)

- **What dispatch buys for the builder.** At this pin the x86 decode-path Q4_K and Q6_K dot products (`ggml-cpu/arch/x86/quants.c:2102-2280`, `:2490-2724`) and `quantize_row_q8_K` (`:505`) are AVX2 only: no `_mm512`, no `dpbusd`. ggml does have AVX-512 VNNI K-quant kernels (`ggml-cpu/tiled/tiled-kernel.cpp:83-207`, `_mm512_dpbusd_epi32`), but `mul_mat_id` uses them only when an expert gets 8 or more rows (`tiled.cpp:1202`), so they serve prefill, not decode. VNNI is otherwise used only through `mul_sum_us8_pairs_float` (Q4_0, Q8_0 and similar), AVX-512 in a few helpers, the repack GEMMs and llamafile sgemm (F32, F16, BF16, Q8_0, Q4_0, Q5_0, IQ4_NL). The repack Q4_K GEMV (`repack.cpp:1464`) has no `_mm512` either, and the builder does not repack (ADR 0039) nor use the AMX buffer type. So for the Qwen3 Q4_K_M experts an AVX-512 variant changes only compiler code generation (EVEX encodings, 32 vector registers, auto-vectorized float ops). So dispatch alone cannot deliver ADR 0045 hypothesis 3 for the builder. Whether KTransformers' 1.34x comes from its iqk VNNI kernels, parallel faulting I/O or the page cache is still open (none of the 0045 addendum stages have run).

## From ADR 0052 (addendum "2026-11-09: build mode per environment", mixed bullet moved whole)

- **Boxes and local dev build native** (`TETRAEDER_GGML_NATIVE=ON`: one CPU variant, static). A fresh box with the DL default would compile 14 CPU variants plus CUDA, an estimated 30 to 40 paid minutes; the native build is the pre-dispatch 9 to 11 minutes. `.cargo/config.toml` `[env]` sets it for local builds and `docker/box/Dockerfile` for the image; `scripts/exit_run.sh` defaults it too. A process env var wins over `[env]`.

## From ADR 0053 Context (mixed paragraph moved whole)

The builder's CPU experts (Qwen3-235B Q4_K_M, one token, `mul_mat_id` over 8 of 128 experts) run ggml's `ggml_vec_dot_q4_K_q8_K` and `ggml_vec_dot_q6_K_q8_K` one row at a time (`ggml-cpu.c:1483-1544`), which are AVX2 only at our pin (`arch/x86/quants.c:2102`, `:2490`). KTransformers decodes 1.34x faster on the 26-core Ice Lake box C (ADR 0045) with ik's iqk kernels on AVX-512 VNNI. The user asked whether this is already solved upstream.

## From ADR 0053 Consequences ("Expected gain, box C", an estimate sized on this session, moved verbatim)

- **Expected gain, box C:** about 8 GB of CPU expert weights per token (87 of 94 layers, gate/up Q4_K, down Q4_K/Q6_K) in 112 ms of compute is about 71 GB/s, against roughly 120 to 140 GB/s of DRAM on an 8-channel 6330N. Compute halved from 15 to 26 threads (ADR 0045), so it is instruction bound, not bandwidth bound. A dpbusd GEMV does about half the integer instructions per block; estimate compute 112 to 70-85 ms, floor about 60 ms from bandwidth. Decode 257 to about 220-230 ms (3.77 to 4.3-4.5 tok/s, +12 to 19%). That closes about half of the 59 ms gap to KTransformers; the streamer's 128 ms (ADR 0045 hypotheses 1 and 2) stays the other lever. No gain on boxes without AVX-512 VNNI. The box run decides; until then this is an estimate.

## From ADR 0054 Context (moved verbatim)

Box C (ADR 0045 addendum, 26-core quota, `bench/results/20261103/exit-main-prefetch-120800.json`): 257 ms per decode token against KTransformers' 198 (5.04 tok/s). Per token:

| Bucket | ms | What it is |
|---|---|---|
| compute | 112 | segments; the CPU experts are about 100 (7.1 per layer, 81 MB per layer from DRAM in about 1.07 ms) |
| prefetch wait | 95 | main thread blocked in `finish_prefetch`: 36 experts, 414 MB per token |
| synchronous misses | 19.5 | device time of 8.4 experts, 97 MB, at 5.0 GB/s |
| rest of the streamer | 13 | mostly the bounce-to-slot copy of those 97 MB on the main thread (about 8 GB/s), lookups |
| alloc, build, readback, inputs | 16 | 95 graphs |

Brief A (ADR 0050 a) targets only the 19.5 ms row. The 95 ms row is larger, and it has a cause in our I/O path, not in the device:

- **The reader is busy 131 ms per token for 414 MB (3.2 GB/s), and the step only has 257.** `Reader` is one thread running `read_batch_with`, which reads a whole sub-batch, waits for all of it, then copies every range into its slot on that same thread (`dio/mod.rs:378-430`). At 5.2 GB/s the device part is about 80 ms; the serial copy adds about 50 ms. ADR 0038 lever 2 (1 MiB chunks at depth 16, implemented, not measured) shortens the device part but leaves the copy serial, so the reader's ceiling stays near 100 to 130 ms per token. Once compute drops (ADR 0053 estimates 27 to 42 ms less), that ceiling is the step.

Per layer the work fits: the reads are 5.4 MB (0.48 experts) per layer, about 1.05 ms at 5.2 GB/s, against 1.19 ms of compute. Fully overlapped, decode is bound by compute plus graph overhead, about 130 ms; after ADR 0053 by the device, about 98 ms of reads per token at today's hit rate.

## From ADR 0054 Options (moved verbatim; the "today" column is measured on box C, the other columns are estimates made before the build)

Both options keep the VRAM hot set and the prefetch prediction (ADR 0038). Per token on box C at today's bytes:

| | today (`main-prefetch`) | A: direct I/O service, arrival order (chosen) | B: page cache, compute in place on mapped pages |
|---|---|---|---|
| bytes read | 511 MB | 511 MB; 440 to 470 with the tier sized to the container (replay decides) | similar: the page cache holds about as many experts as RAM plus reserve; kernel LRU instead of ours |
| device rate under compute | 3.2 GB/s (reader), 5.0 (sync) | 5.2 target: 16 persistent workers, rolling queue | unknown; up to 26 faulting threads, 128 KB readahead per fault, `MADV_WILLNEED` for predictions |
| copies on a serial path | about 63 ms of thread time, 13 on the main thread | none serial: each worker copies its own 1 MiB chunk | none |
| granularity of "compute as bytes arrive" | whole layer | per expert (rows of one expert must stay in one `mul_mat_id`, see Exactness) | per page, inside one `mul_mat_id` |
| exposed wait (estimate) | 95 + 19.5 + 13 | 10 to 25 | 10 to 30 |
| decode step (estimate) | 257 | 145 to 165 | 140 to 165 |
| prefill (32K) | 72 tok/s | unchanged path | op offload copies whole 128-expert tensors per ubatch and refaults the uncached half: llama.cpp's mmap `-ot` prefill is 46 to 50 tok/s on the same boxes |
| control | slot tables, pins, replay | same | kernel reclaim: no pins (a prefetched expert can be dropped before use), no frequency pins (ADR 0050 c), the replay no longer models the tier, `memory()` cannot report used tier bytes (ADR 0037) |

Decode alone does not separate them; prefill and the budget and replay semantics do. llama.cpp `-ot` runs ggml's `mul_mat_id` on mapped pages with all threads faulting and decoded at 0.81x and 0.62x of the builder on boxes A and B, but it also places 7 whole layers in VRAM instead of a per-layer hot set, so those ratios are not evidence against mapped pages as such. Option A keeps every measured advantage and removes the three causes listed above.

## From ADR 0054 Decision, section 3 "RAM tier sized to the container" (moved verbatim)

- The harness reserve (`RAM_RESERVE_GB`, 8 GB) is a guess. Box C peaked at 84.85 of 91.4 GB with the 77-slot tier full (82.6 GB of slots, so about 2.3 GB of everything else), leaving 6.5 GB unused. Under ADR 0031 the tier stays a byte budget; only the reserve changes: 4 GB on the exit boxes, giving 81 slots instead of 77 (`floor(bytes / 1072594944)`), a peak near 89.2 GB, 2.2 GB under the limit. The run's `peak_rss_bytes` checks it; a run above 90 GB goes back to 8.
- Priced offline before the box run: the replay of ADR 0050 (c) on the 235B trace, `tetraeder-bench replay --trace F --vram 10 --ram 77,79,81,83 --policies lru` (prefetch replayed; the `trace` stage of `exit_run.sh` gains these counts), after its fidelity gate (LRU at 77 within 0.5 points of the measured hit rate). The gain is the drop in decode disk bytes per token; at 75 MB per hit-rate point it is about 14 ms of device time per point. Prior from boxes A and B: 0.2 to 0.3 points per slot near 77, so about one point.

## From ADR 0054 "Expected gain" (estimates made before the build from box C, moved verbatim)

- Against `main-prefetch` on box C (257 ms; the next run will also have ADR 0038 levers 2 and 3, still with the serial copy): 145 to 165 ms per token, 6.1 to 6.9 tok/s, 1.2x to 1.37x KTransformers' 5.04. Reasoning: compute and graph overhead stay at 128 ms; the 95 + 19.5 + 13 ms of waits and copies shrink to 10 to 25 ms of exposed reads (device time 98 ms against 112 of compute, hidden per layer behind the hit group and the prefetch window) plus 2 to 5 ms for about 0.4 extra CPU-only graphs and one `xn` readback per layer step with pending experts.
- These estimates overlap ADR 0038's for its levers 1 to 3 (150 to 170 ms): those assumed the reader reaches device rate, which the serial copy prevents. The paired run reports the service alone (`main-iosvc`) and with arrival order (`main-overlap`) apart, so each part has its own number.
- With ADR 0053 (compute 70 to 85 ms) the gains do not add: the step becomes bound by the 98 ms of device time, about 115 to 135 ms. Then bytes per token decide, and the tier sizing (about 14 ms per hit-rate point) and ADR 0050 (c) are the next levers.

## From ADR 0056 Context (moved verbatim)

- **The anchor bands are not candidate bands.** ADR 0040's 9.2% (MoE) and 5.8% (dense) are `Fit::band`, the largest anchor residual of the fit, in sample. On the committed fixtures the candidates' request-time errors are: MoE median 4.3%, 5 of 24 above 9.2%, max 23.1% (c08); dense median 2.1%, 4 of 11 above 5.8%, max 27.8%. A per-result gate at the anchor band would fail on the committed data.

## From ADR 0056 Decision, section 1 "Formula" (moved verbatim)

The reader path is the serial sum because box C shows its prefetch barely hides (95 ms of 257 waiting with the reader busy 131 ms): the sum gives 291 ms there, 13% over. The service formulas are "critical chain or disk busy, whichever is longer", a lower bound until the paired `main-iosvc` and `main-overlap` runs (ADR 0054) measure them. Prefill: misses are read synchronously (ADR 0054 keeps `read_experts` for prefill), so `t_pp` adds `prefill_bytes / disk_rate(sync_depth)` per ubatch, serial. Box C's prefill read 62 s of about 450 s, so the term is not optional.

## From ADR 0056 Decision, section 3 "Snapshot" gate 3 (moved verbatim)

3. per session, the median error of its non-anchor results is at most the class band (MoE 9.2%, dense 5.8%, ADR 0040); today 4.3% and 2.1%;

## From ADR 0056 Implementation notes, part A (moved verbatim)

- **Part A (provenance and ledger).** `MODEL_VERSION` is in `costmodel`; `EnvInfo` gains `session` and prefers `TETRAEDER_GIT_COMMIT`; `exit_regime` writes `provenance` (the io options are omitted until `track-w-iosvc` merges); `scripts/session.sh` exports both variables. `planner::ledger` has `evaluate`, `check`, `write` and `tetraeder-plan ledger [--write] [--accept-worse id=reason]`. Refresh after a version bump or a new fixture: `cargo run -p tetraeder-planner --bin tetraeder-plan -- ledger --write`. `err` is `|predicted - measured| / measured` on the request time. Committed medians: MoE 0.043, dense 0.021. The dense harness files carry no `env`, so its `provenance.json` assumes the `planner-moe` host (noted in the file). A snapshot entry that is missing now, or whose status changed, fails the check; an entry that is new is listed but not a failure until `--write`.

## From ADR 0065 Verdict, "Settled 2026-11-14" (moved verbatim)

- Workload size stays 16 c4 requests: in `bench/results/20261103/gateway/` a c4 run takes 11 s (llama-server) to 27 s (Tetraeder) and the two repeats agree within 0.5%.
- c1 is not the risk: there Tetraeder already matched llama-server (Qwen3-8B 81.1 against 79.8 tok/s `decode_tps_median`, Qwen3-30B-A3B 129.9 against 129.4); the 0.41 to 0.61 gap is c4 only, which is what batching addresses.

## From ADR 0065 Context (moved verbatim)

The plan's Phase 4 exit test: at least llama.cpp server throughput at concurrency 4; prefix hits cut time to first token (TTFT) on a Claude Code replay. ADR 0046 measured one sequence at a time: concurrency 4 gave 0.41 (Qwen3-8B) and 0.59 to 0.61 (Qwen3-30B-A3B) of llama-server `-np 4` throughput, TTFT 4.9 to 6.6x longer.
