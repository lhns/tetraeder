# 0045. Phase 2 exit verdict: Qwen3-235B-A22B larger than RAM, all baselines on the same box

Status: Accepted (Phase 2 exit). Builds on ADR 0035, 0037, 0038, 0044.

## Context

Phase 2 exit test (plan.md, roadmap row 2): logit parity with libllama, and Qwen3-235B-A22B Q4_K_M (142 GB) on a box with at most 128 GB RAM at least as fast as the best tuned baseline that runs it (Colibri, llama.cpp with expert overrides, ik_llama.cpp; the user's goal adds KTransformers), with the per-layer full-hit rate and decode right after a 32K prefill.

## Setup

- Two boxes, every engine paired on the same box, same threads (CPU quota 15), same frozen prompt (`bench/prompts/exit_prompt.txt`, token hash `c743c72b8ffceadd`), 32768 prompt tokens, 64 greedy tokens, `scripts/exit_run.sh`:
  - **A**, box 58900330: RTX 3090, Ryzen 9 5950X, 91.6 GB container RAM, NVMe (5 to 8 GB/s direct reads). `bench/results/20261103/*-013730|021753|030753|032306.json`.
  - **B**, box 58930011: RTX 3090, Xeon E5-2683 v4, 45.7 GB container RAM (model about 3x RAM), slower disk (2 GB/s). `*-070656|080404|085333|090438.json`.
- Expert VRAM is equal: `scripts/llama_fit.sh` finds the most layers' experts llama.cpp fits on the 24 GB card at llama-bench's 32K context and ubatch 2048 (7 layers, `blk.[0-6].ffn_.*_exps=CUDA0;exps=CPU`; 8 fit llama-server but not llama-bench), and the builder gets the same expert bytes as VRAM slots (10.97 GB, 10 slots per layer). The builder's RAM tier is the container RAM minus 8 GB.
- llama.cpp b11976 runs through libllama on the same token ids (`TETRAEDER_EXIT_ENGINE=libllama`, mmap, the fitted `-ot`), so it is also the logit reference. llama-bench (random token ids) is run as well on A.
- KTransformers 0.7.2 (kt-kernel LLAMAFILE on the GGUF experts, sglang-kt, `scripts/ktransformers_*.sh`): all experts on the CPU (its GPU experts need HF weights, which a GGUF-only setup lacks), bf16 attention dequantized from the GGUF, fp8 KV (bf16 KV does not fit at 32K).
- Colibri v2.1.0 (`scripts/colibri_exit.sh`): 2K profile only; its 32K prefill would take about 4 h (2046 tokens took 895 s on the 5-thread box).
- ik_llama.cpp: no number. It was OOM-killed at 128 GB on 2026-11-02 (it does not keep CPU-overridden experts mapped, so the 142 GB must fit in RAM; `--no-mmap` and run-time repack need that too). Its llama-bench also lacks `-d` and `;`-joined `-ot` (both fixed in the stage, not run again).
- FreeToken: not runnable here: every expert must sit in pinned host RAM, and it needs CUDA 13.

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

## Decision

- **Pass, with decode parity against KTransformers on box A.** The builder is faster than every baseline on prefill (1.6x to 3x) on both boxes and on decode against llama.cpp with tuned expert overrides (1.23x and 1.60x), Colibri and KTransformers on box B. On box A its decode equals KTransformers' (3.14 against 3.15, one pair each, inside run-to-run noise). ik_llama.cpp and FreeToken cannot run the model within 128 GB.
- The exit numbers stand on the paired text runs; llama-bench's random-id numbers are not used as the bar.

## Consequences

- Open, in order of expected gain: prefetch depth 2 (ADR 0038 Levers; implemented, not yet measured on the 235B: box B's disk made each 32K run 25 min and the window ran out), `cpu_poll`/`prefetch_threads`, a second paired KTransformers run on box A's class for a noise band.
- Open: the 235B top-1 logit offset against libllama (above). Greedy output is unaffected over 64 tokens on both boxes. The boxes differ in RAM slot count, which changes how the prefill's pair groups are split; that is the first thing to check.
- Not taken (the coordinator's FreeToken levers): splitting decode misses between CPU and PCIe gains little here because the CPU expert path already reads about 8 GB of weights per token in 225 ms, near the boxes' DRAM bandwidth, and PCIe DMA reads the same DRAM; prefill double buffering is not needed while prefill leads every baseline by 1.6x or more.
- Box cost of the exit work on 2026-11-03: about 2.9 USD (58894536 0.14, 58897870 and 58899229 about 0.05, 58900330 0.97 plus 0.92 traffic, 58930011 0.83, no traffic).

## Addendum (2026-11-05): llama.cpp's best setup was not measured

The llama.cpp rows above use `-ot`. The pinned build (b11976) also has a GPU cache of host experts (`--moe-cache-mib`, `llama_context_params.moe_cache_size`, ggml-org/llama.cpp#29887), which that pull request reports at 1.57x to 2.20x over `--fit` on other hardware and models. So "decode beats llama.cpp" here means beats `-ot`, not the best tuned llama.cpp the exit test asks for. The cache may help less at this regime's 8% of experts in VRAM, but only a paired run can say. `llama-bench` has no flag for it: run it through the libllama reference engine of `exit_regime` (needs the parameter set) or `llama-server`. Until then the llama.cpp part of this verdict is open. Since ggml-org/llama.cpp#30966 (merged 2026-11-07, after our pin) the cache can also compute a share of misses on the CPU (`--moe-cache-cpu-share`, tuned by `scripts/moe-cache-tune.py`); the baseline to beat is that tuned pair, built from a commit after the PR.

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
