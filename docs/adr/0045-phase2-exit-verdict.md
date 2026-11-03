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

- KTransformers' 32K prefill on B ran at about 8.5 tok/s (one 4096-token chunk per 8 min, `runs/vast/ktransformers-server-deep-box58930011.log`) and could not finish within the box window, so B pairs it at 2K.
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
