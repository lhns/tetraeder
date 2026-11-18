# Session 20261102

- Status: clean Phase 2 exit run that did not meet the bar (prefill ahead of llama.cpp expert offload, decode behind), plus the paired prefetch off/on runs that passed ADR 0038's 0.6 accuracy gate. Its 32K prompt differed from later boxes (ADR 0038, correction 2026-11-03).
- Code and box: commits of 2026-11-02 (results commit 7281dab and others, see git log of this directory); boxes 58813464 (RTX 3090, 91.6 GB, 15 threads) and 58872072 (RTX 3090, 91.7 GB, 5 threads); driver unknown.
- Raw files: this directory.

## From ADR 0036 (implementation note "CUDA graphs and the resident step (2026-11-01)", expectation and box result, moved verbatim)

Expected on the box: libllama clearly slower with graphs off; the builder with reuse off about as fast with graphs on as off if the context address changes between steps; with reuse on, one capture per shape, a uid reuse on every later decode step, and decode close to libllama. If libllama is not faster with graphs on, CUDA graphs are not the gap and the decode profile (build, allocation, readback) is the next place to look.

Box result (bench/results/20261102/cuda-graphs-*.json): 30B decode 160 to 203 tok/s with reuse (llama.cpp 207), greedy identical, 127 replays, 0 update failures; `reuse_step_graph` now defaults to true.

## From ADR 0037 Consequences (exit run 2026-11-02, moved verbatim)

- **Exit run (2026-11-02, box 58813464, RTX 3090, 91.6 GB container RAM, 15 threads, Qwen3-235B-A22B Q4_K_M 142 GB, `scripts/exit_run.sh`, `bench/results/20261102/exit-*-120520.json`).** Builder `Cpu` policy, VRAM slots 5.5 GB, RAM tier 71.6 GB, op offload, ubatch 2048: 32K prefill 67.8 tok/s, decode after it 2.26 tok/s (442 ms/token: compute 225, streamer 206 of which reads 141; 742 MB read per token at 5.3 GB/s). Same box and threads, llama.cpp `-ot exps=CPU`: pp2048 at depth 32K 38.5, tg32 at depth 32K 2.69; at depth 0 43.0 and 2.60; same expert VRAM (`blk.0-3` experts on the GPU) 47.2 and 3.16. So prefill beats expert offload by about 1.7x and decode does not (0.84x at 32K). Decode is reads then compute in series; overlapping the miss reads with the layer's GPU work and prefetching the next layer are the next levers. ik_llama.cpp `-ot` was OOM-killed at 128 GB (it does not keep CPU-overridden experts mapped), so it has no number in this regime. Ubatch 4096 aborts in CUDA's `mm_ids_helper` (4 B shared memory per row, about 25k rows on Ampere; a pair group at top-8 has 32k): pair groups need a row cap before larger ubatches.

## From ADR 0038 Context (moved verbatim)

The Phase 2 exit run (ADR 0037, bench/results/20261102/exit-main-120520.json, Qwen3-235B-A22B Q4_K_M, RTX 3090, 91.6 GB box) decodes at 2.26 tok/s after a 32K prefill, 0.84x llama.cpp `-ot` (2.69). Per token: 442 ms = compute 225 (GPU attention plus the CPU groups) + streamer 206 (disk reads 141 at 5.3 GB/s for 742 MB, plus the copy from the bounce buffer into the RAM slots) + alloc 8. Every layer step has at least one CPU expert (full-hit rate 0), the RAM tier hits 90%.
The bar is the best tuned baseline at 32K (plan.md, Phase 2 exit): llama.cpp `-ot` with the same expert VRAM (at depth 0: 3.16 tok/s against 2.60 for all experts on the CPU), so about 300 ms per token. Hiding a fraction A of the 206 ms streamer time gets there at A of about 0.7.

## From ADR 0038 Results (moved verbatim)

## Results

Box 58872072 (RTX 3090, container limit 91.7 GB, 5 threads from the CPU quota, disk 3.1 GB/s), bench/results/20261102/*-203803.json and eq-*.json. Qwen3-235B-A22B Q4_K_M, 32K prefill, 64 decode tokens, op offload on, prefetch off and on in the same session:

| | off | on |
|---|---|---|
| prefill tok/s | 57.1 | 60.6 |
| decode tok/s | 2.19 | 2.38 (+8.4%) |
| decode ms per token: compute, streamer | 197, 248 | 185, 225 |
| prediction accuracy (decode, prefill) | | 0.85, 0.88 |
| prefetch read time hidden behind compute (decode, prefill) | | 29%, 48% |

On the 30B (moe_check, `slots32_ram32_cpupolicy_prefetch`): accuracy 0.89, warm decode 38.3 to 39.7 tok/s.

Output: greedy tokens are equal with prefetch off and on in every same-conditions pair (8, 2048, 8192 and 32768 token prefills, op offload off and on). The first 32K run on the box (`main`, off) differed from all later 32K runs, off and on alike, and equals the previous box's first run: a first-run effect unrelated to prefetch, open.

Decision: accepted, the gate (accuracy at least 0.6) holds. Most of the read still waits: the step waits for 71% of the decode prefetch time, likely because the reader thread competes with ggml's spinning CPU threads for a 5.8-core quota. Next levers: a lower ggml thread count while a prefetch is in flight, and reading two layers ahead. Default stays off until prediction runs only where it can prefetch (today it also adds its graph nodes and readback under `MissPolicy::Upload`).

## From ADR 0039 Context (moved verbatim)

- The repack buffer type has no `is_host`, so `ggml_backend_sched` never op-offloads an op whose weights sit there (ggml-backend.cpp, the `op_offload && ggml_backend_buffer_is_host` check). The exit regime's prefill depends on op offload: 67.8 tok/s with it over a 32K prompt, 18.0 without over 2K (exit-main and exit-cpu-prefill-120520.json).

## From ADR 0043 Context (moved verbatim)

The exit regime (Qwen3-235B-A22B Q4_K_M, RTX 3090, model larger than RAM, ADR 0035, 0037, 0038) decodes at 2.2 to 2.4 tok/s after a 32K prefill (bench/results/20261102/exit-main*-203803.json): per token about 450 ms, compute 190 (attention on the GPU, routed experts on the CPU) and streamer 225 to 250 (about 600 MB of missed experts read per token, plus copies), RAM tier hit rate about 0.92. Almost all of that is per expert, not per token: an expert read from disk or streamed through the CPU's memory bus costs the same for one row as for seven. A step that verifies k drafted tokens reads each needed expert once for all k+1 rows, so its cost grows with the union of routed experts, not with k. Consecutive tokens share experts, so the union grows slower than 8(k+1) per layer.

## From ADR 0045 Setup (Colibri and ik_llama.cpp bullets, moved verbatim)

- Colibri v2.1.0 (`scripts/colibri_exit.sh`): 2K profile only; its 32K prefill would take about 4 h (2046 tokens took 895 s on the 5-thread box).
- ik_llama.cpp: no number. It was OOM-killed at 128 GB on 2026-11-02 (it does not keep CPU-overridden experts mapped, so the 142 GB must fit in RAM; `--no-mmap` and run-time repack need that too). Its llama-bench also lacks `-d` and `;`-joined `-ot` (both fixed in the stage, not run again).

## From ADR 0056 Context (moved verbatim)

- **The ties are not small.** `bench/results/20261102/builder-real-202732.json` (Qwen3-30B-A3B Q4_K_M, 5 threads, 32 VRAM slots, CPU miss policy, same box): decode 19.4 tok/s with 32 RAM slots, 33.8 with all 128, 1.74x apart, predicted equal. The builder fixture (`planner-moe-builder`, `docs/m1-open.md` item 6) has never run; it is the open part of the Phase 3a planner exit test.

## From ADR 0056 "Expected value" (moved verbatim)

Note: a prediction made before the `b3-planner` session, not a measurement.

A prediction, to be checked by the `b3-planner` session on the builder fixture: today the analytic pick inside a RAM tie group is the smallest RAM tier, and on the 30B that tier decodes up to 1.74x slower than the largest at the same VRAM slots. With the term, RAM-only candidates are ranked by their disk bytes, the close call compares real neighbours, and regret stays within the 10% bar with Spearman at least 0.8. If it does not, the ledger shows which anchors miss and whether the reader formula or the bytes are wrong.

## From ADR 0056 Implementation notes, part B (moved verbatim; the synthetic fixture numbers are local CI runs)

  On the synthetic builder fixture: with the tier file regret 0 and Spearman 1.000, without it (the box reads no disk) regret 0 and Spearman 0.998, as before; RAM tie groups on the 24 candidates: 4 without the replay, 0 with it. On the committed 30B run (`builder-real-202732.json`, 32 VRAM slots, bytes from its expert counters since no 30B trace is committed), decode tok/s for 32 against 128 RAM slots: measured 19.40 against 33.81; before 69.72 against 69.72; after 13.79 against 66.64 (prior curve) and 10.80 against 66.64 (3.3 GB/s). The sign is right and the gap is too wide (4.8x against 1.74x): an uncalibrated prior, and the cold run's fill counted as decode reads. The trace stage of the `b3-planner` session replaces both.
