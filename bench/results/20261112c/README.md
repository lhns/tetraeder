# Session 20261112c

- Status: diagnosis, not a verdict (b1b-levers diagnostic, one box outside the B1 profile; fit windows throttled under poll 50 before ADR 0061). Its replay meets ADR 0050 (c)'s 10% bar; ADR 0059's admission stays off.
- Code and box: results commit eca4dd0; box 60305047, host 49107, RTX 3090, AMD EPYC 7302P, quota 30.72, 30 threads, THP `madvise`.
- Raw files: this directory.

## From ADR 0035 (addendum "2026-11-12: huge pages for the RAM tier (A/B switch)", the "Why" bullet, moved verbatim)

- **Why.** On box run 20261112c the builder's CPU expert work read about 7 GB of Q4_K expert weights per decode token from an RAM tier of about 80 GB, and the host sampler showed `anon_huge_gb` 0 with THP mode `madvise`: the tier sat on 4 KiB pages, so the matmuls walk millions of page table entries (TLB pressure). Hypothesis: huge pages speed up CPU expert compute. Not established; this is the switch to measure it.

## From ADR 0050 (addendum 2026-11-12 "(c) passes its replay bar; not built now", moved verbatim)

The 235B routing trace of `20261112c` (replay `bench/results/20261112c/replay-235b-vram.json`, VRAM 10, prefetch on, static VRAM tier): the best frequency-pinned RAM policy reads 10.4%, 11.4%, 11.5% and 14.8% fewer disk bytes per decode token than LRU at 77, 79, 81 and 83 RAM slots (`freq:0.5` at 77 to 81, `freq:1` at 83). That meets the 10% bar. It is not built now: every paired run on a fast disk since `20261111g` is `not_io_bound` (disk utilisation 0.11 to 0.25), so fewer disk bytes do not shorten the decode step there; the builder's limit on the profile was cgroup CPU throttling (ADR 0045, `20261112e`). It is the first lever for the slow-disk limitation of ADR 0045's 2026-11-11 addendum, where read waits dominate.

## From ADR 0059 (addendum "2026-11-12: first box result, `20261112c`", moved verbatim)

Box 60305047 (RTX 3090, EPYC 7302P, quota 30.72, 30 threads, disk 6.3 GB/s), interleaved: `main-overlap` 4.767 and 4.335 tok/s, `main-overlap-adapt` (share 0.1) 2.469 and 2.655; adapt over default 0.518 and 0.612, outside both spreads (9.5% and 7.5%). Admission works as designed: decode VRAM pair share 0.165 to 0.372 (the replay's `lru` gives 0.433, llama.cpp's cache 0.353), CPU pairs per token 628 to 473. But compute fell only 115.7 to 105.4 ms per token while the synchronous uploads cost 174 to 208 ms (about 530 MB per token, 2,981 admissions per run, no fall over the four windows). Without the upload time the step would be 188.5 and 194.6 ms against 201.9 and 200.6, a ceiling of 1.03x to 1.07x. The segments with CPU groups barely depend on the pair count (108 ms for 628 pairs, 99 ms for 473). So on this box the VRAM share is not the gap and the async upload is not worth building; the switch stays off. One box at a quota outside the B1 profile, with builder compute at 116 ms against 292 to 551 ms on the EPYC 7642 hosts where the loss was measured: not supported here, not refuted there. Results `bench/results/20261112c/`.

## From ADR 0059 (addendum "2026-11-12b: where the upload time goes, and async admission", moved verbatim; the first line also cites `20261112a`)

This answers point 5's "next step" for the EPYC 7642 hosts, where `20261112a` measured about the same CPU core time per routed CPU pair for both engines (ours 17.8 and 20.8 core-ms at 628 CPU pairs per token, the cache's 16.9 and 17.4 at about 453), so the CPU pair count would set the decode rate there (628 / 453 = 1.39 against the measured 1.35x to 1.54x). `20261112c` showed admission does cut the pairs (628 to 473) but paid 208 ms per token for it: 46.6 admissions of 11.43 MB per token, 4.47 ms each, 2.56 GB/s.

**Our upload, from code.** `Streamer::admit` per expert: `read_slot` allocates a `Vec` of 11.4 MB and, per slab, `get_bytes` allocates a zeroed `Vec` and copies the RAM slot into it (`ggml_backend_tensor_get` on a CPU buffer is a `memcpy`), then appends it to the first: two host passes over freshly allocated pages, so about 5,600 page faults per expert. `write_slot` then calls `set_bytes` three times; the CUDA buffer's `set_tensor` is `cudaMemcpyAsync` on `cudaStreamPerThread` from pageable memory (the driver copies it once more into its own staging buffer, a third host pass) followed by `cudaStreamSynchronize`, so no slab's DMA overlaps the next slab's host work. All of it on the main thread, between a layer's planning and its next segment, while the threadpool's workers spin (`pool_pause` was off; the adapt runs used about 10 core-seconds per token against 5). How the 4.47 ms splits between faults, the two copies, the staging copy and the syncs cannot be read from code; the box stages below separate the host side from the transfer.

**llama.cpp's upload, from source** (b12274, `src/llama-moe-cache.cpp`, `impl::copy`): per layer, in the callback that copies the slot map, after one `ggml_backend_synchronize` for the routing ids, every miss to upload is one `ggml_backend_tensor_set_async(backend, bank, w->data + expert * size, ...)` (consecutive experts into consecutive slots merged) on the compute backend's own stream, straight from the model's host weight; no host copy, no sync per expert (the next layer's callback syncs). The source is pageable in the measured runs: `scripts/llamacpp_cache.sh` loads with `-lm auto` (mmap), and the loader avoids the pinned host buffer type under mmap (`llama-model-loader.cpp`, "avoid using a host buffer when using mmap"); only `-lm none` or `dio` would put CPU experts in `CUDA_Host` memory (`make_cpu_buft_list`). So its enqueue still blocks for the driver's staging copy (one host pass), and its uploads are on the critical path too, but at one host pass per expert instead of our three plus faults and per-slab syncs. `ggml_backend_sched_prefetch_inputs` is a different matter (the CPU split's inputs, ADR 0064) and not needed here.

## From ADR 0060 (addendum "the builder's export does not stall decode; the sampler's smaps_rollup likely does", moved verbatim)

Context: from 20261112c on, one decode window per process spends 40 to 160 ms per token outside `run()` (wall minus `StepProfile::total`). The other windows show a flat 5 to 8 ms. The 1 s engine export of this ADR started that week, so it was the first suspect.

Findings:
- The stall happens without the export. 20261112c ran a commit before e7711c6 (no `telemetry_for`, no `opts.telemetry`), and its main-overlap-repeat still has 96 ms per token in window 0. The steady gap is no larger with the export (5.4 to 8 ms) than without (7 to 8 ms).
- Code: `PeriodicReader` exports on its own thread (`futures_executor::block_on` there). `collect` takes each instrument's lock only to copy the points, and the step path only records. `record_step` locks the ring, which nothing else takes during decode. `report_memory` reads no `/proc` or cgroup file. Its one blocking edge was `DirectFile::bounce_capacity`, which took the bounce-buffer mutex that `read_batch_with` holds for a whole batch. With the single reader thread (`TETRAEDER_EXIT_IO_SERVICE=0`) the step's memory report therefore waited for the prefetch batch in flight. All the stalled runs used the I/O service, so this did not cause them. Fixed anyway: the size is now mirrored in an atomic.
- Not explained by CPU throttling (zero throttled periods in every stalled window) or by cgroup anon growth (flat). The sampler has no reclaim counter, so reclaim is not ruled out.
- The steady 5 to 8 ms is the harness: `margin` sorts a 151,936-entry index per token (`top_k`), 14 ms per token for margin plus argmax on a desktop CPU (`rustc -O`). `decode_tok_s` counts it.

Likely cause: the host sampler's `smaps_rollup` read (72ca4af, between 20261112a, flat, and 20261112c, the first stall). For that file the kernel walks every page table of the engine under its mmap lock, and only drops the lock between mappings. The RAM tier is one 83 GB mapping on 4 KiB pages (`anon_huge_gb` 0), so one walk holds the lock for about a second. An mmap or munmap on the step thread waits for it, for example a 1.2 MB `top_k` index or a 600 KB logits buffer that glibc maps fresh while its threshold settles early in decode. Supporting evidence:
- At a 1 s interval the sampler ticks every 2.2 to 2.6 s.

Not proven on a box.

## From ADR 0064 Context (moved verbatim)

A streamed decode token of Qwen3-235B-A22B (94 layers, 8 of 128 experts) runs 95 segment graphs (`run` in `crates/model/src/qwen3moe.rs`, one `build_segment` per layer boundary) plus the overlap group graphs of ADR 0054 (`run_overlap_layer`, `run_cpu_group` on `sched_cpu`). A fit over the 16 decode windows of `20261112c` (EPYC 7302P, 30 threads) gave 0.88 ms fixed per CPU-carrying segment plus 0.063 ms per CPU pair, so about 70 ms of fixed cost against 40 ms of expert work per token, and 0.44 ms per GPU-only segment. llama.cpp's expert cache (#30966) decoded 1.35x to 1.54x faster on the 7642 hosts (`20261112a`). This ADR counts both engines' dispatch per token from source, says where the fixed term goes, and designs the change that cuts it.
- One graph compute. `llama_context::process_ubatch` reuses the previous graph when `res->can_reuse(gparams)` holds (no build, no `ggml_backend_sched_alloc_graph`), and `graph_compute` calls `ggml_backend_sched_graph_compute_async` once. The `20261112c` log: `graph (pp bs=2048, tg bs=1): nodes = 5944 / 7153, splits = 284 / 284` and `graphs reused = 69` of about 72 decode tokens.
- CPU pairs: 32,623 CPU misses of 54,629 lookups (59.7%, warm-up tokens included), about 449 of 752 pairs per token; 35.3% cache hits.
- 133.5 graph builds, scheduler allocations and computes: 95 segments plus 38.5 CPU group graphs. Build 3.2 ms, alloc 12.8 ms. Every `Sched::alloc_graph` (`crates/ggml-sys/src/ggml/sched.rs`) resets the scheduler, splits and allocates.
- Splits: a CPU segment has 2 (CPU: `get_rows` of `xn`, the RAM group's `mul_mat_id` and `swiglu`; GPU: `set_rows` into `acc`, the compute-tier group, weighting, attention, router), a GPU-only segment 1, a group graph 1: about 78 x 2 + 17 + 38.5 = 212. (`max_splits` 4 is over the run, prefill included.)
- Blocking device operations per CPU segment, from `ggml_backend_sched_copy_input` without events: the CPU split's `xn` input (GPU sync, D2H), the GPU split's four `set_input` index tensors (allocated on the CPU, each a GPU sync and a blocking H2D) and the RAM rows (sync, H2D), the closing `ggml_backend_sched_synchronize`, then the readback of `ids`, `pred` and `pred2` (three blocking D2H, 3.0 ms per token): about 7 syncs and 9 copies.
- About 117 CPU kickoffs (78 CPU segments, 38.5 group graphs). The CPU group runs before the GPU work of its segment, and the GPU waits for it.
- CPU pairs: 628 of 752 (83.5%); VRAM pair share 16.5% at a similar budget (10.97 GB of slots against the cache's 10,458 MiB).
- The scheduler is sized for the whole-step graph: `step_nodes(h) * 4` = 128 x 95 x 4 = 48,640 nodes, so its hash set has 65,537 entries (`ggml_hash_size`). `ggml_backend_sched_reset` memsets `hv_tensor_backend_ids` (262 KB) and `hv_tensor_copies` (2 backends x 8 B, 1.05 MB) on every segment: about 1.3 MB, 124 MB per token. At 15 to 30 GB/s that is 4 to 8 ms of the 12.8 ms alloc. Alternating segment topologies also make `ggml_gallocr_alloc_graph` fail and re-reserve, which synchronizes every backend first (`ggml_backend_sched_alloc_splits`).
### Where the 0.88 ms goes

The fixed term is real; its parts are not yet separated:

- The slope is identified. The `adapt` runs vary pairs per segment: at about 87 CPU segments per token, windows with 640 and 625 pairs took 116.5 and 114.1 ms of CPU segment compute, windows with 426 and 445 pairs 97.6 and 99.2 ms: about 0.09 ms per pair. A 12.24 MB expert (Q4_K gate and up, Q6_K down, inferred from the size: 5.19 bits per weight) at 0.06 to 0.09 ms is 135 to 200 GB/s of reads. The box's probe, 66.5 GB/s, counts bytes copied (`scripts/engine_sample.py`), so it is not the read ceiling and no bandwidth model is fitted here.
- The unthrottled `exit-trace` run of `20261112c` (no overlap, every segment a CPU segment, `throttled_share` 0.00): 127.4 ms over 94 segments, 1.36 ms each, with 6.6 pairs. At 0.06 to 0.09 ms per pair the expert work is 0.4 to 0.6 ms, leaving 0.76 to 0.95 ms fixed, of which the segment's GPU part is at most the 0.44 ms of a GPU-only segment. So the CPU-specific fixed part is about 0.3 to 0.5 ms per CPU segment, 25 to 40 ms per token over 78 CPU segments. The fit's own number, (0.88 - 0.44) x 78 = 34 ms, is inside that range.
- The fit's windows ran at `throttled_share` 0.5 to 0.68 (poll 50, before ADR 0061). Quota stalls fall mostly while the CPU is busy, so part of this fixed term may be throttling.
- Bottom-up, what a CPU segment pays beyond llama.cpp's per-layer structure is the build and alloc above (0.12 ms per graph) and about 4 extra syncs and 6 extra small blocking copies, roughly 10 to 15 us each: 0.1 to 0.15 ms. The rest, 0.15 to 0.35 ms, would be the threadpool and the quota: each CPU segment wakes the pool (cond broadcast when the workers sleep under poll 0 or after a pause, ADR 0061), runs about 6 barriers across 30 threads, and spends quota its spinning workers took. One graph per token does not change the kickoff count: llama.cpp also kicks the pool once per layer.

Which share is syncs and copies (one graph per token cuts it) and which is wake, barriers and quota (it does not) is the open question; the micro-benchmark below measures the CPU side of it.

On the 7642 hosts the per-segment cost follows the CPU quota: 2.74 ms per CPU segment at `throttled_share` 0.05 (`20261112e` poll 0), 3.5 to 5.8 ms at 0.80 to 0.98, 4.9 and 8.5 ms at 0.30 and 0.60 (`20261112f`). One hypothesis for the cache's lead of 1.35x to 1.54x is its CPU pair count: we compute 628 / 449 = 1.40x more CPU pairs per token. At 0.06 to 0.09 ms per pair the 179 extra pairs are 11 to 16 ms on `20261112c`; on the 7642, with slower memory and throttling, more, but not measured. ADR 0059's `main-overlap-adapt` reached a 37.2% VRAM share and still decoded slower (2.47 and 2.65 against 4.77 and 4.33 tok/s) because of admission cost, so the share alone does not settle it. That is a placement question this ADR does not answer.

So cheaper dispatch is worth about 5 to 27 ms of the 202 ms per token on `20261112c` bottom-up (2% to 13%, steps 1 and 2 below), with 56 ms (16 ms build and alloc plus 40 ms) as the bound if the whole CPU-specific fixed part were syncs and copies.

## From ADR 0064 Decision (moved verbatim)

Note: the savings are estimates built on this session's numbers, not measurements.

- Expected: 4 to 8 ms of the 12.8 ms alloc per token on `20261112c` (2% to 4%). The micro-benchmark measures it on a CPU before any box run.
Routing of layer L+1 depends on layer L's MoE output, so it cannot be computed before the graph runs; the ADR 0038 predictions are right 85% (depth 1) and 78% (depth 2) of the time on `20261112c`, so steering compute by them would break parity. The host must see each layer's ids mid-graph, as llama.cpp's callback does.
### Expected savings (`20261112c`, 202 ms per token)

| Item | Today | Step 1 | Step 2 |
| --- | --- | --- | --- |
| Build and alloc | 16.0 ms | 8 to 12 ms | about 2 to 4 ms (group graphs only; the step graph once per 256 tokens) |
| Syncs and small blocking copies per CPU layer (readbacks included) | about 7 and 9 | same | about 4 to 5 and 6 (two split-end syncs, two input-copy syncs; `xn` out, three readbacks, index tensor and `cpu_out` in): 5 to 6 fewer x 10 to 15 us x 95, 5 to 9 ms less |
| Readback (`ids`, `pred`, `pred2`) | 3.0 ms | same | same, inside the callback |
| Parked GPU rows | 0 | 0 | 0 to 8 ms: up to 6.7 extra rows x 12.24 MB / 936 GB/s = 88 us per layer, less if duplicates hit L2 |
| Empty CPU work in GPU-only layers | 0 | 0 | 0 (no CPU node in the step graph on a GPU primary) |

Step 1: 4 to 8 ms (2% to 4%). Step 2 on top: build and alloc 4 to 10 ms, syncs and copies 5 to 9 ms, parking 0 to -8 ms, so 1 to 19 ms (0.5% to 9%); up to about 30 ms more if the CPU-specific fixed part is mostly syncs and copies rather than threadpool and quota, which step 0 and the micro-benchmark decide. Neither is likely to close the 1.35x to 1.54x gap on the 7642, where the per-segment cost follows the quota.

## From ADR 0064 Consequences (moved verbatim)

Note: the gains are estimates built on this session's numbers, not measurements.

- On the 7642 the per-segment cost follows the quota (ADR 0061, 0062), and CPU pairs per token (1.40x the cache's) are the other candidate for the gap. Those stay the levers to pursue first; this ADR removes the dispatch the builder pays beyond llama.cpp's structure.
