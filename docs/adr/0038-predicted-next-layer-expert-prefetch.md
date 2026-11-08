# 0038. Predicted next-layer expert prefetch into the RAM tier

Status: Accepted (Phase 2, measured 2026-11-02, see Results). Extends ADR 0035; does not change it.

## Context

The Phase 2 exit run (ADR 0037, bench/results/20261102/exit-main-120520.json, Qwen3-235B-A22B Q4_K_M, RTX 3090, 91.6 GB box) decodes at 2.26 tok/s after a 32K prefill, 0.84x llama.cpp `-ot` (2.69). Per token: 442 ms = compute 225 (GPU attention plus the CPU groups) + streamer 206 (disk reads 141 at 5.3 GB/s for 742 MB, plus the copy from the bounce buffer into the RAM slots) + alloc 8. Every layer step has at least one CPU expert (full-hit rate 0), the RAM tier hits 90%.

The reads cannot overlap with compute without a prediction. Segment L computes CPU group L-1, attention L and router L, so layer L's experts are known only after segment L, and group L must finish before attention L+1. The reads for layer L's misses sit strictly between segment L and segment L+1. Moving them off the critical path needs a guess of layer L+1's experts before segment L+1 starts.

The bar is the best tuned baseline at 32K (plan.md, Phase 2 exit): llama.cpp `-ot` with the same expert VRAM (at depth 0: 3.16 tok/s against 2.60 for all experts on the CPU), so about 300 ms per token. Hiding a fraction A of the 206 ms streamer time gets there at A of about 0.7.

## Decision

- **Pre-gating.** Segment L also computes layer L+1's router on layer L's FFN input: `top_k(gate_inp[L+1] x rms_norm(h_L) * ffn_norm[L+1])`, where `h_L` is the residual after attention L. One small matmul per layer on the router's backend. The ids are read back with layer L's routing.
- **Async reads into the RAM tier.** After layer L's own misses are staged (synchronously, as today), the predicted experts of layer L+1 that are in neither tier are looked up in layer L+1's RAM cache (which assigns slots) and read by a background reader while segment L+1 computes. Before layer L+1 is planned, its prefetch is waited for. Correct predictions are then RAM hits; wrong ones cost disk bandwidth that was idle and an LRU entry.
- **Safety.** RAM slot tensors are per layer, so a prefetch into layer L+1 never aliases group L, which reads layer L's slots. Slots with a read in flight are pinned until the wait; no other code path reads or evicts them. The reader writes through a raw host pointer of the CPU buffer, so that code lives in `memory` (Rule 7), and the streamer waits for every read before it is dropped. The main thread does no reads while a prefetch is in flight (its reads happen between the wait for L and the prefetch for L+1), so the shard's bounce buffer is never shared concurrently.
- **Scope v1:** `MissPolicy::Cpu` with a RAM tier, decode and prefill, behind `LoadOptions::prefetch` (default off until the exit run shows a gain). Prefetched experts do not count as routed (the hot set reads the counts). Prefill: the union of 2048 predictions covers most experts, so prefetch is capped at the slots the prediction does not need; intra-layer chunk pipelining for prefill (read chunk i+1 while chunk i computes) is a follow-up on the same reader.

## Consequences

- Measure first: per-layer overlap of predicted and actual top-8 (`expert.prefetch` counters: issued, used, wasted bytes) on the 30B in `moe_check` and on the 235B in the exit run, prefetch off and on, paired. If the 235B's accuracy is below about 0.6, pre-gating alone does not pass the exit test and this ADR records that; the reader stays for prefill pipelining.
- New metrics, dashboard panel and alert (Rule 6): prefetch accuracy per layer and wasted read bytes.
- The copy into the RAM slots moves to the reader thread for prefetched experts; the remaining synchronous copy (mispredictions) is a separate lever.
- `exit_run.sh` gains the missing baseline, llama.cpp `-ot` same VRAM at depth 32K, so the number to beat is measured on the same box.

## Implementation notes

Built behind `LoadOptions::prefetch` (default off); tests in `tests/prefetch.rs`, harness in `exit_regime.rs` (`TETRAEDER_EXIT_PREFETCH=1`, stage `main-prefetch` of `exit_run.sh`) and `real_moe.rs` (`slots32_ram32_cpupolicy_prefetch`).

- **Prediction.** Segment L adds `soft_max`, `argsort_top_k` of `gate_inp[L+1] x (rms_norm(ffn_inp_L) * ffn_norm[L+1])` after the real router's nodes (so the CUDA topk-moe window stays intact), copied to a `pred` state tensor and read back with layer L's ids. It runs whenever prefetch is on and experts stream, so accuracy is scored without a RAM tier too. Layer 0 has no prediction.
- **Reader.** `memory::reader::Reader` is one persistent thread running `DirectFile::read_batch_with` per shard, handing each range to a `Send` sink; `copy_sink` + `tensor_dst` is the raw host memcpy (unsafe in `memory` only; `Tensor::host_ptr` in `ggml-sys`). The sink is a boxed closure so a repacking buffer can plug in its own. `tensor_dst` is a safe function whose liveness obligation is documented, not typed: the streamer keeps it by pinning the slots until the wait and declaring the reader before the RAM tier, so it drops (and waits) first.
- **Pinning.** The RAM tier has no hot set, so the in-flight slots use `LayerSlots::set_hot_set`: the predicted experts that are already resident plus the ones being read are pinned, and the new ones are read into free or unpinned LRU slots. At most `RAM slots - resident predicted` experts are read, most-predicted first; experts in the compute tier are skipped. Prefetch lookups touch no routing counts.
- **Order.** Layer L: wait for its prefetch (clears the layer on error), plan and stage as before (prefetched experts are RAM hits), then issue the reads for L+1 after the last synchronous read, so the bounce buffer is never shared. Nothing is in flight after the last layer. The start of a step and `refresh_hot_set` settle any leftover from a failed step.
- **Counters.** `IoCounters`: `prefetch_issued`, `_used` (a flagged expert hit in `fill_ram`), `_wasted` (a flagged expert evicted; never-used ones still resident count in neither), `_bytes`, `_wasted_bytes`, worker read time and the time the step waited (overlap = 1 - wait / read). Prefetch reads are also in `batches`, `experts_read` and `bytes_read` but not in `read_time`. `LayerExpertStats::predicted_hits / predicted_total` give the accuracy. The prefetch lookups count as RAM-tier misses in the slot stats and the later routed lookup as a hit; they are not reported to `expert.cache.requests`.
- **Metrics (Rule 6).** `tetraeder.expert.prefetch` (layer, result issued/used/wasted), `tetraeder.expert.prefetch.wasted` (bytes), `tetraeder.expert.prefetch.accuracy` (histogram per layer step); two dashboard panels and the `TetraederPrefetchAccuracyLow` alert (below 0.4 for 10 minutes while prefetch issues reads).
- **Not done:** no fault-injection hook exists; the error test cuts the model file, which fails whichever read comes first (the reader's own error path is unit-tested in `memory`). Intra-layer pipelining for prefill is still the follow-up. Accuracy and overlap on the 235B are measured by the exit run (Results below).

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

## Levers (2026-11-03)

All off by default; measured later in one box session.

- **Gating.** The prediction nodes and their readback now run only with `prefetch` on, `MissPolicy::Cpu` and a RAM tier, where the reads can be prefetched. Prefetch on with `Upload` (or no RAM tier) predicts nothing and costs nothing; accuracy is no longer scored there.
- **`LoadOptions::prefetch_depth`** (1 or 2, others rejected at load). Depth 2: segment L also predicts layer L+2 from the same input (`rms_norm(ffn_inp_L) * ffn_norm[L+2]`, `gate_inp[L+2]`, softmax, top-k), placed after the real router like the depth-1 prediction, into a second state tensor. After layer L is staged: first a top-up for L+1 from the fresh depth-1 prediction (experts neither resident nor already in flight), then the reads for L+2.
- **Reader.** Several reads can be in flight: `inflight` is a queue. `finish_prefetch(il)` waits for the tickets of layer `il` only (the reader is FIFO) and unpins that layer; `settle` waits for all. A layer's pins are its in-flight experts plus the resident predicted ones, and the read budget is `RAM slots - pins`. The reader has its own `DirectFile` per shard (64 MiB bounce cap, counted in `memory()` as `io`) so it never shares a bounce buffer with the synchronous reads of the layer being staged.
- **Scoring.** `LayerExpertStats::predicted2_hits/_total` and `tetraeder.expert.prefetch.accuracy` with a `tetraeder.prefetch.depth` label; the panel splits by depth, the alert watches depth 1.
- **`LoadOptions::cpu_poll`**: the ggml threadpool poll level (0 to 100, default 50) of the CPU backend. Lower leaves cores to the reader under a tight quota.
- **`LoadOptions::prefetch_threads`**: CPU compute threads while a prefetch read is in flight (threadpool size unchanged; restored after the last wait).
- Harness: `TETRAEDER_EXIT_PREFETCH_DEPTH`, `TETRAEDER_EXIT_CPU_POLL`, `TETRAEDER_EXIT_PREFETCH_THREADS`, recorded in the JSON `options`.

## Correction and default on (2026-11-03)

- **No first-run divergence.** Only one run ever produced `[4370, 69415, ...]`: box 58813464 (`exit-main-120520`). Every 32K run on box 58872072, its first included, gave `[7294, 59549, ...]`. The two boxes ran different commits, and `exit_run.sh` built the prompt from the checkout (`cat AGENTS.md docs/adr/*.md`): 87,924 against 98,881 bytes, first difference at byte 78,551 (about token 20K). 8K prefills shared their text and agreed across boxes. The prompt is now frozen in `bench/prompts/exit_prompt.txt` (the 0956ccf text) and its token hash is in every exit JSON. On box 58900330 the frozen prompt gives `[7294, 59549, ...]` again, and libllama with the same token ids gives the same 64 greedy tokens (top-1 margin 12.9, no near-tie).
- **Default on.** Box 58900330 (RTX 3090, 15 threads, 91.6 GB, bench/results/20261103/exit-main-*-021753.json), 32K, same expert VRAM as llama.cpp's best `-ot` (10.97 GB): decode 2.83 tok/s off, 3.14 on (+11%); prefill 72.2 and 75.8; read time hidden 59% (29% on the 5-thread box). Prediction now runs only where it prefetches (Levers above), so `LoadOptions::prefetch` defaults to true.

## Reads that hide (2026-11-03, box 58958565)

At a 26-core quota the decode step is 257 ms: compute 112, streamer 128. The streamer reads 414 MB per token by the prefetch reader at 3.2 GB/s, 28% hidden, and 97 MB synchronously; synchronous reads of the same layer steps reach 5.2 GB/s. Per layer the next layer's read (about 4.4 MB) has one segment (about 1.2 ms) to land. Levers, in order of expected gain:

1. **Depth 2, actually on** (implemented above; never measured because the harness ran it with prefetch off). It doubles the window. Expected: the 95 ms wait drops to about 40.
2. **Reader queue depth** (implemented): the reader's device reads are 1 MiB at queue depth 16 (`IoOptions::chunk_bytes`, new) instead of whole slabs, three requests per expert. Expected: 3.2 to about 5 GB/s.
3. **Reads start before the layer's own misses** (implemented): `prefetch_ahead` runs right after the layer is planned, not after its synchronous reads. The reader has its own direct files, so both proceed at once. That adds the synchronous read time to the window.
4. **RAM tier by frequency** (not implemented): the LRU RAM tier hits 93.2% with 77 of 128 slots per layer. Keeping the most-routed experts, as the VRAM hot set does, should cut the misses; measure first from recorded routing, offline. Recorder, replay and threshold: ADR 0050 (c).
5. **RAM reserve** 8 to 5 GB (not changed): peak RSS was 84.9 of 91.4 GB, so about 3.6% more slots, a small gain.

Together, 1 to 3 should bring the streamer near the synchronous misses alone: about 150 to 170 ms per token, or 6 to 6.7 tok/s, against KTransformers' 5.04 on the same box. They are measured when boxes are allowed again.
