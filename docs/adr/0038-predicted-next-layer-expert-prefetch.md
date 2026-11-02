# 0038. Predicted next-layer expert prefetch into the RAM tier

Status: Proposed (Phase 2). Accepted once the measurement below holds on the 235B. Extends ADR 0035; does not change it.

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
- **Not done:** no fault-injection hook exists; the error test cuts the model file, which fails whichever read comes first (the reader's own error path is unit-tested in `memory`). Intra-layer pipelining for prefill is still the follow-up. Accuracy and overlap on the 235B are measured by the exit run; the status stays Proposed until then.

