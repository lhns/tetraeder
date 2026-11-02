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
