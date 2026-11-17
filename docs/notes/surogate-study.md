# Surogate study: MoE serving, multi-GPU, scheduler (2026-11-17)

Read-only comparison of Surogate (github.com/invergent-ai/surogate, Apache-2.0, C++/CUDA) with Tetraeder on three topics. Source read: shallow clone of `main` at `c7df4bafb41129da88ab5ba82fa84037be7b5ee7` (2026-11-05). No Surogate code was run or copied; everything below is paraphrased. Surogate numbers are self-reported, mostly on RTX 5090 (PCIe 5) with H100 and DGX Spark rows; none was measured here.

Two limits frame everything:

- **Regime.** Surogate keeps every offloaded expert in page-locked host RAM (`--host-moe-layers`; the docs say that memory cannot be swapped out). There is no NVMe tier and no expert prefetch. Our Phase 2 exit regime is a model larger than RAM (ADR 0035, 0038, 0054), so their MoE design answers the "fits in RAM" case only.
- **Hardware.** Their serving build targets SM89, SM90a, SM120a and SM121a with CUDA 13 and driver 580 or newer (README "Hardware"; no CUDA 12 package). Our verdict boxes (RTX 3090, A40, A6000) are SM86, so no paired run is possible there. At most an informative run on an RTX 4090 box.

Paths below are relative to `csrc/src/serve/` in their repo unless they start with `docs/`, `surogate/` or `DESIGN-`.

## 1. MoE serving: GPU expert cache and CPU/GPU split

### What Surogate does

Files: `family/impl/moe/expert_cache.{h,cpp}`, `api/ops/expert_slot_cache.h`, `ops/expert_slot_cache/expert_slot_cache.cu`, `ops/cpu_expert_compute/cpu_expert_compute.cpp`, `family/impl/moe/banked_experts.cpp`; docs `docs/inference/cli.md` ("Host offload", "MoE expert cache"), `surogate/serve/BENCHMARKS.md`.

- **What is cached.** One device pool of expert slots per GPU, sized from free memory or `--expert-slots`. A single directory covers every layer: flat id `layer * experts + expert`, `slot_of_expert[layers * experts]` and `expert_of_slot[slots]`. So the VRAM pool is one LRU across all layers, not per-layer slots.
- **Eviction.** Each resolve bumps a round stamp. Hits are stamped in parallel; one thread then walks a clock hand over the slots and takes the slot with the oldest stamp, stopping early at an empty slot or one at least 64 rounds old. A slot used in the current round is never a victim. Deterministic order.
- **Admission.** Every miss that is not sent to the CPU is admitted: it is copied into its slot in the same round and computed on the GPU in that round. There is no frequency pinning, no hot-set profile and no prediction (a grep for prefetch or expert prediction in `csrc/src/serve` finds only kernel-level prefetch; `surogate/serve/tools/freq_corpus` holds token counts for the draft head's vocabulary ranking, not routing).
- **Scan resistance.** Rounds wider than 32 tokens count as prompt scans. Their misses go to a ring of trailing slots (one layer's expert count) that the LRU clock never places into, round-robin. So a prefill sweep over all experts cannot evict the decode working set. Enabled when the pool is smaller than the whole model and at least two expert sets large (`SUROGATE_SERVE_NO_SCAN_RING` turns it off).
- **Where resolve runs.** Entirely on the device: the resolve kernel writes the slot table and a miss list whose count is a device word, and the gather kernel reads that count. No host readback of routing ids, so a whole decode step captures into one CUDA graph.
- **The gather.** A kernel copies 16-byte units of each missing expert from the host bank through its device-mapped alias into the pool (four planes per expert, streaming loads). The whole bank is pinned and mapped: 105.7 GiB for Flash-Next and 191.7 GiB for GLM-5.3-Flash in their rows. `family/impl/load/host_bank.cpp` fills each object by parallel first touch and then `cudaHostRegister` with the mapped and portable flags (a comment there says `cudaHostAlloc` pins at about 1.8 GB/s and does not parallelise), with `cudaHostAlloc` as fallback.
- **The split.** With `--cpu-moe-share`, a fixed fraction of a round's distinct misses (interleaved in miss order by a running counter) is not gathered; their (token, expert, weight) jobs go to host workers instead. Hits are never split. The decision is per round (per layer per step): rounds up to 64 tokens use the decode share, wider rounds up to the staging width the prefill share (default 0), wider ones none. Minimum round width 1 (their note: at one user, 1 read 20.2 against 15.3 tok/s for 4).
- **Choosing the share.** `auto` times, at startup and outside capture, one layer's gather and host round both alone and running at the same time (they contend for DRAM), then sets `share = host / (host + PCIe)` from the concurrent rates, clamped to 0.3 to 0.9. The prefill share is 0 unless given: a code comment calls the prompt split a measured loss at every chunk width (host GEMM far below VNNI peak, every combine waits on the host tail). There is code to derive a prefill share from the decode share, but with the default of 0 it does not engage. Example: 189 GB/s host against 46 GB/s gather on their Flash-Next host, 202 against 46 for GLM (83% of misses on 32 host cores).
- **Overlap.** In stream order: resolve, gather, D2H staging of activations and the job list on the main stream, then the host round is forked to a side stream and runs while the GPU computes hits and gathered experts; the combine waits for it. The gather is not overlapped with hit compute (an env probe exists to decide whether that is worth doing).
- **Handshake.** Instead of `cudaLaunchHostFunc` plus an event, the stream raises a flag in mapped pinned memory (`cuStreamWriteValue64`), a persistent coordinator thread runs the round and raises a done flag the stream waits on (`cuStreamWaitValue64`). Raise-and-clear values so it survives graph replay. Self-reported +6% decode over the host-function path; the host-function path stays as fallback. The coordinator spins; a second pipeline stage sharing the host pool keeps the host-function path so as not to add a spinning thread per stage. Host workers default to the cores other jobs leave free (their benchmark note); no cgroup quota handling was found.
- **Host compute.** At load the bank is repacked per half into planar group formats (Q4G32AM, Q5G32AM, W8G32 by source width) read by both the gather and AVX-512 kernels (`vpmaddwd`, activations quantized per 32-group). One worker pool per process, or per NUMA node for pipeline stages.
- **Pipeline rule.** The split is turned off on a stage whose pool would hold nearly all of its experts.

### What Tetraeder does

- Segments per MoE layer with a host round trip for routing ids (ADR 0030, 0036), pair groups with misses computed on the CPU (ADR 0035, policy `Cpu`), static VRAM hot set from routing counts.
- Disk tier with predicted next-layer prefetch (ADR 0038: accuracy 0.85, +11% decode), I/O service and arrival-order compute (ADR 0054), RAM tier per layer with LRU or frequency pins (ADR 0050 c).
- Admission of a share of CPU misses into VRAM (ADR 0059): measured 0.52 to 0.61x of the default because synchronous uploads ran at 2.56 GB/s (4.47 ms per 11.4 MB expert); an `Async` mode with a pinned ring is designed.
- Per-token dispatch cost: about 0.88 ms fixed per CPU-carrying segment, about 70 ms of a roughly 200 ms decode token (ADR 0064).
- Op offload gated by step tokens (ADR 0044), no CPU repack (ADR 0039).

### Difference

1. Their miss path never leaves the device's stream: resolve and gather are kernels, the CPU round is triggered by a flag. Ours reads ids back to the host every layer, which ADR 0064 names as the dominant fixed cost.
2. Their admission is cheap because the source is pinned and mapped and the copy is a kernel inside the round. Ours was host-copy bound (ADR 0059 addendum b).
3. Their VRAM pool is global LRU with a scan ring; ours is a static per-layer hot set, and prefill never admits (ADR 0059).
4. Their split share is measured under DRAM contention on the serving host; ours is "all misses to CPU", with llama.cpp's tuned `--moe-cache-cpu-share` as the baseline (plan.md levers table).
5. They have nothing for disk; our prefetch and I/O work has no counterpart there.

### Ideas, ranked by expected value

1. **A flag handshake in place of step 2's per-layer sync.** ADR 0064 step 2 already designs one graph per decode token with an eval callback per layer; the callback still synchronizes the backend at each observed node, reads ids back and runs the CPU groups inside it, and the ADR puts the remaining syncs and small copies at 4 to 5 and 6 per CPU layer, 10 to 15 us each. Surogate shows the next step: the stream raises a flag, a host thread runs the round, the stream waits on a done flag, so the GPU front end never stops for the host. Two cautions from our own data: their coordinator spins, which ADR 0061 shows costs decode under a cgroup quota (wait with backoff, or count the thread in the quota), and ADR 0064 says the per-segment cost on the 7642 hosts follows the quota more than the syncs. Test first, cheaply: a `gpu_` micro-benchmark in `ggml-sys` on a 3090 box timing a per-layer round trip three ways (backend sync plus ids readback, `cudaLaunchHostFunc` plus event, `cuStreamWriteValue64`/`WaitValue64` with a non-spinning waiter), 94 rounds per token as the 235B, under `CPUQuota`. Pursue only if the memop path saves at least the sync and copy share step 2 leaves (about 5 to 9 ms per token). It needs memop nodes in ggml graphs, so a patch (Rule 8) and its own ADR. Amends ADR 0064.
2. **In-round zero-copy gather instead of synchronous admission upload.** ADR 0059 failed on upload cost, not on the idea. Its own `Async` mode (pinned ring, second stream) was designed but the `main-overlap-adapt-async` stage has no result in `bench/results`, so that arm is the first test: paired with `main-overlap` on an EPYC 7642 host where CPU pairs set the rate. Next, Surogate's form: a kernel gather from `cudaHostRegister`ed memory at link speed with no host copy; measure its GB/s on a 3090 (PCIe 4; far above 2.56 GB/s expected, to be measured). That gather is a new ggml op, so Rule 8. Pinning conflicts with the RAM tier's lazy commit (ADR 0035): register only a bounded staging region or the VRAM-candidate experts. Amends ADR 0059.
3. **Share measured under contention.** If admission or a GPU share returns, set the split from concurrent gather and CPU-group rates at startup, clamped, as `auto` does, instead of a fixed `admit_share`. Test: a load-time probe recorded in the exit JSON, then a paired run of auto against `--moe-cache-cpu-share` tuned by llama.cpp's script on the same box. Amends ADR 0059 and the cost model inputs (ADR 0056).
4. **Scan ring for the tiers.** Answers part of ADR 0050's open "tier contents at the prefill/decode switch" lever, but only for serving with several requests: in the single 32K exit prompt there is no earlier decode set to protect, so frequency pins taken at the end of prefill remain the better lever there. Test: add a `ring:R` RAM policy to `crates/memory/src/replay/sim.rs` (prefill steps place misses only in R trailing slots per layer) and replay a multi-request trace; the saved 235B trace in `runs/traces/` is one request, so first record a routing trace of a few turns (2K prompts plus 128 decode) on the 235B box. Threshold to fix before replaying, for example 25% fewer disk bytes in the first decode window of each turn. Amends ADR 0050 (c).
5. **Global VRAM LRU.** Our RAM-tier replay of a cross-layer pool did not reach its halving bar (ADR 0050 addendum 2026-11-14); a VRAM-tier `glru` with admission was not replayed. Test: replay `vram_policy glru` at VRAM 26 on the saved trace, reporting VRAM pair share and admission bytes per token. Only worth it if idea 2 makes admission cheap. Amends ADR 0050.

Not proposed again: LRU admission with synchronous upload (ADR 0059, measured), frequency RAM pins (ADR 0050 c, built, off), CPU repack (ADR 0039: it would disable op offload for the RAM tier; their repack works because they do not op-offload from the bank).

## 2. Multi-GPU without NVLink

### What Surogate does (serving)

Files: `runtime/engine/pipeline_instance.h`, `pipeline_partition.h`, `pipeline_schedule.h`, `concurrent_executor.h` (`pipelined_iteration`); docs `docs/inference/cli.md` ("Devices", "Pipeline or data-parallel").

- **Stage split.** `--devices A,B,...` gives one stage per GPU, contiguous layer ranges. The partition balances planned device bytes per layer (from the load plan, `targets/registry.cpp`) by binary search on the max stage total, after merging layers that share KV into one unit (a cut never separates a layer from the layer whose KV it reads). Memory, not compute or link time.
- **Micro-batching.** Lanes are split into groups by `lane % groups`, groups default to the stage count, and each group is a "flight" that moves through the stages as a software pipeline: when stage s finishes a flight, its export is parked, and the oldest ready flight is consumed next. The group count shrinks per round so that no group is narrower than a minimum lane count. Prompts ride with decode lanes as mixed flights. Under MTP speculation the synchronous path runs one group.
- **Activation transfer.** The residual crosses a boundary device to pinned host to device, with a host `memcpy` into a per-flight park buffer; no P2P at all. Latency is hidden only by having several flights in flight, not by overlapping copies inside a flight.
- **Their own verdict.** The docs say a pipeline adds no throughput for a model that fits one GPU; `--data-parallel` (one full engine per GPU, sticky routing of a conversation to its replica, rebalanced above 1.25x the mean load) decoded about 40x more on six RTX PRO 6000 cards than a six-stage pipeline (self-reported). Pipelines also keep the full KV reservation at admission (no on-demand growth).
- **Expert parallelism at serving time:** none found by grep at `c7df4ba`. EP exists for training (`csrc/src/runtime/ep/`).

### What Surogate does (training, for the README claim)

`docs/guides/dispatch-pp.md`: stages of about 4 blocks run round-robin over GPUs, frozen base weights stream from pinned host memory once per step, activations and gradients cross through host memory, LoRA only. Code exists (`csrc/src/runtime/dsl/dispatch_pp_plan.h`, `dsl_model_execution.cpp`) but was not read in depth; training only, out of scope for serving.

### What Tetraeder does

ADR 0069 (own executor: per-rank graphs cut at collectives, async issue from one thread, `Copy` ops for stage hops through `ggml_backend_tensor_copy_async`, which the driver stages through host memory without peer access), ADR 0070 (layer split from the search's dynamic program, tensor, expert and tp-times-pp shapes, `HostStaged` links priced from measured curves, patch 0004), ADR 0068 (verdict: one sequence, pp2048 plus tg128). Micro-batches only for prefill (`prefill_micro_batches`).

### Difference

- Their pipeline is a batched-serving design (groups of lanes in flight); ours is judged at batch 1, where a pipeline can only lose (plan.md "Parallel shapes across slow links").
- Their partition balances memory only; ours prices compute, KV and link cost per stage.
- We have tensor and expert parallel; they have neither at serving time.
- Data-parallel replicas are a first-class serving mode there; in our plan they are Phase 6.

### Ideas, ranked

1. **Data-parallel replicas when the model fits one GPU.** Their 40x figure is self-reported and unpaired, but the direction matches our own cost model. Add a `dp-g` candidate to `enumerate_multi` for serving (not for the Phase 5 batch-1 verdict), with sticky conversation routing in the gateway so prefix caches stay warm. Test: simulator fixture `two_3090_hoststaged` with a c4 workload, then a 2x3090 session pairing `dp-2` against `layer-2` and `tp-2` at c4 (ADR 0065 workload). Amends ADR 0070; touches ADR 0028 (gateway).
2. **Decode groups in flight for layer split under `serve --parallel`.** Partition lanes by `lane % stages` and keep one group per stage in flight, reusing ADR 0069's per-micro-batch destination tensors. Test: CPU-rank executor test that results equal the lockstep run, then c4 throughput of `layer-2` with and without groups on a 2x3090 box. Amends ADR 0069. Low until batched multi-GPU serving is judged.

## 3. Scheduler: prefix cache, chunked prefill, batching limits

### What Surogate does

Files: `DESIGN-host-kv-tier.md`, `DESIGN-kv-admission.md`, `family/impl/runtime/host_tier_impl.h`, `family/impl/kv_tier/{host_pool,keys}.*`, `runtime/engine/concurrent_executor.h`, `core/limits.h`; docs `docs/inference/cli.md` ("Host prefix cache", "Scheduling").

- **Objects.** A conversation leaves the GPU as 1,024-token KV chunks (16 pages of 64) plus a boundary object (recurrent state, tail hidden, draft state, the pages past the last whole chunk). Keys are a chained SHA-256 over chunk tokens under a namespace (weights, KV layout and dtype, draft head, pipeline stage, adapter content digest). Chunks are shared by key, so conversations with the same system prompt and tools, and successive turns of one conversation, store them once.
- **Host pool.** One page-locked pool per process: mmap, parallel first touch, `cudaHostRegister` at startup (about a second per 8 GiB, their figure). Fixed-size blocks of one chunk, so no fragmentation. Whole entries evicted LRU across programs; leases protect restores and writes; abandoned writes are poisoned.
- **Background copy.** At every round boundary, if the copy stream is free, the oldest retained idle lane without a current snapshot is copied to the host, only the chunks the pool lacks. A later eviction of that lane then costs nothing. Copies run on their own stream after an event; the compute stream waits only if it reuses the lane's pages first (`fence_lane_copies`). Restores are waited for in stream order, no host sync.
- **Self-reported effect.** RL server on one 5090, ten agent conversations growing to 100k tokens: 85% of later prompts cached with 8 GiB against 15% without the tier, a turn 7.1 s instead of 18.8 s.
- **Not built** (their list): layer-wise restore overlapped with the first chunk, cross-replica lookup, NVMe or external back end (a C table `sinfer_kvtier_v1` is defined for one).
- **Admission.** `--kv-admission ondemand` (default): a request is entitled its prompt plus 256 tokens of headroom; lanes grow at round boundaries, oldest first; when pages run out, idle prefixes go first, then the newest running request is preempted and later recomputed (prompt plus generated tokens), strict FCFS, oldest never preempted by a newer one. A request whose full reservation can never fit is refused up front.
- **Chunked prefill and batching.** `--max-num-batched-tokens` 2048 by default (multiple of 128); prompts share rounds with decoding lanes (mixed rounds, up to 64 prefills per round, `runtime/contract/types.h`); a burst of prompts arriving while nothing decodes is prefilled together. Round width at most 128 columns (`core/limits.h`); `--max-num-seqs` defaults to 1 and more than 128 needs an explicit 256 (docs, not checked further). `--batch-invariant` fixes chunk cut points for reproducible logits.

### What Tetraeder does

ADR 0049 (RAM prefix cache of whole-sequence snapshots, LRU, 8192 MiB), ADR 0065 (unified pool with mask, FIFO admission, reservation of prompt plus min(max_tokens, 1024), growth, preemption of the newest with a snapshot, step budget, resident idle sequences), ADR 0067 (chunked layout), ADR 0074 (save through pinned staging; snapshot spare pool). Session `20261117e` (box 61060914): with the spare pool, admission `save_ms` median fell from 46 to 11 ms (8B) and 31 to 10 ms (30B), but the 8B p95 is 48 to 51 ms against the 25 ms bar (one fallback per run) and the 30B passes only with huge pages.

### Difference

- They copy a conversation when it goes idle, before an admission needs its rows (and still copy at takeover when no snapshot exists yet); we copy at admission, on the worker thread, with live sequences stalled. ADR 0074's argument that overlap buys nothing assumes the copy starts at admission. A copy that starts at retirement has the gap until the rows are reused: in `20261117e` (`server-q8b-thp1.log`) the 8B probe's sequence went idle 17 to 55 ms before the admission that saved it, while that save took 10 to 48 ms through pageable memory. A DMA straight into registered blocks needs well under that if it reaches pinned rates (78 MB at 10 GB/s is 8 ms; the rate on a 3090 is to be measured).
- Their host memory is registered once at startup in fixed blocks, so no first-touch faults or huge-page compaction on any save; our spare pool prepares buffers on a thread and still falls back.
- Their entries share chunks by hash; ours are whole snapshots, and a prefix of a longer entry is replaced, not shared.
- Admission rules are close: both reserve less than prompt plus max_tokens and preempt the newest; their headroom is 256, our GROW 1024.

### Ideas, ranked

1. **Save at retirement, on a copy stream, into a startup-registered block pool.** When a sequence finishes, enqueue its rows' D2H copies on a second backend stream (the second backend instance of ADR 0059's `Async` mode is the pattern) into fixed blocks of a pool registered at load; an admission that reuses the rows fences only on copies not yet done. Over the spare pool of ADR 0074 it adds: no host copy at all (the DMA lands in the entry's blocks), no fallback allocation (the 8B p95 failure in `20261117e` was one fallback), no first-touch or compaction on any save, and the copy overlaps the decode steps of other sequences. Test: CPU unit tests of the block pool (no block handed out twice, leases survive eviction) and a property test that rows are never overwritten before their copy is fenced; a `gpu_` test of D2H rate into registered memory on a 3090; then `scripts/gpu_kv_save2.sh` with ADR 0074's bar, reading both `save_ms` (expected near 0 where the gap covers the copy) and the fence wait, then the D2b TTFT replay. Amends ADR 0074 (narrows its "overlap buys nothing" consequence to copies started at admission) and ADR 0049 (storage).
2. **Chunk-keyed entries with chained hashes.** Claude Code sends a main conversation and subagents that share the system prompt and tools; whole snapshots store that prefix once per entry. Test offline first: run the recorded Claude Code request log (`bench/private/`) through a `PrefixCache` simulator with whole entries against 1,024-token chunk entries at 8192 MiB, reporting host bytes per save, entries held and RAM-hit tokens; build only if hit tokens rise or save bytes fall by a margin fixed beforehand. Then the D2b replay pair. Amends ADR 0049; ADR 0069's one-rank snapshot layout already makes the bytes plan-independent, which a chunk key needs.
3. **Smaller reservation headroom.** Their 256-token headroom against our GROW 1024 matters only when the pool is tight; ADR 0067's simulator can price it on workload W in CI. Amends ADR 0065.

## README claims: checked or not

| Claim | Status |
| --- | --- |
| GPU expert caching and CPU/GPU split for every supported MoE family | Code supports it: `banked_experts.cpp` configures `ExpertCache`, and qwen3_moe, qwen3_5_moe, gemma4_moe, lfm2, glm5_next and qwen4exp bind banked experts (grep, not exhaustive) |
| Multi-GPU pipelines without P2P (serving) | Verified: device, pinned host, device copies (`pipeline_instance.h`) |
| Dispatch pipeline parallelism over PCIe without P2P (training) | Docs only (`docs/guides/dispatch-pp.md`); code not checked |
| Expert parallelism | Training only; no serving EP found by grep |
| Up to 128 active sequences | 128 is the round width (`kMaximumBatchColumns`); the concurrency cap is not 128 and the docs allow 256 |
| Prefix caching with host tier | Verified in code and design doc |
| 7.0x llama.cpp, GLM-5.3-Flash, 8x 5090, 16 users | Their own notes: llama.cpp was an unreleased PR build that served the 16 streams nearly one after another, and the rows predate a kernel change. A batching gap in that baseline, not a PCIe technique |
| 1.69x llama.cpp with experts in host RAM | Baseline is `-cmoe -b 4096 -ub 4096` (all experts on CPU); `moe-cache` does not appear in `BENCHMARKS.md`, so not against llama.cpp's expert cache with a CPU share, our Phase 2 baseline |
| +6% decode from the memop handshake | Code present; number self-reported |
| Rationale in `design/INFERENCE.md`, `design/serve-engine-flash-next.md` | Cited in code, not in the public repo; unverifiable |

## Are their benchmarks usable for us?

No, as numbers. Different GPUs (5090 PCIe 5, H100, GB10), CUDA 13, NVFP4 and FP8 artifacts on most rows, staggered closed-loop clients, and comparisons against llama.cpp configurations weaker than our baselines. Usable as hypotheses only: the split rule (host over host plus PCIe under contention), the scan ring, background KV offload, and the pipeline-versus-replica result. Their method notes are worth keeping: staggered clients (in-phase clients made an 8-stage pipeline read 381 tok/s against 276 staggered) and a 60 to 90 s steady-state window (shorter windows read up to 25% high). If we ever want Surogate as a third engine, it needs an SM89 box with driver 580 or newer (RTX 4090), recorded as informative, never gating.

## Top ideas across topics

1. Save KV at retirement into a startup-registered pool (topic 3, idea 1): on the Phase 4 critical path, ADR 0074.
2. Run ADR 0059's never-measured async admission arm, then a zero-copy gather if it is still upload-bound (topic 1, idea 2): ADR 0059.
3. Flag handshake instead of the per-layer sync left in ADR 0064 step 2, gated by a round-trip micro-benchmark under a quota (topic 1, idea 1): ADR 0064.
4. Chunk-keyed prefix entries, priced offline on the Claude Code log (topic 3, idea 2): ADR 0049.
5. Data-parallel replicas as a serving shape (topic 2, idea 1): ADR 0070, Phase 6 or earlier.
