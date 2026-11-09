# 0054. Decode expert reads: a persistent I/O service, experts computed in arrival order, RAM tier sized to the container

Status: Proposed (Phase 2 exit follow-up, ADR 0045 hypotheses 1 and 2). Supersedes ADR 0050 (a), stage 1 (Brief A); ADR 0050 (a) stage 2 (GPU group asynchronous) keeps its gate, now measured on this path; ADR 0050 (b) and (c) are unchanged. Supersedes the reason ADR 0035 gives against a memory map (below); its decision stands. Extends ADR 0031 (direct I/O), 0038 (prefetch, pins, ordering). Brief: `docs/levers-briefs.md`, Brief A (replaced).

## Context

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
- **There is no io_uring on the boxes.** Every box C run reports `io_backend: direct thread-pool`: Docker's default seccomp refuses io_uring, so `dio::pool::Pool` spawns up to `queue_depth` scoped threads per `read_batch_with` call and joins them, under a 26-core quota with 26 spinning ggml threads.
- **A layer waits for all its prefetched experts before it computes any.** `finish_prefetch(seg)` blocks before `plan_cpu`, so one late expert stalls the 6.6 RAM hits that could compute meanwhile.

KTransformers v0.7.2 decodes this regime from an `np.memmap` of the GGUF with no prefetch and no repack. Its 26 pinned workers split each layer into 32-row tasks; a worker that touches a missing page faults it in itself. So up to 26 reads are in flight, each one wakes exactly the thread that needs it, there is no copy, and the waiting thread gives its CPU quota back. It also has no GPU experts (`--kt-num-gpu-experts 0`), so its page cache (the container minus its own memory, about 85 GB) must also hold the 10 hot experts per layer we keep in VRAM: it holds at most as many experts per layer as our 77 RAM plus 10 VRAM slots. So its reads per token are likely not fewer than ours; the sampler stage of ADR 0045 measures it.

Per layer the work fits: the reads are 5.4 MB (0.48 experts) per layer, about 1.05 ms at 5.2 GB/s, against 1.19 ms of compute. Fully overlapped, decode is bound by compute plus graph overhead, about 130 ms; after ADR 0053 by the device, about 98 ms of reads per token at today's hit rate.

## Options

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

Decode alone does not separate them. Prefill, the budget and replay semantics, and the measured `-ot` numbers do: llama.cpp `-ot` is ggml's `mul_mat_id` on mapped pages with all threads faulting, the closest measured stand-in for option B, and it decoded at 0.81x and 0.62x of the builder on boxes A and B. Option A keeps every measured advantage and removes the three causes listed above.

ADR 0035 rejected a memory map because "a page-cache fault is a synchronous small random read". KTransformers shows that 26 threads faulting concurrently are a deep queue, so that reason is withdrawn. The reasons that stand are the ones in the table: prefill, control, accounting.

## Decision

### 1. One persistent I/O service in `memory`

- `memory::io::IoService` replaces `memory::reader::Reader` for every read that runs beside compute: prefetch (all steps) and the urgent misses of decode (section 2). Synchronous reads (prefill misses, hot-set loads) keep `read_experts` on the main thread's files.
- Persistent workers (`IoOptions::workers`, default 16) each own a `DirectFile` per shard opened with `queue_depth: 1` (so a read is inline, no thread spawn; Windows gets one handle per thread, Linux one fd per thread), and a bounce of one chunk. A worker takes the next chunk, reads it with `read_batch_with`, copies it into its destination from the same thread, and marks it done. Copies run on 16 threads in parallel with the device, never after it.
- A unit is one expert: its three slabs split into chunks of `IoOptions::chunk_bytes` (default 1 MiB, ADR 0038 lever 2). A unit completes when its last chunk has landed; its waiters wake. Splitting a slab at chunk boundaries widens each chunk to 4 KiB alignment on its own (at most 8 KiB extra per MiB, under 1% more device bytes).
- Order: earliest deadline first. The deadline is `(step, layer)` of the layer that needs the unit; urgent units (a miss of the layer being planned) go before prefetches with the same deadline. FIFO within a deadline. So the current layer's misses jump the prefetches of the next layers, and a prefetch for layer L+1 jumps one for L+2.
- No new I/O syscalls and no new `unsafe` outside `memory` (Rule 7): workers reuse `DirectFile` and `HostDst`. Native AIO or io_uring stay a follow-up, only if the box shows the workers cannot reach device rate while the CPU is busy (see "What could overturn this").
- Dropping the service waits for every queued unit; the streamer declares it before the RAM tier (same ordering as ADR 0038).
- Behind `IoOptions::service: bool` (default `false`); the `Reader` path stays until a paired run defaults the service on, then it is deleted.

### 2. Experts computed in arrival order (decode)

- Scope: `LoadOptions::overlap` (default `false`, needs `service`), `MissPolicy::Cpu` with a RAM tier, steps without op offload (`!offloads(..)`, decode and verify), one CPU chunk. Other steps take today's path.
- The layer no longer waits for its prefetch before planning. After the readback: `plan_cpu`, `prefetch_ahead` (unchanged, same place), then `stage_overlap`: the RAM lookups of the CPU experts in `fill_ram`'s order and with its counters (an expert whose prefetch is in flight already has its slot, so it is a hit), urgent units for the misses, and the split into `ready` (resident and landed) and `pending` (in flight or just submitted).
- No pending expert: today's path, unchanged (the CPU group runs inside segment L+1). Otherwise the host reads back `xn` (16 KB at decode), computes the `ready` experts as one group graph on a CPU-only scheduler, then repeatedly waits for any pending expert to land and computes all that have landed as the next group. Groups write unweighted outputs into a host tensor `cpu_out_host`; one copy moves the rows into `cpu_out` on the primary backend, and segment L+1 starts with `acc = set_rows(acc, cpu_out[:, 0..p], pairs)` (Brief A's mechanism, kept).
- Pins: every expert in flight for a layer stays pinned until it has landed and the layer has used it, including mispredicted ones (they are released without blocking when they land, or waited for at the next `finish_prefetch` of that layer). In decode an in-flight expert is among the layer's most recently inserted entries, so it is never the LRU victim of the layer's own misses: cache behavior, and so the replay's fidelity, is unchanged. A test checks it.
- `set_compute_threads` and `prefetch_threads` keep their meaning: with reads in flight the CPU can run one thread fewer, so the workers are not starved by the quota.

### 3. RAM tier sized to the container

- The harness reserve (`RAM_RESERVE_GB`, 8 GB) is a guess. Box C peaked at 84.85 of 91.4 GB with the 77-slot tier full (82.6 GB of slots, so about 2.3 GB of everything else), leaving 6.5 GB unused. Under ADR 0031 the tier stays a byte budget; only the reserve changes: 4 GB on the exit boxes, giving 81 slots instead of 77 (`floor(bytes / 1072594944)`), a peak near 89.2 GB, 2.2 GB under the limit. The run's `peak_rss_bytes` checks it; a run above 90 GB goes back to 8.
- Priced offline before the box run: the replay of ADR 0050 (c) on the 235B trace, `tetraeder-bench replay --trace F --vram 10 --ram 77,79,81,83 --policies lru` (prefetch replayed), after its fidelity gate (LRU at 77 within 0.5 points of the measured hit rate). The gain is the drop in decode disk bytes per token; at 75 MB per hit-rate point it is about 14 ms of device time per point. Prior from boxes A and B: 0.2 to 0.3 points per slot near 77, so about one point.
- The page cache's advantage (it grows to free memory) is this reserve and nothing else: our direct reads leave the page cache empty, so free memory is simply unused.

### Exactness

Groups are per expert, so each expert keeps all its rows, its row order and its K-quant kernel path (ADR 0035); the inputs are the same f32 bytes; the merge is a copy. Logits are bitwise equal with `overlap` and `service` on and off, on CPU and GPU builds. Computing row blocks of one expert as they arrive, across separate graphs, is not allowed: changing an expert's row count changes the CPU K-quant kernel on AVX (0.18 to 0.36 logit differences, ADR 0035 implementation notes). Row-level interleaving is free only inside one `mul_mat_id`, which is what option B gets and option A gives up.

### Metrics (Rule 6)

- `tetraeder.io.inflight` (histogram, chunks in flight, sampled at each submit) by `tetraeder.io.class` (`urgent`, `prefetch`).
- `tetraeder.moe.read.hidden` (histogram, ratio, per decode layer step with pending experts): `1 - read_wait / read_span`, where `read_span` runs from the layer's plan to its last landing and `read_wait` is the time the main thread blocked.
- `tetraeder.moe.layer.duration` (histogram, s) by `tetraeder.part`: `cpu_ready`, `read_wait`, `cpu_landed`, `copy`.
- Bytes per token: existing `tetraeder.tier.read.bytes` over `tetraeder.tokens`.
- `StepProfile` gains `cpu_ready`, `read_wait`, `cpu_landed`, `copy`, inside `streamer` so `total` still adds up.
- Panel "Expert reads under compute" in the "Experts and cache" row: chunks in flight p50 by class, hidden share p50, disk bytes per token.
- Alert `TetraederExpertReadsNotHidden`: overlap on, hidden share p50 below 0.3 for 15 minutes while urgent or prefetch reads occur. Cause: few chunks in flight means the workers are starved by the CPU quota (lower `prefetch_threads`); many in flight means the disk is saturated (grow the RAM tier, ADR 0050 c).

### Expected gain

- Against `main-prefetch` on box C (257 ms; the next run will also have ADR 0038 levers 2 and 3, still with the serial copy): 145 to 165 ms per token, 6.1 to 6.9 tok/s, 1.2x to 1.37x KTransformers' 5.04. Reasoning: compute and graph overhead stay at 128 ms; the 95 + 19.5 + 13 ms of waits and copies shrink to 10 to 25 ms of exposed reads (device time 98 ms against 112 of compute, hidden per layer behind the hit group and the prefetch window) plus 2 to 5 ms for about 0.4 extra CPU-only graphs and one `xn` readback per layer step with pending experts.
- These estimates overlap ADR 0038's for its levers 1 to 3 (150 to 170 ms): those assumed the reader reaches device rate, which the serial copy prevents. The paired run reports the service alone (`main-iosvc`) and with arrival order (`main-overlap`) apart, so each part has its own number.
- With ADR 0053 (compute 70 to 85 ms) the gains do not add: the step becomes bound by the 98 ms of device time, about 115 to 135 ms. Then bytes per token decide, and the tier sizing (about 14 ms per hit-rate point) and ADR 0050 (c) are the next levers.

### What could overturn this (ADR 0045 sampler stages)

- Our `main` decode already at `disk_queue_depth_mean` near 16 and `read_mb_s` at the device rate: the disk is saturated, so deeper queues buy nothing; only the copy removal and fewer bytes (section 3, ADR 0050 c) help. Build section 1 only; skip section 2.
- KTransformers' `read_bytes_per_token` under 70% of ours: its cache keeps more useful experts than our LRU at a similar size. Then the replay's `freq` policies and the reserve go first, and option B is reconsidered only if the replay cannot explain the gap.
- `main-iosvc` reaching under 4 GB/s with fewer than 8 chunks in flight while compute runs, and over 5 with compute idle: the workers lose the CPU to the spinning pool even with `prefetch_threads`. That is the case where faulting in the compute threads (option B, or native AIO with one submitter) wins; this ADR is then superseded for decode.
- `main-native` or ADR 0053 closing most of the gap: compute was the gap, and this lever becomes the floor after it rather than the fix.

## Consequences

- `memory` gains the service (one module, threads and a priority queue, no new dependency); `Reader` goes away once the service is the default.
- Up to 0.4 extra small CPU graphs per decode layer step, only when an expert is still in flight at plan time.
- Two engines coexist in the streamer until the paired run decides; the brief keeps them behind one enum.
- Memory: 16 workers times the shard count times one chunk of bounce (about 80 MB at 5 shards) replaces the reader's 64 MiB per shard; reported as `io`.
- Option B is not built. It returns if the third bullet above holds.
- All box measurements wait for the hold to lift; local tests use the synthetic MoE of `crates/model/tests/common/synth.rs` with an injected per-chunk delay.
