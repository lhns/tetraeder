# 0054. Decode expert reads: a persistent I/O service, experts computed in arrival order, RAM tier sized to the container

Status: Accepted (2026-11-15: built and measured; in serve). Earlier: Proposed (Phase 2 exit follow-up, ADR 0045 hypotheses 1 and 2). Supersedes ADR 0050 (a), stage 1 (Brief A); ADR 0050 (a) stage 2 (GPU group asynchronous) keeps its gate, now measured on this path; ADR 0050 (b) and (c) are unchanged. Supersedes the reason ADR 0035 gives against a memory map (below); its decision stands. Extends ADR 0031 (direct I/O), 0038 (prefetch, pins, ordering). Brief: `docs/levers-briefs.md`, Brief A (replaced). Outcome: the service and arrival-order compute meet the decode bar on the paired 235B run (`20261110c`) and with CPU matched (`20261110e`), and are the `serve` defaults; the hidden share bar is missed and is no longer the measure (`20261111a`).

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

## Context

Box C (ADR 0045 addendum, 26-core quota, `20261103`): the builder decoded well behind KTransformers. Per token, compute (mostly the CPU experts) and the main thread blocked in `finish_prefetch` were the two large buckets; synchronous misses, the bounce-to-slot copy on the main thread, and allocation, build, readback and inputs were smaller.

Brief A (ADR 0050 a) targets only the synchronous misses. The prefetch wait is larger, and it has a cause in our I/O path, not in the device:

- **The reader is busy for most of the step.** `Reader` is one thread running `read_batch_with`, which reads a whole sub-batch, waits for all of it, then copies every range into its slot on that same thread (`dio/mod.rs:378-430`), so the serial copy adds to the device time (`20261103`). ADR 0038 lever 2 (1 MiB chunks at depth 16, implemented, not measured) shortens the device part but leaves the copy serial, so the reader's ceiling stays. Once compute drops (ADR 0053), that ceiling is the step.
- **There is no io_uring on the boxes.** Every box C run reports `io_backend: direct thread-pool`: Docker's default seccomp refuses io_uring, so `dio::pool::Pool` spawns up to `queue_depth` scoped threads per `read_batch_with` call and joins them, under a 26-core quota with 26 spinning ggml threads.
- **A layer waits for all its prefetched experts before it computes any.** `finish_prefetch(seg)` blocks before `plan_cpu`, so one late expert stalls the 6.6 RAM hits that could compute meanwhile.

KTransformers v0.7.2 decodes this regime from an `np.memmap` of the GGUF with no prefetch and no repack. Its 26 pinned workers split each layer into 32-row tasks; a worker that touches a missing page faults it in itself. So up to 26 reads are in flight, each one wakes exactly the thread that needs it, there is no copy, and the waiting thread gives its CPU quota back. It also has no GPU experts (`--kt-num-gpu-experts 0`), so its page cache (the container minus its own memory, about 85 GB) must also hold the 10 hot experts per layer we keep in VRAM: it holds at most as many experts per layer as our 77 RAM plus 10 VRAM slots. So its reads per token are likely not fewer than ours; the sampler stage of ADR 0045 measures it.

Per layer the work fits: at device rate a layer's reads take about as long as its compute (`20261103`). Fully overlapped, decode is bound by compute plus graph overhead; after ADR 0053, by the device at today's hit rate.

## Options

Both options keep the VRAM hot set and the prefetch prediction (ADR 0038). Box C's numbers for today and the per-token estimates of both options are in `20261103`.

- **A: direct I/O service, arrival order (chosen).** The same bytes, fewer with the tier sized to the container (the replay decides). 16 persistent workers on a rolling queue aim at device rate under compute. No serial copy: each worker copies its own 1 MiB chunk. Compute as bytes arrive is per expert (rows of one expert must stay in one `mul_mat_id`, see Exactness). Prefill keeps its path. Control stays: slot tables, pins, replay.
- **B: page cache, compute in place on mapped pages.** Similar bytes: the page cache holds about as many experts as RAM plus reserve, with the kernel's LRU instead of ours. Device rate under compute unknown: up to 26 faulting threads, 128 KB readahead per fault, `MADV_WILLNEED` for predictions. No copies. Compute as bytes arrive is per page, inside one `mul_mat_id`. Prefill: op offload copies whole 128-expert tensors per ubatch and refaults the uncached half; llama.cpp's mmap `-ot` prefill is slower than ours on the same boxes (`20261103`). Control goes to kernel reclaim: no pins (a prefetched expert can be dropped before use), no frequency pins (ADR 0050 c), the replay no longer models the tier, `memory()` cannot report used tier bytes (ADR 0037).
- The estimated exposed wait and decode step are about the same for both.

Decode alone does not separate them; prefill and the budget and replay semantics do. llama.cpp `-ot` runs ggml's `mul_mat_id` on mapped pages with all threads faulting and decoded slower than the builder on boxes A and B (`20261103`), but it also places 7 whole layers in VRAM instead of a per-layer hot set, so those ratios are not evidence against mapped pages as such. Option A keeps every measured advantage and removes the three causes listed above.

ADR 0035 rejected a memory map because "a page-cache fault is a synchronous small random read". KTransformers shows that 26 threads faulting concurrently are a deep queue, so that reason is withdrawn as stated. A narrower hazard of option B stands: Linux turns mmap readahead off for a file once its fault misses outrun its hits by `MMAP_LOTSAMISS` (100; `mmap_miss` in `do_sync_mmap_readahead`, `mm/filemap.c`), after which every fault reads 4 KiB, the likely mechanism of the `-ot` collapse once the model exceeds RAM; KTransformers may avoid it only through `MADV_WILLNEED`. The reasons that stand are prefill, control, accounting, and that hazard.

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
- Pins: today `finish_prefetch` clears the layer's pins (resident predicted experts and reads in flight) before its lookups. The overlap path first reaps landed units without blocking, then sets the layer's pins to exactly its experts still in flight, dropping the resident predicted ones as today, and only then looks up. In-flight experts stay pinned until they have landed and the layer has used them, mispredicted ones included (released without blocking when they land, or waited for at the next `finish_prefetch` of that layer). In decode an in-flight expert is among the layer's most recently inserted entries, so it is never the LRU victim of the layer's own misses. And `start_prefetch(il)` first waits for leftover units of `il` from earlier steps (long landed under earliest-deadline order), so `prefetch_plan` sees the same in-flight set as today. Cache behavior, and so the replay's fidelity, is unchanged; a test checks it.
- `set_compute_threads` and `prefetch_threads` keep their meaning: with reads in flight the CPU can run one thread fewer, so the workers are not starved by the quota.

### 3. RAM tier sized to the container

- The harness reserve (`RAM_RESERVE_GB`, 8 GB) is a guess: box C left part of the container unused with the 77-slot tier full (`20261103`). Under ADR 0031 the tier stays a byte budget; only the reserve changes: 4 GB on the exit boxes, giving 81 slots instead of 77 (`floor(bytes / 1072594944)`). The run's `peak_rss_bytes` checks it; a run above 90 GB goes back to 8.
- Priced offline before the box run: the replay of ADR 0050 (c) on the 235B trace, `tetraeder-bench replay --trace F --vram 10 --ram 77,79,81,83 --policies lru` (prefetch replayed; the `trace` stage of `exit_run.sh` gains these counts), after its fidelity gate (LRU at 77 within 0.5 points of the measured hit rate). The gain is the drop in decode disk bytes per token, as device time per hit-rate point; the prior from boxes A and B is about one point (`20261103`).
- The page cache's advantage (it grows to free memory) is this reserve and nothing else: our direct reads leave the page cache empty, so free memory is simply unused.

### Exactness

Groups are per expert, so each expert keeps all its rows, its row order and its K-quant kernel path (ADR 0035); the inputs are the same f32 bytes; the merge is a copy. Logits are bitwise equal with `overlap` and `service` on and off, on CPU and GPU builds. Computing row blocks of one expert as they arrive, across separate graphs, is not allowed: changing an expert's row count changes the CPU K-quant kernel on AVX (0.18 to 0.36 logit differences, ADR 0035 implementation notes). Row-level interleaving is free only inside one `mul_mat_id`, which is what option B gets and option A gives up. (No session directory yet.)

### Metrics (Rule 6)

- `tetraeder.io.inflight` (histogram, chunks in flight, sampled at each submit) by `tetraeder.io.class` (`urgent`, `prefetch`).
- `tetraeder.moe.read.hidden` (histogram, ratio, per decode layer step with pending experts): `1 - read_wait / read_span`, where `read_span` runs from the layer's plan to its last landing and `read_wait` is the time the main thread blocked.
- `tetraeder.moe.layer.duration` (histogram, s) by `tetraeder.part`: `cpu_ready`, `read_wait`, `cpu_landed`, `copy`.
- Bytes per token: existing `tetraeder.tier.read.bytes` over `tetraeder.tokens`.
- `StepProfile` gains `cpu_ready`, `read_wait`, `cpu_landed`, `copy`, inside `streamer` so `total` still adds up.
- Panel "Expert reads under compute" in the "Experts and cache" row: chunks in flight p50 by class, hidden share p50, disk bytes per token.
- Alert `TetraederExpertReadsNotHidden`: overlap on, hidden share p50 below 0.3 for 15 minutes while urgent or prefetch reads occur. Cause: few chunks in flight means the workers are starved by the CPU quota (lower `prefetch_threads`); many in flight means the disk is saturated (grow the RAM tier, ADR 0050 c).

### Expected gain

- Decode bound by compute and graph overhead instead of reads and copies, faster than `main-prefetch` and than KTransformers on box C; the estimates are in `20261103`. They overlap ADR 0038's for its levers 1 to 3, which assumed the reader reaches device rate, which the serial copy prevents. The paired run reports the service alone (`main-iosvc`) and with arrival order (`main-overlap`) apart, so each part has its own number.
- With ADR 0053 the gains do not add: the step becomes bound by the device's read time. Then bytes per token decide, and the tier sizing and ADR 0050 (c) are the next levers.

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

## Implementation notes

Built as in Brief A, both behind default-off options (`IoOptions::service`, `LoadOptions::overlap`).

- `memory::io::IoService` (`crates/memory/src/io.rs`), `HostDst::sub`, `IoOptions::{service, workers}`. In the model: `Prefetcher` enum in the streamer (reader or service), `Streamer::{begin_step, stage_overlap, wait_landed, finish_overlap}`, `LoadOptions::overlap`, `State.host` (`xn_host`, `cpu_out_host`), `Slot.cpu_out`, `sched_cpu`, `ExpertSrc::Copied`. `expert_rows` is now the one definition of the expert FFN ops for segment groups and CPU groups.
- Metrics, panel "Expert reads under compute" (id 54), alert `TetraederExpertReadsNotHidden`. Harness env `TETRAEDER_EXIT_IO_SERVICE`, `_IO_WORKERS`, `_OVERLAP`; JSON `decode_overlap`. Box stages `main-iosvc`, `main-overlap`, `main-overlap-d2`, `main-overlap-ram`, `main-overlap-repeat`, replay counts 79, 81, 83 (written, not run).

Deviations and local choices:

- `tetraeder.io.inflight` is sampled by the streamer after each submit (`io.inflight()`), not inside the service, which has no metrics handle.
- The wait for leftover units in `start_prefetch` is `begin_step` (settle at the start of every `run`) plus a settle at the end of an overlapped step. Leftovers are long landed, so it is free, and `ExpertStats` is complete when a step returns.
- `batches` counts one per submitting call (prefetch call, or miss batch), not per landing. A unit's `took` runs from submit to its last chunk, so `read_time` and `prefetch_read_time` sum latencies and exceed device time when units overlap.
- Pinned mispredicted reads can leave too few RAM slots for a layer step on a very small tier (RAM slots below experts routed plus experts stranded in flight). `stage_overlap` then waits for the stranded reads first. The fixture (12 RAM slots, top-8) takes that path; production tiers (77 or more) do not.
- With `overlap` and no op offload, any step with one CPU chunk takes the overlap path, prefill included (the brief gates only on `!offloads`).
- In the gated case `plan_cpu` runs before `finish_prefetch` (it does not depend on it); `finish_prefetch` runs only when the step is not overlapped and has CPU experts. A step with no CPU experts leaves its layer's reads for the next settle.
- `LoadOptions::io_delay` (a per-chunk sleep) is gone; ADR 0056 replaces it with `LoadOptions::io_factory` and the virtual-clock `SimIo`.
- The branch is based on `origin/track-u-parallel-reads-adr` (this ADR and the brief were not on `main` yet).

Tests: `crates/memory/src/io.rs` (7 unit tests), `crates/model/tests/overlap.rs` (bitwise service and overlap at depth 1, 2 and off, cache counters equal, pending groups, read error and recovery, GPU case skipped without a device), proptest `overlap_groups_partition_all_pairs`. `replay_fidelity.rs` was not extended (open item).

Local synthetic exit regime (release, 2 threads, 64 prefill, 32 decode, 2 VRAM slots, RAM tier a quarter of the experts, prefetch depth 1; greedy tokens identical in all five runs), decode per token:

| Run | tok/s | total ms | streamer ms | read_wait ms | hidden share |
|---|---|---|---|---|---|
| reader (serial) | 6.50 | 153.8 | 13.0 | | |
| service | 6.73 | 148.6 | 14.7 | | |
| service + overlap | 6.21 | 161.0 | 64.7 (cpu_ready 32.8, cpu_landed 26.2) | 1.5 | 0.96 |
| service, 3 ms per chunk | 6.41 | 156.0 | 12.8 | | |
| service + overlap, 3 ms per chunk | 6.34 | 157.7 | 66.5 (cpu_ready 31.2, cpu_landed 25.7) | 6.0 | 0.87 |

On this machine reads are already hidden by the prefetch (the page cache is warm and 16 workers absorb the delay), so overlap has nothing to win and costs its extra graphs; the numbers say nothing about box C. The box pairs decide. (No session directory yet.)

Defaults in serve: `tetraeder serve` takes `--io-service on|off` and `--overlap on|off` (also with `--plan`). Unset, the service is on when the builder has a RAM tier (`--ram-slots`, or `ram_slots` in the plan) and overlap is on when the service is on and the miss policy is `cpu` (a plan with a RAM tier always misses on the CPU). Without a RAM tier both are off. `--overlap on` without the service, or without a RAM tier, is a CLI error. The defaults follow the paired box result on the 235B (`20261110c`, addendum 2026-11-10). The old `Reader` path stays for `--io-service off`. The metrics above are exported through the load options' telemetry, so the panel and alert cover a served run. `--prefetch on|off` follows the same rule (on with a RAM tier, as in every measured 235B run) and `--prefetch-depth` defaults to 2, the depth of those runs.

## Addendum (2026-11-10, box 60057461): paired result, defaults on

Bars of the paired run against `main-prefetch` on one box: decode at least 15% above it, hidden share at least 0.5. The decode bar was met and the hidden share bar missed (`20261110c`). The gain was measured with spare cores of quota, so a box with a quota near the thread count had to confirm it.

- Decision: the service and overlap become the defaults of `serve` where they apply (RAM tier; overlap also needs the `cpu` miss policy). The `Reader` path stays until the starved-case run, then is deleted.

## Addendum (2026-11-10, box 60072027): starved case

With both engines at the quota's whole cores and decode CPU matched, the decode gain holds and the hidden share stays under the 0.5 bar (`20261110e`), so the spare-core confound of the previous addendum is closed. The `Reader` path now only serves `--io-service off`; deleting it is a cleanup, not a measurement question.

## Addendum (2026-11-11): where the read wait goes

From the committed overlap runs (`20261110c`, `20261110e`, `20261110f`): the wait is not all urgent misses (`urgent_wait_time` counts every wait in `wait_landed`, late prefetches included, and many prefetches are still in flight at their layer's plan), and the disk is not the bound. Candidate causes: a layer step submits its misses after the next layers' prefetch, and with 16 workers every chunk of a layer step is in service at once, so deadline order never applies and the device shares its rate among them; and the workers' CPU under the quota while the ggml threads poll.

- `LoadOptions::urgent_first` (default off, harness `TETRAEDER_EXIT_URGENT_FIRST`) submits the misses first; the cache and the logits are unchanged. `decode_overlap` splits the pending experts (urgent, depth-1, depth-2, prefetch lead) and the wait (steps with and without an urgent miss). Stages `main-overlap-uf`, `-w4`, `-uf-w4`, `-w8` and `-poll0` pair with `main-overlap`; the bar stands until they have run.

## Addendum (2026-11-11, box 60151711): first lever results

`urgent_first` raised the hidden share but slowed decode, and 4 I/O workers slowed it more (`20261111a`). Hidden share rose while throughput fell, so it is not the measure to optimize: `urgent_first` stays off, 16 workers stay. The bar question goes to exposed wait per token and decode tok/s, judged on paired runs.
