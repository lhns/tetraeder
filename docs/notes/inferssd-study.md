# InferSSD study: expert streaming, caching, prefill, benchmark baselines (2026-11-19)

Read-only comparison of InferSSD (github.com/wici-ai/inferssd, Apache-2.0, a llama.cpp/ggml fork in C++) with Tetraeder on four topics. Source read: shallow clone of `main` at `050d3e1bfc2154def3ed7d0223133664b62ae4dc` (2026-11-19). No InferSSD code was run or copied; everything below is paraphrased. Their numbers are self-reported; none was measured here. Most of what follows comes from code and the long comments beside it; claims that rest on docs only are marked (docs).

Paths are relative to `engine/ggml/src/` in their repo unless they start with `docs/`, `framework/`, `bench/` or `engine/src/`.

Three limits frame everything:

- **Regime.** Their headline models are 505 to 508 GB GGUFs (DeepSeek-V4.1-Flash Q8_0, Muse-Spark-Open UD-Q4_K_XL) on machines with 8 to 128 GB of RAM (README "Test hardware"). Ours is a 142 GB model on 91 to 128 GB (ADR 0031). They are deeper in the disk-bound corner than we are.
- **Prompt.** Their tables use a 4K prompt; ours are judged at 32K under the CUDA row cap of ADR 0037.
- **Revision.** The README rates postdate the archived artifacts (`docs/benchmark.md`, "Data provenance"), and the Mac runs used a predecessor revision (`8010461f`). The mechanisms below are read from current code, not isolated by their data.

## 1. Expert streaming from SSD

### What InferSSD does

Files: `ggml-ssd/store/expert-layout.h`, `ggml-ssd/platform/file.cpp`, `ggml-ssd/io/{ring-common.h,linux-ring.cpp,win-ring.cpp,pool-ring.cpp}`, `ggml-cpu/ssd/cpu-sink.{h,cpp}`, `engine/src/llama-graph.cpp` (`build_moe_prefetch_hint`), `engine/src/ssd/cpu-runtime.h`, `ggml-cuda/ssd/transport.cpp`, `ggml-ssd/host/page-lock.h`; `docs/adr/0005-one-read-ring.md`, `docs/architecture.md`, `docs/configuration.md`.

- **I/O path.** Direct I/O everywhere: `O_DIRECT` on Linux, `FILE_FLAG_NO_BUFFERING` plus overlapped on Windows, `F_NOCACHE` on macOS (`platform/file.cpp`). One read ring per OS (their ADR 0005): io_uring through raw syscalls on Linux, IoRing on Windows, reader threads on macOS, plus a portable pread pool. Configured read failures never fall back to mmap. No GPUDirect Storage (grep for cuFile finds nothing).
- **Granularity.** One read per expert role (up, gate, down), the whole role in one request, widened to 4 KiB units (`ssd_expert_read_of`). Reads land straight in a 4 KiB-aligned RAM cache slot or ring slot; there is no bounce buffer and no copy. An optional pack format lays experts out 4 KiB-aligned (docs). `IORING_OP_READ_FIXED` with registered buffers is used for the ring's own slots only, as a policy option.
- **Queue depth.** The decode ring holds 8 experts (`ring_experts = 8` in `cpu-runtime.h`), a submission queue of 24, so up to 24 role reads in flight. Next-layer prefetch uses 3 reader threads beside it, one read each.
- **Order within a layer.** Up and gate are read first for every expert, down after ("role-major"); up/gate compute starts as soon as an expert's two roles land and down is scheduled without blocking (`cpu-sink.cpp` scheduler flags).
- **Prefetch.** The graph applies layer L+1's router to layer L's FFN input and hands the full top-k ranking to the sink as a hint (same idea as our ADR 0038). Their `prefetch.depth` (default 64) counts predicted ranks of the next layer, not layers; there is no two-layer lookahead. Prefetch reads run only in the drive's idle window, from the end of the layer's own reads to the start of the next layer's run. Each read is issued while it pays: the chance of use (measured per hint rank) must exceed the share of the read that would still be left when the window closes (`ssd::place::prefetch_read_value`). Reads not started when the next layer begins are dropped; a started read the next layer needs becomes its wait.
- **Decode critical path.** Router on the GPU, ids to the sink, which is one multi-threaded CPU op in the ggml graph. Experts the VRAM tier holds are launched on the GPU first; RAM hits compute on the CPU at once; misses compute as their roles land. A per-layer list schedule over measured lines (`ggml-ssd/place/layer-plan.h`) may move host experts to the GPU when the copy finishes before the CPU would.
- **Pinned staging.** No whole-tier pinning. The ring's buffers are page-locked for GPU copies, and single RAM cache slots are page-locked lazily by a rent-or-buy rule: a slot is locked once the copies it would have sped up add up to one registration's cost, one slot at a time, only while a layer waits for the drive (`host/page-lock.h`; their comment gives about 1.1 to 1.3 ms per 19 MB registration on an RTX 5090).
- **Threads.** Compute participants poll a phase word for up to 3 ms, then park on wait-on-address. The decode pool is sized to the fewest threads that reach the best measured streaming rate, less the CPUs other work used over recent tokens (`docs/architecture.md`, "Threads"; `ssd_cpu_sink_decode_threads`).

### What Tetraeder does

ADR 0031 (`O_DIRECT` with page-aligned bounce buffers, io_uring where allowed, thread pool otherwise; boxes have no io_uring under Docker seccomp, ADR 0054), ADR 0054 (16 persistent workers, 1 MiB chunks, earliest-deadline order, each worker copies its chunk from its bounce into the RAM slot, experts computed in arrival order per expert), ADR 0038 (pre-gating, depth 2 layers, default on), ADR 0058 (bytes per token against the disk curve), ADR 0061 (poll level from the cgroup quota), ADR 0064 (segment per MoE layer, host round trip per layer).

### Difference

1. Their reads are whole roles into the final slot with no copy; ours are 1 MiB chunks through a bounce with one copy per chunk. On fast disks our decode is `not_io_bound` (ADR 0050, 0058), so this matters on slow disks and in prefill (topic 3), not in today's decode verdicts.
2. Their prefetch is one layer ahead and value-gated; ours is two layers ahead and issues the top predictions up to the slot budget at once.
3. Their decode thread count follows measured bandwidth saturation and other CPU load; ours follows the quota (ADR 0061).
4. Pinning: they page-lock individual slots on demand; ADR 0059's `Async` mode designs a pinned staging ring plus a host copy.

## 2. Caching

### What InferSSD does

Files: `ggml-ssd/host/expert-cache.h`, `ggml-cuda/ssd/vram-slot-table.h`, `ggml-cuda/ssd/vram-tier.cpp`, `ggml-cpu/ssd/cpu-sink-vram.inc`, `ggml-ssd/host/{ledger.h,room.h,arbiter.cpp}`; `docs/configuration.md`.

- **Tiers.** A VRAM tier (CUDA and ROCm only) and a RAM cache. On unified memory with a GPU tier the RAM cache keeps only the slots prefetch lands in; the tier holds the rest in the same DRAM.
- **RAM cache.** Per-layer slots, the same quota for every layer ("uniform"). Production policy `count_decay2`: per layer and expert a selection count (once per token, routing weight ignored), decayed by half every 2 tokens. Victim: empty slot, then a RAM copy of an expert the VRAM tier also holds, then the lowest decayed count, ties by recency. Admission filter: a miss whose own score is below the victim's is not cached (bypass). Kept across requests; slots are given back on memory pressure and regrown at token boundaries.
- **VRAM tier.** One global pool of equal slots (sized for the largest expert) across all layers. A decode miss replaces a victim only when `its count x saving` beats `victim count x saving`, where saving is the measured time a tier hit saves against a RAM or drive use (`vram_slot_table.h`, `update_admission`); a free slot takes any expert. Uploads are deferred to their own thread and stream and published at the layer's next use, off the critical path. On a shrink the most valuable entries are compacted into the slots of the least valuable ones first.
- **Prefill and the switch.** A prompt pass seeds empty VRAM slots with the experts it routes most per layer, published when the pass ends, at score zero, so the first decode misses evict unused prompt heat first. At load, before any request, both caches are filled with each layer's first experts as placeholders that count as empty.
- **Capacity.** No byte setting by default. A host ledger with two terms: the envelope (cgroup v2 `memory.max`, Windows job limit or `host.budget_mib`, less 512 MiB) and the system term (MemAvailable less a floor). The caches size themselves from it at the first decode token and follow it at every token boundary; every other allocation takes room back from the caches first. The VRAM tier grows into the card once device memory has settled, keeping `vram.reserve_mib` free.
- **Observation.** An optional online miss-ratio curve per layer (LRU stack distances), so the hit rate of any per-layer size is known while serving.

### What Tetraeder does

ADR 0035 (static per-layer VRAM hot set from routing counts, CPU for every other expert), ADR 0037 (byte budgets, equal slots per layer), ADR 0050 (c) (per-layer RAM LRU; `freq:P` pins built, off, for slow disks; cross-layer pool replayed and not built), ADR 0054 section 3 (RAM tier from a fixed 4 GB reserve), ADR 0059 (decode admission into VRAM; lost to synchronous uploads; `Async` unmeasured).

### Difference

1. Their VRAM tier is global, adaptive and cost-priced, with uploads off the critical path; ours is a static per-layer hot set. Their RAM cache is per layer like ours; the replay already showed per-layer slots are not our bound (ADR 0050, 2026-11-14).
2. They manage the prefill-to-decode switch explicitly (seeds at score zero, RAM cache kept); ADR 0050 names the tier contents at the switch as our next lever and has no mechanism yet.
3. Their capacity follows the cgroup at token boundaries; ours is fixed at load.
4. Their RAM policy is close to a short-window LFU with a bypass filter; nothing like it is in our replay.

## 3. Prefill

### What InferSSD does

Files: `framework/src/inferssd/strategy/launch.py`, `framework/src/inferssd/launch/typed.py` (`one_pass`, `ONE_PASS_BATCH`), `ggml-cpu/ssd/cpu-prefill.{h,cpp}`, `ggml-cpu/ssd/cpu-sink.cpp` (`ssd_cpu_sink_stream_hybrid`, lookahead), `ggml-cpu/ssd/cpu-sink-vram.inc` (`select_prefill_transient`), `ggml-ssd/place/layer-plan.h`.

- **One pass per prompt.** When the launch strategy runs, it sets `-b` and `-ub` as large as the context, so a 4K prompt is one ubatch (`launch.py`: "prefills each prompt in one pass"). The archived Mac run used 4096/4096 (docs).
- **Each expert once per layer.** The grouped prefill groups the ubatch's routes by expert in first-touch order and reads every distinct expert once (`cpu-prefill.cpp`); each weight row stays hot across that expert's token columns.
- **Reads stream under compute.** The union goes through the same ring; experts compute as they land. Heavy experts (many tokens) go to the GPU: the cost table's makespan rule moves experts in order of route count and keeps the fewest moves within one read of the best predicted time, then issues reads in the order the lanes need them. A moved expert is copied from the page-locked ring or RAM slot to a device scratch slot and computed there.
- **Lookahead.** Layer L's graph also applies layer L+1's router to every token. While the GPU runs attention and routing of L+1, the drive reads the experts that prediction names into RAM cache slots, by the same value rule as decode prefetch, borrowing coldest slots of other layers when the layer's own are full.

Why 5 to 46x over llama.cpp (my reading, not isolated by their data): at ubatch 512 llama.cpp touches nearly every routed expert of every layer once per 512 tokens, eight times for a 4K prompt, through mmap faults; InferSSD reads each once per prompt at drive rate with compute overlapped. A one-pass read of a 508 GB model at 7 to 14 GB/s takes about 35 to 70 s, which puts 4K prompts at roughly 60 to 120 tok/s, the range of their 75 to 165 tok/s CUDA rows (estimate). Decode gains come from direct reads at depth instead of 4 KiB faults once readahead is off (ADR 0054 discusses that hazard), the RAM and VRAM caches and the one-layer prefetch.

### What Tetraeder does

Ubatch 2048 at 32K, 16 ubatches. Pair groups (ADR 0035); RAM-tier misses read synchronously on the main thread (`read_experts`, ADR 0054 section 1 keeps prefill there); op offload for steps of 32 tokens or more (ADR 0044); prefill prefetch capped at the slots the prediction does not need (ADR 0038). Measured on `20261113b` (`exit-main-overlap-023738.json`, 3090, 23-core quota): 32,768 tokens in 1,040 s (31.5 tok/s), 577 GB read (17.6 MB per token, about four passes over the model), at 0.55 GB/s against a loaded disk curve of 6.1 GB/s at depth 4 to 8 (`disk_util` 0.09). Per ubatch 64.3 s, of which the streamer is 43.6 s and compute 20.1 s; GPU utilization under 1% on average. Prefill is bound by serial synchronous reads, neither disk nor compute.

### Difference

1. Their prefill reads at drive rate with compute on arrival; ours reads at a tenth of the drive with compute after.
2. They read each expert once per prompt; we read each RAM miss once per 2048-token ubatch.
3. They place heavy experts on the GPU per expert by a cost table; we offload whole CPU groups by token count.

## 4. Benchmark claims and which baseline they beat

- **Hardware** (README, `docs/benchmark.md`): MacBook Pro M5 Max 48 GB unified, 2 TB internal SSD; RTX 5060 Ti 16 GB with 8 GB RAM, PCIe 4 x4 SSD; RTX 5090 32 GB with 32 GB RAM (archived runs also had a 13.98 GB process limit), PCIe 5 x4; DGX Spark 128 GB unified, PCIe 5 x4. Models 505 to 508 GB, so 4x to 60x larger than RAM.
- **Mac rows (documented).** Baseline: the `vcruz305/llama.cpp` DeepSeek fork at `5210c7c5`, batch flags omitted (defaults 2048/512), Metal dense with CPU routed experts, same GGUF, prompt, template and threads (8 decode, 12 prefill). No expert cache, no op-offload or cache flags named; mmap is not stated but is the default and the only way a 508 GB file runs in 48 GB. InferSSD: predecessor revision, 4096/4096, 8 GiB LFU cache. One run each, page cache not controlled.
- **Linux and DGX rows (undocumented).** Tester-supplied summaries; no llama.cpp command, build or flags is recorded anywhere. A grep of `bench`, `scripts`, `framework` and `docs` for `cmoe`, `cpu-moe`, `override-tensor`, `moe-cache`, `no-mmap` finds only InferSSD's own launch and Mac harness (`--cpu-moe --no-op-offload` for their engine), never a baseline command. The largest multiple (DGX Spark, 46.7x prefill) sits on the slowest llama.cpp prefill row (2.73 tok/s) despite the most memory, which a weak baseline configuration would explain.
- **Against our baselines.** The Mac baseline is closest to our earliest one, llama.cpp `-ot`/`-cmoe` on mmap at default ubatch (ADR 0035, 0037), which we beat 1.23x to 1.60x. Our verdict baseline is llama.cpp with `--moe-cache-mib` and a tuned `--moe-cache-cpu-share`, ubatch 2048, `--poll 0` (`20261113b`, flags in its exit JSON). Their 5x to 46x are therefore not comparable to our 1.2x to 1.7x; they say mostly how much ubatch 512 on mmap loses when the model is many times RAM.

## Ideas, ranked by expected value for Tetraeder

1. **Stream prefill misses through the I/O service with compute on arrival.** Our prefill waits on 0.55 GB/s synchronous reads with a 6.1 GB/s disk and an idle GPU (`20261113b`). Route prefill misses of a ubatch's pair groups through `IoService` (ADR 0054) and run the CPU groups per expert as they land, as decode overlap does; with op offload, upload and run landed experts in arrival order. At device rate the 577 GB take about 95 s instead of 657 s of read time. Exactness as ADR 0054: whole experts per group. Test: CPU test on the synthetic MoE with ADR 0056's `SimIo` (bitwise logits, read time hidden under the virtual clock), then a 235B 32K box pair against `main-overlap` reading prefill tok/s, `read_mb_s` and streamer time per ubatch. Amends ADR 0054 (service for prefill) and ADR 0038 (prefill prefetch cap).
2. **Layer-major prefill across ubatches.** Run MoE layer L for all ubatches of the prompt before layer L+1, so each expert is read at most once per prompt per layer, while every `mul_mat_id` still sees at most one ubatch of rows (ADR 0037 cap kept). Attention of chunk j at layer L needs only KV of chunks before j at layer L, which this order has already written, so each chunk computes the same graph and logits stay bitwise equal. Cost: hidden states of the whole prompt (32K x 4096 x 4 B, about 0.5 GB) and one segment loop inverted. Bytes per prompt fall from about four passes to under one. Test: first count bytes offline from the saved trace (`runs/traces/routing-235b-20261114-041805.trace`: distinct missing experts per layer per prompt against per ubatch, under the replayed RAM tier); then a CPU parity test against ubatch-major order; then the box pair after idea 1. Amends ADR 0030 and 0035 (step order), ADR 0031 (prefill budget). Ranked second because it pays fully only once reads stream (idea 1).
3. **Tier contents at the prefill-to-decode switch.** ADR 0050 found the first 16-token decode window reads several times the later ones under every online policy. Seed free VRAM slots (and the RAM tier) with the last prefill ubatches' most-routed experts at score zero, as InferSSD does, so they cost nothing if decode does not use them. Test: add a `seed` VRAM and RAM policy to `crates/memory/src/replay/sim.rs` and replay the saved trace with `--decode-window 16`; fix the bar first (for example 30% fewer bytes in the first window, no worse later). Box only if it passes, on D1 beside the planned `main-overlap-poll0-hot` arm (`docs/m2-plan.md`). Amends ADR 0050 (c).
4. **ADR 0059 `Async`, InferSSD's way.** Instead of a pinned staging ring plus a host copy, page-lock individual RAM slots by rent-or-buy while the layer waits on the drive, upload on a second stream from the locked slot, publish at the layer's next use, and admit by `count x measured saving` against the victim's in a global VRAM pool. Test: a `gpu_` micro-benchmark of `cudaHostRegister` per 11.4 MB slot and of DMA from it on a 3090; then the unrun `main-overlap-adapt-async` stage on D1 with this upload path. Amends ADR 0059; overlaps the Surogate note's idea 2 (zero-copy gather), which stays the alternative if registration is too slow.
5. **Value-gated decode prefetch.** Issue a predicted read only while its measured per-rank chance of use exceeds the share of it that would still be in flight when the next layer starts, and drop reads not started by then. This trims wasted prefetch bandwidth that competes with urgent misses, without reordering them (`urgent_first` was measured negative, ADR 0054). Test: timed replay (ADR 0056) of the saved trace, which carries predictions, comparing exposed wait and wasted bytes against today's depth-2 rule; fix the bar before running. Amends ADR 0038 and 0054. Mostly a slow-disk lever (ADR 0045 limitation).
6. **Count-decay RAM policy with bypass.** Add `cdecay:<half-life>` (counts per token, halved every h tokens, a miss scoring below the victim is not cached) to the replay beside `lru` and `freq:P`. Cheap; matters on slow disks only. Amends ADR 0050 (c).
7. **Decode threads from measured saturation.** Size the decode pool to the fewest threads that reach the best measured expert streaming rate, less CPUs other work used, instead of from the quota alone; ADR 0058's CPU attribution shows a 1.9x compute swing nothing explains, and this measures its input. Test: CPU sweep on one box, recorded per decode window. Amends ADR 0061. Low.

Not proposed: a RAM pool across layers (ADR 0050, measured short of its bar), `urgent_first` and fewer I/O workers (ADR 0054, measured negative), synchronous admission (ADR 0059), io_uring and `READ_FIXED` (unavailable under the boxes' seccomp and not the bound; whole-role reads without a bounce matter only after idea 1), an online miss-ratio curve (our replay already answers sizing offline).

## Top ideas across topics

1. Prefill misses through the I/O service with compute on arrival (idea 1): the measured prefill bound, ADR 0054.
2. Layer-major prefill, priced on the saved trace first (idea 2): ADR 0030, 0035, 0031.
3. Seeded tier contents at the switch, replay first (idea 3): ADR 0050 (c).
4. Slot-wise page-locking and cost-priced global admission for ADR 0059's unrun async arm (idea 4).
5. Value-gated prefetch for slow disks (idea 5): ADR 0038, 0054.
