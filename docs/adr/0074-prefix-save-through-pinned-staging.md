# 0074. Prefix-cache save on a GPU: pinned staging and prepared snapshot buffers

Status: Accepted (owner, 2026-11-19; proposed 2026-11-17). The save bar is met in `bench/results/20261117h/`; the TTFT bar (median and p90 at most 1.15x llama-server, saves at most 10% of TTFT) is met in both rounds of `bench/results/ttft7-20261119/` (2026-11-19).

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

## Context

- An admission that reuses an idle sequence first saves it to the host RAM prefix cache (ADR 0049, 0065), on the worker thread, with every live sequence stalled. On a 3090 that save was the largest remaining loss at concurrency 4 (`20261116pc`).
- The chunked layout (ADR 0067) gives a sequence one contiguous run, so both builders read one range per K or V tensor per layer. The cost was how each range was read: a zero-filled temporary `Vec`, a device copy into pageable memory (bounced by the driver), and a second copy into the snapshot.
- Overlapping the save with the next step buys nothing: the admission prefills into the same rows in that step, which waits for the copy in stream order. Real overlap needs a device-side copy of the rows, and ADR 0065 (2026-11-16 addendum) rules out spare device room for that.
- With the device copy made cheap, the save was bound by page faults on the fresh snapshot memory (`20261117c`). Advising that memory for huge pages on the admission path stalled the server in direct compaction (`20261117d`), and advising it even off the admission path stalled the server on a host with fragmented memory (`20261117h`).

## Decision

- **Staging.** A model whose KV is on a GPU (the scheduler-path dense builder and the unsharded MoE builder) holds a `Readback` (`crates/model/src/readback.rs`): one tensor of two halves of 8 MiB in the device's host buffer type (pinned on CUDA), allocated at load and reused by every save, reported in `memory()` as `host`, `ram`, kind `staging`. A save enqueues the device-to-host copies of its ranges into one half and appends the other half (landed at the previous sync) to the snapshot without a zero fill; DMA into one half overlaps the host copy out of the other. The device is synchronized before the first copy and after the last. Without a GPU, and on the sharded paths, ranges are appended straight into the snapshot. Snapshot bytes and order are unchanged on every path.
- **Prepared snapshot buffers.** Admission never first-touches snapshot memory and never waits on a huge-page allocation. `tetraeder_model::snapshot_pool` (one per process) keeps up to `--parallel` (at least 2; `TETRAEDER_SNAPSHOT_SPARES` overrides, 0 restores a plain allocation) buffers ready, prepared on its own thread: allocate, touch every page, clear. Huge-page advice only with `TETRAEDER_SNAPSHOT_THP=1`, default off: with the pages touched off the admission path it gains nothing, and on fragmented memory it costs seconds.
  - `seq_snapshot` takes the smallest ready spare that fits and is at most about twice the snapshot, else the largest one too small (growing it faults only the tail), else a plain allocation without advice, and wakes the preparer.
  - Target capacity: the largest size hinted or taken, rounded to 2 MiB. The scheduler hints when it places a request: the prompt plus at most 512 tokens it may still generate (clients send max_tokens of tens of thousands). No buffer, and not all spares together, exceed the prefix cache budget.
  - Every buffer the cache lets go of (eviction, replacement, an entry over budget) and every restored snapshot goes back to the pool while it has room and is as large as the target.
  - Every backend reports `kv_row_bytes` so the hint is right (the dense backend did not, so its first save fell back).
- **Restore** keeps its blocking writes.
- **Metrics and alerts.** `tetraeder.kv.prefix.save.duration` and `tetraeder.kv.prefix.restore.duration`; `tetraeder.kv.prefix.save.buffers{tetraeder.buffer=spare|fallback}`, gauges `tetraeder.kv.prefix.spares` and `tetraeder.kv.prefix.spare_memory`; dashboard panels 86 and 87; alerts `PrefixSaveSlow` (p95 save over 25 ms) and `PrefixSaveFallback` (over 50% fallbacks for 10 minutes). Skipped live saves: counter `tetraeder.kv.prefix.save.skipped` (panel 87). The admission debug log carries `save_ms`, `restore_ms` and the pool counters.
- **The bar,** judged on the server path: admission `save_ms` over the 12 nonzero admissions per model of `scripts/gpu_kv_save2.sh`'s probe, p95 at most 25 ms (Qwen3-8B) and 18 ms (Qwen3-30B-A3B), max at most 50 ms. `crates/model/tests/gpu_kv_save.rs` asserts only the absolute bounds (25 and 18 ms); staged against straight is printed, not asserted, because the straight copy's speed depends on the host (`s2d-20261118`).

## Consequences

- 16 MiB of pinned memory per GPU model, plus up to `--parallel` spare buffers of the largest snapshot size (bounded by the cache budget), which can double the cache's host memory in the worst case.
- A save still falls back to a plain allocation when every spare is in use.
- Rejected: huge-page advice on the admission path (branch `track-save-thp`, never merged) and on by default in the preparer.

## Decision: no save of what the live hit keeps (2026-11-18, before the change)

- Why: on the Claude Code replay nearly every admission reuses the idle sequence of its own conversation in place (source `live`), and the scheduler first snapshots the whole sequence to RAM, 2 to 5 GB at those lengths, to keep only the few hundred tokens the new request cuts off. That save is most of the TTFT gap to llama-server; without it a hit costs what llama-server's does (`ttft-20261118`).
- Rule: before reusing an idle sequence in place, save it only when recomputing the cut tail would cost more than the save: cut rows times the prefill cost per row above the sequence's rows times the save cost per row. Until both costs are measured per model, the rule is: save only when the cut is more than a quarter of the sequence (prefill about 0.36 ms per token against a save of about 0.06 ms per token on the 8B; the quarter is the ratio rounded up). A sequence evicted for another conversation is still saved as before. Also skipped, whatever its size: a cut that lies wholly in the tokens the sequence generated itself (the common prefix reaches the end of its prompt). A client that resends history without those tokens verbatim (Claude Code drops the thinking) will not match them later, so saving them keeps gigabytes nobody reuses (`ttft2-20261118`: the three slowest hits saved 0.9 to 2.4 s each to keep only their own output).
- The bar, judged on the same replay as `ttft-20261118` (one box, interleaved with llama-server, two rounds): tetraeder reuse on median and p90 TTFT each at most 1.15x llama-server's, and the sum of `save_ms` at most 10% of the sum of TTFT. The admission log gains the restore time.

## Decision: a new conversation takes a free slot, not an idle one (2026-11-19, before the change)

- Why: after the live no-save rules the remaining save share is six requests that share only the client's system prompt with the idle conversation; admission placed each in the idle conversation's slot (longest common prefix) and so saved that conversation first, 0.4 to 6.4 s each, although other slots were free (`ttft3-20261118`).
- Rule: when reusing an idle sequence in place would require a save (the cut is neither small nor only its own output) and a free slot exists, the request goes to the free slot; the common prefix is copied there from the idle sequence on the device when the backend supports a row copy (`seq_cp`), else prefilled; the idle sequence stays resident. With no free slot, admission evicts as before (oldest idle, saved).
- Implementation: the admission log says `source=copy` for such a placement and the counter `tetraeder.kv.prefix.placed_free` counts it (panel 87). A failed copy falls back to the in-place path with its save. `seq_restore` of the scheduler-path dense and unsharded MoE builders writes through the same 2x8 MiB pair (`Readback::write`: host copy into one half while the async host-to-device copies of the other run); without a GPU it writes directly.
- Restores from RAM use the pinned staging pair as saves do (the other open lever); judged by the same bar as before, on the same replay.

## Decision: no avoidable save or restore on the admission path (2026-11-19, after ttft4)

- Why: the p90 misses of `ttft4-20261119` are admissions that saved or restored a whole conversation when the pool had room for the prompt, and a log that reported those saves as queueing or under another request's skip reason (diagnosis in that README).
- Admission reserves the prompt plus a stretch of generation (ADR 0067). When that does not fit and an idle sequence would be evicted for it, or nothing is live, it reserves only the prompt plus one (and all of a RAM entry it restores); the steps grow the run as it generates and evict then if they must. With only live sequences in the way the request waits for the stretch as before, so it is not admitted just to be preempted.
- A free-slot copy never evicts its own source. When only that eviction would make room, the request reuses the source in place, saved by the rules above, and nothing is restored.
- A chunked run that can neither grow in place nor move to a free gap slides down over itself into the free rows right below it, on the device, in copies no longer than the slide and applied in order. A lone sequence no longer goes through RAM to grow (ADR 0067 history, 2026-11-18). Moving other sequences' runs to make room stays open.
- The admission log reports every placement attempt of a request: `queued_ms` excludes earlier attempts, `place_ms`, `save_ms` and `restore_ms` sum them, `attempts=` counts them, and `saved=seq:positions,...` names each sequence saved for it.
- Consequence: an eviction that generation really needs now stalls a decode step instead of delaying the first token.
- Judged by the bar of the 2026-11-18 section, on the same replay.

## History

- 2026-11-19, after `ttft6-20261119` (`track-ttft-spares`): spares grow with the target instead of being replaced, and frees run on the preparer thread in pieces.
  - Why: the target (the longest prompt plus 512 tokens so far) rises a little at most admissions, and the preparer replaced every spare below it, freeing and touching GBs each time. Frees that overlapped a prefill stalled it, and the churn left saves without a fitting, touched spare. The saves were the same sizes as in ttft5 but several times slower per byte. Diagnosis and numbers: `bench/results/ttft6-20261119/README.md`, last section. The stall mechanism, the serving thread's address-space calls waiting behind a long unmap, is consistent with the logs but not proven.
  - Rule: a spare below the target is grown by realloc (glibc moves the mapping, so nothing is copied) and only its new tail is touched. A spare is freed only when the budget no longer holds `spares` buffers at the target (the smallest goes), or when growing one would exceed the budget. A released buffer of any size is kept while there is room and grown later. One without room is freed by the preparer, not by the caller. Frees on Linux give memory back 64 MiB at a time (`shrink_to`), yielding in between, so a free still overlaps steps but no single unmap runs long. Snapshot reads reserve exactly, so a too-small spare grows to the snapshot's size, not to twice its own.
  - Step trace (`TETRAEDER_STEP_TRACE=1`): `snapshot take` (need, capacity, bytes already touched, source fit, small or fallback), `snapshot spare touched` (bytes touched, capacity before and after, `grow_ms` of the realloc) and `snapshot spare freed` (bytes, pieces). Pool counters `touched_bytes` and `freed_bytes`.
  - Judged on the next replay by the same bar. Spares on: no `snapshot spare freed` inside a prefill except budget frees, those in pieces; #1 to #10 level with r2; every save `source=fit` or small by at most the generated tail, at about the ttft5 rate per GB.
- 2026-11-20 (`track-save-metrics`): saves inside a step (`make_room`: evict, preempt) are timed and traced as `step save reason=evict|preempt seq=<slot> save_ms=<ms> prefilling=<slots>`, and `ttft_join.py` adds them to the save share of the requests still without a first token (the latest admit of each prefilling slot); the `admit` line is unchanged. The preparer's touched and freed bytes are exported as `tetraeder.kv.prefix.spare.work` and shown on the snapshot buffer pool panel. Observability only, no decision changed.

## Open

- Request #53 of the replay (a miss) prefills about 3x slower than llama-server (`ttft-20261118`); not yet explained.
- Long sequences evicted for another conversation still pay a multi-GB save; save at idle (off the admission path), pinned restores or chunk-shared cache entries remain the levers if the replay shows they matter.
- Saves ran much slower in the second round of `ttft4-20261119` than in the first, on the same box and code; not yet explained (host memory state, the spare preparer running beside a save, or a spare too small for the snapshot are candidates).
- Prefill of 1k to 2.5k new tokens at 13k to 33k positions is slower than llama-server's (`ttft4-20261119`); with the admission saves gone it is expected to hold p90 over the bar.
