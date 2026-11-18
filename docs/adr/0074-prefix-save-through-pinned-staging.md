# 0074. Prefix-cache save on a GPU: pinned staging and prepared snapshot buffers

Status: Proposed (2026-11-17). The save bar below is met in `bench/results/20261117h/`; the TTFT replay (`20261117l`) shows saves and restores of long conversations still dominate, see "Open".

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
- Restores from RAM use the pinned staging pair as saves do (the other open lever); judged by the same bar as before, on the same replay.

## Open

- Request #53 of the replay (a miss) prefills about 3x slower than llama-server (`ttft-20261118`); not yet explained.
- Long sequences evicted for another conversation still pay a multi-GB save; save at idle (off the admission path), pinned restores or chunk-shared cache entries remain the levers if the replay shows they matter.
