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
- **Metrics and alerts.** `tetraeder.kv.prefix.save.duration` and `tetraeder.kv.prefix.restore.duration`; `tetraeder.kv.prefix.save.buffers{tetraeder.buffer=spare|fallback}`, gauges `tetraeder.kv.prefix.spares` and `tetraeder.kv.prefix.spare_memory`; dashboard panels 86 and 87; alerts `PrefixSaveSlow` (p95 save over 25 ms) and `PrefixSaveFallback` (over 50% fallbacks for 10 minutes). The admission debug log carries `save_ms` and the pool counters.
- **The bar,** judged on the server path: admission `save_ms` over the 12 nonzero admissions per model of `scripts/gpu_kv_save2.sh`'s probe, p95 at most 25 ms (Qwen3-8B) and 18 ms (Qwen3-30B-A3B), max at most 50 ms. `crates/model/tests/gpu_kv_save.rs` asserts only the absolute bounds (25 and 18 ms); staged against straight is printed, not asserted, because the straight copy's speed depends on the host (`s2d-20261118`).

## Consequences

- 16 MiB of pinned memory per GPU model, plus up to `--parallel` spare buffers of the largest snapshot size (bounded by the cache budget), which can double the cache's host memory in the worst case.
- A save still falls back to a plain allocation when every spare is in use.
- Rejected: huge-page advice on the admission path (branch `track-save-thp`, never merged) and on by default in the preparer.

## Open

- The bar was set and met at about 533-token sequences (about 78 MB on the 8B). Claude Code conversations are 15K to 34K tokens, 2 to 5 GB of KV each, and in the TTFT replay prefix hits with a few hundred new tokens still take seconds (`20261117l`). The breakdown session decides the next lever: save when a sequence goes idle, pinned restores, or chunk-shared cache entries; its bar is written here before the change.
