# 0074. Prefix-cache save on a GPU reads through a reused pinned staging pair

Status: Proposed (2026-11-17).

## Context

- An admission that reuses an idle sequence first saves it to the host RAM prefix cache (ADR 0049, 0065), on the worker thread, with every live sequence stalled. On one RTX 3090 (`bench/results/20261116pc/pin-confirm/`, `save_ms` in the admission log) that takes 41 to 44 ms for Qwen3-8B and 30 to 33 ms for Qwen3-30B-A3B per sequence of about 533 tokens (78 and 52 MB of F16 KV, so about 1.8 GB/s); three saves in one pass serialize to 91 to 121 ms. It is the largest remaining loss at concurrency 4.
- The chunked layout (ADR 0067) gives a sequence one contiguous run, so both builders already read one range per K or V tensor per layer. The cost is how each range is read: `get_bytes` zero-fills a temporary `Vec`, `ggml_backend_tensor_get` copies device memory into that pageable buffer (the CUDA driver bounces it through its own small pinned buffer and waits), and `extend_from_slice` copies it again into the snapshot.
- Overlapping the save with the next step (async copies, one sync before the rows are overwritten) buys nothing here: the admission cuts the saved sequence and prefills into the same rows in that next step, which then waits for the copy in stream order. Real overlap needs a device-side copy of the rows first, and the 2026-11-16 addendum of ADR 0065 rules out spare device room for that.

## Decision

- A model whose KV is on a GPU (the scheduler-path dense builder and the unsharded MoE builder, what `serve --engine dense|moe` runs) holds a `Readback` (`crates/model/src/readback.rs`): one tensor of two halves of 8 MiB in the device's host buffer type (`alloc_host_staging`, pinned on CUDA), allocated at load and reused by every save. It is reported in `memory()` as `host`, `ram`, kind `staging` (16 MiB per model).
- A save enqueues the device-to-host copies of its ranges (`ggml_backend_tensor_get_async`, `Tensor::read_to_async`) into one half, then appends the other half, which landed at the previous sync, to the snapshot (`Tensor::get_bytes_into`, no zero fill), then syncs. DMA into one half overlaps the host copy out of the other. Ranges larger than a half are cut. The device is synchronized before the first copy and after the last, so nothing is in flight outside a save.
- Without a GPU (CPU KV), and on the sharded paths, ranges are appended straight into the snapshot with `get_bytes_into`: one copy instead of two. The snapshot bytes and their order do not change on any path, so saved entries and `seq_restore` are unchanged.
- Restore keeps its blocking writes; saves are on every admission of an idle id, restores only on a RAM hit.
- `tetraeder.kv.prefix.save.duration` times each save, next to `tetraeder.kv.prefix.restore.duration`; the dashboard shows both and `PrefixSaveSlow` fires when the p95 save stalls the worker for more than 25 ms.

## Consequences

- Expected on the 3090: the DMA runs at pinned speed and hides under the host copy, so a save costs about one host copy of its bytes into fresh snapshot memory plus one half of DMA; about 10 to 20 ms for 8B and 7 to 14 ms for 30B instead of 41 and 31 (to be measured: `crates/model/tests/gpu_kv_save.rs` times both paths paired in one process).
- 16 MiB of pinned memory per GPU model, fixed whatever the context length.
- The fresh snapshot memory (first-touch page faults) stays on the stall path. If the measurement shows it dominating, next steps are recycling snapshot buffers in the prefix cache or `MADV_HUGEPAGE` on them; moving the save off the worker thread needs the device-side copy ruled out above.

## Addendum: the host side dominates (2026-11-17, `bench/results/20261117c/kv-save/`, box 61042290, RTX 3090)
- Staged save 47.5 ms (8B) and 28.6 ms (30B) against 55.4 and 35.6 ms straight: `gpu_kv_save` failed its 25 and 18 ms bounds; admission `save_ms` stayed at 43 to 49 ms (8B) and 29 to 35 ms (30B).
- Copying into fresh memory alone took 48.1 and 28.1 ms, and 15.7 and 9.0 ms with `MADV_HUGEPAGE`: the save is bound by page faults on the new snapshot, not by the device copy. Advising the snapshot before its first write (`track-save-thp`, 99988e3) was tried next.

## Addendum: the test passes, the server path stalls (2026-11-17, `bench/results/20261117d/kv-save/`, box 61046208, THP `madvise`, defrag `madvise`)
- `gpu_kv_save` passes with the advice: staged 15.6 ms (8B) and 10.1 ms (30B). The 30B server probe saves take 10 to 13 ms.
- The 8B server probe saves, in admission order: 17, 17, 104, 432, 2041, 1979, 771, 234, then 157, 131, 127, 140 ms. Without advice (`20261117c`) they were 43 to 49 ms. The likely cause is direct compaction (defrag `madvise` stalls the faulting thread on advised memory), not yet proven.
- Decision: admission must not first-touch snapshot memory nor wait on a huge-page allocation. Spare snapshot buffers are prepared off the admission path; when none is ready, a save falls back to a plain allocation without advice (the old 44 ms). The advice is not merged on the admission path (`track-save-thp` stays a branch).
- The bar, judged on the server path and not on `gpu_kv_save` alone: admission `save_ms` over the 12 nonzero admissions per model of the `gpu_kv_save.sh` probe, p95 at most 25 ms (8B) and 18 ms (30B), max at most 50 ms.
