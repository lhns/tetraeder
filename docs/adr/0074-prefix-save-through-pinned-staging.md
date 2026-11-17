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

## Addendum: the host side dominates; huge pages on the snapshot (2026-11-17, `bench/results/20261117c/kv-save/`, box 61042290, RTX 3090)
- Measured: staged save 47.5 ms (8B) and 28.6 ms (30B) against 55.4 and 35.6 ms straight, so `gpu_kv_save` failed its 25 and 18 ms bounds; the server's admission `save_ms` stayed at 43 to 49 ms (8B) and 29 to 35 ms (30B), no gain over 41 to 44 and 30 to 33.
- Why: copying into fresh memory alone took 48.1 ms (8B) and 28.1 ms (30B), and 15.7 and 9.0 ms with `MADV_HUGEPAGE`. The save is bounded by page faults on the new snapshot, not by the device copy.
- Decision: `Readback::read` advises huge pages on the snapshot's reserved capacity before the first write. Linux only, a hint; where THP is off it changes nothing. The bounds stay; the next box reruns `gpu_kv_save`.
