# Session 20261111a

- Status: clean but thin; Phase 2 met against KTransformers on AVX-512 at `serve` defaults, margin within the baseline's spread (ADR 0045). The ADR 0054 lever runs are a diagnosis (both slower than `main-overlap`).
- Code and box: results commit a9588ef; box 60151711, host 209278, A10 24 GB, Xeon Gold 6342 (AVX-512), 30 threads, 91 GB balloon, driver 570.86.16.
- Raw files: this directory.

## From ADR 0045 (addendum "AVX-512 at `serve` defaults", moved verbatim)

## Addendum (2026-11-11, box 60151711): AVX-512 at `serve` defaults

A10 24 GB, Xeon Gold 6342 (Ice Lake, AVX-512 VNNI), host 209278, ballooned to 91 GB, 30 threads for both engines; KTransformers ran its avx512_vbmi kernels. Decode `main-overlap`/KTransformers: adjacent 1.124 and 1.038, cross 1.043 and 1.119, means 4.283 vs 3.967 tok/s = 1.08. Overlap pair 0.4% apart, KTransformers pair 7.8%. Prefill 1.11 and 1.21. Greedy tokens equal across the builder runs; overlap and `cpu_experts` GPU parity tests pass. KTransformers read 160 to 180 MB per token through mmap at 84% CPU; the builder 585 MB per token at 2.4 GB/s at 99% CPU. Verdict: Phase 2 met at `serve` defaults on both CPU classes measured; on AVX-512 the margin is within the baseline's spread. Results `bench/results/20261111a/`.

## From ADR 0054 (addendum "first lever results", moved verbatim)

Same box, after the `main-overlap` pair (4.29 and 4.27 tok/s, hidden share 0.455 and 0.451, miss latency 4.27 ms). `main-overlap-uf` (urgent first): hidden share 0.522 but decode 3.76 tok/s (-12%), miss latency 4.67 ms. `main-overlap-w4` (4 I/O workers): hidden share 0.343, decode 2.78 tok/s (-35%), miss latency 6.29 ms, prefetch lead 12.4 ms. Greedy tokens equal in all four. Hidden share rose while throughput fell, so it is not the measure to optimize: `urgent_first` stays off, 16 workers stay. The bar question goes to exposed wait per token and decode tok/s, judged on paired runs. Results `bench/results/20261111a/`.

## From ADR 0056 (addendum "I/O attribution of the exit results", moved verbatim)

`io_class` (with `io_class_missing`) is a provisional, report-only label: no gate, no alert. `disk_util` >= 0.7 is `disk_bandwidth` whatever the stall share. Else with s the larger of `io_some_share` and `read_wait_share`: s < 0.25 is `not_io_bound`; else a queue depth below 2 is `io_latency` (serial reads or page faults); else `io_under_issue`. A missing input gives `unknown` and lists it. The thresholds are constants in `engine_sample.py` (`IO_BOUND_SHARE`, `DISK_UTIL_BOUND`, `SERIAL_QUEUE_DEPTH`). Known miss: `main-overlap-w4` of `20261111a` (read wait 0.24, slower because 4 workers under-issue) is `not_io_bound` without a disk curve; with one, its `disk_util` and `tok_s_ceiling_disk` show it.

## From ADR 0058 Context (moved verbatim)

- **Capacity:** the disk's loaded rate at the observed queue depth (ADR 0056 disk curve). Only some boxes recorded it, so a slow disk and an engine asking for too little looked the same. `main-overlap-w4` waited 24% of its step on a 7 GB/s disk because 4 workers issued too few reads; `20261111b`, the slow-disk run, has no curve at all.

The existing signals mislead on their own: the builder's read-wait share misses prefetch-only runs (it reads 0 there) and undercounts (4 workers: step +124 ms, read wait +60); `decode_io.read_gb_s` is prefetch bytes over waited time (19 GB/s on a 2.4 GB/s disk); ADR 0054's hidden share rose to 0.52 under `urgent_first` while decode got 12% slower. `tetraeder.bottleneck.share` (ADR 0013) is emitted by `tetraeder doctor` only, not by `serve`.
