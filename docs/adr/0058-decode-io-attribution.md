# 0058. Decode I/O attribution: report capacity and demand per window before any bottleneck label

Status: Proposed (2026-11-11, owner request after the slow-disk loss in ADR 0045)

## Context

One Phase 2 loss (`20261111b`, KTransformers 1.6x faster on a 0.5 GB/s disk) took three successive explanations before the data settled it, and a KTransformers outlier (`20261111c`, 0.52 tok/s) was first blamed on thread count when its host block showed page-fault latency (69 MB/s at queue depth 0.37, about 2,600 major faults per token). What was missing was not a label but two measurements:

- **Capacity:** the disk's loaded rate at the observed queue depth (ADR 0056 disk curve). Only some boxes recorded it, so a slow disk and an engine asking for too little looked the same. `main-overlap-w4` waited 24% of its step on a 7 GB/s disk because 4 workers issued too few reads; `20261111b`, the slow-disk run, has no curve at all.
- **Demand over time:** disk bytes per token per decode window, for every engine. The builder read 3.5x KTransformers' bytes per token in the first 64 tokens after a 32K prefill and the same bytes afterwards; one window hid that.

The existing signals mislead on their own: the builder's read-wait share misses prefetch-only runs (it reads 0 there) and undercounts (4 workers: step +124 ms, read wait +60); `decode_io.read_gb_s` is prefetch bytes over waited time (19 GB/s on a 2.4 GB/s disk); ADR 0054's hidden share rose to 0.52 under `urgent_first` while decode got 12% slower. `tetraeder.bottleneck.share` (ADR 0013) is emitted by `tetraeder doctor` only, not by `serve`.

## Decision

1. **Harness numbers, every engine, every B1 session:** `disk-curve` in every B1 stage list; per phase and per decode window (4 windows): disk `bytes_per_token`, `read_mb_s`, `disk_queue_depth_mean`, `disk_util` (achieved over the loaded curve at that depth), cgroup `io_some_share` where readable, the builder's `read_wait_share` (flagged unreliable on prefetch-only runs), and `tok_s_ceiling_disk` = capacity at the observed depth over bytes per token. `decode_io.read_gb_s` is renamed to what it measures.
2. **A label, report-only:** `io_class` (`disk_bandwidth` when `disk_util >= 0.7`; else `not_io_bound` below 0.25 stall share; else `io_latency` below queue depth 2; else `io_under_issue`; `unknown` with the missing inputs). Provisional thresholds, no gate, no alert.
3. **Later, gated on evidence:** a `serve` read-wait/disk-utilisation metric with a panel and an alert replacing the ADR 0054 hidden-share alert, and a detector input adapter for exit runs. Both wait for at least one paired slow-disk run with a disk curve and the new fields, so the `disk_bandwidth` branch has a real case; before that, check whether ADR 0056's tier term (disk reads charged from the curve) already predicts the measured decode rate, which would make it the `serve` signal instead of a second classifier.

## Consequences

- Every future exit result answers "how many bytes per token, against what capacity, over time" for every engine, without forensics.
- The label can be wrong and says what it lacks; nothing acts on it until a slow-disk run with a curve validates it.
- Rule 6 applies when item 3 is built, not now: item 1 and 2 are bench harness outputs, not runtime components.

## Addendum 2026-11-12: CPU attribution

The builder's CPU compute per decode token ranged from 292 to 551 ms over five runs on the same CPU model, binary and bytes (`bench/results/20261111g`, `20261111h`, `20261112a`), and nothing recorded explained it; one run used 24 threads on a 23.04-core quota. `scripts/engine_sample.py` now emits, per phase and decode window next to the I/O fields: `cpu_some_share` and `cpu_full_share` (cpu.pressure, source in `cpu_psi_src`), `throttled_share` (nr_throttled over nr_periods) and `throttled_ms_per_s` (cpu.stat v2 or v1, source in `throttle_src`), `cpu_steal_share` (host-wide /proc/stat) and `cpu_mhz_mean` (/proc/cpuinfo at window start and end). Missing files give null. `scripts/clamp_threads.sh` (used by `session.sh` and `exit_run.sh`) lowers a thread count above the known whole-core quota to the quota and logs it; `THREADS_ALLOW_OVER_QUOTA=1` keeps it. The fields label; they claim no cause.
