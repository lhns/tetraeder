# 0058. Decode I/O attribution: report capacity and demand per window before any bottleneck label

Status: Accepted (2026-11-15: harness numbers and report-only io_class are in committed results; item 3 stays gated on a slow-disk run). Earlier: Proposed (2026-11-11, owner request after the slow-disk loss in ADR 0045)

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

## Context

Two Phase 2 results needed forensics before the data settled them: a loss on a slow disk (`20261111b`, KTransformers faster) took three successive explanations, and a KTransformers outlier (`20261111c`) was first blamed on thread count when its host block showed page-fault latency. What was missing was not a label but two measurements:

- **Capacity:** the disk's loaded rate at the observed queue depth (ADR 0056 disk curve). Only some boxes recorded it, so a slow disk and an engine asking for too little looked the same: `main-overlap-w4` waited a large share of its step on a fast disk because 4 workers issued too few reads (`20261111a`); `20261111b`, the slow-disk run, has no curve at all.
- **Demand over time:** disk bytes per token per decode window, for every engine. The builder read several times KTransformers' bytes per token in the first tokens after a 32K prefill and the same bytes afterwards (`20261110c`); one window hid that.

The existing signals mislead on their own: the builder's read-wait share misses prefetch-only runs (it reads 0 there) and undercounts (with 4 workers the read wait grew by less than half the step's growth); `decode_io.read_gb_s` is prefetch bytes over waited time, far above the disk's rate; ADR 0054's hidden share rose under `urgent_first` while decode got slower (`20261111a`). `tetraeder.bottleneck.share` (ADR 0013) is emitted by `tetraeder doctor` only, not by `serve`.

## Decision

1. **Harness numbers, every engine, every B1 session:** `disk-curve` in every B1 stage list; per phase and per decode window (4 windows): disk `bytes_per_token`, `read_mb_s`, `disk_queue_depth_mean`, `disk_util` (achieved over the loaded curve at that depth), cgroup `io_some_share` where readable, the builder's `read_wait_share` (flagged unreliable on prefetch-only runs), and `tok_s_ceiling_disk` = capacity at the observed depth over bytes per token. `decode_io.read_gb_s` is renamed to what it measures.
2. **A label, report-only:** `io_class` (`disk_bandwidth` when `disk_util >= 0.7`; else `not_io_bound` below 0.25 stall share; else `io_latency` below queue depth 2; else `io_under_issue`; `unknown` with the missing inputs). Provisional thresholds, no gate, no alert.
3. **Later, gated on evidence:** a `serve` read-wait/disk-utilisation metric with a panel and an alert replacing the ADR 0054 hidden-share alert, and a detector input adapter for exit runs. Both wait for at least one paired slow-disk run with a disk curve and the new fields, so the `disk_bandwidth` branch has a real case; before that, check whether ADR 0056's tier term (disk reads charged from the curve) already predicts the measured decode rate, which would make it the `serve` signal instead of a second classifier.

## Consequences

- Every future exit result answers "how many bytes per token, against what capacity, over time" for every engine, without forensics.
- The label can be wrong and says what it lacks; nothing acts on it until a slow-disk run with a curve validates it.
- Rule 6 applies when item 3 is built, not now: item 1 and 2 are bench harness outputs, not runtime components.

## Addendum 2026-11-12: CPU attribution

The builder's CPU compute per decode token varied widely over five runs on the same CPU model, binary and bytes, and nothing recorded explained it; one run used more threads than its whole-core quota (`20261111g`, `20261111h`, `20261112a`). `scripts/engine_sample.py` now emits, per phase and decode window next to the I/O fields: `cpu_some_share` and `cpu_full_share` (cpu.pressure, source in `cpu_psi_src`), `throttled_share` (nr_throttled over nr_periods) and `throttled_ms_per_s` (cpu.stat v2 or v1, source in `throttle_src`), `cpu_steal_share` (host-wide /proc/stat) and `cpu_mhz_mean` (/proc/cpuinfo at window start and end). Missing files give null. `scripts/clamp_threads.sh` (used by `session.sh` and `exit_run.sh`) lowers a thread count above the known whole-core quota to the quota and logs it; `THREADS_ALLOW_OVER_QUOTA=1` keeps it. The fields label; they claim no cause. Further probes (same file): per window `ctxsw_vol_per_s`, `ctxsw_invol_per_s` (context switches of the process tree, summed over threads) and `anon_huge_gb` (AnonHugePages at window end), `env.thp_mode` (THP setting), and `engine_sample.py membw` (copy bandwidth of N processes, 10 s before each builder and baseline stage in `exit_run.sh`, stored as `env.membw` of builder results and as `membw-<stage>-<stamp>.json` beside every stage; `MEMBW_PROBE=0` skips it).
