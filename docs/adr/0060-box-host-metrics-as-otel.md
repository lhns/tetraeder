# 0060. Box and host metrics as OpenTelemetry metrics: one collector per box run, the result's host block derived from its file

Status: Proposed (2026-11-12, owner decision). Extends ADR 0012 (one OTel data path) and ADR 0058 (host fields per phase and decode window); the sampler path of ADR 0058 stays until a box run shows both paths agree.

## Context

`scripts/engine_sample.py` samples host, cgroup, process-tree and GPU counters once a second into `samples-*.jsonl` and summarizes them into each result's `host` block (ADR 0056 addendum, ADR 0058 and its addenda). That is a second data path beside ADR 0012's: the fields never reach a dashboard, and a new signal means a new reader, a new JSONL column and a new summary. Box runs start no collector, and `exit_regime.rs` builds no `Telemetry`, so the builder's own metrics (step split, expert reads, prefetch) never reach a bench result either.

Constraints:

- **Baselines export nothing.** llama.cpp and KTransformers have no OTLP metrics, so every signal compared across engines must come from outside the engine: collector receivers or the sampler.
- **Decode windows are 5 to 7 s.** Host data needs a 1 s grid; the engine SDK exports every 10 s by default (`TelemetryConfig`).
- **The builder runs at 94 to 98% of its CPU quota.** Anything added runs for every engine's stage alike, and its cost is measured, not assumed.
- **Committed JSON stays** (Rule 3). The `host` block keeps its field names and math.

## Decision

1. **One collector per box run.** `exit_run.sh` starts `otelcol-contrib` 0.116.1 (`scripts/otelcol_box.sh`, config `deploy/otelcol/config-file.yaml`) before the first stage and stops it before `DONE`, so the files are flushed before fetch. It writes OTLP JSON lines to `runs/vast/otel/<start stamp>/metrics.jsonl` (a new directory per start: the file exporter truncates on open), which `taxameter fetch` packs (`[results] extra = runs/vast`, `*.jsonl`). A failed start is logged and the run goes on with the sampler path alone. `GOMAXPROCS=1` and `GOMEMLIMIT=200MiB` bound the collector. Empty provenance variables become `unknown` (an empty value stops the resource processor from building).
2. **Sources, one per field.** Each `host` field has exactly one source in the OTLP path:
   - **hostmetrics receiver, 1 s:** disk (`system.disk.io`, `system.disk.operations`, `system.disk.pending_operations`, `system.disk.weighted_io_time`, by device name; loop and ram devices excluded) and CPU (`system.cpu.time`). A `metricstransform` step keeps `system.cpu.time` per state summed over CPUs (8 series: steal share) and adds `tetraeder.host.cpu.busy` per CPU (all states but idle and wait: the busiest-CPUs view). Per CPU and per state at 1 s would be about 200 KB/s on a host showing 128 cores; this is about 30 KB/s. Also memory (`system.memory.usage`), network (`system.network.io`, eth and en devices), and the process scraper for the builder and llama.cpp binaries (names `exit_regime-.*`, `tetraeder.*`, `llama-.*`), with `process.paging.faults`, `process.context_switches`, `process.threads` and `process.disk.operations` enabled. These process series feed the dashboard only.
   - **The sampler, OTLP/HTTP JSON over `urllib` to the same collector, 1 s:** everything hostmetrics does not cover or measures differently. The cgroup (CPU time, quota, throttling, `io.stat` reads, `memory.current`, `memory.max`, file and anon memory), pressure (PSI cpu and io), mean CPU MHz, the engine's process tree (CPU time, major faults, context switches summed over threads, RSS, AnonHugePages), the GPU through `nvidia-smi` (utilization, memory, power, temperature, SM and memory clocks, throttle reasons, PCIe link generation and width), and once per probe the membw result and the disk curve points. The process scraper filters by executable name only: KTransformers runs as `python3`, like the sampler and the RAM balloon. And the sampler sums context switches over every thread's `/proc/PID/task/TID/status` because the process file counts the leader only (ADR 0058 addendum); whether hostmetrics' `process.context_switches` does the same was not checked. So every process field of the `host` block comes from the sampler's pid tree.
   - **Not from nvidia-smi `--query-gpu`:** PCIe throughput (only `nvidia-smi dmon`, another process). Left out.
   - Stdlib `urllib` instead of the Python SDK: the box runs the system `python3`, often with `-I`, and installs no packages; the payload is a fixed set of points. The post has a 0.5 s timeout, never raises, and stops after 10 straight failures. `TETRAEDER_HOST_OTLP` (the collector's base URL) turns it on; the harness sets it when the collector is up, and child scripts (KTransformers, llama.cpp cache) inherit it. `SAMPLER_OTLP=0` turns it off. The JSONL file is written as before.
3. **Engine metrics.** `TelemetryConfig::from_env` reads the standard `OTEL_METRIC_EXPORT_INTERVAL` (ms). `exit_regime.rs` builds a `Telemetry` when `OTEL_EXPORTER_OTLP_ENDPOINT` is set and passes it in `LoadOptions::telemetry`; the harness sets the endpoint and `OTEL_METRIC_EXPORT_INTERVAL=1000` for the builder only (`ENGINE_OTLP=0` turns it off). Export every second rather than a flush at window edges: a flush between windows would land inside `decode_s`. New gauges: `tetraeder.cpu.threads` (configured compute threads) and `tetraeder.cpu.quota` (the cgroup's CPU limit in cores). The VRAM admission and segment-split metrics of other tracks reach the same file through the catalog with no harness change; deriving engine fields into the result is a later step.
4. **Names.** Semantic-convention names where one exists and means the same: `container.cpu.time`, `container.disk.io`, `container.memory.usage`, `hw.gpu.utilization`, `hw.gpu.memory.usage`, `hw.gpu.memory.limit`, `hw.power`, `hw.temperature`, and hostmetrics' own `system.*` and `process.*`. Everything else is `tetraeder.host.*` (list below). A sampler series whose meaning differs from a hostmetrics series of the same semconv name (process tree versus process, threads versus leader) keeps the prefix. Resource attributes: `service.name` (`tetraeder-host-sampler`, `tetraeder`, or the collector's), `host.id` (Vast container id), `host.name`, `host.cpu.model.name`, `tetraeder.session`, `tetraeder.git.commit`, `tetraeder.cpu.quota`, `tetraeder.gpu.name`, `tetraeder.gpu.driver`, `tetraeder.host.thp_mode`, `tetraeder.host.disk.device`; the sampler and the builder also carry `tetraeder.stage`. The collector spans all stages, so hostmetrics series carry no stage: the window timestamps assign them. No series is labelled by expert id.
5. **Derivation** (`scripts/otlp_host.py`, `engine_sample.py otlp-merge`). The OTLP lines are read back into the sampler's row shape and the existing `host_summary()` runs unchanged, so field names and math are the same by construction. Rows sit on the sampler's ticks. The two tick on their own clocks, so a hostmetrics value as of the tick would lag up to one interval, which over a 6 s window bracket moves a disk rate by more than the 5% diff threshold: the cumulative disk counters are interpolated linearly to the tick; queue length and CPU times are taken as of the tick (at most 2.5 s old). Per-CPU totals are the scrape times (a CPU accrues one second per second), so the busy share is busy seconds over elapsed seconds. Units convert back exactly (microseconds, kB, MiB). Windows come from each result: the builder's `decode_windows`, else a `host_windows` object the KTransformers and llama.cpp cache scripts now store. After the collector stops, `exit_run.sh` writes `host_otlp` next to `host` in every result of the run, and `host_otlp_diff`: the fields where the two differ by more than 5% or one is null.
6. **Overhead plan.** The collector runs for every engine's stage equally; only the builder exports engine metrics. On the next box, in one session, as stages next to `main-prefetch`:
   - engine export off: `VARIANTS="main-prefetch-nootlp:TETRAEDER_EXIT_PREFETCH=1,OTEL_EXPORTER_OTLP_ENDPOINT="` with the name in `B1_STAGES` (a variant's settings come after the harness's in the `env` line, and an empty endpoint means no exporter);
   - collector off: a relaunch with `OTEL_COLLECTOR=0` and `B1_STAGES=main-prefetch-nocol`, `VARIANTS="main-prefetch-nocol:TETRAEDER_EXIT_PREFETCH=1"` (the collector and the environment are per launch; `RESUME=1` skips the stages that already ran).
   The gate is the noise band of the paired ratio, as every Vast comparison.
7. **Install.** `scripts/otelcol_install.sh` fetches the release tarball by version and checks its sha256; the provision hook (`box_deps.sh`) runs it when `/opt/otelcol/otelcol-contrib` is missing, and CI runs it to `validate` both configs and to run `otelcol_box.sh` with the sampler end to end (`test_box_collector_end_to_end`). The image tag only changes with the engine pins, so the image is not the install path. Row in `docs/sources.md`.

`tetraeder.host.*` series from the sampler (counters are cumulative):

| Series | Kind, unit | Attributes | `host` fields |
|---|---|---|---|
| `container.cpu.time` | counter, s | | `cpu_busy_pct` |
| `container.disk.io` | counter, By | `disk.io.direction=read` | `cgroup_read_bytes` |
| `tetraeder.host.cgroup.disk.operations` | counter, {operation} | `disk.io.direction=read` | |
| `container.memory.usage`, `tetraeder.host.cgroup.memory.limit` | updown, By | | |
| `tetraeder.host.cgroup.memory.usage` | updown, By | `tetraeder.memory.kind=file,anon` | `cache_file_gb_max`, `anon_gb_max` |
| `tetraeder.host.cgroup.cpu.quota` | gauge, {cpu} | | |
| `tetraeder.host.cgroup.cpu.periods`, `.throttled_periods`, `.throttled_time` | counter, {period} or s | `tetraeder.source` | `throttled_share`, `throttled_ms_per_s` |
| `tetraeder.host.pressure.stall` | counter, s | `tetraeder.pressure.resource=cpu,io`, `tetraeder.pressure.kind=some,full`, `tetraeder.source` | `io_*_share`, `cpu_*_share` |
| `tetraeder.host.cpu.frequency` | gauge, Hz (mean over CPUs) | | `cpu_mhz_mean` |
| `tetraeder.host.process.cpu.time` | counter, s | | `cpu_busy_pct` without a cgroup |
| `tetraeder.host.process.paging.faults` | counter, {fault} | `type=major` | `majflt` |
| `tetraeder.host.process.context_switches` | counter, {count} | `type=voluntary,involuntary` | `ctxsw_*_per_s` |
| `tetraeder.host.process.memory.usage`, `.anon_huge` | gauge, By | | `rss_max_gb`, `anon_huge_gb` |
| `hw.gpu.utilization` | gauge, 1 | `hw.id`, `hw.type=gpu`, `hw.gpu.task=general` | `gpu_busy_pct` |
| `hw.gpu.memory.usage`, `hw.gpu.memory.limit` | updown, By | `hw.id`, `hw.type` | `gpu_mem_max_mib` |
| `hw.power`, `hw.temperature` | gauge, W or Cel | `hw.id`, `hw.type` | |
| `tetraeder.host.gpu.clock` | gauge, Hz | `tetraeder.gpu.clock=sm,memory` | |
| `tetraeder.host.gpu.throttled` | gauge, 0 or 1 | `tetraeder.gpu.throttle_reason` | |
| `tetraeder.host.gpu.pcie.link.generation`, `.width` | gauge | | |
| `tetraeder.host.membw` | gauge, By/s | `tetraeder.membw.threads` | (`env.membw` stays from its file) |
| `tetraeder.host.disk.curve.rate` | gauge, By/s | `tetraeder.disk.curve=idle,loaded`, `tetraeder.disk.queue_depth` | (`disk_util` stays from the curve file) |

Disk fields (`read_*`, `disk_queue_depth_mean`, `disk_inflight_max`), `cpu_busy_pct_top_cpus` and `cpu_steal_share` come from hostmetrics as listed in item 2.

## Consequences

- Box host signals reach the compose stack's dashboard and alerts through the same names (`deploy/otelcol/config.yaml` runs the same receivers and transform, with `root_path: /hostfs`), and a new signal is one reader plus one table row.
- Until a box run shows `host_otlp_diff` empty (or explained), `host` stays sampler-derived; then a later change makes `host` OTLP-derived and removes the JSONL path.
- About 120 MB of OTLP per box hour at 1 s, fetched with the results and billed per GB; the files are not committed.
- Timestamps: hostmetrics points lag a tick by up to a second, so a field over a 5 s window can differ from the sampler's by up to one interval's share; `host_otlp_diff` shows how much.
- Metric names of hostmetrics come from the 0.116.0 receiver docs and the CI `validate` step; their values on a Vast container are confirmed by the first box run, not before.

Fixes after the first box comparison (20261112e): context switches are emitted per thread (tetraeder.host.process.thread.context_switches) so host_otlp uses the sampler positive-delta rule; cpu_busy_pct and the top-CPU mean divide by the threads summarize ran with (stored as host_windows.threads, else the membw probe threads), not options.threads, which a stage may lower; disk_inflight_max is an instantaneous gauge read on two clocks and may differ legitimately.

## Addendum: zstd for the OTLP files (2026-11-12)

Context: 530 MB per 40 minutes, 1 to 2 GB per session, too big to fetch cheaply or commit.

Decision: compress after the collector stops (`otelcol_box.sh compress`: `zstd -T2 -10 --rm` on every `*.jsonl` of the run's directory, called by `exit_run.sh` after `stop`), not through the file exporter's `compression: zstd`. Why:
- A file that is plain while the collector runs survives a crash or a killed box readable; a compressed stream cut mid-frame loses its unflushed tail. The exporter's framing (one zstd frame per export, as we understand it) was not checked against the 0.116.1 source from here, and rotation with it is unverified.
- Rotation (100 MB) stays the exporter's plain business, and no compression CPU runs inside the collector (`GOMAXPROCS=1`, beside an engine at its quota). Compression runs when no engine does.
- No collector config change, so the compose stack's config and CI `validate` stay as they are.

Readers (`otlp_host.open_lines`, `otlp-merge`) accept `*.jsonl` and `*.jsonl.zst` in one tree: stdlib `compression.zstd` (Python 3.14), else the `zstd` CLI, which the box image (Dockerfile) and `box_deps.sh` install. No pip package, so no `docs/sources.md` row. A run that crashed before `compress` leaves plain files, which still merge. `taxameter.toml` fetches `*.zst`. The files are small enough to commit under `bench/results/<date>/otel/` when they total under 200 MB (docs/sessions.md); this replaces "the files are not committed" in the Consequences.

## Addendum: the builder's export does not stall decode; the sampler's smaps_rollup likely does (2026-11-14)

Context: from 20261112c on, one decode window per process spends 40 to 160 ms per token outside `run()` (wall minus `StepProfile::total`). The other windows show a flat 5 to 8 ms. The 1 s engine export of this ADR started that week, so it was the first suspect.

Findings:
- The stall happens without the export. 20261112c ran a commit before e7711c6 (no `telemetry_for`, no `opts.telemetry`), and its main-overlap-repeat still has 96 ms per token in window 0. The steady gap is no larger with the export (5.4 to 8 ms) than without (7 to 8 ms).
- Shape: at most one stalled window per process, always window 0 or 1. A periodic export would touch every window. In 20261113o (checked there only) the step histogram, exported every second, has one export interval with no step and no step time inside the stalled window. The four intervals around it hold 2.3 s of step time in 4 s, so the missing 1.7 s is the window's whole excess. The step thread was stalled; the exporter kept its schedule.
- Code: `PeriodicReader` exports on its own thread (`futures_executor::block_on` there). `collect` takes each instrument's lock only to copy the points, and the step path only records. `record_step` locks the ring, which nothing else takes during decode. `report_memory` reads no `/proc` or cgroup file. Its one blocking edge was `DirectFile::bounce_capacity`, which took the bounce-buffer mutex that `read_batch_with` holds for a whole batch. With the single reader thread (`TETRAEDER_EXIT_IO_SERVICE=0`) the step's memory report therefore waited for the prefetch batch in flight. All the stalled runs used the I/O service, so this did not cause them. Fixed anyway: the size is now mirrored in an atomic.
- Not explained by CPU throttling (zero throttled periods in every stalled window) or by cgroup anon growth (flat). The sampler has no reclaim counter, so reclaim is not ruled out.
- The steady 5 to 8 ms is the harness: `margin` sorts a 151,936-entry index per token (`top_k`), 14 ms per token for margin plus argmax on a desktop CPU (`rustc -O`). `decode_tok_s` counts it.

Likely cause: the host sampler's `smaps_rollup` read (72ca4af, between 20261112a, flat, and 20261112c, the first stall). For that file the kernel walks every page table of the engine under its mmap lock, and only drops the lock between mappings. The RAM tier is one 83 GB mapping on 4 KiB pages (`anon_huge_gb` 0), so one walk holds the lock for about a second. An mmap or munmap on the step thread waits for it, for example a 1.2 MB `top_k` index or a 600 KB logits buffer that glibc maps fresh while its threshold settles early in decode. Supporting evidence:
- At a 1 s interval the sampler ticks every 2.2 to 2.6 s.
- In both 20261113o runs a sampler tick starts 0.1 to 0.2 s before the hole.

Not proven on a box.

Decision:
- No exporter change. A guard test covers the OTLP path (`step_path_does_not_wait_for_a_hung_collector`): the real exporter against a collector that never answers, exporting every 20 ms, keeps every step-path call under 250 ms.
- The sampler reads `smaps_rollup` only for stages named `*thp*`, which need `anon_huge_gb` (ADR 0035). `SAMPLER_SMAPS=1` or `0` forces it on or off, for example for an A/B pair or the levers-snap `thp` arm. Elsewhere `anon_huge_gb` is null.
- The exit harness reports, per decode window, the token with the most time outside `run()` (`gap` in `decode_windows`). It splits that time into the part inside `eval` and the harness's own, each with the step thread's faults and user and system CPU time. A gap far above its CPU time means the thread was blocked.
- `tests/test_smaps_stall.py` (Linux CI) shows the mechanism without a box: reading `smaps_rollup` of a process with 1 GiB on 4 KiB pages slows that process's mmap plus munmap more than threefold.

Consequence: the next box session with windows confirms the cause or not. If the stall is gone and `gap.max_ms` stays near the steady gap, this is settled. If it remains, `gap` says whether the step thread was blocked (wall far above CPU) or busy in the kernel (`sys_ms`, faults), which points at reclaim. A run with `SAMPLER_SMAPS=1` against one without is the direct A/B.
