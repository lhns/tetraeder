# 0059. Box and host metrics as OpenTelemetry metrics: one collector per box run, the result's host block derived from its file

Status: Proposed (2026-11-12, owner decision). Extends ADR 0012 (one OTel data path) and ADR 0058 (host fields per phase and decode window); the sampler path of ADR 0058 stays until a box run shows both paths agree.

## Context

`scripts/engine_sample.py` samples host, cgroup, process-tree and GPU counters once a second into `samples-*.jsonl` and summarizes them into each result's `host` block (ADR 0056 addendum, ADR 0058 and its addenda). That is a second data path beside ADR 0012's: the fields never reach a dashboard, and a new signal means a new reader, a new JSONL column and a new summary. Box runs start no collector, and `exit_regime.rs` builds no `Telemetry`, so the builder's own metrics (step split, expert reads, prefetch) never reach a bench result either.

Constraints:

- **Baselines export nothing.** llama.cpp and KTransformers have no OTLP metrics, so every signal compared across engines must come from outside the engine: collector receivers or the sampler.
- **Decode windows are 5 to 7 s.** Host data needs a 1 s grid; the engine SDK exports every 10 s by default (`TelemetryConfig`).
- **The builder runs at 94 to 98% of its CPU quota.** Anything added runs for every engine's stage alike, and its cost is measured, not assumed.
- **Committed JSON stays** (Rule 3). The `host` block keeps its field names and math.

## Decision

1. **One collector per box run.** `exit_run.sh` starts `otelcol-contrib` 0.116.1 (`scripts/otelcol_box.sh`, config `deploy/otelcol/config-file.yaml`) before the first stage and stops it before `DONE`, so the files are flushed before fetch. It writes OTLP JSON lines to `runs/vast/otel/<start stamp>/metrics.jsonl` (a new directory per start: the file exporter truncates on open), which `taxameter fetch` packs (`[results] extra = runs/vast`, `*.jsonl`). A failed start is logged and the run goes on with the sampler path alone. `GOMAXPROCS=1` and `GOMEMLIMIT=200MiB` bound the collector.
2. **Sources, one per field.** Each `host` field has exactly one source in the OTLP path:
   - **hostmetrics receiver, 1 s:** disk (`system.disk.io`, `system.disk.operations`, `system.disk.pending_operations`, `system.disk.weighted_io_time`, by device name; loop and ram devices excluded) and CPU (`system.cpu.time`). A `metricstransform` step keeps `system.cpu.time` per state summed over CPUs (8 series: steal share) and adds `tetraeder.host.cpu.busy` per CPU (all states but idle and wait: the busiest-CPUs view). Per CPU and per state at 1 s would be about 200 KB/s on a host showing 128 cores; this is about 30 KB/s. Also memory (`system.memory.usage`), network (`system.network.io`, eth and en devices), and the process scraper for the builder and llama.cpp binaries (names `exit_regime-.*`, `tetraeder.*`, `llama-.*`), with `process.paging.faults`, `process.context_switches`, `process.threads` and `process.disk.operations` enabled. These process series feed the dashboard only.
   - **The sampler, OTLP/HTTP JSON over `urllib` to the same collector, 1 s:** everything hostmetrics does not cover or measures differently. The cgroup (CPU time, quota, throttling, `io.stat` reads, `memory.current`, `memory.max`, file and anon memory), pressure (PSI cpu and io), mean CPU MHz, the engine's process tree (CPU time, major faults, context switches summed over threads, RSS, AnonHugePages), the GPU through `nvidia-smi` (utilization, memory, power, temperature, SM and memory clocks, throttle reasons, PCIe link generation and width), and once per probe the membw result and the disk curve points. The process scraper filters by executable name only: KTransformers runs as `python3`, like the sampler and the RAM balloon, and its `process.context_switches` reads `/proc/PID/status`, which counts the leader thread only. So every process field of the `host` block comes from the sampler's pid tree.
   - **Not from nvidia-smi `--query-gpu`:** PCIe throughput (only `nvidia-smi dmon`, another process). Left out.
   - Stdlib `urllib` instead of the Python SDK: the box runs the system `python3`, often with `-I`, and installs no packages; the payload is a fixed set of points. The post has a 0.5 s timeout, never raises, and stops after 10 straight failures. `TETRAEDER_HOST_OTLP` (the collector's base URL) turns it on; the harness sets it when the collector is up, and child scripts (KTransformers, llama.cpp cache) inherit it. `SAMPLER_OTLP=0` turns it off. The JSONL file is written as before.
3. **Engine metrics.** `TelemetryConfig::from_env` reads the standard `OTEL_METRIC_EXPORT_INTERVAL` (ms). `exit_regime.rs` builds a `Telemetry` when `OTEL_EXPORTER_OTLP_ENDPOINT` is set and passes it in `LoadOptions::telemetry`; the harness sets the endpoint and `OTEL_METRIC_EXPORT_INTERVAL=1000` for the builder only (`ENGINE_OTLP=0` turns it off). Export every second rather than a flush at window edges: a flush between windows would land inside `decode_s`. New gauges: `tetraeder.cpu.threads` (configured compute threads) and `tetraeder.cpu.quota` (the cgroup's CPU limit in cores). The VRAM admission and segment-split metrics of other tracks reach the same file through the catalog with no harness change; deriving engine fields into the result is a later step.
4. **Names.** Semantic-convention names where one exists and means the same: `container.cpu.time`, `container.disk.io`, `container.memory.usage`, `hw.gpu.utilization`, `hw.gpu.memory.usage`, `hw.gpu.memory.limit`, `hw.power`, `hw.temperature`, and hostmetrics' own `system.*` and `process.*`. Everything else is `tetraeder.host.*` (list below). A sampler series whose meaning differs from a hostmetrics series of the same semconv name (process tree versus process, threads versus leader) keeps the prefix. Resource attributes: `service.name` (`tetraeder-host-sampler`, `tetraeder`, or the collector's), `host.id` (Vast container id), `host.name`, `host.cpu.model.name`, `tetraeder.session`, `tetraeder.git.commit`, `tetraeder.cpu.quota`, `tetraeder.gpu.name`, `tetraeder.gpu.driver`, `tetraeder.host.thp_mode`, `tetraeder.host.disk.device`; the sampler and the builder also carry `tetraeder.stage`. The collector spans all stages, so hostmetrics series carry no stage: the window timestamps assign them. No series is labelled by expert id.
5. **Derivation** (`scripts/otlp_host.py`, `engine_sample.py otlp-merge`). The OTLP lines are read back into the sampler's row shape and the existing `host_summary()` runs unchanged, so field names and math are the same by construction. Rows sit on the sampler's ticks; each hostmetrics series joins as of the tick (its last point at or before it, so at most one interval old, as the sampler's own grid). Per-CPU totals are the scrape times (a CPU accrues one second per second), so the busy share is busy seconds over elapsed seconds. Units convert back exactly (microseconds, kB, MiB). Windows come from each result: the builder's `decode_windows`, else a `host_windows` object the KTransformers and llama.cpp cache scripts now store. After the collector stops, `exit_run.sh` writes `host_otlp` next to `host` in every result of the run, and `host_otlp_diff`: the fields where the two differ by more than 5% or one is null.
6. **Overhead plan.** The collector runs for every engine's stage equally; only the builder exports engine metrics. On the next box, one paired run: `main-prefetch` with and without `ENGINE_OTLP` and the collector (`OTEL_COLLECTOR=0`), in the same session, then the gate is the noise band of the paired ratio, as every Vast comparison.
7. **Install.** `scripts/otelcol_install.sh` fetches the release tarball by version and checks its sha256; the provision hook (`box_deps.sh`) runs it when `/opt/otelcol/otelcol-contrib` is missing, and CI runs it to `validate` both configs. The image tag only changes with the engine pins, so the image is not the install path. Row in `docs/sources.md`.

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
