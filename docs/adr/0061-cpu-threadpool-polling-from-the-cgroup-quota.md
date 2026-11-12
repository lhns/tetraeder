# 0061. CPU threadpool polling chosen from the cgroup CPU quota

Status: Proposed (2026-11-12). Extends ADR 0036 (one CPU threadpool per backend) and the `cpu_poll` switch of ADR 0038; evidence in ADR 0045 (`20261112e`).

## Context

ggml's threadpool workers spin between graphs before they sleep: level `poll` (0 to 100, default 50, `ggml_threadpool_params_init`) bounds the spin at `128 * 1024 * poll` relax instructions (`ggml_graph_compute_poll_for_work`, ggml-cpu.c at the pinned commit `2e0d86be`). The builder runs about 95 segment graphs per decode token with CPU, GPU and read-wait gaps between them, so with 23 threads on a 23-core cgroup quota the idle workers spend the quota that the next CPU graph then waits for.

`bench/results/20261112e` (host 130210, EPYC 7642, quota 23, 23 threads): cgroup `throttled_share` 0.80 to 0.98 in every run with the default level; `cpu_poll` 0 cut it to 0.047, CPU segment compute from 297 to 317 to 232 ms per token, and decoded 3.03 against 2.33 tok/s (1.30x, one run, one-sided). Voluntary context switches rose to about 16,000/s. 19 threads on the same quota (tq4) were still throttled in 82% of periods and decoded 0.80x, so fewer threads are not the fix. Polling off has not been measured on a host without a quota, where spinning costs nothing but power and SMT siblings, and waking a sleeping worker costs a futex round trip per graph.

## Decision

1. **Auto by default.** `LoadOptions::cpu_poll = None` means auto (`tetraeder_model::cpu_poll::resolve`): with a detected quota `q`, level 0 when `q` is below the host's online CPUs, or when `threads >= floor(q)`; otherwise ggml's default 50. Without a quota (no cgroup limit, not Linux): 50. An explicit level always wins (clamped to 100).
   - The first clause covers the measured case and the tq4 run: a quota below the host is binding even with fewer threads (the main thread, the I/O service and the GPU driver share it). The second clause matters only when the quota is at or above the host's online CPUs, or the online count is unknown; it is unmeasured.
2. **Detection.** The quota from cgroup v2 `cpu.max`, else v1 `cpu.cfs_quota_us` over `cpu.cfs_period_us` (`cpu` or `cpu,cpuacct`), through `tetraeder_telemetry::cgroup_cpu_quota` (now with a root-path variant for tests). Online CPUs from `/sys/devices/system/cpu/online`, not `available_parallelism()`, which already respects the quota. Read once at load; the pool reads `poll` only at creation (`ggml_threadpool_new_impl`), so it is resolved before `Backend::cpu_with_poll`.
3. **Record.** `Backend::cpu_poll()` returns the level ggml received. `Qwen3Moe::cpu_poll()` returns `CpuPoll { level, auto, quota, online }`; load logs it (`cpu threadpool polling`). The exit JSON's `options.cpu_poll` is now the level in effect, with `cpu_poll_auto` and `cpu_quota_detected` beside it (null for the libllama reference). `TETRAEDER_EXIT_CPU_POLL` stays the explicit override. `serve` uses auto with no change, as it never sets `cpu_poll`.
4. **Metrics (Rule 6).** Gauge `tetraeder.cpu.poll` (level in effect, attribute `tetraeder.cpu.poll.auto`), recorded at load when telemetry is on; `tetraeder.cpu.quota` already exists (ADR 0060). Series "engine poll level" on the Container CPU panel. Alert `TetraederCpuPollingUnderQuota`: poll above 0 while `threads + 1 > quota` (that is, `threads >= floor(quota)` without `floor`). Auto never fires it; an explicit level can. The throttling alerts' advice now points at polling, not fewer threads.

## Pausing the pool in GPU and read-wait gaps

Read in the vendored ggml (`ggml-cpu.c`, OpenMP off in our build, so this path is the one we run):

- `ggml_threadpool_pause` sets `pause` under the pool mutex and broadcasts. A polling worker leaves its spin at once (`ggml_graph_compute_thread_ready` checks `pause`) and sleeps on the condition variable in the pause loop of `ggml_graph_compute_secondary_thread`.
- `ggml_graph_compute_kickoff` resumes a paused pool by itself (it clears `pause` and broadcasts), so no resume call is needed, and the pause has to be issued again after every graph that precedes a gap. Calling it between `ggml_backend_graph_compute` calls is safe: all workers have passed the final barrier then.
- The symbol is exported by the CPU backend module (`GGML_BACKEND_API` in `ggml-cpu.h`), so it resolves through `CpuProcs` like `ggml_threadpool_new`.
- `poll` only bounds the spin between graphs. `ggml_barrier` between ops inside a graph spins without bound at any level, so neither poll 0 nor pausing touches intra-graph waits.

So a pause after the last CPU graph before a GPU-only segment or a read wait is a small binding (one `CpuProcs` entry and `Backend::pause_cpu_pool`). Its use is the larger part: the builder would have to know, per segment, that the next graph is not on the CPU or that it is about to wait on reads, and the wake cost on resume equals poll 0's. Its value is as an alternative to poll 0: keep spinning between adjacent CPU graphs and sleep only in the gaps, which may avoid most of the 16,000 context switches per second. It changes the decode hot path with no benchmark yet (Rule 3), so it is not built here.

## Consequences

- On a quota box the default changes: `main-overlap` now runs at poll 0, so stages that differ only by `TETRAEDER_EXIT_CPU_POLL=0` duplicate it. The default-polling baseline needs an explicit `TETRAEDER_EXIT_CPU_POLL=50` stage, paired and interleaved with the auto stage.
- Next steps: (a) a paired run of auto against 50 on a quota box with a repeat, for the noise band; (b) poll 0 against 50 on a box without a quota, to test the threads clause; (c) the gap pause above, paired against poll 0.
- Context switches at poll 0 are a known cost; `ctxsw_voluntary_per_s` in the result's `host` block tracks them.

## Addendum 2026-11-12: pause the pool in the gaps (`pool_pause`)

Step (c) above, built behind an option so it can be paired against poll 0 and poll 50.

- `LoadOptions::pool_pause: bool` (default false). The polling level is unchanged; the builder calls `Backend::pause_cpu_pool` (ggml `ggml_threadpool_pause`) where no CPU graph follows soon: before a segment without RAM-tier groups when the primary backend is a GPU, before `finish_prefetch` (both places in `run`) and before each `wait_landed` in the overlapped layer step. The next graph compute resumes the pool by itself. All calls sit between graph computes on the one thread that issues them.
- The CPU registry does not return `ggml_threadpool_pause` at the pin, and a `GGML_BACKEND_DL` build cannot link it directly. Patch `0003-cpu-registry-threadpool-pause.patch` adds the three-line `strcmp` branch (row in `docs/sources.md`; tested through `cpu_pool_pause_is_exact_and_counts_pauses`, which fails without it as no pause is issued). `CpuProcs::threadpool_pause` is optional, so a module without the patch makes the option a no-op.
- Exactness: pausing moves no data; logits are bitwise equal with it on and off (`crates/model/tests/prefetch.rs`).
- Metric (Rule 6): counter `tetraeder.cpu.pool.pauses`, added per step, series "pool pauses per s" on the Container CPU panel. No alert: a pause count has no threshold of its own.
- Harness: `TETRAEDER_EXIT_POOL_PAUSE=1` in `exit_regime.rs`, recorded as `options.pool_pause`. The pairing for step (c) is `TETRAEDER_EXIT_CPU_POLL=50 TETRAEDER_EXIT_POOL_PAUSE=1` against the auto stage (poll 0) and against poll 50 alone, with a repeat. No benchmark result yet (Rule 3): the option stays off until one exists.

## Note 2026-11-12: evidence

The auto default rests on one pair (`20261112e`: poll 0 cut throttling from 0.80 to 0.05 and decoded 1.30x, one-sided). On `20261112f` (same host, invalid as a verdict) poll 0 left throttling at 0.30 and 0.60. Revisit the default after the testbed pairs poll 50, poll 0 and pool pause under `CPUQuota` with repeats.
