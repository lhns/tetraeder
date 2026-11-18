# Session 20261112e

- Status: provisional, partial session (trace skipped, repeats cut): builder levers only, no baseline pairing; the poll 0 gain is one one-sided run. ADR 0061's auto polling and ADR 0062's threshold rest on it.
- Code and box: results commit f3e2ec5; box 60326318, host 130210, EPYC 7642, quota 23, 23 threads, THP `madvise`.
- Raw files: this directory.

## From docs/m2-plan.md, section "Facts from the read-only checks" (moved verbatim)

  - ADR 0061's tq4 run (19 threads on a 23 quota) stayed throttled in 82% of periods and decoded 0.80x.

## From ADR 0045 (addendum "tuner too slow for the 235B; thin margin at 23 threads", the `20261112e` paragraph, moved verbatim)

`20261112e`, builder levers in the loss regime (box 60326318, host 130210, EPYC 7642, quota 23, 23 threads, THP `madvise`, disk 6.5 GB/s), order A1, thp, A2, tq4, poll0, thp-repeat (the repeats of tq4 and poll0 and the trace did not fit: builder runs took 19.5 min, prefill 28 to 29 tok/s): decode `main-overlap` 2.294 and 2.334 (spread 1.7%); `cpu_poll` 0: 3.032, 1.30x against A2, one-sided; 19 threads (tq4): 1.868, 0.80x; huge pages: 2.501 then 1.629 (1.09x then 0.70x, not reproducible; `anon_huge_gb` 0 during the first run's decode, 82.6 in the repeat's). **The CPU quota throttles the builder:** cgroup `throttled_share` 0.80 to 0.98 in every run with ggml's default polling, 0.047 with `cpu_poll` 0, which cut CPU segment compute from 297 to 317 to 232 ms per token. Idle workers spinning between the about 95 segment graphs per token spend the quota the compute then waits for; fewer threads do not help (tq4 still throttled 0.82). This is the likely source of the builder's unexplained 292 to 551 ms spread across sessions. Not yet paired against the llama.cpp cache. Results `bench/results/20261112e/`.

## From ADR 0061 Context (moved verbatim)

`bench/results/20261112e` (host 130210, EPYC 7642, quota 23, 23 threads): cgroup `throttled_share` 0.80 to 0.98 in every run with the default level; `cpu_poll` 0 cut it to 0.047, CPU segment compute from 297 to 317 to 232 ms per token, and decoded 3.03 against 2.33 tok/s (1.30x, one run, one-sided). Voluntary context switches rose to about 16,000/s. 19 threads on the same quota (tq4) were still throttled in 82% of periods and decoded 0.80x, so fewer threads are not the fix. Polling off has not been measured on a host without a quota, where spinning costs nothing but power and SMT siblings, and waking a sleeping worker costs a futex round trip per graph.
So a pause after the last CPU graph before a GPU-only segment or a read wait is a small binding (one `CpuProcs` entry and `Backend::pause_cpu_pool`). Its use is the larger part: the builder would have to know, per segment, that the next graph is not on the CPU or that it is about to wait on reads, and the wake cost on resume equals poll 0's. Its value is as an alternative to poll 0: keep spinning between adjacent CPU graphs and sleep only in the gaps, which may avoid most of the 16,000 context switches per second. It changes the decode hot path with no benchmark yet (Rule 3), so it is not built here.

## From ADR 0061 (note 2026-11-12 "evidence", moved verbatim; same line also filed under 20261112f)

The auto default rests on one pair (`20261112e`: poll 0 cut throttling from 0.80 to 0.05 and decoded 1.30x, one-sided). On `20261112f` (same host, invalid as a verdict) poll 0 left throttling at 0.30 and 0.60. Revisit the default after the testbed pairs poll 50, poll 0 and pool pause under `CPUQuota` with repeats.

## From ADR 0062 Context and Decision (moved verbatim)

On the Vast boxes the container's CPU quota throttled the builder in 80 to 98% of CFS periods at `threads` = quota with ggml's default polling (`bench/results/20261112e`, ADR 0045 last paragraph). Idle workers spinning between the about 95 segment graphs per token spent the quota the compute then waited for. With `cpu_poll` 0 the throttled share fell to 0.05 and decode ran 1.30x faster. The run looked like plain CPU-bound compute: the `limits` list said "limited by cpu" and no fault fired. `cpu_cotenant` does not cover it either: nothing else uses the cores, the engine's own threads do.
- **Rule.** Fires when `cpu_throttled_share >= 0.3` and `cpu` is busy at least `MATERIAL_SHARE` (10%) of the step and slower than calibrated beyond the band (or no calibrated rate). 0.3 is between the healthy runs (under 0.05 in the box and sim runs) and the faulty ones (0.80 to 0.98); the host alert `TetraederCpuThrottled` warns earlier at 0.2 on the quota alone.
- **Cost.** With a calibrated CPU rate, the excess CPU seconds as a share of the step (as for `cpu_cotenant`). Without: `cpu busy share x throttled share x 0.3`, which gives about 0.23 for the 20261112e case (CPU 0.8 busy, 0.9 throttled), the measured 1.30x.

## From ADR 0064 Context ("Where the 0.88 ms goes", moved verbatim; same line also filed under 20261112c and 20261112f)

On the 7642 hosts the per-segment cost follows the CPU quota: 2.74 ms per CPU segment at `throttled_share` 0.05 (`20261112e` poll 0), 3.5 to 5.8 ms at 0.80 to 0.98, 4.9 and 8.5 ms at 0.30 and 0.60 (`20261112f`). One hypothesis for the cache's lead of 1.35x to 1.54x is its CPU pair count: we compute 628 / 449 = 1.40x more CPU pairs per token. At 0.06 to 0.09 ms per pair the 179 extra pairs are 11 to 16 ms on `20261112c`; on the 7642, with slower memory and throttling, more, but not measured. ADR 0059's `main-overlap-adapt` reached a 37.2% VRAM share and still decoded slower (2.47 and 2.65 against 4.77 and 4.33 tok/s) because of admission cost, so the share alone does not settle it. That is a placement question this ADR does not answer.
