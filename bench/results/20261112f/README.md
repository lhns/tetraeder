# Session 20261112f

- Status: null, invalid as a verdict: the RAM balloon was OOM-killed, so the cache ran with the model resident (ADR 0045 balloon rule); no DONE file.
- Code and box: results commit dbcc186; box 60342483, host 130210, EPYC 7642, quota 23.04, 23 threads.
- Raw files: this directory.

## From ADR 0045 (addendum "tuner too slow for the 235B; thin margin at 23 threads", the `20261112f` paragraph, moved verbatim)

`20261112f` (box 60342483, host 130210, EPYC 7642, quota 23.04, 23 threads), builder with `cpu_poll` 0 interleaved with the cache at 0.95 in dio and mmap: **invalid as a verdict.** The RAM balloon was OOM-killed during the first cache stage and not restarted, so the later stages ran with the 142 GB model resident (llama-server RSS 131 to 141 GB, 0 major faults, 0 disk bytes per token): not the larger-than-RAM regime. Numbers for the record: builder 1.683 and 1.133 tok/s (39% apart, CPU compute 457 and 752 ms per token), cache dio 3.565 and 2.078, mmap 2.147 and 2.162; the cache led every pair. `cpu_poll` 0 did not reproduce `20261112e`'s throttling cut (throttled share 0.30 and 0.60, not 0.05). The memory-bandwidth probe before each stage read 42 GB/s before the first builder run and 28 before its repeat on this host (24 to 26 before the cache stages, 37 to 53 on `20261112e`), so the shared host's memory bandwidth moves between stages by up to 2x. It does not explain the builder's spread: on `20261112e`, same host, probes of 42.1 and 40.4 GB/s came with CPU compute of 312 and 513 ms per token, and the fastest run (237 ms) had the lowest probe (37.2); the probe also runs about 25 minutes before decode. Across the twelve builder runs of `20261112c`, `e` and `f`, compute follows the throttled share more closely than the probe, but neither explains it alone; the cause of the 2x swing on the 7642 hosts is unknown. The harness now has to keep the balloon alive and flag any stage that ran without it. Results `bench/results/20261112f/`.

## From ADR 0061 (note 2026-11-12 "evidence", moved verbatim; same line also filed under 20261112e)

The auto default rests on one pair (`20261112e`: poll 0 cut throttling from 0.80 to 0.05 and decoded 1.30x, one-sided). On `20261112f` (same host, invalid as a verdict) poll 0 left throttling at 0.30 and 0.60. Revisit the default after the testbed pairs poll 50, poll 0 and pool pause under `CPUQuota` with repeats.

## From ADR 0064 Context ("Where the 0.88 ms goes", moved verbatim; same line also filed under 20261112c and 20261112e)

On the 7642 hosts the per-segment cost follows the CPU quota: 2.74 ms per CPU segment at `throttled_share` 0.05 (`20261112e` poll 0), 3.5 to 5.8 ms at 0.80 to 0.98, 4.9 and 8.5 ms at 0.30 and 0.60 (`20261112f`). One hypothesis for the cache's lead of 1.35x to 1.54x is its CPU pair count: we compute 628 / 449 = 1.40x more CPU pairs per token. At 0.06 to 0.09 ms per pair the 179 extra pairs are 11 to 16 ms on `20261112c`; on the 7642, with slower memory and throttling, more, but not measured. ADR 0059's `main-overlap-adapt` reached a 37.2% VRAM share and still decoded slower (2.47 and 2.65 against 4.77 and 4.33 tok/s) because of admission cost, so the share alone does not settle it. That is a placement question this ADR does not answer.
