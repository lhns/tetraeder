# Session ci-poll-20261112

- Status: diagnosis on a CI runner: confirms the throttling mechanism under a quota (ADR 0061), not the decode gain (the synthetic run is bound by its reads).
- Code and box: results commit 1519448; CI run 43275262090, 4 cores, quota 2, synthetic MoE.
- Raw files: this directory.

## From ADR 0061 (note 2026-11-12 "evidence", CI check, moved verbatim)

CI check (`bench/results/ci-poll-20261112`, 4-core runner, quota 2 cores, threads 4, 8 interleaved rounds, synthetic MoE): poll 50 throttled in 98% of periods, poll 0 in 13%, pool pause in 15%; CPU compute per token 24.6, 13.9 and 13.1 ms. Decode tok/s did not move (ratios 1.002 to 1.003), because the synthetic run is bound by its reads, so this confirms the throttling mechanism but not the decode gain. Without a quota poll 0 cost about 6 ms per token of compute (13.8 against 7.2, one round).
