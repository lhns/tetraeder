# Session 20261112e

- Status: provisional, partial session (trace skipped, repeats cut): builder levers only, no baseline pairing; the poll 0 gain is one one-sided run. ADR 0061's auto polling and ADR 0062's threshold rest on it.
- Code and box: results commit f3e2ec5; box 60326318, host 130210, EPYC 7642, quota 23, 23 threads, THP `madvise`.
- Raw files: this directory.

## From docs/m2-plan.md, section "Facts from the read-only checks" (moved verbatim)

  - ADR 0061's tq4 run (19 threads on a 23 quota) stayed throttled in 82% of periods and decoded 0.80x.

## From ADR 0045 (addendum "tuner too slow for the 235B; thin margin at 23 threads", the `20261112e` paragraph, moved verbatim)

`20261112e`, builder levers in the loss regime (box 60326318, host 130210, EPYC 7642, quota 23, 23 threads, THP `madvise`, disk 6.5 GB/s), order A1, thp, A2, tq4, poll0, thp-repeat (the repeats of tq4 and poll0 and the trace did not fit: builder runs took 19.5 min, prefill 28 to 29 tok/s): decode `main-overlap` 2.294 and 2.334 (spread 1.7%); `cpu_poll` 0: 3.032, 1.30x against A2, one-sided; 19 threads (tq4): 1.868, 0.80x; huge pages: 2.501 then 1.629 (1.09x then 0.70x, not reproducible; `anon_huge_gb` 0 during the first run's decode, 82.6 in the repeat's). **The CPU quota throttles the builder:** cgroup `throttled_share` 0.80 to 0.98 in every run with ggml's default polling, 0.047 with `cpu_poll` 0, which cut CPU segment compute from 297 to 317 to 232 ms per token. Idle workers spinning between the about 95 segment graphs per token spend the quota the compute then waits for; fewer threads do not help (tq4 still throttled 0.82). This is the likely source of the builder's unexplained 292 to 551 ms spread across sessions. Not yet paired against the llama.cpp cache. Results `bench/results/20261112e/`.
