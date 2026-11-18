# Session 20261112e

- Status: provisional, partial session (trace skipped, repeats cut): builder levers only, no baseline pairing; the poll 0 gain is one one-sided run. ADR 0061's auto polling and ADR 0062's threshold rest on it.
- Code and box: results commit f3e2ec5; box 60326318, host 130210, EPYC 7642, quota 23, 23 threads, THP `madvise`.
- Raw files: this directory.

## From docs/m2-plan.md, section "Facts from the read-only checks" (moved verbatim)

  - ADR 0061's tq4 run (19 threads on a 23 quota) stayed throttled in 82% of periods and decoded 0.80x.
