# Session ci-overhead-20261114

- Status: clean CI micro-benchmark of the segment dispatch floor (ADR 0064); a 4-CPU runner cannot show the 26-thread cost.
- Code and box: results commit 775347f; CI run 43535138852, Xeon 6975P-C, 4 CPUs, no quota.
- Raw files: this directory.

## From ADR 0064 (addendum 2026-11-14 "floor on the CI runner", moved verbatim)

`bench/results/ci-overhead-20261114/` (Xeon 6975P-C, 4 CPUs, no quota). Per graph kickoff at 4 threads after an idle gap of 100 to 500 us: 11 to 17 us at poll 0, 1.0 to 1.3 us at poll 50; barrier 1.3 to 1.9 us either way. 8 threads on 4 CPUs (`n_2x`) cost milliseconds per kickoff, so the pool must never exceed the quota. End to end on the synthetic 48-layer model: segment scheduler 88.0 against whole-step 83.8 tok/s (alloc 0.88 against 1.22 ms per token), greedy equal.
At about 95 segment graphs per token, poll 0 wake-ups cost about 1 to 2 ms per token at 4 threads; the 26-thread cost is not measurable on a 4-CPU runner. 20261113o ran the builder at poll 0 (auto, ADR 0061: quota 30.7 below the host) with 26 threads, while KTransformers busy-spins. With 4.7 CPUs of headroom, poll 50 is the cheap arm: the next point-3 box runs `main-overlap-poll50` and its repeat interleaved with `main-overlap`, and reads `throttled_share`.
