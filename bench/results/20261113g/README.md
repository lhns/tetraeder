# Session 20261113g

- Status: null; builder clean, but the llama.cpp cache thrashed mmap, so no ratio (ADR 0045 thrashing rule; origin of the 150 GB cgroup rule).
- Code and box: commit on main at the time of box 60497639 (see git log of the results commit); host 70142, RTX 3090, EPYC 7663, quota 53.8, 24 threads.
- Raw files: this directory.

## From ADR 0045 (addendum "tuner too slow for the 235B; thin margin at 23 threads", the `20261113g` paragraph, moved verbatim)

`20261113g` (box 60497639, host 70142, EPYC 7663, RTX 3090, quota 53.8, 24 threads both engines; guards clean, outside load under 3 cores): **builder clean, the cache run gives no ratio.** Decode builder 4.841 and 5.640 tok/s, cache (mmap, poll 0, share 0.95) 0.440 and 0.437, 4 to 5x below the same configuration on `20261113b`. The cache was thrashing mmap: it read 1.6 GB from disk per decode token against 0.46 GB on `20261113b`, its RSS peaked at 72 and 74 GB against 87, and its measured decode was only twice its warm-up (0.19 and 0.23), where `20261113b` reached 2.6 to 3.1x. This box differs in one parameter from every earlier valid pairing: `memory.max` itself was the 91 GB budget (balloon 0.55 GB), where earlier boxes had a 183 GB cgroup with a 92 GB balloon. Same rule as `20261112c`: a thrashing cache gives no ratio. Pairings now require a cgroup limit of at least 150 GB, so the balloon sets the budget as on `20261113b` (docs/box-driver.md). Results `bench/results/20261113g/`.
