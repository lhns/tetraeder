# Session 20261113b

- Status: clean; Phase 2 point 1 verdict pairing: builder ahead in every pair against the llama.cpp cache at its best runnable configuration (mmap, poll 0, share 0.95) (ADR 0045).
- Code and box: commit on main at the time of box 60437503 (see git log of the results commit); host 130210, same profile as `20261113a`.
- Raw files: this directory.

## From ADR 0045 (addendum "tuner too slow for the 235B; thin margin at 23 threads", the `20261113b` paragraph, moved verbatim)

`20261113b` (box 60437503, host 130210, same profile and launch as `20261113a`, cache at `--poll 0`; guards clean on every verdict stage, outside load 3.0 to 8.3 cores): **builder ahead in every pair against the cache at poll 0 in mmap.** Order builder, cache, builder, cache. Decode 3.018, 1.786, 2.743, 2.231 tok/s: builder over cache 1.69, 1.54 and 1.23 adjacent, 1.35 cross, 1.44 mean over mean; spread builder 9.5%, cache 22%; prefill 1.49x and 1.66x. Poll 0 removed the cache's throttling (12.9 and 1.7 ms of each second, against 329 and 371 in `20261113a`); the builder was at 28.8 and 41.6 ms. Hit rate 0.353 and decode faults 2700 and 2884 per token, as before. The dio stage was cut by the deadline; dio has no valid run under the balloon (`20261112f` was invalid), so the cache in dio at poll 0 is the one configuration left to pair. Results `bench/results/20261113b/`.
