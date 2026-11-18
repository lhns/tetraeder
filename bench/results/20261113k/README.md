# Session 20261113k

- Status: clean; point 1 confirmed on a second host: builder ahead of the llama.cpp cache (mmap, poll 0, share 0.95) in every pair (ADR 0045).
- Code and box: commit on main at the time of box 60510586 (see git log of the results commit); host 88910, RTX 4090, EPYC 7B13, quota 61.4, 24 threads.
- Raw files: this directory.

## From ADR 0045 (addendum "tuner too slow for the 235B; thin margin at 23 threads", the `20261113k` paragraph, moved verbatim)

`20261113k` (box 60510586, host 88910, EPYC 7B13, RTX 4090, quota 61.4, both engines at 24 threads, `memory.max` 519 GB with the balloon setting the 91 GB budget, disk 5.2 GB/s; guards clean on every verdict stage, outside load 1.1 to 15.8 cores in decode): **builder ahead in every pair; point 1 confirmed on a second host.** Order builder, cache (mmap, poll 0, share 0.95), builder, cache. Decode 3.621, 2.996, 4.547, 2.985 tok/s: builder over cache 1.21, 1.52 and 1.52 adjacent, 1.21 and 1.52 cross, 1.37 mean over mean; spread builder 23%, cache 0.4%; prefill 1.12x and 1.51x. Neither engine was throttled. The cache was healthy: decode 1.5 to 1.9x its warm-up, RSS 86 GB, 0.43 to 0.45 GB read per token (`20261113b`: 0.46), hit rate 0.354. The baseline's CUDA build targets sm_86 and ran on the 4090 (sm_89), which runs sm_86 code natively. With `20261113b` (1.23x to 1.69x, host 130210) this is the second clean session on a second host, each with the builder ahead in every pair. The builder's own run-to-run spread (10% and 23%) is the weak side of both. Results `bench/results/20261113k/`.
