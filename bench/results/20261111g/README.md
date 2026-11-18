# Session 20261111g

- Status: clean but thin; builder ahead of the llama.cpp cache at share 0.5 only (not its best share); the tuner timed out (ADR 0045).
- Code and box: results commit 1041981; box 60236780, host 130210, RTX 3090, AMD EPYC 7642, quota 23, 91 GB balloon.
- Raw files: this directory.

## From ADR 0045 (addendum "tuner too slow for the 235B; thin margin at 23 threads", moved verbatim)

## Addendum (2026-11-11, box 60236780): tuner too slow for the 235B; thin margin at 23 threads

RTX 3090, AMD EPYC 7642, host 130210, CPU quota 23 (the B1 profile asks 24 to 26), 91 GB balloon, disk 6.4 GB/s at depth 16. llama.cpp's tuner at 8K context fit this time but timed out (rc 124 after 40 min) inside its first probe: that probe generates 2,048 tokens (warm-up at about 1.2 tok/s, 29 min) at a 13,048 MiB cache, above `--max-mib`; by estimate its six probes need over 2 hours (not measured beyond this 40-minute limit). The tuner fits per-machine rates, so its pick is only meaningful paired on the box it ran on; it always picks the largest cache that fits and the lowest share within 5% of the fastest of 0, 0.5, 0.8, 0.9, 0.95 and 0.98. `llamacpp-cache` at 10,458 MiB, share 0.5: decode 1.403 and 1.344 tok/s against overlap 1.578 and 1.504; ratios 1.125 and 1.119 adjacent, 1.174 and 1.072 cross; spreads 4.7% (builder) and 4.4% (cache); prefill 2.0x. The builder spent 533 of 627 ms per token computing and 11 waiting on reads: at 23 threads it is CPU-bound, while the cache runs most expert work on the GPU, so the margin shrinks with cores (2.3x at 46 threads, `20261111c`). Share 0.5 is probably not the cache's best: on `20261110f` it went from 1.22 tok/s at share 0 to 3.36 at 0.9, and the only pairing at that share is 1.66x (30 threads). Shares 0.8 to 0.98 have not been paired at 23 to 26 threads, where a similar lift would put the cache ahead. Greedy tokens equal across the builder runs; prompt hash equal in all four runs. Results `bench/results/20261111g/`.
