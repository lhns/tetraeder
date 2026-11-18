# Session 20261110f

- Status: clean; builder ahead of the llama.cpp expert cache at shares 0 and 0.9 on AVX2 (ADR 0045); later superseded as the llama.cpp verdict by the share-0.95 pairings.
- Code and box: commit on main at the time of box 60083507 (see git log of the results commit); RTX 3090, EPYC 7742, 91 GB, 30 threads.
- Raw files: this directory.

## From ADR 0045 (addendum "closing run against the llama.cpp expert cache", moved verbatim)

## Addendum (2026-11-10, box 60083507): closing run against the llama.cpp expert cache

No RTX 3090 offer had an AVX-512 CPU, so the run used the b1c host again: EPYC 7742 (AVX2), RTX 3090, 91 GB, 30 threads = floor(quota) for both engines, KTransformers not rerun. llama.cpp `-cmoe --moe-cache-mib 10458` (the builder's slot bytes), mmap, same prompt hash. Decode `main-overlap`/cache: 4.65, 4.59, 4.60 at share 0; 1.66 (adjacent pair), 1.68, 1.66 at share 0.9 (cache 1.22 and 3.36 tok/s). Prefill 2.8x to 2.9x. Three overlap runs 1.4% apart in decode, greedy tokens equal; overlap and `cpu_experts` GPU parity tests pass. Builder decode used 28 to 30 cores, hidden share 0.41 to 0.43. Verdict: Phase 2 met against KTransformers and the llama.cpp expert cache at `serve` defaults on AVX2; on AVX-512 the prefetch path ties KTransformers and overlap is unmeasured against KTransformers (see the next addendum). Results `bench/results/20261110f/`.
