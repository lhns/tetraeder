# Session 20261111a

- Status: clean but thin; Phase 2 met against KTransformers on AVX-512 at `serve` defaults, margin within the baseline's spread (ADR 0045). The ADR 0054 lever runs are a diagnosis (both slower than `main-overlap`).
- Code and box: results commit a9588ef; box 60151711, host 209278, A10 24 GB, Xeon Gold 6342 (AVX-512), 30 threads, 91 GB balloon, driver 570.86.16.
- Raw files: this directory.

## From ADR 0045 (addendum "AVX-512 at `serve` defaults", moved verbatim)

## Addendum (2026-11-11, box 60151711): AVX-512 at `serve` defaults

A10 24 GB, Xeon Gold 6342 (Ice Lake, AVX-512 VNNI), host 209278, ballooned to 91 GB, 30 threads for both engines; KTransformers ran its avx512_vbmi kernels. Decode `main-overlap`/KTransformers: adjacent 1.124 and 1.038, cross 1.043 and 1.119, means 4.283 vs 3.967 tok/s = 1.08. Overlap pair 0.4% apart, KTransformers pair 7.8%. Prefill 1.11 and 1.21. Greedy tokens equal across the builder runs; overlap and `cpu_experts` GPU parity tests pass. KTransformers read 160 to 180 MB per token through mmap at 84% CPU; the builder 585 MB per token at 2.4 GB/s at 99% CPU. Verdict: Phase 2 met at `serve` defaults on both CPU classes measured; on AVX-512 the margin is within the baseline's spread. Results `bench/results/20261111a/`.
