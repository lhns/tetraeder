# Session 20261110c

- Status: clean paired run (b1b-levers): ADR 0054's decode bar met, its hidden share bar missed; the ADR 0054 overlap path ahead of KTransformers on AVX2, but CPU not matched (I/O workers beyond the 26 threads), closed by `20261110e`.
- Code and box: results commits 35f424d and 023f2cc; box 60057461, RTX 3090, AMD EPYC 7742 (AVX2), quota 30.72, 26 threads, 91 GB balloon, driver 595.71.05.
- Raw files: this directory.

## From ADR 0045 (addendum "an AVX2 box needs the ADR 0054 overlap", moved verbatim)

## Addendum (2026-11-10, box 60057461): an AVX2 box needs the ADR 0054 overlap

Session `b1b-levers`: AMD EPYC 7742 (AVX2 only, so no iqk VNNI kernels on either engine), 26 threads, 91 GB. KTransformers ran its `avx2` variant (spread 6.3%). Decode builder/KT: `main-prefetch` 0.92, `main-repeat` 0.89, so the b1-exit path is behind on this CPU class; with ADR 0054's I/O service and overlap 1.27 and 1.21 (`main-iosvc` 1.21). Prefill 2.1x to 2.6x. KT decodes with 171k major faults per run and reads 161 to 171 MB per token; the builder reads 550 to 565 MB.

- Not pooled with b1-exit: different CPU class.
- The verdict holds on both classes only with overlap, which `serve` now enables by default (ADR 0054 addendum). The builder's I/O workers used CPU beyond the 26 threads, which KTransformers did not; a quota-tight run (quota about threads plus one, as b1-exit) of `main-overlap` against KTransformers closes that question.

## From ADR 0045 (addendum "owner decision on slow disks; correction", 2026-11-11, the correction paragraph, moved verbatim)

Correction to the box 60190564 addendum: the extra decode bytes are not mainly prefetch. On `20261110c` (same box and options) the 64-token runs read 549 MB per token and the 256-token trace run 262, so tokens 65 to 256 read about 167 MB per token, the same as KTransformers; without prefetch the first 64 tokens still read 443 (`20261103`). The builder's excess is warm-up of the RAM tier in the first tokens after the 32K prefill; KTransformers' page cache shows none. Cause under diagnosis; the replay sentence above stands as data, not as the lever.

## From ADR 0050 (addendum 2026-11-14 "RAM-tier policies across layers in the replay", findings on this session's trace moved verbatim)

- The 256-token trace run of `20261110c` (prefetch, no overlap) read 262 MB per token. Its replay gives 262.3 for `lru` at VRAM 10, RAM 77 with prefetch, and a decode RAM hit rate of 0.9637 against 0.9637 measured. The same configuration at 64 tokens (`main-prefetch`, same session) read 549 MB per token, so tokens 64 to 255 read about 167 MB per token. Most of the excess is in the first 32 tokens after the prefill, not in the steady state.
- On the same trace, per-layer `belady` without prefetch reads 39.9 MB per token against 186.8 for `lru` without prefetch. Per-layer slots do not bound the result; the eviction choice and the tier's contents when decode starts do.
