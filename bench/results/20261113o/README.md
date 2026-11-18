# Session 20261113o

- Status: unclean, loss: KTransformers ahead in both pairs, builder stages fail the co-tenant guard, so no clean verdict (ADR 0045). Also the decode window data for ADR 0050 (c) and ADR 0060's stall diagnosis.
- Code and box: results commit 417c184; box 60560915, host 60400, RTX 4090 48 GB, Zen 4 engineering sample (AVX-512), quota 30.7, 26 threads, THP `madvise`.
- Raw files: this directory.

## From ADR 0045 (addendum "tuner too slow for the 235B; thin margin at 23 threads", the `20261113o` paragraph, moved verbatim)

`20261113o` (box 60560915, host 60400, Zen 4 engineering sample with AVX-512 VNNI, VBMI and BF16, no AMX, RTX 4090 48 GB, quota 30.7, 26 threads both engines by `scripts/kt_threads.sh`, `memory.max` 195 GB with the balloon setting the 91 GB budget, download 360 MB/s): **KT ahead in both pairs; the builder stages fail the co-tenant guard, so no clean verdict, but the direction is against us.** Order builder, KT (its own pick `avx512_bf16`, `kt_cpu_variant_ok`), builder, KT. Decode 4.856, 5.658, 3.748, 5.989 tok/s: builder over KT 0.86 and 0.63 adjacent, 0.81 and 0.66 cross; prefill 0.79 and 0.67. KT ran clean (outside load 6.1 and 9.2 cores, limit 9.6, no throttling), spread 5.7%. The builder ran with 11.4 and 25.0 cores busy outside the container (spread 26%). Even the builder's better run is 14% behind KT's worse one. This is the first run against KT's bf16 kernel on Zen 4; on Ice Lake (`20261111a`, `avx512_vbmi`) the builder led 1.04 to 1.12. Results `bench/results/20261113o/`.

## From ADR 0050 (addendum 2026-11-14 "RAM-tier policies across layers in the replay", question and findings moved verbatim)

Question: on `20261113o` the builder reads 561 MB per decode token where KTransformers reads about 160 MB with the same RAM. KTransformers runs experts in place from an mmap, so the page cache acts as one pool for all layers. Do our fixed per-layer RAM slots (77 of 128 per layer at the 83 GB tier) miss more than one pool across layers would?
- 561 MB is the mean of 64 decode tokens. The four 16-token windows read 1147, 549, 215 and 335 MB per token, the same in every session since `20261111h`.
- KTransformers' host disk counters show the same shape on `20261113o`: 508, 330 and 135 MB per token in its first three windows (160 over the run), and 612, 1048, 645 and 208 in the repeat (314 over the run). Its 160 is the better of two runs that differ by 2x, not a steady rate.
- Calibration gate: `lru`'s 16-token windows (VRAM 10, RAM 77, prefetch on) must come within 5% of the trace run's own `decode_windows`, which the stage now records (`TETRAEDER_EXIT_DECODE_WINDOWS=16`), before any policy delta counts. The first four should also lie near `main-overlap`'s 1147, 549, 215 and 335 MB. If they do not, overlap changes the reads and the replay misses that.

## From ADR 0060 (addendum "the builder's export does not stall decode; the sampler's smaps_rollup likely does", moved verbatim)

- Shape: at most one stalled window per process, always window 0 or 1. A periodic export would touch every window. In 20261113o (checked there only) the step histogram, exported every second, has one export interval with no step and no step time inside the stalled window. The four intervals around it hold 2.3 s of step time in 4 s, so the missing 1.7 s is the window's whole excess. The step thread was stalled; the exporter kept its schedule.
- In both 20261113o runs a sampler tick starts 0.1 to 0.2 s before the hole.
