# Session 20261113o

- Status: unclean, loss: KTransformers ahead in both pairs, builder stages fail the co-tenant guard, so no clean verdict (ADR 0045). Also the decode window data for ADR 0050 (c) and ADR 0060's stall diagnosis.
- Code and box: results commit 417c184; box 60560915, host 60400, RTX 4090 48 GB, Zen 4 engineering sample (AVX-512), quota 30.7, 26 threads, THP `madvise`.
- Raw files: this directory.

## From ADR 0045 (addendum "tuner too slow for the 235B; thin margin at 23 threads", the `20261113o` paragraph, moved verbatim)

`20261113o` (box 60560915, host 60400, Zen 4 engineering sample with AVX-512 VNNI, VBMI and BF16, no AMX, RTX 4090 48 GB, quota 30.7, 26 threads both engines by `scripts/kt_threads.sh`, `memory.max` 195 GB with the balloon setting the 91 GB budget, download 360 MB/s): **KT ahead in both pairs; the builder stages fail the co-tenant guard, so no clean verdict, but the direction is against us.** Order builder, KT (its own pick `avx512_bf16`, `kt_cpu_variant_ok`), builder, KT. Decode 4.856, 5.658, 3.748, 5.989 tok/s: builder over KT 0.86 and 0.63 adjacent, 0.81 and 0.66 cross; prefill 0.79 and 0.67. KT ran clean (outside load 6.1 and 9.2 cores, limit 9.6, no throttling), spread 5.7%. The builder ran with 11.4 and 25.0 cores busy outside the container (spread 26%). Even the builder's better run is 14% behind KT's worse one. This is the first run against KT's bf16 kernel on Zen 4; on Ice Lake (`20261111a`, `avx512_vbmi`) the builder led 1.04 to 1.12. Results `bench/results/20261113o/`.
