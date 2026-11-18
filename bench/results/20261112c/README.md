# Session 20261112c

- Status: diagnosis, not a verdict (b1b-levers diagnostic, one box outside the B1 profile; fit windows throttled under poll 50 before ADR 0061). Its replay meets ADR 0050 (c)'s 10% bar; ADR 0059's admission stays off.
- Code and box: results commit eca4dd0; box 60305047, host 49107, RTX 3090, AMD EPYC 7302P, quota 30.72, 30 threads, THP `madvise`.
- Raw files: this directory.

## From ADR 0035 (addendum "2026-11-12: huge pages for the RAM tier (A/B switch)", the "Why" bullet, moved verbatim)

- **Why.** On box run 20261112c the builder's CPU expert work read about 7 GB of Q4_K expert weights per decode token from an RAM tier of about 80 GB, and the host sampler showed `anon_huge_gb` 0 with THP mode `madvise`: the tier sat on 4 KiB pages, so the matmuls walk millions of page table entries (TLB pressure). Hypothesis: huge pages speed up CPU expert compute. Not established; this is the switch to measure it.
