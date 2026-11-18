# Session 20261114

- Status: provisional; at the default slot budget the builder and KTransformers are level within KT's spread, vramfill ahead in one unrepeated run, `noiqk` failed the co-tenant guard (ADR 0045); point 3 is judged over this session and `20261114b` (owner decision 2026-11-14). Its replay is clean for ADR 0050 (c) (the pool across layers is not built); the KV layout step 1 files are offline, not from the box.
- Code and box: results commits ff7ee80 and 51523c8; box 60591327, host 60400, RTX 4090 48 GB, Zen 4 (AVX-512), quota 30.7, 26 threads.
- Raw files: this directory.

## From ADR 0040 (addendum "2026-11-14: fill tier", moved verbatim)

Top-k and its doublings stop at 16 slots on a 48 GB card, but filling the free VRAM (about 26 slots per layer on Qwen3-235B) decoded 1.51x faster (ADR 0045, 20261114). `feasible_vram_tiers_kv` now also offers the largest slot count that fits the budget at the smallest ubatch and RAM tier, deduplicated, ascending. The builder campaign's "vram-large" anchor is that tier.

## From docs/m2-plan.md, section "Facts from the read-only checks" (moved verbatim)

  - At 77 RAM slots with prefetch, the global policies cut disk bytes per token compared with per-layer LRU: gfreq:0.5 reads 164 MB, glru 172 MB, LRU 266 MB. Synchronous misses fall from 47 to 27 MB.
  - On a 48 GB card the planner picks at most 16 slots. The measured winner was 26.
  - In `20261114` and `20261114b`, poll 50 at 26 threads on a 30.7 quota was throttled 92 to 95% but still won 1.16 to 1.22x, on one host. The default stays unchanged until a paired run settles it.

## From docs/m2-plan.md, section "Order" (moved verbatim)

   - the planner's own fill tier, last (it may OOM): 35 slots (38.4e9 B) by its estimate on a 49,140 MiB (51.5e9 B) card. Measured besides slots: 13.9e9 B at the exit run's ubatch 2048 (40,852 MiB peak with 28.97e9 B of slots), which matches the planner formula at ubatch 2048, so 35 slots fit only at ubatch 256 (estimate 50.5e9 B); candidates are fit-checked per ubatch (`builder.rs`), so serve would pick 35 slots at ubatch 256. Record its peak and prefill rate (the harness's 22.5 GB reserve gives 26 slots at any ubatch);
