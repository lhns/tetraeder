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

## From ADR 0045 (addendum "tuner too slow for the 235B; thin margin at 23 threads", the `20261114` paragraph, moved verbatim)

`20261114` (box 60591327, host 60400 again, same CPU, card, quota 30.7, 26 threads both engines, `memory.max` 195 GB with the 91 GB balloon, download 300 MB/s; guards clean on every stage but `noiqk`): **same configuration still not settled; the builder with the free VRAM as expert slots is ahead of both KT runs.** Order builder, KT (`avx512_bf16`), builder at poll 50, builder, KT, builder at poll 50, then the diagnostic arms. Decode builder 5.04 and 5.33, KT 5.91 and 4.62 tok/s: builder over KT 0.85 and 1.15 adjacent (KT spread 22%, so the pairs disagree within it). Poll 50: 5.83 and 6.45, 1.16x and 1.21x the adjacent poll-0 builder with 92 to 95% of decode periods throttled, and 0.99x and 1.40x KT. `vramfill` (29.0 GB of expert slots against 10.97, peak 39.6 of 48 GB): 7.63 tok/s, 1.51x the first builder run, 1.29x KT's faster run and 1.65x its slower one; one run, no repeat. `noiqk` (iqk kernels off) 5.59, co-tenant guard failed (11.5 cores outside), no reading. The harness gap is 5.3 ms per 16-token window after the first (ADR 0060 fix holds). KT runs with `--kt-num-gpu-experts 0` because its GPU experts need HF weights (above), so on a GGUF setup it cannot use the free VRAM; the builder's 10.97 GB default was matched to llama.cpp's `-ot` for point 1, not to KT. Results `bench/results/20261114/` (routing trace not committed, 53 MB).

## From ADR 0050 (addendum 2026-11-14 "the replay on the real trace; no pool across layers", moved verbatim)

Trace from `20261114` (box 60591327, 256 decode tokens after the 32K prompt; local copy `runs/traces/`, not committed). Calibration gate passes: `lru` at VRAM 10, RAM 77, prefetch on, gives 1147 MB per token in the first 16-token window, the trace run's own `decode_windows[0].io.bytes_per_token` is 1146.9 MB.

Prefetch on, RAM 77 slots, MB per decode token (all 256; first 64):
- VRAM 10 (`bench/results/20261114/replay-235b-ram.json`): best per-layer `freq:0.5` 252; best across layers `gfreq:0.5` 164 (0.65).
The build threshold (at most half the bytes of the best per-layer policy) is not met at either VRAM size, so the pool across layers is not built. At VRAM 26 the gain left is about 6 MB of synchronous reads per token, about 2 ms at 3 GB/s. The first 16-token window reads 3 to 4x the later ones under every online policy, so the warm-up is the cost; the cheaper lever named above (tier contents at the switch) stays the next candidate. The VRAM `prefill` policy raises the VRAM pair share from 0.35 to 0.51 but doubles RAM-tier disk bytes (101 to 224 MB per token at `lru`); whether the GPU share pays for that is a box question (`main-overlap-poll0-hot` arm, `docs/m2-plan.md` D1).

## From ADR 0067 (addendum "2026-11-14: step 1 result, chunked", moved verbatim)

`bench/results/20261114/kv-layout/` (`sim.json`, `kv-gather.json`; branch `track-c6-kv`).
- Memory: paged needs 355 units of 256 rows for W, chunked 553: s = 35.8% (33.1% without idle prefixes), at least X.
- Reads: y_gather 0.184 and y_sparse 0.056 (256-row blocks, 8 threads; 0.136 and 0.100 at 64-row blocks), both above Y.
- y_mask: -0.172 with each layout at its own minimal pool, +0.087 with both at chunked's pool. The definition above does not fix the pool, but its concern is that scattered blocks make n_kv follow the pool's high-water mark, which is a comparison at one pool size; at its own smaller pool paged's n_kv is capped by the pool, so the memory saving would count a second time, in Y as well as in X. So y_mask is read at the common pool: +0.087, above Y.
- Outcome 4: **chunked**, no patch. C4 and C5 use chunked (`--kv-layout` default `chunked`); paged stays in `kvcache` for the simulator and tests.
- Caveats: the timings ran on an SSE-only dev VM under load from parallel builds (attention about 15 ms at L 2048, one run). A faster kernel makes gather overhead relatively larger, so the y values likely understate the cost on real hardware; the verdict does not depend on them being exact (all exceed Y). Revisit only if a later workload needs more than about 1.5x the sequences per VRAM that chunked gives.
