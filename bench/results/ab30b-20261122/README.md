# Session ab30b-20261122

- Status: not run yet (criteria written before measuring).
- Purpose: a same-box A/B of the own Qwen3-30B-A3B MoE engine (`tetraeder serve --engine moe`), to see whether final main regressed against the Phase 4 verdict build. Background: `d2a30b-20261122` (probe c4 0.910 under the 0.925 line; own engine 267 tok/s against 288 to 291 on the Phase 4 machine) and `20261117k`. Node `bench-ab30b`. Diagnostic only: nothing is judged against llama-server and the D2a null (`bench-d2a-30b`) does not change.
- A: Phase 4 verdict commit 6b07795, built on the box from a `git archive` tarball in its own checkout (own target dir, its own patch queue). B: final main, 922031c (the branch `box-ab30b-20261122` head before the results commit), the checkout taxameter pushes. Both built release with CUDA; the same server binary flags.
- Code: `scripts/ab_moe_box.sh` (server launch, warm-up, bench calls and sampler logging as the tetraeder-moe side of `scripts/phase4_box_d2a.sh`), `scripts/ab_moe_summary.py`.

## Criteria (fixed now, before any box)

- Workload per run, one fresh server process per run: `--engine moe --ngl -1 --parallel 4 --ctx 32768 -t 23 --accept-any-model` (THREADS=23, as D2a), one warm-up request, then c4 `--prompt-tokens 512 --max-tokens 128 --concurrency 4 --requests 16`, then c1 the same with `--concurrency 1 --requests 8`. Seeds per round (c4 round*10+4, c1 round*10+1), the same on A and B.
- Order: three rounds, A B, B A, A B. A round is one run per side.
- Gating fields: c4 `throughput_tps` and c1 `decode_tps_median`; round ratio = B / A.
- A run is clean when errors are 0 in both tests, no co-tenant flag, and the 1 Hz sampler shows no sample in the run's windows with a bit outside 0x1, 0x2, 0x4, 0x100, 0x200, 0x400 (so no 0x20 thermal, 0x8, 0x40, 0x80; ADR 0065 sampler rule). The board-limit share (0x200 or 0x400) and the before and after throttle snapshots are reported per run, not judged. A round is clean when both of its runs are. Unclean rounds are listed with the reason and excluded from the median.
- Reading, by the median B/A over the clean rounds, for c4 and c1 separately:
  - no regression evidence: the median is at least 0.98 for both c4 and c1;
  - regression: the median is below 0.97 for either, with every clean round of that metric below 1 (all on the same side of 1);
  - inconclusive: anything else, and always when fewer than 2 rounds are clean.
- Machine: RTX 3090 at the default power limit, gates `cuda: ok`, power, idle, co-tenant; any CPU class (the same box serves A and B), at least 16 cores preferred; cap 0.90 USD/h all-in, skip list per `docs/box-driver.md`. CPU model, host CPUs and quota are recorded.
- Not judged: any ratio against llama-server, absolute tok/s across boxes (shown for orientation only).
