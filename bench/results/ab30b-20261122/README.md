# Session ab30b-20261122

- Status: no regression evidence. The median B/A ratio over three clean rounds is 0.992 for c4 and 0.992 for c1, both at least 0.98; final main (05d43b9 code) is within 1% of the Phase 4 verdict build (6b07795) on the same box.
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

## Results

- Run: box 61890814, host 45726 (Japan), RTX 3090, driver 595.84, power 390 W at default (gate ok), CPU AMD Ryzen 9 5950X (16 cores, 32 host CPUs, cgroup quota 15), 125 GB RAM. Gates at SSH: `cuda: ok`, power gate, idle gate 10/10, co-tenant sample (0.35 busy cores outside the run). `THREADS=15` (the quota; see Deviations). Code: branch `box-ab30b-20261122` at 05d43b9 (B; `scripts/ab_moe_box.sh` is the only difference to final main 922031c), A built from `git archive 6b07795`.
- Files: `ab/ab-summary.json` (rounds, medians, reading), `ab/ab-<A|B>-<c4|c1>-r<n>.json`, `ab/env.json`, throttle snapshots `ab/gpu-*.txt`, `ab/gpu-samples.csv`, server and build logs.
- All 12 runs clean: errors 0, no co-tenant flag, no sampler sample with a disqualifying bit, board-limit share 0.00 on every run (15 window samples each). No round excluded.

| round | order | c4 A tok/s | c4 B tok/s | c4 B/A | c1 A tok/s | c1 B tok/s | c1 B/A | clean |
|---|---|---|---|---|---|---|---|---|
| 1 | A B | 301.3 | 301.1 | 0.999 | 189.7 | 187.5 | 0.988 | yes |
| 2 | B A | 303.4 | 300.9 | 0.992 | 188.9 | 187.4 | 0.992 | yes |
| 3 | A B | 304.8 | 301.6 | 0.990 | 188.7 | 187.3 | 0.992 | yes |
| median | | | | 0.992 | | | 0.992 | |

- c4 is `throughput_tps`, c1 is `decode_tps_median`. Reading by the fixed rule: both medians at least 0.98, so no regression evidence. All nine ratios are below 1 (0.988 to 0.999), a small consistent B deficit of about 1% that the rule treats as noise-level; it is far from the 0.97 line.
- The earlier gap (267 tok/s on `d2a30b-20261122` against 288 to 291 on the Phase 4 machine) is a machine effect: on this box both builds give 301 to 305 tok/s at c4. The `d2a-30b` null stands; this session judges nothing against llama-server.

## Deviations

- Offer 57767359 vanished between scan and `create --offer` (three searches); the box was created with `create --auto --min-cpu-cores 16 --min-inet 1000` and the skip list in `--exclude-hosts` (offer 48414826 at host 45726, 16 cores, 0.307 USD/h plus traffic). It stayed in `loading` for about 10 min and came up at the stall limit, so it was kept.
- `THREADS=15`, not 23: the quota here is 15 and AGENTS.md requires threads within the quota. Both sides ran 15 threads (server logs).
- Only the 30B model was provisioned (`--models Qwen/Qwen3-30B-A3B-GGUF:Qwen3-30B-A3B-Q4_K_M.gguf`); the A source went to the box as a tarball with `taxameter put` (box has no history).
- Reading rule: board-limit share and the before/after snapshots are reported, not judged (as written in the criteria). Each run's bench window is about 6 s per test, short because the model was in the page cache and servers started warm; the three rounds are the repetition.
- Box life 30 min, credit 31.39 to 31.17 USD (0.22 USD), destroyed after a verified fetch (`results-61890814-20261122-031225.tgz`); `status` shows no instance.
