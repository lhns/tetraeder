ADR 0074 bar (TTFT median and p90 at most 1.15x llama-server): MET in both rounds. r1 median 0.98x (0.308 vs 0.313 s), p90 0.92x (0.759 vs 0.825 s); r2 (run with TETRAEDER_SNAPSHOT_SPARES=0) median 0.94x (0.300 vs 0.318 s), p90 0.90x (0.771 vs 0.859 s). Save share of TTFT (bar 10%): met in r1 (4.98 s of 54.0 s, 9.2%), MISSED in r2 (8.47 s of 57.7 s, 14.7%, spares off).

# Session ttft6-20261119

- Status: clean. Rule: ADR 0074 incl. the 2026-11-19 "no avoidable save or restore" section, plus the ADR 0079 checks from `ttft5-20261119` ("Long-context tail"). Same replay as ttft4/ttft5 (`scripts/ttft_diag.sh`, `scripts/ttft_join.py`), interleaved tet r1, llama r1, llama r2, tet r2. Request log sha256 f79b4a6d...5f7b35 (105 requests, 95 streamed with status 200).
- Code: origin/main 366d42a (ADR 0079, mask buffer reused, step trace), branch box-ttft6-20261119. `TETRAEDER_STEP_TRACE=1` in both tet rounds; tet r2 also had `TETRAEDER_SNAPSHOT_SPARES=0` (no spare snapshot buffers), r1 default.
- Box 61432082, host 34742 (Utah), 1x RTX 3090, driver 595.71.05, AMD EPYC 7B13, CPU quota 30, power limit 350 W = default, idle gate 10/10, `cuda: ok`, llama.cpp baseline fc504f91. Throttle bits 0x0 before every run, 0x4 (power cap) after. Box age 2h10m at 0.296 USD/h all-in, plus a first box (61430991, host 392144) destroyed after 8 min at the power gate (limit 360 W, default 420 W). Cost about 0.88 USD including traffic (estimate: the taxameter state does not list billed traffic after destroy).
- Raw tetraeder logs (35 MB each) are not committed: `server-replay-q8b-tet-r*.admit.log` keep the admit, `tetraeder::trace` (steps, snapshot spares freed/touched) and WARN/ERROR lines, ANSI stripped. `joined-r*.json` is the `ttft_join.py` output.

## TTFT

Over the 95 streamed status-200 requests (median / p90, s; `ttft_join.py`):

| round | tetraeder | llama-server | ratio med / p90 |
|---|---|---|---|
| r1 | 0.308 / 0.759 | 0.313 / 0.825 | 0.98 / 0.92 |
| r2 (spares off) | 0.300 / 0.771 | 0.318 / 0.859 | 0.94 / 0.90 |

Sum TTFT tetraeder 54.0 / 57.7 s against llama-server 69.8 / 70.7 s. Same host as ttft5 (r1 0.348 / 1.358 s, ratios 1.12 / 1.65): the p90 miss is gone. Sum `save_ms` 4.98 s (r1) and 8.47 s (r2) over 3 saves per round (#25, #42, #102); no restores, `queued_ms` 0.

Saves are slower than in ttft5 on the same host (r1 total 4.98 s against 2.02 s): #42 2.37 s (fallback buffer, `pool_fallback` 1), #102 1.76 s, #25 0.85 s; ttft5 had 0.62, 0.85, 0.56 s. Without spares (r2) they are slower again (#25 2.38 s, #102 3.85 s, #42 2.24 s) and #102 is 0.52 s behind llama-server. Not explained; the r1 save share (9.2%) is close to the bar.

## Check 1: ADR 0079 (one-input steps attend over their own run)

Decode ms per token, `(total_s - ttft_s) / (output_tokens - 1)`, tetraeder against llama-server, same round:

| requests | r1 ratio median / mean | r2 ratio median / mean | max TTFT gap tet - llama |
|---|---|---|---|
| #16 to #24 | 0.995 / 0.995 | 0.995 / 0.994 | 0.00 s / 0.01 s |
| #45 to #61 | 1.004 / 1.010 | 1.006 / 1.017 | 0.01 s / 0.02 s |
| all other requests | 0.997 median | 0.997 median | |

ttft5 had 1.34x to 1.45x for these two ranges and gaps up to 0.40 s. Now every request in both ranges is within 0.02 s of llama-server (most are 0.01 to 0.08 s ahead; #16, #45, #58 are 2.4 to 3.0 s ahead because of llama-server's slow miss prefill). Pass. One outlier: #53 decodes at 13.35 vs 11.48 ms/token (1.16x) in r1 and 13.71 vs 11.64 (1.18x) in r2, while its TTFT is 2.4 s ahead (the ttft4 miss).

## Check 2: stalls (#76 to #91, #1 to #15)

Per request: the `tetraeder::trace` steps inside its TTFT window, and the spare thread's events overlapping it.

- #76 to #91: no stalls in either round. Largest gap tet - llama is +0.00 s (r2 #91), most are 0.02 to 0.11 s ahead. Per request: host phases (total_ms - compute_ms) 6 to 47 ms, minflt 0, step masks up to 79.1 MiB (#89) and 33.1 MiB (#88), 32.7 MiB (#82) level. In r1 `snapshot spares freed` (0.5 to 1.0 s, 4 to 8 GB) overlapped most of these steps and still caused nothing. The six slow requests of ttft5 (masks over 32 MiB) are gone; with the mask buffer reused (minflt 0) the page-fault cause fits, and the spares thread is not a factor here.
- #1 to #15: r1 has four slow requests, all with a `snapshot spares freed` of 0.33 to 0.73 s ending inside the prefill: #1 gap +0.72 s (0.89 vs 0.17; one step, compute 640 ms for 313 tokens against about 120 ms normally, host 185 ms, minflt 1332; freed 6.9 GB, 730 ms), #2 +0.26 s (compute 310 ms, host 74 ms, minflt 470; freed 7.1 GB, 697 ms), #3 +0.15 s (compute normal 176 ms, host 225 ms, minflt 471; freed 5.5 GB, 534 ms), #10 +0.23 s (compute normal 87 ms, host 233 ms, minflt 474; freed 2.6 GB, 330 ms). The extra time is in compute_ms for #1 and #2 and in host phases (no `compute` growth) for #3 and #10, always with a few hundred minor faults. Frees that end after the step ended (#4, #5, #8, #13 to #15) and touches alone cost nothing.
- r2 (spares off, no `snapshot spare` events): the same four requests are level (#1 +0.01, #2 +0.01, #3 -0.05, #10 -0.02 s); no request in r2 is more than 0.10 s behind llama-server except the save of #102.
- Conclusion: the stall hypothesis holds for the first requests (a spare free overlapping a step costs 0.15 to 0.7 s, gone with spares off) and not for #76 to #91, whose stalls were the fresh mask buffer.

## Check 3: first_kind and the ten slowest

- `first_kind` is `thinking` for all 95 streamed requests in both engines and rounds; there is no tool-first request in this log, so the tool-call gap stays unmeasured by this replay.
- Requests more than 0.1 s behind llama-server: r1 #1 (+0.72), #2 (+0.26), #10 (+0.23), #3 (+0.15), all the spares-free overlaps above; r2 #102 (+0.52, save 3.85 s). Median gap -0.03 s, mean -0.17 / -0.14 s (r1 / r2).

Ten slowest tetraeder requests (TTFT s; llama-server same round):

| r1 # | prompt | new | source | attempts | save ms | TTFT | llama |
|---|---|---|---|---|---|---|---|
| 42 | 21576 | 14407 | live | 2 | 2368 | 6.73 | 7.14 |
| 102 | 11225 | 10734 | live | 2 | 1757 | 4.83 | 7.28 |
| 25 | 11300 | 10808 | copy | 1 | 854 | 3.34 | 4.58 |
| 0 | 11246 | 11246 | live | 1 | 0 | 2.70 | 2.86 |
| 58 | 11306 | 10815 | copy | 1 | 0 | 2.50 | 5.44 |
| 16 | 11231 | 10740 | copy | 1 | 0 | 2.46 | 4.86 |
| 45 | 8877 | 8860 | copy | 1 | 0 | 1.96 | 4.53 |
| 1 | 11559 | 313 | live | 1 | 0 | 0.89 | 0.17 |
| 6 | 15751 | 2487 | live | 1 | 0 | 0.88 | 0.91 |
| 12 | 19535 | 2043 | live | 1 | 0 | 0.76 | 0.83 |

| r2 # | prompt | new | source | attempts | save ms | TTFT | llama |
|---|---|---|---|---|---|---|---|
| 102 | 11225 | 10734 | live | 2 | 3847 | 7.46 | 6.94 |
| 42 | 21576 | 14407 | live | 2 | 2242 | 6.97 | 7.37 |
| 25 | 11300 | 10808 | copy | 1 | 2383 | 4.93 | 4.94 |
| 0 | 11246 | 11246 | live | 1 | 0 | 2.74 | 2.84 |
| 58 | 11306 | 10815 | copy | 1 | 0 | 2.55 | 5.60 |
| 16 | 11231 | 10740 | copy | 1 | 0 | 2.51 | 4.85 |
| 45 | 8877 | 8860 | copy | 1 | 0 | 1.98 | 4.88 |
| 6 | 15751 | 2487 | live | 1 | 0 | 0.87 | 0.91 |
| 33 | 17582 | 2033 | live | 1 | 0 | 0.79 | 0.81 |
| 12 | 19535 | 2043 | live | 1 | 0 | 0.77 | 0.86 |

## Reading

ADR 0079 closes cause A of ttft5 (decode 1.0x, gaps within 0.02 s) and the reused mask buffer closes the mask-sized stalls (B in #76 to #91); p90 is now below llama-server's. What remains: the saves (#42, #102, #25) are the slowest requests and run 2 to 3 times slower than in ttft5 on the same host, much slower without spares (r2), and a spare free overlapping an early prefill costs up to 0.7 s (r1 #1 to #10). Open for the ADR: why the saves slowed since ttft5, and whether the spare thread should avoid freeing while a step runs.

## Diagnosis of the spare frees and the slow saves (2026-11-19, from the committed files)

Snapshot sizes are positions times 147456 bytes (`kv_row_bytes` of the 8B; the touched spare of 1734344704 bytes is 11246 + 512 positions rounded to 2 MiB).

**Why spares are freed at all.** The pool's target is the longest prompt plus 512 tokens so far. On this replay it rises at most admissions, by 0.3 to 5%. The preparer freed every spare below the new target and touched replacements of the full size. In r1 that is 32 `snapshot spares freed` lines (2.6 to 8.3 GB, 0.33 to 1.3 s each) and 78 full touches (1.7 to 5.1 GB, 0.70 to 1.0 s/GB). Buffers that went stale while being touched were also dropped, without a trace line. There are 4 spares (`--parallel 4`) until the 8 GiB budget holds fewer. The frees that ended inside a prefill are the stalls of check 2. These fit the serving thread's own address-space calls (mmap, faults) waiting behind a long unmap, but the logs cannot tell that apart from contention in the page allocator.

**Why saves slowed.** The same requests saved the same sizes as in ttft5, so the cost is per byte (ms per GB of snapshot):

| save | positions | ttft5 r1 | ttft6 r1 | ttft6 r2 (spares off) |
|---|---|---|---|---|
| #25 | 21651 | 174 | 267 | 746 |
| #42 | 21682 | 192 | 741 | 701 |
| #102 | 34392 / 34285 | 168 | 348 | 761 |

- r2 gives this host's cost of fresh pages, 0.70 to 0.76 s/GB, the same as the preparer's touch rate. ttft5 r2 ran with spares on, so r2 is a baseline, not a regression.
- #42 r1 (fallback): the admission at #41 raised the target by 0.26%. The preparer dropped the 3.21 GB spare it had just touched (11:21:31.28), freed the ready one (31.95, 379 ms) and finished the replacement at 34.51. The save started at 34.17, so no spare was ready. In ttft5 the save also started 4.37 s after #41's admission and found a spare. The pool code is the same, so the difference is the preparer's speed on that day. That is inferred: ttft5 has no trace.
- #25 r1: it took the largest spare, 3.137 GB for a 3.193 GB snapshot (55 MB short), while a 3.14 GB stale spare was freed beside it (15.55 to 15.96 s; save 15.55 to 16.40).
- #102 r1: it took a 5.048 GB spare for 5.056 GB (7.7 MB short; the sequence generated 571 tokens, more than the 512 hinted), while a 5.06 GB touch ran beside it (29.77 to 34.85 s; save 30.33 to 32.08). Its read reserved with `Vec::reserve`, which doubles a too-small buffer's capacity (to about 10 GB, untouched).
- `place_ms - save_ms` is 558 ms (#102 r1), 1102 ms (#102 r2) and 360 ms (#42 r2), against 166 ms for #102 in ttft5. It is part of TTFT but not of `save_ms`, and the logs do not attribute it. Candidates are the inline free of an evicted entry (r2, pool off) and `seq_rm`.
- Pool counters also move between admissions with no logged save (r1: `pool_fallback` 1 to 2 between 11:23:30 and 11:23:42). These are saves from `make_room` or `preempt` inside a step (KV full). They are not in the admission `save_ms`, so the 10% bar does not count them.

**Change** (`track-ttft-spares`, ADR 0074 History 2026-11-19): spares grow with the target (realloc, touch only the tail), and frees happen only over the budget. They run on the preparer thread in 64 MiB pieces, including released buffers without room. Snapshot reads reserve exactly. The step trace logs each take's source and the preparer's grows, touches and frees.

**Next replay checks.**
- Spares on: no `snapshot spare freed` except budget frees (`pieces=` over 1).
- #1 to #10 level with r2.
- Every save's `snapshot take` is `fit`, or `small` by at most the generated tail, at about 170 to 200 ms/GB.
- `grow_ms` of the touches stays at a few ms.
- `place_ms - save_ms` of #102 is attributed.
