ADR 0074 bar after the snapshot-pool fix (spares grow, frees in pieces): MET in both rounds, both with default settings (spares on). r1 TTFT median 0.94x (0.246 vs 0.261 s), p90 0.84x (0.631 vs 0.750 s); r2 median 0.93x (0.248 vs 0.266 s), p90 0.84x (0.630 vs 0.750 s). Save share of TTFT (bar 10%): 2.9% in both rounds (1.21 s of 42.1 s, 1.22 s of 42.1 s).

# Session ttft7-20261119

- Status: clean. Rule: ADR 0074 bar (median and p90 TTFT at most 1.15x llama-server, sum `save_ms` at most 10% of sum TTFT) plus the six checks of `ttft6-20261119` ("Next replay checks"), after `track-ttft-spares` (ADR 0074 History 2026-11-19: spares grow with the target, frees only over the budget, in 64 MiB pieces). Same replay as ttft4 to ttft6 (`scripts/ttft_diag.sh`, `scripts/ttft_join.py`), interleaved tet r1, llama r1, llama r2, tet r2. Request log sha256 f79b4a6d...5f7b35 verified on the box (105 requests, 95 streamed with status 200).
- Code: origin/main a94d6e2, branch box-ttft7-20261119. `scripts/ttft_diag.sh` does only two tetraeder rounds, so both ran with defaults (no `TET_R2_ENV`, no `TETRAEDER_SNAPSHOT_SPARES=0` baseline round). Env: `SESSION=ttft7-20261119`, `TETRAEDER_STEP_TRACE=1`.
- Box 61469527, host 642116 (Utah), 1x RTX 3090, driver 610.57.04, Intel Core i7-14700K (28 threads, no container quota; both engines ran `-t 13`), power limit 420 W = default, idle gate 10/10, host CPU idle over a 20 s `/proc/stat` sample (about 40 busy jiffies of 56000), `cuda: ok`. Throttle bits 0x0 before every run, 0x4 (power cap) after. Box age 1h59m at 0.281 USD/h (0.276 box + traffic), about 0.60 USD with traffic. A different host and CPU class than ttft5/ttft6 (EPYC, 34742): absolute times are lower (tetraeder median 0.25 s against 0.31 s), so compare only the paired ratios. Five earlier boxes were destroyed at the gates (see Deviations); all five together about 0.25 USD, so the session cost about 0.85 USD.
- Raw tetraeder logs (35 MB each) are not committed: `server-replay-q8b-tet-r*.admit.log` keep the admit, `tetraeder::trace` (steps, snapshot take/spare lines) and WARN/ERROR lines, ANSI stripped. `joined-r*.json` and `join-r*.txt` are the `ttft_join.py` output; `analysis.txt` is the output of the check script behind the numbers below (it read the raw logs).

## TTFT

Over the 95 streamed status-200 requests (median / p90, s):

| round | tetraeder | llama-server | ratio med / p90 |
|---|---|---|---|
| r1 | 0.246 / 0.631 | 0.261 / 0.750 | 0.94 / 0.84 |
| r2 | 0.248 / 0.630 | 0.266 / 0.750 | 0.93 / 0.84 |

Sum TTFT tetraeder 42.1 / 42.1 s against llama-server 51.4 / 51.7 s. Sum `save_ms` 1.21 s (r1) and 1.22 s (r2) over 3 saves per round (#25, #42, #102): 2.9% of sum TTFT. No restores, `queued_ms` 0.01 s. The two default rounds agree within 0.003 s (the pool behaves the same run to run: same spare events, same takes).

## Checks

1. **Spare frees.** 8 `snapshot spare freed` lines per round, 2.0 to 4.2 GB each, `pieces=` 31 to 63 (none at 1), 47 to 94 ms each (ttft6 r1: 32 lines, 0.33 to 1.3 s). They are the only spare events besides `touched`: no stale-buffer drops or target-triggered frees appear. Two of them fall inside the TTFT window of #102 (0.74 s and 0.81 s after admit) and one inside #42's (0.51 s); the steps there show no stall (below).
2. **#1 to #10 level with llama-server.** TTFT gap tet minus llama, r1: -0.002, -0.009, -0.045, -0.013, -0.013, -0.065, -0.034, -0.001, -0.010, -0.014 s; r2 the same within 0.003 s. ttft6 r1 had +0.72, +0.26, +0.15, +0.23 s on #1, #2, #3, #10. The +0.15 s stalls are gone.
3. **Takes.** Six `snapshot take` lines per round, three of them the admission saves:

   | save | need (GB) | cap of the spare (GB) | source | save_ms | ms per GB |
   |---|---|---|---|---|---|
   | #25 | 3.193 | 3.137 | `small` by 55 MB (373 positions; the sequence generated 736 tokens) | 349 | 109 |
   | #42 | 3.197 | 3.217 | `fit` | 341 | 107 |
   | #102 | 5.056 | 5.048 | `small` by 7.7 MB (52 positions; it generated 571 tokens) | 519 | 103 |

   Both `small` takes are within the generated tail. The other three takes (need 2.75, 2.37 and 3.65 GB, all `fit`, 8.4 s after admit #40, 2.6 s after #53, 10.0 s after #63) are not admission saves (not in `save_ms`; presumably make_room or preempt inside a step). ms per GB is 103 to 109 here against 170 to 200 in ttft5 (a faster host: i7-14700K). No `fallback` (`pool_fallback` 0 in all three admits). r2 is identical.
4. **`grow_ms`.** 97 `snapshot spare touched` lines per round: 10 full touches (from_cap 0, avg 677 ms, max 1184 ms; the first spares and the replacements after a free) and 87 grows that touch only the tail (avg 19 ms, max 87 ms). `grow_ms` (the realloc) is 0 in 80 of 97 lines in r1 and 7 ms at most (r2 8 ms): a few ms as hoped. ttft6 had 78 full touches of 0.7 to 1.0 s per GB.
5. **place_ms minus save_ms.** #102: place 519.4 ms, save 519.3 ms, difference 0.07 ms (r2 0.08 ms); #42: 341.6 vs 341.5 ms, 0.07 ms (r2 0.09 ms). ttft6 had 558 ms (#102 r1), 1102 ms (#102 r2) and 360 ms (#42 r2); the unattributed time was the slow save path and is gone. Where the rest of the TTFT goes (r1, from the step trace):
   - #102: TTFT 2.796 s (llama 3.859 s) = place 0.519 s + 6 prefill steps of 2048 tokens or fewer, sum `total_ms` 2241 ms (compute 2203 ms) + 35 ms between the steps and the first token. Steps are 355, 387, 422, 451 ms and so on, `minflt` 0, `plan_ms` 1 to 4, `reserve_ms` 0.
   - #42: TTFT 4.309 s (llama 5.222 s) = place 0.342 s + 8 steps, sum `total_ms` 3916 ms (compute 3805 ms) + 51 ms. Steps 457 to 555 ms.
   - So the time after the save is prefill compute (the 10.7k and 14.4k new tokens), as in llama-server, and tetraeder is 0.9 to 1.1 s ahead on both requests. The spare frees inside these windows (76 and 68 ms on #102, 70 ms on #42, run on the preparer thread) did not slow the steps.
6. **Requests more than 0.1 s behind llama-server.** None, in either round. The largest gap is +0.009 s (r1) and +0.008 s (r2); median gap -0.022 / -0.023 s, mean -0.098 / -0.101 s (the means are the slow llama-server misses #16, #45, #58 and the long saves). Slowest tetraeder requests r1 (TTFT s, llama): #42 4.31 (5.22), #102 2.80 (3.86), #25 2.64, #0 2.42, #58 2.29, #16 2.28, #45 1.81 (the last four misses are 0.2 to 3 s ahead of llama-server), then hits #6 0.76, #12 0.67, #33 0.63. See `join-r1.txt`.

## Reading

All of the ttft6 open items are closed on this host: no free overlaps an early prefill with a stall, saves cost about 105 ms/GB with no fallback and no hidden place time, grows take a few ms, and the save share is 2.9% (bar 10%). Median 0.93 to 0.94x and p90 0.84x are inside the 1.15x bar in both rounds. Caveat: this is another host and CPU class than ttft5/ttft6, and there was no spares-off baseline round, so the gain over ttft6 is shown by the check values, not by a paired spares-on/off run on this box.

## Deviations

- Two default rounds instead of default plus `TETRAEDER_SNAPSHOT_SPARES=0` (the script only does two tetraeder rounds, as briefed).
- Box choice: the cheap 3090 offers kept failing the gates, so five boxes were destroyed before this one: 61461677 (host 443829) and 61463709 (host 451840) stayed in `loading` over 10 minutes; 61465827 (host 93197, power limit 250 of 350 W) and 61467767 (host 601059, 300 of 350 W) and 61468289 (host 79869, 300 of 370 W) failed the power gate. This box (host 642116) is the sixth; the 4-GPU driver's boxes ran in parallel and were not touched.
- The session finished at box age 1h59m; each round took 22 to 28 minutes (build 3 min).
