# Session ttft4-20261119

- Status: ADR 0074 bar MISSED, on p90 only. Median TTFT is met (r1 0.492 s vs 0.467 s = 1.05x; r2 0.495 s vs 0.492 s = 1.01x); p90 is 1.50x (1.547 vs 1.030 s) and 2.55x (2.657 vs 1.044 s) against the bar of 1.15x; the save share of TTFT is met: 2.75 s of 86.4 s (3.2%) in r1, 6.29 s of 104.2 s (6.0%) in r2 (bar 10%).
- Rule: ADR 0074 incl. the free-slot decision (a new conversation takes a free slot; restores through pinned staging). Same replay as `ttft3-20261118`, interleaved tet r1, llama r1, llama r2, tet r2 (`scripts/ttft_diag.sh`, `scripts/ttft_join.py`, both unchanged). Request log sha256 f79b4a6d...5f7b35 (105 requests, 95 streamed with status 200).
- Code and box: origin/main 9fdaa3e (branch box-ttft4-20261119), box 61353173, host 456900 (Quebec), 1x RTX 3090, CPU quota 21, power limit 350 W = default, idle gate 10/10, llama.cpp baseline pin fc504f91. Throttle bits 0x0 before every run, 0x20 after (0x24 once). Box 2h38m at 0.276 USD/h plus traffic: about 0.9 USD.
- The raw tetraeder debug logs (42 MB each) are not committed; `server-replay-q8b-tet-r*.admit.log` keep the admit and WARN/ERROR lines.

## Restore test

`gpu_restore_through_staging` (8B; the 30B skipped, not provisioned) passed: restore 78.6 MB, staged 13.17 ms (5.97 GB/s), direct 18.27 ms (4.30 GB/s), ratio 0.72; bytes identical (the test asserts it). Log: `gpu_restore_through_staging.log`.

## TTFT

Over the 95 streamed status-200 requests (median / p90, s):

| round | tetraeder | llama-server | ratio med / p90 |
|---|---|---|---|
| r1 | 0.492 / 1.547 | 0.467 / 1.030 | 1.05 / 1.50 |
| r2 | 0.495 / 2.657 | 0.492 / 1.044 | 1.01 / 2.55 |

(ttft3-20261118 on this host: 0.79 / 1.23 and 0.80 / 1.27; the median is now worse, 0.38 to 0.49 s.) Sum TTFT tetraeder 86.4 / 104.2 s against llama-server 82.8 / 84.7 s.

## Saves, skips, placements

Same in both rounds (105 admissions: 98 live, 4 copy, 3 ram): 75 skipped `own_output`, 4 `small_cut`, 26 without a skip reason.
- Saves taken: 2 per round, #24 (1217 ms r1, 4172 ms r2) and #53 (1532 / 2119 ms; reason shown `-` for #24, `own_output` for #53 but still saved with place_ms = save_ms). ttft3 had 6.
- Placements into a free slot (source=copy): 4 (#16, #25, #45, #58), place_ms about 1 ms, no save. These are the ttft3 saves of 0.4 to 3.7 s; their TTFT is now 4.4 to 5.7 s against 4.2 to 5.1 s on llama-server (prefill of 8.9k to 10.8k new tokens).
- Restores from RAM (3, through pinned staging): #39 413 / 445 ms, #42 402 / 412 ms, #102 604 / 634 ms. #53 no longer restores (it stays live). ttft3 restored #53 in 3.6 s.

## Ten slowest tetraeder requests

r1:

| # | prompt | cached | new | source | place ms | save ms | restore ms | skipped | TTFT | llama |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 21576 | 7169 | 14407 | ram | 402 | 0 | 402 | - | 6.38 | 7.09 |
| 102 | 11225 | 491 | 10734 | ram | 910 | 0 | 604 | - | 5.78 | 5.90 |
| 58 | 11306 | 491 | 10815 | copy | 1 | 0 | 0 | - | 5.70 | 5.08 |
| 16 | 11231 | 491 | 10740 | copy | 1 | 0 | 0 | - | 5.31 | 4.70 |
| 25 | 11300 | 492 | 10808 | copy | 1 | 0 | 0 | - | 5.02 | 4.50 |
| 45 | 8877 | 17 | 8860 | copy | 1 | 0 | 0 | - | 4.35 | 4.23 |
| 0 | 11246 | 0 | 11246 | live | 0 | 0 | 0 | - | 2.96 | 3.01 |
| 53 | 23088 | 22242 | 846 | live | 1532 | 1532 | 0 | own_output | 2.07 | 2.44 |
| 39 | 21138 | 20579 | 559 | ram | 592 | 0 | 413 | - | 1.69 | 0.49 |
| 24 | 17924 | 17478 | 446 | live | 1224 | 1217 | 0 | - | 1.55 | 0.39 |

r2:

| # | prompt | cached | new | source | place ms | save ms | restore ms | skipped | TTFT | llama |
|---|---|---|---|---|---|---|---|---|---|---|
| 102 | 11225 | 491 | 10734 | ram | 877 | 0 | 634 | - | 9.55 | 6.08 |
| 39 | 21138 | 20579 | 559 | ram | 586 | 0 | 445 | - | 8.89 | 0.49 |
| 42 | 21576 | 7169 | 14407 | ram | 645 | 0 | 412 | - | 8.73 | 7.26 |
| 58 | 11306 | 491 | 10815 | copy | 1 | 0 | 0 | - | 5.79 | 5.21 |
| 16 | 11231 | 491 | 10740 | copy | 1 | 0 | 0 | - | 5.48 | 4.77 |
| 25 | 11300 | 492 | 10808 | copy | 1 | 0 | 0 | - | 5.10 | 4.56 |
| 24 | 17924 | 17478 | 446 | live | 4180 | 4172 | 0 | - | 4.50 | 0.40 |
| 45 | 8877 | 17 | 8860 | copy | 1 | 0 | 0 | - | 4.42 | 4.28 |
| 0 | 11246 | 0 | 11246 | live | 0 | 0 | 0 | - | 2.97 | 3.03 |
| 53 | 23088 | 22242 | 846 | live | 2120 | 2119 | 0 | own_output | 2.66 | 2.51 |

## Reading

The save share is now within the bar and the miss prefill matches llama-server. What remains is the tail: a few requests whose TTFT is far above their place_ms plus a small prefill, e.g. #39 (new 559 tokens, restore 0.45 s, TTFT 8.89 s in r2 vs 1.69 s in r1) and #24 (a 1.2 to 4.2 s save of a plain live hit, not skipped), and queueing (sum queued_ms 3.2 s r1, 16.6 s r2, up to 2 s on #102). r2 is slower than r1 on the same code, so the p90 is also noisy on this host. Cause of the #39 and #24 outliers is not diagnosed here.

## Diagnosis (2026-11-19, from the admit lines; no new run)

The replay sends one request at a time, and no request queues behind another: every nonzero `queued_ms` (#39, #42, #102, the only ones above 1 ms) belongs to a request whose first placement attempt put it back in the queue. The admit line timed only the last attempt, so the first attempt's saves showed as queueing. Saves are counted by the pool counters (`pool_spare` plus `pool_fallback` rise by one per save). The pool is 40960 rows, one run per sequence (ADR 0067), and two conversations of about 20k tokens each are resident from #16 on.

| # | what happened | evidence |
|---|---|---|
| 24 | Not its own save: its cut is 446 tokens, under the 512 minimum, hence no skip reason. Admission reserved prompt + 1024 rows; seq 1's run could not grow that far (seq 0's run before it, the pool end after it), so the other conversation (seq 0, about 21.6k positions) was evicted and saved: 1217 / 4172 ms. The prompt alone fit. | pool counter +1, `save_ms` = `place_ms`, `queued_ms` 0.1 |
| 53 | Same: `own_output` is seq 0's cut, skipped; the save is the eviction of seq 1 (the conversation of #45 to #52, about 15.5k positions) for the stretch: 1532 / 2119 ms, through a fallback buffer (`pool_fallback` +1). | as #24 |
| 39 | Seq 0's run sat behind seq 1's and could not grow past the pool end. Admission evicted seq 1 (save), still could not grow seq 0 where it lay, freed seq 0 itself (second save) and placed the request again from RAM (restore 413 / 445 ms). Both saves are in `queued_ms`: 732 ms in r1, 7934 ms in r2. Prefill of 559 tokens is the rest. | `pool_spare` +2, `queued_ms` with no request ahead, `source=ram` with the idle sequence's own prefix |
| 42, 102 | A new conversation went to the free slot (copy of 7169 / 491 tokens from the idle sequence), but the free slot's run fit only if that idle sequence was evicted: it was saved, the copy then failed (`cannot copy a busy sequence: unknown sequence 0` / `1`, the WARN lines just before them), and the request was placed again, restoring what had just been saved. Save in `queued_ms` (391 / 2747 ms for #42, 2034 / 5892 ms for #102), restore 402 to 634 ms. Reusing the idle sequence in place costs the save only. | WARN before each, pool counter +1, `source=ram` |
| 61 (TPOT) | One save during its decode (pool counter +1 at #62): the run grew into the other conversation and evicted it. Not on the TTFT path. | |

Why r2 is slower than r1 on the same requests: the same saves took 1.4 to 3.4x longer (#24 1217 vs 4172 ms, #39 732 vs 7934 ms for two saves, #102 2034 vs 5892 ms); restores and prefill did not change. Not explained by these logs. The rest of r2's p90 follows from those saves.

All the outlier saves above were avoidable on the admission path; the fixes and their rules are in ADR 0074 (2026-11-19 section, after ttft4), on branch `track-ttft-outliers`. A GPU re-run of this replay judges them.
