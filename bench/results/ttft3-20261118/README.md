# Session ttft3-20261118

- Status: ADR 0074 bar MISSED in both rounds. Median TTFT is met (0.79x and 0.80x llama-server); p90 is 1.23x and 1.27x (bar 1.15x); the save share of TTFT is 12.6 / 13.9% (bar 10%).
- Rule: ADR 0074 (no save of a small cut, no save of a cut that is only the sequence's own output; bar: median and p90 TTFT each at most 1.15x llama-server's, sum of `save_ms` at most 10% of the sum of TTFT). Same replay as `ttft2-20261118`, interleaved tet r1, llama r1, llama r2, tet r2 (`scripts/ttft_diag.sh`, `scripts/ttft_join.py`, both unchanged).
- Code and box: main 00e756c (branch box-ttft3-20261118), box 61320193, host 456900 (Quebec), 1x RTX 3090, CPU quota 21, power limit 350 W = default, idle gate 10/10, llama.cpp baseline pin fc504f91. Request log sha256 f79b4a6d...5f7b35 (105 requests, 95 streamed with status 200). Throttle bits 0x0 before and 0x20 after each run. Box 2h15m at 0.276 USD/h plus traffic: about 0.8 USD.
- Raw files: this directory (`summary.txt`, `joined-tet-r*.json`, `replay-q8b-*.json`, `server-replay-q8b-tet-r*.admit.log`, llama-server logs). The raw tetraeder debug logs (35 MB each) are not committed.
- This box was slower per request than 34742 (ttft2): llama-server median 0.48 s against 0.33 s. Only the paired ratios compare.

## Findings

TTFT over the 95 streamed status-200 requests (median / p90, s; p90 by `statistics.quantiles`, n=10):

| round | tetraeder | llama-server | ratio med / p90 |
|---|---|---|---|
| r1 | 0.382 / 1.275 | 0.486 / 1.035 | 0.79 / 1.23 |
| r2 | 0.387 / 1.312 | 0.481 / 1.033 | 0.80 / 1.27 |

(ttft2-20261118: 1.00 / 1.55x.)

Save share: sum `save_ms` 9.55 s of 76.0 s TTFT (12.6%) in r1, 10.76 s of 77.2 s (13.9%) in r2. Sum TTFT tetraeder 76.0 / 77.2 s against llama-server 81.4 / 80.4 s.

Saves and skips (same in both rounds; 105 admissions: 104 live, 1 from RAM): 74 skipped `own_output`, 3 skipped `small_cut`, 28 without a skip reason (6 saved, 22 had nothing to save). 6 saves taken (ttft2: 10): #16, #25, #58, #102 (common 491 of about 11.2k), #42 (7169 of 21576) and #45 (17 of 8877). All are misses where a long sequence evicted for another conversation is saved before the prefill. The in-place hits of ttft2 (#69 to #71, 0.9 to 2.4 s) no longer save.

Save times vary by run on this host: #102 1.8 s in r1 and 6.4 s in r2, #25 3.7 s in r1 and 0.36 s in r2.

Restore: #53 (23,088 prompt, 22,242 from RAM) restore 3.6 / 3.7 s (0.5 s on 34742), TTFT 4.5 / 4.6 s against 3.2 / 2.5 s on llama-server.

Ten slowest tetraeder requests (r1; skipped is `-` unless named):

| # | prompt | cached | new | place ms | save ms | restore ms | skipped | TTFT | llama |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 21576 | 7169 | 14407 | 2757 | 2621 | 0 | - | 8.04 | 7.37 |
| 25 | 11300 | 492 | 10808 | 3722 | 3722 | 0 | - | 6.51 | 4.52 |
| 102 | 11225 | 491 | 10734 | 2067 | 1838 | 0 | - | 4.87 | 5.96 |
| 53 | 23088 | 22242 | 846 | 3799 | 0 | 3613 | - | 4.49 | 3.22 |
| 58 | 11306 | 491 | 10815 | 479 | 479 | 0 | - | 3.39 | 5.10 |
| 16 | 11231 | 491 | 10740 | 439 | 439 | 0 | - | 3.36 | 4.69 |
| 0 | 11246 | 0 | 11246 | 1 | 0 | 0 | - | 2.98 | 2.99 |
| 45 | 8877 | 17 | 8860 | 689 | 452 | 0 | - | 2.93 | 4.19 |
| 81 | 28803 | 27717 | 1086 | 0 | 0 | 0 | own_output | 1.29 | 0.84 |
| 6 | 15751 | 13264 | 2487 | 0 | 0 | 0 | own_output | 1.26 | 1.03 |

r2 has the same set; #102 leads there (save 6384 ms, TTFT 9.54 s against 6.01 s).

Reading: the in-place hits now cost about what llama-server's do. The remaining save share is entirely the six saves of sequences evicted for another conversation, and the p90 gap is the RAM restore of #53 plus hits of 1.0 to 1.3 s against 0.8 to 1.0 s; these are the "Open" levers of ADR 0074 (save at idle, pinned restores).
