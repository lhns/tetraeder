# Session ttft2-20261118

- Status: ADR 0074 bar MISSED in both rounds. Median TTFT meets it (0.99x and 1.00x llama-server); p90 is 1.55x (bar 1.15x) and the save share of TTFT is 14.3 / 14.9% (bar 10%).
- Rule: ADR 0074, Decision of 2026-11-18 (tetraeder reuse on; median and p90 TTFT each at most 1.15x llama-server's; sum of `save_ms` at most 10% of the sum of TTFT). Same replay as `ttft-20261118`, interleaved tet r1, llama r1, llama r2, tet r2 (`scripts/ttft_diag.sh`, `scripts/ttft_join.py`, both unchanged).
- Code and box: main 938f04c (branch box-ttft2-20261118), box 61298735, host 34742 (Utah; 456900 was not on offer), 1x RTX 3090, CPU quota 30 (21 last time), power limit 350 W = default, idle gate 10/10, llama.cpp baseline pin fc504f91. Request log sha256 f79b4a6d...5f7b35 (105 requests, 95 streamed with status 200). Throttle bits 0x0 before and 0x4 (SW power cap) after each run. Box 2h08m at 0.285 USD/h plus about 0.02 USD/GB traffic: about 0.65 USD.
- Raw files: this directory (`summary.txt`, `joined-tet-r*.json`, `replay-q8b-*.json`, `server-replay-q8b-tet-r*.admit.log`, llama-server logs). The raw tetraeder debug logs (35 MB each) are not committed.
- Join note: the admit line now has `restore_ms` between `save_ms` and `prompt`, which the unchanged `ttft_join.py` regex does not match. The join ran on scratch copies of the admit logs with ` restore_ms=...` stripped; `restore_ms` below is read from the committed `.admit.log` files.

## Findings

TTFT over the 95 streamed status-200 requests (median / p90, s):

| round | tetraeder | llama-server | ratio med / p90 |
|---|---|---|---|
| r1 | 0.327 / 1.386 | 0.332 / 0.893 | 0.99 / 1.55 |
| r2 | 0.329 / 1.327 | 0.328 / 0.857 | 1.00 / 1.55 |

(ttft-20261118: tetraeder 1.75 / 3.7 to 4.8 s against 0.48 / 0.98 s.)

Bar: median met in both rounds; p90 missed (1.55x); sum `save_ms` 9.62 s of 67.2 s TTFT (14.3%) in r1 and 9.74 s of 65.5 s (14.9%) in r2, missed (was 61 / 67%). Sum TTFT tetraeder 67.2 / 65.5 s against llama-server 74.5 / 71.9 s: in total tetraeder is faster, because its long-prefill misses beat llama-server's.

Saves: per round 106 admit lines for 105 requests: 104 live and 2 from RAM. 94 live admissions had `save_ms` 0 (skipped, or nothing to save: the log has no separate marker, and the `tetraeder.kv.prefix.save.skipped` counter is not in the fetched files); 10 saved (82 of 105 last time). The same 10 in both rounds, sequences of 8.9k to 21.2k rows, 0.48 to 2.4 s each, two over 1 s (#70 2.2 / 2.4 s, #102 1.2 s). Three are in-place hits with a small cut that still saved (`pool_fallback=1`): #70 (19,761 of 20,279 rows kept, cut 518) saved 2.2 / 2.4 s, TTFT 2.85 / 3.02 s against 0.32 s on llama-server; #69 and #71 saved 0.93 s each. The quarter rule does not explain those; the code path that still saves there is the open item. The other saves are misses (cut above a quarter), where the save is a part of a long prefill.

Restore: two admissions from RAM per round, #53 (place 455 / 492 ms) and the 24,321-token one (501 / 516 ms); `restore_ms` is the whole of their `place_ms`. All others 0.

Ten slowest tetraeder requests (r1; r2 has the same set in a slightly different order). Restore 0 for all.

| # | prompt | cached | new | place ms | save ms | TTFT | llama |
|---|---|---|---|---|---|---|---|
| 42 | 21576 | 7169 | 14407 | 921 | 921 | 6.15 | 7.39 |
| 45 | 8877 | 17 | 8860 | 1747 | 883 | 3.81 | 4.26 |
| 102 | 11225 | 491 | 10734 | 1157 | 1157 | 3.76 | 6.66 |
| 16 | 11231 | 491 | 10740 | 774 | 774 | 3.51 | 5.09 |
| 58 | 11306 | 491 | 10815 | 744 | 744 | 3.39 | 5.51 |
| 25 | 11300 | 492 | 10808 | 558 | 558 | 3.21 | 4.46 |
| 0 | 11246 | 0 | 11246 | 0 | 0 | 2.95 | 2.82 |
| 70 | 20279 | 19761 | 518 | 2534 | 2243 | 2.85 | 0.32 |
| 69 | 19761 | 19516 | 245 | 1208 | 927 | 1.39 | 0.21 |
| 71 | 21192 | 20279 | 913 | 930 | 930 | 1.39 | 0.49 |

Six of the ten are long-prefill misses where tetraeder beats llama-server even with the save; #70, #69 and #71 are the in-place hits that still saved.

Request #53 (23,088 prompt): not a miss this time. 22,242 cached from RAM (source ram, restore 455 / 492 ms), TTFT 1.29 / 1.25 s against 2.73 / 2.78 s on llama-server; the 7.3 s prefill of ttft-20261118 (17 cached) did not recur. Not investigated further.

p90 gap: it sits on the saved hits (#69 to #71, 1.4 to 3.0 s against 0.2 to 0.5 s). Also #15 (1.30 s against 0.42 s) and #1 (1.19 s against 0.17 s) in r1 with `save_ms` 0, which are not save related and came back to 0.75 s in r2.
