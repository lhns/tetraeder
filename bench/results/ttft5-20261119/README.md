# Session ttft5-20261119

- Status: ADR 0074 bar (section "no avoidable save or restore on the admission path") MISSED, on p90 only. Median TTFT 1.12x (r1 0.348 vs 0.312 s) and 1.07x (r2 0.332 vs 0.310 s) of llama-server: met (bar 1.15x). p90 1.65x (1.358 vs 0.823 s) and 1.62x (1.326 vs 0.820 s): missed (bar 1.15x). Save share of TTFT met: 2.02 s of 69.7 s (2.9%) in r1, 2.21 s of 67.5 s (3.3%) in r2 (bar 10%). The KV slide GPU tests passed (13 of 13).
- Rule: ADR 0074 incl. the 2026-11-19 "no avoidable save or restore" section; same replay as ttft4 (`scripts/ttft_diag.sh`, `scripts/ttft_join.py`, both unchanged), interleaved tet r1, llama r1, llama r2, tet r2. Request log sha256 f79b4a6d...5f7b35 (105 requests, 95 streamed with status 200).
- Code and box: origin/main c528d32 (branch box-ttft5-20261119), box 61386070, host 34742 (Utah), 1x RTX 3090, AMD EPYC 7B13, CPU quota 30.7, power limit 350 W = default, idle gate 10/10, llama.cpp baseline pin fc504f91. Throttle bits 0x0 before every run, 0x4 (power cap) after. Box 2h15m at 0.285 USD/h plus traffic: about 0.75 USD.
- The raw tetraeder debug logs (42 MB each) are not committed; `server-replay-q8b-tet-r*.admit.log` keep the admit and WARN/ERROR lines (the only WARN is the model-load token warning).

## GPU slide tests

`TETRAEDER_REQUIRE_GPU=1 cargo test --release -p tetraeder-model --features tetraeder-ggml-sys/cuda --test gpu_multi_seq --test gpu_dense -- --nocapture` (log `gpu-slide-tests.log`): gpu_dense 7 passed (incl. `gpu_dense_slide_keeps_the_bytes`, `cpu_dense_slide_keeps_the_bytes`, `gpu_qwen3_8b_matches_libllama`), gpu_multi_seq 6 passed (incl. `gpu_seq_cp_snapshot_and_move`); 0 failed.

## TTFT

Over the 95 streamed status-200 requests (median / p90, s):

| round | tetraeder | llama-server | ratio med / p90 |
|---|---|---|---|
| r1 | 0.348 / 1.358 | 0.312 / 0.823 | 1.12 / 1.65 |
| r2 | 0.332 / 1.326 | 0.310 / 0.820 | 1.07 / 1.62 |

Sum TTFT tetraeder 69.7 / 67.5 s against llama-server 65.6 / 65.3 s. ttft4 on host 456900: 0.492 / 1.547 and 0.495 / 2.657 (llama-server 0.467 / 1.030); this host is faster for both engines, so compare ratios only. The r2 slowdown of ttft4 is gone here: r1 and r2 agree within 0.03 s at the p90.

## Checks from ttft4

- #24, #53, #39: `attempts=1`, no `saved=`, `save_ms` 0, `restore_ms` 0, source live; TTFT 0.41 / 0.39 s (#24), 0.46 / 0.48 s (#53), 0.30 / 0.31 s (#39) in r1 / r2 (ttft4: 1.55 / 4.50, 2.07 / 2.66, 1.69 / 8.89 s). Pass.
- #42, #102: `source=live`, `attempts=2`, exactly one `saved=` each (`2:21682`, `0:34392`), `restore_ms` 0, no `cannot copy` WARN (none in either log). Pass. #42 place 615 / 600 ms; #102 place 1016 / 1202 ms (save 850 / 1037 ms through a fallback buffer, `pool_fallback` 1).
- Saves overall: 3 per round (#25, a copy into a free slot that saved `0:21651`, 556 / 570 ms; #42; #102), the same in both rounds; 103 of 105 admissions with `attempts=1`. Save times r1 vs r2: 2.02 vs 2.21 s in total (ttft4: up to 11x slower in r2).
- Sources: 91 live, 4 copy (#16, #25, #45, #58), 0 ram in both rounds. No restore at all.

## Ten slowest tetraeder requests

r1:

| # | prompt | cached | new | source | attempts | saved | place ms | save ms | restore ms | TTFT | llama |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 21576 | 7169 | 14407 | live | 2 | 2:21682 | 615 | 615 | 0 | 5.52 | 6.67 |
| 58 | 11306 | 491 | 10815 | copy | 1 | - | 0.8 | 0 | 0 | 5.06 | 5.17 |
| 16 | 11231 | 491 | 10740 | copy | 1 | - | 0.8 | 0 | 0 | 4.69 | 4.59 |
| 45 | 8877 | 17 | 8860 | copy | 1 | - | 1.0 | 0 | 0 | 3.91 | 4.26 |
| 102 | 11225 | 491 | 10734 | live | 2 | 0:34392 | 1016 | 850 | 0 | 3.88 | 5.90 |
| 25 | 11300 | 492 | 10808 | copy | 1 | 0:21651 | 557 | 556 | 0 | 3.20 | 4.32 |
| 0 | 11246 | 0 | 11246 | live | 1 | - | 0.4 | 0 | 0 | 2.83 | 2.82 |
| 76 | 25298 | 23861 | 1437 | live | 1 | - | 0.1 | 0 | 0 | 1.46 | 0.74 |
| 89 | 33120 | 31874 | 1246 | live | 1 | - | 0.1 | 0 | 0 | 1.43 | 0.74 |
| 81 | 28803 | 27717 | 1086 | live | 1 | - | 0.1 | 0 | 0 | 1.36 | 0.60 |

r2:

| # | prompt | cached | new | source | attempts | saved | place ms | save ms | restore ms | TTFT | llama |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 21576 | 7169 | 14407 | live | 2 | 2:21682 | 600 | 600 | 0 | 5.45 | 6.83 |
| 58 | 11306 | 491 | 10815 | copy | 1 | - | 0.7 | 0 | 0 | 5.05 | 5.13 |
| 16 | 11231 | 491 | 10740 | copy | 1 | - | 0.7 | 0 | 0 | 4.75 | 4.52 |
| 45 | 8877 | 17 | 8860 | copy | 1 | - | 0.8 | 0 | 0 | 3.96 | 4.07 |
| 102 | 11225 | 491 | 10734 | live | 2 | 0:34392 | 1202 | 1037 | 0 | 3.84 | 5.91 |
| 25 | 11300 | 492 | 10808 | copy | 1 | 0:21651 | 570 | 570 | 0 | 3.20 | 4.23 |
| 0 | 11246 | 0 | 11246 | live | 1 | - | 0.5 | 0 | 0 | 2.79 | 2.79 |
| 81 | 28803 | 27717 | 1086 | live | 1 | - | 0.1 | 0 | 0 | 1.42 | 0.60 |
| 12 | 19535 | 17492 | 2043 | live | 1 | - | 0.1 | 0 | 0 | 1.34 | 0.82 |
| 76 | 25298 | 23861 | 1437 | live | 1 | - | 0.1 | 0 | 0 | 1.33 | 0.72 |

## Reading

The admission-path fixes work as predicted: no restores, no avoidable saves, no queueing (sum `queued_ms` 0.00 s), and #24, #53 and #39 are back to cache-hit TTFT. The big misses (#58, #16, #45, #0, #42, #102, #25) are at or below llama-server. The p90 miss is the tail the ttft4 README predicted: long-context hits with 0.5k to 2.5k new tokens (#76, #89, #81, #6, #21, #78, #12, #33, #88, #14, #82) with `place_ms` under 1 ms and no save, TTFT 0.8 to 1.5 s against llama-server's 0.25 to 0.9 s, about 0.4 to 0.7 s behind. That is prefill speed at 13k to 33k positions, not placement. Median is within the bar.

## Long-context tail (diagnosis, 2026-11-19, from the committed files)

Method: llama-server's own prompt-eval time per request (`print_timing` in `server-replay-q8b-llama-r1.log`) stands in for the prefill time, and the decode ms per token (`(total_s - ttft_s) / (output_tokens - 1)` for tetraeder, `eval time` for llama-server) measures the attention width, since decode is level wherever the two engines attend over the same rows. Gap = tetraeder TTFT - llama prefill - 0.09 s (llama-server's own TTFT minus its prefill is 0.07 to 0.11 s on every hit but #53). Misses (more than 3000 new tokens) are left out.

The tail is not prefill speed at long context: #33 (15.5k cached + 2033 new), #64 (14.6k + 1397), #68 (18.1k + 1400) and #74 (22.2k + 995) are level. Two separate causes:

- **A. Run offset (code-certain).** A sequence whose chunked run sits above another sequence's attends over rows `0..n_kv` including the other run (`KvPool::step_n_kv`, `attn::attention_seqs_out` `kv_view(0, n_kv)`); libllama with `-kvu` and flash attention reads only the sequence's own cells (`fused_kv_rows`: llama-context.cpp 384-390 and 710-716, llama-graph.cpp 102-130 and 3055-3066, llama-kv-cache.cpp 1391-1420 at fc504f91). Seq 1 (#16 to #24, above seq 0's 22k rows) and seq 0 (#45 to #61, above seq 2) decode at 1.34x to 1.45x llama-server's ms per token, every other phase at 0.97 to 1.03x; their gaps grow with the new tokens (#17 220 new 0.05 s, #18 679 0.21 s, #19 1579 0.31 s, #21 2046 0.40 s) and repeat within 0.03 s between r1 and r2. 21 requests, 3.2 s of TTFT per round.
- **B. Episodic stalls of 0.45 to 0.85 s (mechanism open).** In #76 to #91 (seq 0 alone at the bottom, decode 1.00x) the six slow requests are exactly the six with a step mask over 32 MiB (#88 33.1, #82 32.7 MiB slow; #87 28.9, #84 28.1 MiB level). In #62 to #75 masks up to 52.6 MiB are level. What differs: in #76 to #91 (and #1 to #15) every admission raises the snapshot pool's target (`expect_snapshot`, the longest prompt so far), so the spare thread frees the outgrown multi-GB spares and touches new ones while the prefill runs; in #62 to #75 it does not. A fresh per-step mask above glibc's 32 MiB mmap threshold page-faults on every page, and the stall varies between rounds (#14 0.62 / 0.00 s, #89 0.71 / 0.54 s), as host contention would. 14 (r1) and 11 (r2) requests, 7.0 and 4.9 s.

| without | median ratio | p90 ratio (this README's p90 index) |
|---|---|---|
| nothing (measured) | 1.12 / 1.07 | 1.72 / 1.63 |
| A | 1.03 / 1.02 | 1.72 / 1.63 |
| B | 1.04 / 1.02 | 1.05 / 1.06 |
| A and B | 1.02 / 1.01 | 0.97 / 0.98 |

(r1 / r2, removing each cause's gap from the affected requests.) B sets the p90, A the median and the mean.

Tool calls: the gateway sends a tool call whole (`anthropic.rs`, `OutputParser::drain_block`), llama-server streams its arguments (`input_json_delta`, server-task.cpp 1464 at fc504f91), and the replay counts `partial_json` as first content. A response that starts with a tool call therefore shows its whole call in our TTFT. It cannot be the B pattern (it does not depend on mask size or on the target rising), but its share is unmeasured: the replay now records `first_kind`.

Changes on `track-longctx-prefill`: one-input steps attend over their own run (A, ADR 0066 History); the pool reuses its mask buffer (B candidate); `TETRAEDER_STEP_TRACE=1` traces each prefill step (reserve and moves, plan, state, upload, build, compute, read, minor faults) and the spare thread's frees and touches.
