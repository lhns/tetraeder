# Session 20261117l

- Status: Phase 4 D2b, TTFT pairs pass (ADR 0065); this session's own 8B throughput pairs are not used (one unclean).
- Code and box: e0758c7, box 61086097, host 34742, RTX 3090, driver 595.
- Raw files: this directory.

## Findings

- Prefix hits with a few hundred new tokens still take seconds with reuse on (request #55: 23,684 prompt tokens, 23,492 cached, 192 new, TTFT 4.1 s); the 83 hits have a median of 1.15 s and a p90 of 2.7 s. Misses are about 1 s slower with reuse on than off (#53: 8.2 against 7.2 s). Prompts: median 17.6K, max 33.7K tokens; at about 147 KB per token (8B) a conversation is 2 to 5 GB of KV.
- The 10 requests excluded as "status mismatch" were HTTP 500s in the recording (logged 500, 200 on replay): a 33,586-token prompt rejected with "does not fit the KV cache", retried by Claude Code (under diagnosis, `track-kv-full`).

## From docs/m2-plan.md, section "D2a verdict of record" (moved verbatim)

- D2b (`20261117l`): prefix reuse cuts the 8B's TTFT on the Claude Code replay to 0.20 to 0.22 of reuse off (median) and 0.32 to 0.35 (p90). Phase 4's exit test passes (ADR 0065 addendum); Phase 5 is next.

## From ADR 0065 (addendum "the Phase 4 exit test passes", moved verbatim)

- Throughput: both counting models pass in `20261117k` (main 6b07795; addendum above).
- TTFT: `bench/results/20261117l/phase4/` (box 61086097, host 34742, RTX 3090, driver 595.71; code e0758c7, the same engine as 6b07795), Qwen3-8B, the Claude Code log recorded on that box (105 requests, `log_sha256` f79b4a6d...5f7b35, kept in `bench/private/`). Pair (on, off): median 1.218 against 6.059 s (ratio 0.201), p90 3.340 against 10.445 s (0.320); pair (off, on): median 1.287 against 5.954 s (0.216), p90 3.630 against 10.249 s (0.354); every ratio at most 1 minus the band (0.925). Both pairs clean by the sampler rule (0 disqualifying samples, share 0), errors 0 on both sides; 10 requests per run without status 200 in both are excluded as the Fields rule says. `ttft.pass` true.
- That session's own 8B throughput pairs are not used (one is unclean on a 0x4 before sample; the 8B verdict is `20261117k`'s). The informative llama-server TTFT run was skipped for time.
- `phase4_pass`: true, combining `20261117k` (q8b, q30b) and `20261117l` (ttft) as the addendum "models judged in separate sessions" allows. The milestone M2 continues with Phase 5.
