# Session 20261117l

- Status: Phase 4 D2b, TTFT pairs pass (ADR 0065); this session's own 8B throughput pairs are not used (one unclean).
- Code and box: e0758c7, box 61086097, host 34742, RTX 3090, driver 595.
- Raw files: this directory.

## Findings

- Prefix hits with a few hundred new tokens still take seconds with reuse on (request #55: 23,684 prompt tokens, 23,492 cached, 192 new, TTFT 4.1 s); the 83 hits have a median of 1.15 s and a p90 of 2.7 s. Misses are about 1 s slower with reuse on than off (#53: 8.2 against 7.2 s). Prompts: median 17.6K, max 33.7K tokens; at about 147 KB per token (8B) a conversation is 2 to 5 GB of KV.
- The 10 requests excluded as "status mismatch" were HTTP 500s in the recording (logged 500, 200 on replay): a 33,586-token prompt rejected with "does not fit the KV cache", retried by Claude Code (under diagnosis, `track-kv-full`).
