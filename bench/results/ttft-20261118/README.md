# Session ttft-20261118

- Status: diagnosis, no verdict. TTFT on the Claude Code replay, Qwen3-8B, tetraeder reuse on against llama-server, interleaved (tet r1, llama r1, llama r2, tet r2); throttle 0x4 most of the time on both engines, one 0x24 sample.
- Code and box: ab337db-era main (branch box-ttft-20261118, 1d47f37), box 61269408, host 456900, 1x RTX 3090, CPU quota 21; request log sha256 f79b4a6d...5f7b35 (105 requests).
- Raw files: this directory (`summary.txt`, `joined-tet-r*.json`, `server-replay-q8b-tet-r*.admit.log`).

## Findings

- TTFT over the 95 streamed requests with status 200 (median / p90): tetraeder 1.747 / 3.684 s (r1), 1.750 / 4.820 s (r2); llama-server 0.475 / 0.988 s (r1), 0.477 / 0.974 s (r2).
- Sum of `save_ms` 117.6 / 151.5 s against a TTFT sum of 191.4 / 227.0 s (61 / 67%); `place_ms` is almost all `save_ms`; `queued_ms` 0.
- Every admission was source `live` (one slot at concurrency 1): no restore ran. 82 of 105 admissions saved the idle sequence before reusing it in place; 46 / 48 saves took over 1 s, up to 5.0 / 7.8 s; `pool_fallback` ended at 20 / 17; estimated save rate median 2.4 / 2.2 GB/s.
- On hits, TTFT minus `place_ms` is about 0.3 s, the same as llama-server (0.34 to 0.8 s). Example #55: 23,684 prompt, 23,492 cached, save 3.05 s, TTFT 3.57 s, llama-server 0.36 s.
- Misses: prefill matches or beats llama-server, plus the save in front (0.4 to 2.4 s); exception #53 (23,088 prompt, 17 cached): prefill 7.3 s against 2.4 s on llama-server in both rounds, not investigated.
