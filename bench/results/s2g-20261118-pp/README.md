# Session s2g-20261118-pp (ADR 0077 box confirmation)

- Status: informational, not judged (no clean rule applied; 3 runs per model and round, one rep each). Same box and session as `../s2g-20261118/` (box 61328297, host 24953, 2x RTX 3090, driver 615.71.09, power limit = default 350 W), run right after its rehearsal. Script: `scripts/pp_pipeline_box.sh`.
- Runs: `tetraeder bench --engine dense|moe --plan <layer-2 plan> -p 2048 -n 0 --ubatch 512` with `prefill_micro_batches = 4` (m4, the planner's value) and the same plan with `= 1` (m1), and llama.cpp `llama-bench -sm layer -ub 512 -p 2048 -n 0`. Round 1 order m4, m1, llama; round 2 reversed. All runs exited 0 (JSON present).
- Throttle: `gpu-samples.csv` (1 Hz, both GPUs) recorded for the whole script; not judged per run. Sample bits: mostly 0x0, 0x4, 0x400 and 0x600; two samples carry 0x20 (0x424).

## pp2048 tok/s

| model | round | m4 | m1 | llama.cpp layer | m4 / llama | m1 / llama | m4 / m1 |
|---|---|---|---|---|---|---|---|
| 30B-A3B | 1 | 7283 | 5702 | 8401 | 0.867 | 0.679 | 1.277 |
| 30B-A3B | 2 | 7282 | 5582 | 8374 | 0.870 | 0.667 | 1.305 |
| 32B | 1 | 1725 | 1316 | 2045 | 0.844 | 0.644 | 1.311 |
| 32B | 2 | 1705 | 1310 | 2036 | 0.838 | 0.643 | 1.302 |

- Pipelining raises layer-2 prefill by 1.28 to 1.31x on both models; the gap to `-sm layer` goes from 0.64 to 0.68 down to 0.84 to 0.87 of its speed. It does not tie yet: llama.cpp layer stays 15 to 19% faster.
- The rehearsal's layer-2 measurement (30B 7201 pp, 32B 1692 pp, with M = 4 from the plan) agrees with the m4 runs here.
