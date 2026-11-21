# Cost model 5 offline check (ADR 0082, "The check")

Version 4 and version 5 predictions against the measured candidates of every committed session, from the replica
(`scripts/costmodel_replica.py`, `scripts/costmodel_v5_check.py`). One fit per session, the same in both versions (the
layer anchor is held out and no session has it yet). Clocks of the 4-GPU sessions s3, s3b and s3c: `decode-clocks-host20210.json`
(busy 1905 for both models, low 1695; from s3e part C by `scripts/decode_clocks.py --style diag`, the 32B's 1905 is the
30B's, an upper bound). s3f2: 1590 and 1590 (its own `gpu-samples.csv`). s2f, s2g and v2-20261120 (branch `box-v2-20261120`,
2 GPUs): none. `e_pp` and `e_tg` are `predicted / measured - 1`; `moved` is the largest relative change of pp or tg from v4 to v5.

Reproduce: `python scripts/costmodel_v5_check.py --v2-dir <bench/results/v2-20261120/phase5 of box-v2-20261120>`.
Files: `results.json` (every row and residual), `fit-s3b.json` (the version-5 graph and rows of s3b, which
`crates/planner/tests/costmodel_v2.rs` predicts in Rust and compares at 1e-9).

## Items of the check

1. 30B layer-4 on host 20210, predicted layer-4 / single tg against measured: s3 0.902 against 0.930 (-0.028), s3b 0.903
   against 0.930 (-0.027), s3c 0.930 against 0.910 (+0.020); all within 0.03. `e_tg` +0.001, -0.003, +0.067 (v4 +0.093, +0.089,
   +0.167); all within 0.07. Pass, equal to the ADR's prototype.
2. s3f2 with its own clocks: `busy = low = 1590`, k = 1, every prediction bit-identical to v4 (`moved` 0). Out of sample, 30B
   layer-4 truncated to 36 layers: measured 231.8, v4 and v5 230.9 (-0.004). Pass.
3. 32B layer-4 `e_tg` moves by 0.011, 0.011, 0.010 (s3, s3b, s3c); every other candidate of s2f, s2g, s3, s3b, s3c, s3f2 and
   v2-20261120 moves by 0 in `e_pp` and `e_tg` (`moved` is 0.0000 in the table). Pass.
4. Rust and replica: `predict_reproduces_the_v5_replica_on_s3b` (CI) compares Rust with the replica's version-5 s3b rows at 1e-9.
5. Box confirmation (not part of done): the next 4-GPU session on a host where single boosts, with `LAYER_ANCHOR=1`.

## v4 and v5 per candidate

| session | model | candidate | pp meas | tg meas | e_pp v4 | e_pp v5 | e_tg v4 | e_tg v5 | moved |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| s2f | 30B | single | 4813 | 177.0 | -0.043 | -0.043 | +0.007 | +0.007 | 0.0000 |
| s2f | 30B | layer-2 | 5230 | 198.4 | -0.122 | -0.122 | -0.105 | -0.105 | 0.0000 |
| s2f | 30B | tp-2 | 4316 | 181.3 | -0.036 | -0.036 | +0.010 | +0.010 | 0.0000 |
| s2f | 30B | ep-2 | 4502 | 174.9 | -0.003 | -0.003 | +0.058 | +0.058 | 0.0000 |
| s2f | 32B | single | 1084 | 27.7 | -0.002 | -0.002 | +0.086 | +0.086 | 0.0000 |
| s2f | 32B | layer-2 | 1107 | 38.6 | -0.026 | -0.026 | -0.222 | -0.222 | 0.0000 |
| s2f | 32B | tp-2 | 1322 | 53.7 | +0.009 | +0.009 | -0.071 | -0.071 | 0.0000 |
| s2g | 30B | single | 5547 | 194.0 | -0.085 | -0.085 | +0.018 | +0.018 | 0.0000 |
| s2g | 30B | layer-2 | 7215 | 199.8 | +0.021 | +0.021 | -0.016 | -0.016 | 0.0000 |
| s2g | 30B | tp-2 | 3854 | 185.3 | +0.006 | +0.006 | -0.022 | -0.022 | 0.0000 |
| s2g | 30B | ep-2 | 4009 | 180.1 | +0.039 | +0.039 | +0.014 | +0.014 | 0.0000 |
| s2g | 32B | single | 1269 | 38.0 | -0.018 | -0.018 | +0.005 | +0.005 | 0.0000 |
| s2g | 32B | layer-2 | 1692 | 39.6 | +0.144 | +0.144 | -0.037 | -0.037 | 0.0000 |
| s2g | 32B | tp-2 | 1205 | 58.8 | +0.038 | +0.038 | -0.001 | -0.001 | 0.0000 |
| s3 | 30B | single | 5586 | 195.4 | -0.080 | -0.080 | +0.032 | +0.032 | 0.0000 |
| s3 | 30B | layer-4 | 8318 | 181.8 | +0.285 | +0.285 | +0.093 | +0.001 | 0.0848 |
| s3 | 30B | ep-4 | 4006 | 142.5 | +0.214 | +0.214 | +0.392 | +0.392 | 0.0000 |
| s3 | 30B | tp2-pp2 | 4470 | 180.1 | +0.294 | +0.294 | +0.015 | +0.015 | 0.0000 |
| s3 | 32B | single | 1301 | 38.4 | -0.025 | -0.025 | +0.013 | +0.013 | 0.0000 |
| s3 | 32B | layer-4 | 2077 | 38.7 | +0.350 | +0.350 | +0.003 | -0.008 | 0.0108 |
| s3 | 32B | tp-4 | 1492 | 80.4 | +0.093 | +0.093 | +0.068 | +0.068 | 0.0000 |
| s3 | 32B | tp2-pp2 | 1358 | 59.6 | +0.511 | +0.511 | -0.010 | -0.010 | 0.0000 |
| s3b | 30B | single | 5568 | 196.2 | -0.107 | -0.107 | +0.026 | +0.026 | 0.0000 |
| s3b | 30B | layer-4 | 11642 | 182.4 | -0.053 | -0.053 | +0.089 | -0.003 | 0.0849 |
| s3b | 30B | ep-4 | 3987 | 138.5 | +0.197 | +0.197 | +0.496 | +0.496 | 0.0000 |
| s3b | 30B | tp2-pp2 | 5085 | 181.8 | +0.247 | +0.247 | +0.020 | +0.020 | 0.0000 |
| s3b | 32B | single | 1299 | 38.4 | -0.017 | -0.017 | +0.016 | +0.016 | 0.0000 |
| s3b | 32B | layer-4 | 2918 | 38.7 | -0.027 | -0.027 | +0.005 | -0.006 | 0.0109 |
| s3b | 32B | tp-4 | 1481 | 80.3 | +0.093 | +0.093 | +0.123 | +0.123 | 0.0000 |
| s3b | 32B | tp2-pp2 | 1408 | 59.7 | +0.483 | +0.483 | -0.010 | -0.010 | 0.0000 |
| s3c | 30B | single | 5591 | 205.3 | -0.083 | -0.083 | +0.044 | +0.044 | 0.0000 |
| s3c | 30B | layer-4 | 11918 | 186.9 | -0.047 | -0.047 | +0.167 | +0.067 | 0.0853 |
| s3c | 30B | ep-4 | 4040 | 179.4 | +0.088 | +0.088 | -0.084 | -0.084 | 0.0000 |
| s3c | 30B | tp2-pp2 | 5079 | 184.4 | +0.249 | +0.249 | +0.002 | +0.002 | 0.0000 |
| s3c | 32B | single | 1299 | 38.6 | -0.037 | -0.037 | +0.022 | +0.022 | 0.0000 |
| s3c | 32B | layer-4 | 2934 | 38.9 | -0.051 | -0.051 | +0.019 | +0.009 | 0.0096 |
| s3c | 32B | tp-4 | 1504 | 81.5 | +0.070 | +0.070 | -0.032 | -0.032 | 0.0000 |
| s3c | 32B | tp2-pp2 | 1425 | 60.1 | +0.448 | +0.448 | -0.021 | -0.021 | 0.0000 |
| s3f2 | 30B | single | 5287 | 179.0 | -0.051 | -0.051 | +0.026 | +0.026 | 0.0000 |
| s3f2 | 30B | layer-4 | 11424 | 177.5 | -0.029 | -0.029 | +0.005 | +0.005 | 0.0000 |
| s3f2 | 30B | ep-4 | 4524 | 150.1 | +0.084 | +0.084 | -0.061 | -0.061 | 0.0000 |
| s3f2 | 30B | tp2-pp2 | 6486 | 156.5 | +0.089 | +0.089 | +0.002 | +0.002 | 0.0000 |
| s3f2 | 32B | single | 1250 | 38.0 | -0.074 | -0.074 | +0.005 | +0.005 | 0.0000 |
| s3f2 | 32B | layer-4 | 2758 | 38.3 | -0.067 | -0.067 | -0.007 | -0.007 | 0.0000 |
| s3f2 | 32B | tp-4 | 1630 | 72.9 | +0.108 | +0.108 | -0.030 | -0.030 | 0.0000 |
| s3f2 | 32B | tp2-pp2 | 2192 | 56.5 | +0.045 | +0.045 | +0.007 | +0.007 | 0.0000 |
| v2 | 30B | single | 5661 | 201.3 | -0.023 | -0.023 | +0.043 | +0.043 | 0.0000 |
| v2 | 30B | layer-2 | 8971 | 203.7 | -0.022 | -0.022 | -0.015 | -0.015 | 0.0000 |
| v2 | 30B | tp-2 | 4632 | 187.0 | -0.045 | -0.045 | -0.016 | -0.016 | 0.0000 |
| v2 | 30B | ep-2 | 4803 | 181.2 | -0.064 | -0.064 | +0.024 | +0.024 | 0.0000 |
| v2 | 32B | single | 1273 | 37.7 | +0.002 | +0.002 | +0.015 | +0.015 | 0.0000 |
| v2 | 32B | layer-2 | 2030 | 39.8 | -0.007 | -0.007 | -0.046 | -0.046 | 0.0000 |
| v2 | 32B | tp-2 | 1420 | 59.0 | -0.002 | -0.002 | -0.001 | -0.001 | 0.0000 |

## 30B layer-4 against single

| session | measured layer-4 / single tg | v4 predicted | v5 predicted | v5 error | e_tg v4 | e_tg v5 |
|---|---:|---:|---:|---:|---:|---:|
| s3 | 0.930 | 0.986 | 0.902 | -0.028 | +0.093 | +0.001 |
| s3b | 0.930 | 0.987 | 0.903 | -0.027 | +0.089 | -0.003 |
| s3c | 0.910 | 1.017 | 0.930 | +0.020 | +0.167 | +0.067 |

## s3f2 out of sample: 30B layer-4 truncated to 36 layers, tg128

measured 231.8, v4 230.9, v5 230.9
