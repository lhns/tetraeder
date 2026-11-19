# Session s3b-20261119

- Status: unclean, verdict null for D4, M4 fails on the baseline gate and prediction (phase5_pass false). Phase 5 S3b: second 4-GPU rehearsal, ROUNDS=2 (plus reruns), after the pipeline issue-order fix (ADR 0077 History 2026-11-19) and the prefill-balanced layer split (ADR 0070 History 2026-11-19). Every stage ran, every gate passed; vLLM skipped (no AWQ/GPTQ checkpoint).
- Rule: ADR 0068 (band 0.08 provisional, throttle by sampler window per GPU, per-configuration share comparison, two clean rounds per configuration, up to two reruns).
- Code: origin/main 9fdaa3e (branch box-s3b-20261119). Box 61353006, host 20210 (Finland), 4x RTX 3090, driver 595.91.07, AMD EPYC 7282 (32 threads, CPU quota 30.7), memory.max 259 GB, NCCL, p2p none.
- Power limit 350 W = default on all four GPUs (max 400 W). Idle gate 10/10, co-tenant sample at SSH 0.18 host cores busy over 20 s, `cuda: ok`.
- Topology: GPU0-GPU1 PHB, GPU2-GPU3 PHB, every other pair NODE, one NUMA node (same as s3-20261119).
- Box: 1.192 USD/h plus about 0.03 USD/h traffic, created 02:20, destroyed 03:07 (47.5 min), after a verified fetch. Cost about 0.95 USD box plus about 0.1 traffic (credit counter shared with the other box, so not separable). First offer, no gate failure.
- Timing (box clock): provision 13 min (46 GB), phase5_box.sh 01:34 to 02:06 (32 min: build 8.7, gpu-parity 6.6, anchors 2, rounds and reruns 12).
- Not committed: `gpu-parity.log.full` (stayed on the box), plans/ (not fetched; the plans are in candidates-*.json).

## Gates

- gpu-parity ok, comm_probe ok (gates.json); every test result line 0 failed.
- 4 GPUs, power gate, idle gate, co-tenant sample at SSH all passed before provisioning.

## Layer splits (candidates-*.json)

- Qwen3-32B (64 layers): layer-4 16/16/16/16, tp2-pp2 32/32.
- Qwen3-30B-A3B (48 layers): layer-4 12/12/12/12, tp2-pp2 24/24.
- 32B candidates: single, layer-4, tp-4, tp2-pp2. 30B: single, layer-4, ep-4, tp2-pp2 (tp-4 pruned: 3 blocks over 4 ranks).

## Calibration (8 anchors, cost model v2, one fit over both models)

- Band 0.102 (s3: 0.074). Residuals -0.085 (30B single L24 pp) to +0.097 (30B tp-2 L12 pp); all tg residuals within 0.015; fitted all-reduce latency share 0.259 (s3: 0.445).

## Predicted vs measured (err = pred/meas - 1; `x` marks beyond 0.15; clean rounds in the last column)

| Case | Candidate | pred pp / tg | meas pp / tg | err pp / tg | clean rounds |
|---|---|---|---|---|---|
| D4 | single | 1277 / 39.1 | 1298 / 38.4 | -0.016 / +0.018 | 1 |
| D4 | layer-4 | 2840 / 39.0 | 2903 / 38.7 | -0.022 / +0.008 | 1 |
| D4 | tp-4 | 1617 / 88.1 | 1481 / 80.3 | +0.092 / +0.098 | 2 |
| D4 | tp2-pp2 | 2090 / 59.8 | 1415 / 59.6 | +0.477 x / +0.003 | 2 |
| M4 | single | 4915 / 202.2 | 5568 / 196.2 | -0.117 / +0.031 | 2 |
| M4 | layer-4 | 10883 / 200.4 | 11642 / 182.4 | -0.065 / +0.099 | 2 |
| M4 | ep-4 | 5383 / 200.0 | 3987 / 138.5 | +0.350 x / +0.444 x | 2 |
| M4 | tp2-pp2 | 6507 / 191.0 | 5085 / 181.8 | +0.280 x / +0.050 | 2 |

Against s3 (before the fixes): layer-4 pp is now within 0.07 on both models (s3: +0.35 and +0.27), measured 32B layer-4 pp 2903 against 2077 and 30B 11642 against 8318. tp2-pp2 pp is still over-predicted (+0.48 and +0.28) and ep-4 is over-predicted on both pp and tg (tg 200 predicted against 138 measured; s3: 152 against 143, so the tg prediction of ep-4 got worse).

## Planner picks and baselines

- D4: pick tp-4 (predicted T 2.719 s, measured 2.977 s; measured T: tp-4 2.977, tp2-pp2 3.594, layer-4 4.014, single 4.912). Winner set empty in verdict.json (single and layer-4 have 1 clean round); informally {tp-4}, the pick is the fastest.
- M4: pick layer-4 (predicted T 0.827 s, measured 0.878 s; measured T: layer-4 0.878, single 1.020, tp2-pp2 1.107, ep-4 1.438). Winner set {layer-4}, pick_ok true, prediction_ok false (ep-4, tp2-pp2 beyond 0.15).
- Baselines (llama.cpp, T for pp2048 + tg128): D4 layer 3.928 s, tensor 3.053 s (best); pick 2.977 s, ratio 1.026, baseline_ok true. M4 layer 0.805 s (best), tensor 1.236 s; pick 0.878 s, ratio 0.917, baseline_ok false (needs above 1 - band = 0.92; misses by 0.003).
- A/A: D4 single 4.9117 s at start, 4.9093 s at end (-0.05%); M4 single 1.0200 s, 1.0184 s (-0.16%).
- verdict.json: phase5_pass false, band 0.08 (provisional cuda-default), comm_impl nccl, p2p none. M4: clean true, pass false (baseline_ok false, prediction_ok false). D4: clean false, pass null; reasons "single: 1 clean round(s), need 2", "layer-4: 1 clean round(s), need 2". predict sha256: 32B 64549816..., 30B cabb627c....

## Throttle and co-tenant

- 14 of 38 judged runs unclean, all on Qwen3-32B: 13 by one or two window samples with bit 0x20 (12 on GPU 0, one on GPU 3 for layer-4 r4), one by a 0x4 share spread (tp-4 r2, GPU 2, 0.22 against median 0.11). Qwen3-30B-A3B: 0 unclean. The single and layer-4 candidates and both llama-bench baselines hit 0x20 in most rounds; reruns (r3, r4) did not give single and layer-4 a second clean round. Board share 0 everywhere.
- Co-tenant: 0 of 38 runs flagged (`cotenant_ok` true on all); outside cores 0.10 to 0.32 of 32.

## Failures

- None: no failed stage, no rc other than 0, no gate failure. D4 is null only for the clean-round count.
