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

## Cost model diagnosis (v3, ADR 0078)

Offline, `scripts/costmodel_replica.py` (branch `track-costmodel-v3`). Version 2 on each session's `calib.json` reproduces its `predict-*.json` bit for bit (S2g, S3, S3b, pipelined prefill included; `tests/test_costmodel_replica.py`). Each "before" below is the replica's own version-2 refit, so before and after share the fit.

**ep-4 decode (pred 200, meas 138.5; s3 152 against 143).**
- Per token the model gave ep-4 3.31 ms compute plus 1.52 ms for 96 all-reduces (4 ranks); measured 7.22 ms.
- The fitted share times the anchored pair's probe floor is the in-step latency the anchors see: 0.445 x 27.9 us = 12.4 us (s3), 0.259 x 43.4 us = 11.2 us (s3b). The share follows the floor's noise; applied to the 4-rank floor it gave 28 us (s3) and 13 us (s3b) per all-reduce, the whole move of the ep-4 decode prediction between the two sessions (same host 20210, measured 142.5 and 138.5).
- Backed out of measured decode, a 4-rank all-reduce costs about 21 us in tp-4 (dense 32B, no routing) and about 36 us in ep-4. One rank-dependent latency cannot give both; the ep-4 remainder (about 1.4 ms per token) is ep-specific.
- Version 3 fits the latency `a` directly: 10.9 us on s3b, 12.0 to 12.4 on s3 (from starts at 12 and 25 us), 12.0 to 12.5 on s2g. Without a second anchored group size the 4-rank all-reduce pays `a`, and the ep-4 decode miss becomes stable (+0.39, +0.49, cross-box +0.43). The ring step count `2 (p - 1)` as the other bound fits ep-4 decode (+0.04, -0.05) but over-prices tp-4 (-0.10 on s3b, -0.16 on s3 and cross-box), and tp-4 has no routing term: rejected.

**ep prefill (ep-4 +0.35, ep-2 +0.11).**
- With tp-2's all-reduces priced right on the same box (s2g: tp-2 within 0.03), ep-2 measured 263 ms of compute against 211 predicted, ep-4 249 against 114. The slowest rank's expert work that implies is about 0.67 of the layer's at ep-2 and 0.68 at ep-4, where uniform routing gives 0.50 and 0.25.
- `tetraeder bench` feeds byte ids 0 to 255 (`crates/tetraeder/src/bench.rs`) in prefill and decode, so concentrated routing is expected; `tetraeder-plan predict` passes no routing frequencies (`features_multi(..., None)`), so expert parallel is priced at uniform routing.
- Bounds in the replica: decode at `top_k` local experts and prefill mass 1 over-shoot (ep-4 pp -0.18 to -0.20, ep-2 -0.18). Inside them lies the measured rate; which value is a property of the judged token stream, to be recorded (ADR 0070's routing input), not fitted.
- Not a host sync: at Q4_K and 512 rows `mul_mat_id` takes MMQ with the on-device ids helper (`ggml_cuda_mul_mat_id_needs_sync`), and CUDA graphs stay on.

**tp2-pp2 prefill (+0.48 on 32B, +0.28 on 30B).**
- It ran pipelined: one 2048-token step (`n_eval` 1), prefill ranges 4x the decode ranges.
- Stage times are right: stage 0 is the tp-2 L/2 anchor on the same GPUs (32B residual +0.034). Per 512-token micro-batch, 32B: stage 0 99 ms compute plus 98 ms all-reduce, stage 1 97 plus 78; 30B: 33 plus 30 and 33 plus 25.
- Measured against the stages run in turn: 32B 1454 against 1513 ms (s3: 1508 against 1524), about 11% of the GPipe credit; 30B 403 against 492 ms, about half. layer-4 overlaps as modeled (-0.03, -0.07).
- Contention of the two groups' host-staged all-reduces is not it: the 4-rank probe is about 1.1x the 2-rank one, and by a rough count serialized all-reduces with overlapping compute would still give about 1000 to 1100 ms on 32B. ggml's NCCL all-reduce does not block the host (`ggml_backend_cuda_comm_allreduce_nccl`: group start, all-reduce, group end, BF16 casts on the stream). The mechanism is not found offline; the measurement that decides it is tp2-pp2 at `prefill_micro_batches` 1 against 4 on one box. No overlap factor is introduced (one constant cannot fit 11% and 50%, and it would be a fit to one candidate).

**Before (v2 refit) and after (v3), err = pred / meas - 1, each session fit on its own anchors, plus s3 anchors judged on s3b; `x` beyond 0.15.**

| fit -> eval | model | candidate | v2 pp / tg | v3 pp / tg |
|---|---|---|---|---|
| s2g | 30B | single | -0.096 / +0.020 | -0.096 / +0.018 |
| s2g | 30B | layer-2 | +0.009 / -0.012 | +0.009 / -0.014 |
| s2g | 30B | tp-2 | +0.033 / -0.025 | +0.033 / -0.023 |
| s2g | 30B | ep-2 | +0.111 / +0.011 | +0.111 / +0.013 |
| s2g | 32B | single | -0.018 / +0.007 | -0.018 / +0.006 |
| s2g | 32B | layer-2 | +0.145 / -0.035 | +0.145 / -0.035 |
| s2g | 32B | tp-2 | +0.039 / -0.002 | +0.039 / -0.001 |
| s3 | 30B | single | -0.091 / +0.034 | -0.091 / +0.032 |
| s3 | 30B | layer-4 | +0.269 x / +0.102 | +0.269 x / +0.099 |
| s3 | 30B | ep-4 | +0.354 x / +0.066 | +0.377 x / +0.391 x |
| s3 | 30B | tp2-pp2 | +0.329 x / +0.010 | +0.329 x / +0.014 |
| s3 | 32B | single | -0.024 / +0.015 | -0.024 / +0.014 |
| s3 | 32B | layer-4 | +0.350 x / +0.005 | +0.350 x / +0.005 |
| s3 | 32B | tp-4 | +0.086 / -0.094 | +0.093 / +0.066 |
| s3 | 32B | tp2-pp2 | +0.511 x / -0.012 | +0.511 x / -0.010 |
| s3b | 30B | single | -0.117 / +0.031 | -0.117 / +0.027 |
| s3b | 30B | layer-4 | -0.065 / +0.099 | -0.065 / +0.095 |
| s3b | 30B | ep-4 | +0.350 x / +0.444 x | +0.353 x / +0.494 x |
| s3b | 30B | tp2-pp2 | +0.280 x / +0.050 | +0.279 x / +0.018 |
| s3b | 32B | single | -0.017 / +0.018 | -0.017 / +0.018 |
| s3b | 32B | layer-4 | -0.027 / +0.008 | -0.027 / +0.008 |
| s3b | 32B | tp-4 | +0.092 / +0.098 | +0.093 / +0.120 |
| s3b | 32B | tp2-pp2 | +0.484 x / +0.003 | +0.483 x / -0.010 |
| s3 -> s3b | 30B | single | -0.088 / +0.030 | -0.088 / +0.028 |
| s3 -> s3b | 30B | layer-4 | -0.035 / +0.098 | -0.035 / +0.096 |
| s3 -> s3b | 30B | ep-4 | +0.361 x / +0.096 | +0.383 x / +0.431 x |
| s3 -> s3b | 30B | tp2-pp2 | +0.287 x / +0.001 | +0.287 x / +0.005 |
| s3 -> s3b | 32B | single | -0.023 / +0.015 | -0.023 / +0.015 |
| s3 -> s3b | 32B | layer-4 | -0.033 / +0.005 | -0.033 / +0.005 |
| s3 -> s3b | 32B | tp-4 | +0.094 / -0.093 | +0.101 / +0.066 |
| s3 -> s3b | 32B | tp2-pp2 | +0.485 x / -0.012 | +0.485 x / -0.011 |

- s3's layer-4 rows are from before the issue-order fix (s3 README); they are not a cost model error.
- Anchors' worst residual unchanged (0.103, 0.072, 0.097).
- Picks (smallest predicted T) unchanged by v3 and equal to the measured fastest in every case: s2g layer-2 and tp-2, s3 and s3b layer-4 and tp-4, and s3 -> s3b layer-4 and tp-4.
- Ring-step bound (`2 (p - 1)` steps at the anchored per-step latency): ep-4 tg +0.043 (s3b), -0.045 (s3), -0.018 (cross); tp-4 tg -0.104, -0.156, -0.156.

**What closes the misses (owner, ADR 0068 reopen rule: unanchored shapes, missing physical terms).**
- A truncated `tp-4` anchor on Qwen3-32B (L/4, L/2) fits the latency per ring step `b` (ADR 0078); without it 4-rank all-reduces pay the 2-rank latency.
- Routing frequencies of the judged token stream (byte ids) for Qwen3-30B-A3B, recorded in the calibration stage and passed to `predict`, set expert parallel's slowest rank (ADR 0070's input, not wired on the box).
- tp2-pp2: one paired run at `prefill_micro_batches` 1 and 4 on a 4-GPU box shows whether the tp stages pipeline at all.
