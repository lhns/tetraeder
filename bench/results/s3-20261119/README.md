# Session s3-20261119

- Status: unclean, verdict null (phase5_pass false). Phase 5 S3: first 4-GPU session (gpu4_ gates, comm-probe, rehearsal of D4 and M4), ROUNDS=1. Every stage ran; vLLM skipped (no AWQ/GPTQ checkpoint). Every gate passed.
- Rule: ADR 0068 (band 0.08 provisional, throttle by sampler window per GPU, per-configuration share comparison). One round means no configuration has the two clean rounds the rule needs; the numbers below are a rehearsal, not a verdict.
- Code: origin/main 8341384 (branch box-s3-20261119). Box 61338045, host 20210 (Finland), 4x RTX 3090, driver 595.91.07, AMD EPYC 7282 (CPU quota 30 cores, 32 threads), memory.max 259 GB, 80 GB disk, NCCL 2.25.1, no P2P.
- Power limit 350 W = default on all four GPUs (max 400 W). Idle gate 10/10, co-tenant sample at SSH 0.19 cores busy on the host, `cuda: ok`.
- Topology (`nvidia-smi topo -m`): GPU0-GPU1 PHB, GPU2-GPU3 PHB, every other pair NODE; one NUMA node. PCIe max gen4 x16 per GPU (idle reads gen1 x8). The pairs (0,1) and (2,3) share a root port; the two pairs are the tp2-pp2 groups.
- Box choice: attempt 1 (host 124072, 0.652 USD/h + 0.039 traffic, 24 cores, Xeon Silver 4310) was ready after 7 min but failed the co-tenant sample at SSH (host 3.7 to 5.5 cores busy against a container idle at 0.0, limit max(4, 15% of 24 physical) = 4, `top` 16% user and 18 to 22% iowait): destroyed after 13 min (about 0.14 USD). Attempt 2 (host 20210, 1.192 USD/h, 32 cores) ready after 8 min, host idle.
- Timing: box 00:23 to 01:11 (48 min); phase5_box.sh 24.5 min (build 8.6 min, gpu-parity 6.5 min, anchors to summary about 8 min). Cost about 1.0 USD for the box (1.192 USD/h) plus about 0.1 for 45 GB traffic, plus 0.14 for the destroyed attempt.
- Raw files: phase5/ (`.rc` files and `gpu-parity.log.full` not committed).

## Gates

- gpu-parity ok, comm_probe ok (gates.json). Every package ran with `--no-fail-fast`; 0 failed in every `test result` line.
- `gpu4_all_reduce_over_groups` ok: all four ranks and two pairs through NCCL (1024 collective bytes counted for the four, `{("all_reduce","nccl"): 1024}`).
- `gpu4_shapes_match_single` (Qwen3-8B, dense) ok: libllama tensor against one GPU worst 7.87e-3 of the largest logit; tp-4 4 ranks, NCCL, bound max(5e-3, 1.18e-2 of largest logit), worst delta 0.67x the bound, ranges [4672 x4] over 64 decode tokens; layer-4 delta 0 (bit-equal), ranges [64 x4]; tp2-pp2 worst 0.55x the bound, ranges [2368 x4].
- `gpu4_moe_shapes_match_single` (Qwen3-30B-A3B) ok: libllama tensor worst 7.77e-2 of the largest logit; every shape 0 of 65 steps over the bound: ep-4 worst 0.56x, tp2-pp2 worst 0.43x, layer-4 0.00x (bit-equal), greedy equal in the logged steps.
- Also ok on this box: `gpu2_pipelined_prefill_matches` 400 tokens serial 127.9 ms, pipelined 87.9 ms (1.46x); `gpu2_moe_pipelined_prefill_matches` 157.6 ms against 112.2 ms (1.40x); `gpu2_*` shapes (anchored by the four-way split, ADR 0069 known limit).

## Comm probe (calib-comm.json, path nccl, 20 repeats, 4 KiB to 64 MiB)

- 4-way all-reduce (gpu0-3): 63.7 us at 4 KiB, 103 us at 64 KiB, 247 us at 1 MiB, 2.70 ms at 16 MiB, 10.53 ms at 64 MiB.
- 2-way all-reduce: gpu0+1 27.9 us at 4 KiB, 209 us at 1 MiB, 9.47 ms at 64 MiB; gpu2+3 28.3 us, 172 us, 7.97 ms.
- Copy (12 directed pairs, flat across pairs, host-staged): 13.5 to 16 us at 4 KiB, 180 us at 1 MiB, 10.95 ms at 64 MiB (about 6.1 GB/s); no pair is faster than another.

## Prepass

- llama.cpp baseline: `layer` ok and `tensor` ok on both models; `row` rejected (rc 1, "device CUDA0 does not support split buffers"). The all-reduce path of `-sm tensor` is not named in its log (comm null).
- ik_llama.cpp: layer, row and graph rejected on both models (rc 0, no result rows); report-only anyway.

## Enumeration (fixed before calibration)

- Qwen3-32B (D4): 4 candidates, 0 pruned: `single`, `layer-4` (prefill_micro_batches 4), `tp-4`, `tp2-pp2`.
- Qwen3-30B-A3B (M4): 4 candidates, 1 pruned: `single`, `layer-4`, `ep-4`, `tp2-pp2`; `tp-4` pruned (3 blocks over 4 ranks, expert down rows).

## Calibration (8 anchors, cost model v2, one fit over both models)

- Band 0.074; residuals -0.059 (30B single L24 pp) to +0.071 (30B tp-2 L12 pp); all tg residuals within 0.017; fitted all-reduce latency share 0.445.

## Predicted vs measured (one round; err = pred/meas - 1; the 15% line is exceeded where marked `x`)

| Case | Candidate | pred pp / tg | meas pp / tg | err pp / tg |
|---|---|---|---|---|
| D4 | single | 1269 / 39.0 | 1301 / 38.4 | -0.024 / +0.015 |
| D4 | layer-4 | 2806 / 38.9 | 2077 / 38.7 | +0.350 x / +0.005 |
| D4 | tp-4 | 1620 / 72.8 | 1492 / 80.4 | +0.086 / -0.094 |
| D4 | tp2-pp2 | 2052 / 58.9 | 1358 / 59.6 | +0.511 x / -0.012 |
| M4 | single | 5077 / 202.1 | 5586 / 195.4 | -0.091 / +0.034 |
| M4 | layer-4 | 10558 / 200.3 | 8318 / 181.8 | +0.269 x / +0.102 |
| M4 | ep-4 | 5425 / 151.8 | 4006 / 142.5 | +0.354 x / +0.066 |
| M4 | tp2-pp2 | 5939 / 182.0 | 4470 / 180.1 | +0.329 x / +0.010 |

Every candidate has `rounds_clean` 1. All decode rates are within 0.15; the prefill rates of every shape with a pipeline or an expert split are over-predicted by 0.27 to 0.51 (single and tp-4 are within).

## Planner picks and baselines

- D4: pick tp-4 (predicted T 3.023 s, measured 2.966 s, the fastest measured of the four: layer-4 4.295, tp2-pp2 3.655, single 4.907). Informal winner set {tp-4}.
- M4: pick layer-4 (predicted T 0.833 s, measured 0.950 s); measured single 1.022, tp2-pp2 1.169, ep-4 1.410; layer-4 is the fastest candidate.
- Baselines (llama.cpp, T for pp2048 + tg128, one round): D4 layer 3.928 s, tensor 3.035 s; M4 layer 0.818 s, tensor 1.234 s. Informal: D4 pick 2.966 s against best baseline 3.035 s (tensor), 1.02x faster; M4 pick 0.950 s against layer 0.818 s, 1.16x slower (ratio 0.86, below 1 - band).
- A/A: D4 single 4.907 s at start, 4.914 s at end (+0.1%); M4 single 1.022 s at start, end run unclean (GPU 0 sample with 0x20).
- verdict.json: phase5_pass false, band 0.08 (provisional cuda-default), comm_impl nccl, p2p none; D4 and M4 pass null, clean false. Reasons: every configuration and both baselines have 1 clean round, need 2 (ROUNDS=1); M4 A/A end run unclean (0x20 on GPU 0). predict sha256 in verdict.json: 32B 3108df95..., 30B 230876b7....

## Throttle and co-tenant

- 1 of 30 judged runs unclean by the throttle rule: `qwen3-30b-a3b-single-aa` (one window sample with 0x20 on GPU 0). Co-tenant: 0 of 30 runs flagged (`cotenant ok` on all; outside cores 0.14 to 0.19 on the 30B runs).
- Capped-clock (0x4) share per configuration highest on 32B single GPU 0 (0.48; one GPU loaded alone), at most 0.17 elsewhere; board share 0 everywhere; each configuration has one run (A/A aside), so the share comparison did not apply.

## Findings for the planner (rehearsal, one round, so indicative)

- Decode is predicted well for every shape (largest error +0.102, 30B layer-4).
- Prefill is over-predicted for layer-4 (+0.27 and +0.35, pipelined with M=4 micro-batches over four stages), tp2-pp2 (+0.33 and +0.51) and ep-4 (+0.35). The model credits more overlap or less communication than the box delivers; measured layer-4 pp gains 1.6x on 32B and 1.5x on 30B over `single`, against 2.2x and 2.1x predicted. These misses decide the case on the 15% rule; the cause is not diagnosed here.
- The planner's picks were the measured winners in both cases, but the D4 pick passes the baseline gate only narrowly and M4 fails it (llama.cpp `-sm layer` is faster at 0.818 s).
