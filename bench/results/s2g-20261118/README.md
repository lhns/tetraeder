# Session s2g-20261118

- Status: unclean, verdict null (phase5_pass false). Phase 5 S2 rehearsal with cost model v2 and pipelined layer prefill, ROUNDS=2 (rounds up to r4 for configurations short of two clean runs). Every stage ran; vLLM skipped (no AWQ/GPTQ checkpoint).
- Rule: ADR 0068 (band 0.08 provisional, throttle by sampler window per GPU, per-configuration share comparison).
- Code: origin/main e8d8b3f plus scripts/pp_pipeline_box.sh (branch box-s2g-20261118). Box 61328297, host 24953 (Oregon, US), 2x RTX 3090 (PCIe gen4 x16 max), driver 615.71.09, AMD EPYC 7702 (CPU quota 30 cores), memory.max 129.6 GB, no P2P (topo PHB), NCCL 2.25.1.
- Gates: idle 10/10, power limit 350 W = default on both GPUs, co-tenant sample ok at SSH (8.1 cores outside, limit 9.6); gpu_parity, comm_probe ok; cuda: ok.
- Box choice: attempts 1 (host 1276) and 2 (host 142) stayed `loading` more than 10 minutes and were destroyed; attempt 3 ready after 7 min.
- Raw files: phase5/ (`.rc` files and `gpu-parity.log.full` not committed). Pipelining bench: ../s2g-20261118-pp/.

## Pipelined prefill tests (gpu-parity, ADR 0077)

- `gpu2_pipelined_prefill_matches` (dense, Qwen3-8B): 400 tokens, serial 127.3 ms, pipelined 87.5 ms (1.45x); every checked step bit-equal true, top-1 equal.
- `gpu2_moe_pipelined_prefill_matches`: 400 tokens, serial 156.2 ms, pipelined 111.7 ms (1.40x); every checked step bit-equal true, greedy equal.

## Findings

- Calibration (8 anchors, v2): band 0.108; residuals -0.062 to +0.103 (largest: 30B tp-2 L12 pp +0.103, 30B single L24 pp -0.062); fitted all-reduce latency share 0.468.
- Predicted vs measured (pp tok/s, tg tok/s; err = (pred-meas)/meas; rounds_clean in brackets). All within 0.15 on both rates:
  - 30B single 5012 / 198 vs 5550 / 194 (pp -0.097, tg +0.020) [2]; layer-2 7282 / 197 vs 7201 / 200 (+0.011, -0.011) [2]; tp-2 3980 / 181 vs 3873 / 185 (+0.028, -0.025) [2]; ep-2 4454 / 182 vs 4014 / 180 (+0.110, +0.009) [2].
  - 32B single 1246 / 38.2 vs 1269 / 38.0 (-0.018, +0.007) [2]; layer-2 1937 / 38.2 vs 1692 / 39.6 (+0.145, -0.035) [1]; tp-2 1251 / 58.6 vs 1196 / 58.9 (+0.046, -0.005) [1].
- Planner picks: 30B layer-2 (winner set {layer-2}, pick_ok true, prediction_ok true); 32B tp-2 (winner set empty: layer-2 and tp-2 have one clean round, so null).
- Baselines (llama.cpp, time T for pp2048 + tg128): 30B layer 0.853 s (tensor: no clean round); 32B layer 4.211 s, tensor 3.890 s. baseline_ratio null (best baseline needs two clean rounds). Informal: 30B pick layer-2 0.926 s vs layer 0.853 s (1.09x slower); 32B pick tp-2 3.884 s vs tensor 3.890 s (about level).
- A/A: 30B single 1.028 s at start, end run unclean (12.2 cores busy outside the container, limit 9.6); 32B single 4.982 s at start, end run unclean (GPU 0 sample 0x20).
- verdict.json: phase5_pass false; M2 and D2 pass null, clean false. Reasons: 30B llama-bench tensor 0 clean rounds, A/A co-tenant; 32B layer-2, tp-2, llama layer and tensor each 1 clean round, A/A 0x20.
- Throttle: 21 of 42 runs have a disqualifying window sample (0x20 SW thermal, one or two samples each) or a co-tenant flag (30B single r3, llama layer r4, A/A: 11.5 to 13.3 cores outside, limit 9.6 on this 64-core host); capped-clock (0x4) share up to 0.62 per GPU (32B single GPU 0), board share up to 0.8.

## Pipelining bench (s2g-20261118-pp)

Both layer-2 plans carry `prefill_micro_batches = 4` (the planner's value); the M=1 copy differs only in that line. pp2048, ubatch 512, one rep per run, two rounds (round 2 reversed), same session and box as the rehearsal. Results in ../s2g-20261118-pp/README.md.
