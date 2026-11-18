# Session s2f-20261118

- Status: unclean, verdict null. Phase 5 S2 rehearsal (one round, so no configuration reaches the two clean rounds the verdict needs). Every stage ran: gates, enumerate, probe, prepass, anchors, calibrate, predict, rounds, A/A, summary. vLLM skipped (no AWQ/GPTQ checkpoint on the box).
- Rule: ADR 0068 (band 0.08 provisional, throttle by sampler window per GPU).
- Code and box: cf3cffb, box 61273508, host 522460 (Florida), 2x RTX 3090, driver 595.91.07, AMD EPYC 7352, no P2P (topo NODE), NCCL 2.25.1, memory.max 129.5 GB, THP madvise, idle gate 10/10, co-tenant ok (outside 0.04 cores).
- Raw files: phase5/. All gates ok (gpu_parity, comm_probe, baseline_modes).

## Findings

- Candidates: 30B single, layer-2, tp-2, ep-2; 32B single, layer-2, tp-2. Baselines accepted: llama.cpp layer and tensor. Rejected: llama.cpp row ("device CUDA0 does not support split buffers"), all ik_llama modes (rc 0, none accepted).
- Calibration fit (8 anchors, GPU only): band 0.486, residuals 0.11 to 0.40 for 30B (pp +0.25..+0.40, tg +0.11 and -0.13..-0.19 for tp-2), -0.35 to -0.12 for 32B (pp), tg +0.08 / -0.12..-0.16.
- Predicted vs measured (pp tok/s, tg tok/s; err = (pred-meas)/meas; none within the band):
  - 30B single 7760 / 198 vs 4813 / 177 (pp +0.61, tg +0.12); layer-2 7710 / 196 vs 5230 / 198 (+0.47, -0.01); tp-2 5860 / 147 vs 4316 / 181 (+0.36, -0.19); ep-2 6250 / 148, run unclean (GPU 1 sample 0x20), no measurement.
  - 32B single 859 / 30.7 vs 1084 / 27.7 (-0.21, +0.11); layer-2 858 / 30.7, run unclean (GPU 1 sample 0x20), no measurement; tp-2 1141 / 45.5 vs 1322 / 53.7 (-0.14, -0.15).
- Planner picks: 30B single (measured fastest of the clean ones is layer-2 on tg, single on pp), 32B tp-2. Winner sets empty (no configuration with two clean rounds), pick_ok, prediction_ok and baseline_ratio null.
- Baselines (llama.cpp, pp / tg): 30B layer 7136 / 206, tensor 3967 / 181; 32B layer 1670 / 39.3, tensor 1291 / 53.6. Informal time ratio (not a verdict field): 30B pick single is 21% slower than llama.cpp layer; 32B pick tp-2 3.93 s vs tensor 3.97 s (about 1.01x).
- A/A (first candidate again): 30B single 1.1487 s vs 1.1494 s; 32B 6.516 s vs 6.541 s.
- verdict.json: phase5_pass false; M2 and D2 pass null, clean false ("N clean round(s), need 2" for every configuration).
- Throttle: capped-clock share 0 to 0.58 per GPU (board share 0); one disqualifying 0x20 SW thermal sample each on 30B ep-2 (GPU 1) and 32B layer-2 (GPU 1).
- Stage times: build 9 min, gpu-parity 5 min, prepass to rounds 6 min, rounds 4 min; launch to done 22 min.

## Power

- Both GPUs peaked at about 250 W (limit lowered; `s2e` peaked at 300 W, `s2d` at 349 W). Median SM clock of loaded samples: 32B single 1350 MHz, 32B layer-2 1830 / 1740 MHz, 32B tp-2 1695 / 1740 MHz, 30B shapes 1740 to 1950 MHz. The 32B single run was power-capped; the shape ranking of this session is partly a host artifact.
