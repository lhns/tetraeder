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

## Cost model diagnosis (2026-11-18, offline, ADR 0076)

From the committed files. The lines are reproduced by `python scripts/costmodel_lines.py bench/results/s2f-20261118/phase5`; the breakdowns and refits came from a Python re-implementation of `predict` (not committed) that matches `predict-*.json` to the printed digit. Two measured rows come from unclean runs (one 0x20 sample each, see Findings): 32B layer-2 and 30B ep-2; they are used here as indications only.
- Lines per shape: `T(L') = t_step + L' * t_layer` through the L/4 and L/2 anchor medians, per rate, predict the full models (pp / tg tok/s, err): 30B single 4816 / 178.8 (+0.001, +0.010), tp-2 4358 / 183.4 (+0.010, +0.011); 32B single 1099 / 28.2 (+0.014, +0.021), tp-2 1314 / 54.6 (-0.005, +0.017). layer-2 and ep-2 have no anchors.
- Committed prediction breakdown (calib.json: F 61.1 TFLOP/s, B 630 GB/s, c0 1.22 us; c_range 3e-12 s and c_step 1e-10 s, both driven to zero by the fit): 30B single pp 264 ms against 425 measured (head 30 ms of it, charged for all 2048 rows); 32B single pp 2384 ms against 1890 (head 76 ms); 30B tp-2 tg 6.82 ms per token, all-reduce 2.95 of it (96 x 30.7 us from the probe); 32B tp-2 tg 22.0 ms, all-reduce 4.76. The layer-2 hop copy is 1.5 ms of 264 (30B pp) and 3.8 of 2384 (32B pp). The attention FLOP term is not identifiable from these anchors (a separate attention rate fits to an unphysical 2.5e15 FLOP/s).
- Refits of the analytic model on these anchors (not committed code): with a separate expert-matmul rate the fit gives 74 to 77 TFLOP/s dense and 21 to 22 TFLOP/s expert; that, the head on output rows, every expert read at prefill and a per-class roofline bring single and tp-2 pp within 0.06, tg within 0.18 (30B single tg +0.17, 30B and 32B tp-2 tg -0.15), and layer-2 to -0.13 (30B pp) and -0.21 (32B tg). Scaling the all-reduce curves by a fitted in-step factor (0.81 here, 0.63 on s2e) narrows single and tp-2 to 0.13 but leaves layer-2 at -0.17 and -0.22. Adding a duty-cycle power term (speed `min(1, a / duty)`) fits the anchors within 0.04 to 0.07 and misses layer-2 tg by -0.11 to +0.66, depending on whether B is fitted or pinned.
- Power: both GPUs behave as capped at 250 W (sampler maximum 251.75 and 251.52 W, with bit 0x4 at the peaks; s2e maximum 300.4 W). No power limit was recorded, so the cap is inferred. Under the cap the 32B single run computed at 780 to 870 MHz SM, layer-2 at 1590 to 1830, tp-2 at 1155 to 1425 (window samples with bit 0x4). 32B decode per 16 layers: single 8.5 ms here against 6.84 on s2e, tp-2 4.38 against 4.10; measured 32B layer-2 tg 38.6 against single 27.7.
- Cross-box: the lines fitted on s2e anchors predict this session's single and tp-2 within 0.28 (worst 32B single tg); s2e has two truncations only, so no out-of-sample check on its own box.
- Engine, not cost model: the anchors' line predicts the 30B single pp at 4816 against 4813 measured, so pp per GPU is as the anchors say. llama.cpp `-sm layer` with two GPUs and all layers offloaded enables pipeline parallelism (`llama-context.cpp:528`), overlapping the 4 ubatches across the stages; our layer-2 runs `prefill_micro_batches = 1`. llama.cpp layer over our layer-2: pp 1.36x (30B), 1.51x (32B); tg 1.04x and 1.02x. Phase 4 (`20261117k`) measured 407-token TTFT on another box, not comparable for pp2048; its decode parity (moe builder 196 against llama-server 198 tok/s) shows no per-GPU deficit.
- `baseline_ratio` null: by design, the gate needs two clean rounds of the pick and the baselines (`phase5_summary.py`, baseline gate). Informal ratios as above.
