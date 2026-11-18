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

## Cost model diagnosis (2026-11-18, offline, ADR 0076)

From the committed files. The lines are reproduced by `python scripts/costmodel_lines.py bench/results/s2f-20261118/phase5`; the breakdowns and refits came from a Python re-implementation of `predict` (not committed) that matches `predict-*.json` to the printed digit. Two measured rows come from unclean runs (one 0x20 sample each, see Findings): 32B layer-2 and 30B ep-2; they are used here as indications only.
- Lines per shape: `T(L') = t_step + L' * t_layer` through the L/4 and L/2 anchor medians, per rate, predict the full models (pp / tg tok/s, err): 30B single 4816 / 178.8 (+0.001, +0.010), tp-2 4358 / 183.4 (+0.010, +0.011); 32B single 1099 / 28.2 (+0.014, +0.021), tp-2 1314 / 54.6 (-0.005, +0.017). layer-2 and ep-2 have no anchors.
- Committed prediction breakdown (calib.json: F 61.1 TFLOP/s, B 630 GB/s, c0 1.22 us; c_range 3e-12 s and c_step 1e-10 s, both driven to zero by the fit): 30B single pp 264 ms against 425 measured (head 30 ms of it, charged for all 2048 rows); 32B single pp 2384 ms against 1890 (head 76 ms); 30B tp-2 tg 6.82 ms per token, all-reduce 2.95 of it (96 x 30.7 us from the probe); 32B tp-2 tg 22.0 ms, all-reduce 4.76. The layer-2 hop copy is 1.5 ms of 264 (30B pp) and 3.8 of 2384 (32B pp). The attention FLOP term is not identifiable from these anchors (a separate attention rate fits to an unphysical 2.5e15 FLOP/s).
- Refits of the analytic model on these anchors (not committed code): with a separate expert-matmul rate the fit gives 74 to 77 TFLOP/s dense and 21 to 22 TFLOP/s expert; that, the head on output rows, every expert read at prefill and a per-class roofline bring single and tp-2 pp within 0.06, tg within 0.18 (30B single tg +0.17, 30B and 32B tp-2 tg -0.15), and layer-2 to -0.13 (30B pp) and -0.21 (32B tg). Scaling the all-reduce curves by a fitted in-step factor (0.81 here, 0.63 on s2e) narrows single and tp-2 to 0.13 but leaves layer-2 at -0.17 and -0.22. Adding a duty-cycle power term (speed `min(1, a / duty)`) fits the anchors within 0.04 to 0.07 and misses layer-2 tg by -0.11 to +0.66, depending on whether B is fitted or pinned.
- Power: both GPUs behave as capped at 250 W (sampler maximum 251.75 and 251.52 W, with bit 0x4 at the peaks; s2e maximum 300.4 W). No power limit was recorded, so the cap is inferred. Under the cap the 32B single run computed at 780 to 870 MHz SM, layer-2 at 1590 to 1830, tp-2 at 1155 to 1425 (window samples with bit 0x4). 32B decode per 16 layers: single 8.5 ms here against 6.84 on s2e, tp-2 4.38 against 4.10; measured 32B layer-2 tg 38.6 against single 27.7.
- Cross-box: the lines fitted on s2e anchors predict this session's single and tp-2 within 0.28 (worst 32B single tg); s2e has two truncations only, so no out-of-sample check on its own box.
- Engine, not cost model: the anchors' line predicts the 30B single pp at 4816 against 4813 measured, so pp per GPU is as the anchors say. llama.cpp `-sm layer` with two GPUs and all layers offloaded enables pipeline parallelism (`llama-context.cpp:528`), overlapping the 4 ubatches across the stages; our layer-2 runs `prefill_micro_batches = 1`. llama.cpp layer over our layer-2: pp 1.36x (30B), 1.51x (32B); tg 1.04x and 1.02x. Phase 4 (`20261117k`) measured 407-token TTFT on another box, not comparable for pp2048; its decode parity (moe builder 196 against llama-server 198 tok/s) shows no per-GPU deficit.
- `baseline_ratio` null: by design, the gate needs two clean rounds of the pick and the baselines (`phase5_summary.py`, baseline gate). Informal ratios as above.

## Cost model version 2 (2026-11-18, offline)

The fixes of ADR 0070 History 2026-11-18, refitted on this session's committed anchors (ADR 0068's: `single` and `tp-2` at L/4 and L/2, 16 rates) and on S2e's. Reproduced by `python scripts/costmodel_replica.py bench/results/s2f-20261118/phase5` (and `bench/results/s2e-20261118/phase5 --eval bench/results/s2f-20261118/phase5`). The replica gives version 1's `predict-*.json` to the digit. Its version-2 fits are in `costmodel-v2/`; `crates/planner/tests/costmodel_v2.rs` checks that Rust predicts the same rows on those graphs and that Rust's own fit lands on them. Caveats: 32B layer-2 and 30B ep-2 are unclean runs (indications only), and every run here was on a 250 W cap, which slows `single` most (Power). So the 30B, `single` pp and `tp-2` carry the most weight.

Fit on this session (version 1 is the committed `calib.json`):

| | v1 | v2 |
|---|---|---|
| F, F_exp (TFLOP/s) | 61.1, none | 73.8, 22.0 |
| B (GB/s), c0 (us) | 630, 1.22 | 627, 1.46 |
| c_step, c_range (us) | 0.0001, 0.000003 | 82, 0.0 |
| all-reduce latency share | none | 0.327 |
| anchor band (worst residual) | 0.486 | 0.078 (32B tp-2 L32 tg -0.075) |

Predicted / measured and error, pp and tg (tok/s):

| Candidate | v1 pp | v1 tg | v2 pp | v2 tg |
|---|---|---|---|---|
| 30B single | 7760 / 4813 (+0.61) | 198 / 177 (+0.12) | 4521 (-0.061) | 179.2 (+0.012) |
| 30B layer-2 | 7710 / 5230 (+0.47) | 196 / 198 (-0.01) | 4506 (-0.138) | 178.7 (-0.099) |
| 30B tp-2 | 5860 / 4316 (+0.36) | 147 / 181 (-0.19) | 4304 (-0.003) | 182.1 (+0.005) |
| 30B ep-2 (unclean) | 6250 / 4502 (+0.39) | 148 / 175 (-0.15) | 4931 (+0.095) | 184.0 (+0.053) |
| 32B single | 859 / 1084 (-0.21) | 30.7 / 27.7 (+0.11) | 1082 (-0.002) | 30.1 (+0.088) |
| 32B layer-2 (unclean) | 858 / 1107 (-0.23) | 30.7 / 38.6 (-0.21) | 1080 (-0.025) | 30.1 (-0.221) |
| 32B tp-2 | 1141 / 1322 (-0.14) | 45.5 / 53.7 (-0.15) | 1333 (+0.009) | 49.7 (-0.074) |

Within 0.15 on both rates: 0 of 7 candidates with v1, 6 of 7 with v2. The miss is 32B layer-2 tg, an unclean run that decoded faster on two GPUs than `single` on one, which batch 1 rules out at full power. The other two layer-2 rates (30B pp -0.14 and tg -0.10) point the same way: under the cap the half-idle stages clock higher.

Cross-box, fit on S2e (host 608696, about 300 W) and predict this session: v2 fits F 79.1 and F_exp 23.8 TFLOP/s, B 770 GB/s, c0 1.56 us, c_step 120 us and share 0.249; anchor band 0.073. Errors (pp, tg):
- v2: 30B single +0.011, +0.085; layer-2 -0.073, -0.034; tp-2 +0.071, +0.010; ep-2 +0.175, +0.056. 32B single +0.070, +0.306; layer-2 +0.044, -0.064; tp-2 +0.125, +0.056.
- v1: 30B single +0.74, +0.26; layer-2 +0.59, +0.11; tp-2 +0.44, -0.30; ep-2 +0.47, -0.27. 32B single -0.14, +0.36; layer-2 -0.16, -0.03; tp-2 -0.04, -0.11.
- The 32B single tg miss is this host's cap: S2e's single decodes 32B at 0.43 ms per layer, this session at 0.54.

Per fix: version 2 refitted on this session with one fix left out (`--drop`):
- Expert rate: without it the fit finds one F of 50 TFLOP/s, the band rises to 0.55 (worst residual 0.44), and pp misses by +0.48 (30B single), +0.32 (30B tp-2) and -0.32 (32B single). This fix carries most of the pp gain. At prefill an expert layer of the 30B costs about 2.0 ms of FLOPs per ubatch at F_exp, against 0.62 ms to read all its experts.
- Head on output rows: without it F absorbs the head (83.8 TFLOP/s), and 32B single pp goes from -0.002 to +0.085. The head was 76 ms of 2384 in the committed 32B single pp, and 30 ms of 264 in the 30B's.
- Every expert at prefill: no change on `single` and `tp-2`. Their expert prefill is bound by FLOPs at F_exp, so the extra bytes never bind. It matters when it binds: smaller ubatches, or `ep` with few pairs per rank.
- Roofline per op class: no change on this data. Every class is bound by compute at prefill and by bytes at decode, where the sum and the max agree. It matters in mixed regimes.
- Single on the ranks path: without it (the pre-ranks single features, with no expert rate and no output rows) the band rises to 0.52 and 30B single pp misses by +0.64.
- All-reduce latency share: without it (share 1) the worst residual is 0.16, and tg misses by -0.16 (30B tp-2), -0.16 (32B tp-2) and +0.16 (30B single). The fit then trades c0 against the all-reduce.
- Host link held at the prior: no change once the share is fitted (c_step 84 us instead of 82). With the multiplicative form the free link had fitted to 1.4 GB/s and stood in for c_step.
- One factor on the whole all-reduce time (`--ar-form scale`, the form not taken) fits 0.814 here and 0.633 on S2e. Its anchor residuals stay at up to 0.13 here, with tp-2 pp high and tg low on every anchor. Candidates: 30B single tg +0.120, 30B tp-2 tg -0.112, 32B tp-2 tg -0.126, ep-2 pp +0.155. Cross-box: 30B single tg +0.213, 30B tp-2 pp +0.167 and tg -0.131, 32B tp-2 pp +0.224.
- Not identified: `c_range` against the share (ADR 0070 History); `c_range` fits to 0 in every fit.
- Pick (ADR 0068, smallest predicted T = 2048 / pp + 128 / tg), v2 on this fit. 30B predicted: ep-2 1.11 s, single 1.17, layer-2 1.17, tp-2 1.18. Measured: layer-2 1.04, single 1.15, tp-2 1.18, ep-2 1.19 (unclean). The pick moves from `single` to `ep-2`, which measured slowest; the model has ep-2 prefill faster than single, and measured it is slower. Layer-2 is the measured best only under this host's cap. 32B: predicted tp-2 4.11, single 6.15, layer-2 6.15; measured tp-2 3.93, layer-2 5.17, single 6.51. Pick tp-2, the measured best.
- With M = 4 plans (ADR 0077) the model pipelines layer-2's prefill. On the `two_3090` prior fixture this raises the 30B layer-2 pp 1.45x (`crates/planner/tests/snapshots/multi_predictions.json`). Applied to this fit, that would make layer-2 the 30B pick; not computed on the fit, since the replica does not pipeline.
- Known gap under pipelining: the MoE shard builder skips the last layer's FFN for the chunks before the last (ADR 0077). The model charges it for every chunk, a few percent of 30B layer-2 pp.
- Pipelined prefill (ADR 0077) is in the model since; this session's candidate files carry `prefill_micro_batches = 1`, so the rows above are unchanged by it. New layer-N and tp-pp plans carry 4 and are predicted pipelined.
