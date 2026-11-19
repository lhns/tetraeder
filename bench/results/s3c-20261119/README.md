# Session s3c-20261119

- Status: unclean for D4 (verdict null: `llama-bench -sm tensor` on the 32B has 1 clean round, need 2), M4 clean and failed (`prediction_ok` false). `phase5_pass` false. Phase 5 S3c: third 4-GPU rehearsal, ROUNDS=2 (reruns for unclean runs), with recorded routing frequencies in `predict`, the tp-4 anchor and the tp2-pp2 overlap stage (ADR 0068 History 2026-11-19, ADR 0078 History).
- Rule: ADR 0068 (band 0.08 provisional, throttle by sampler window per GPU, per-configuration share comparison, two clean rounds per configuration, up to two reruns).
- Code: 27c6e00 (branch box-s3c-20261119). Box 61430359, host 20210 (Finland), 4x RTX 3090, driver 595.91.07, AMD EPYC 7282 (32 threads, CPU quota 30), memory.max 259 GB, NCCL 2.25.1, p2p none.
- Decode toggles: `TETRAEDER_RANGE_UID=1` and `TETRAEDER_ASYNC_INPUTS=1` were set for this run, because the keep rule of ADR 0069 History 2026-11-19 was met by each toggle alone in both 30B rounds with the parity step green (`bench/results/s3c-20261119-decode-decode/README.md`). Every our-engine number below (anchors, candidates, A/A, pp-overlap) is with both on. The llama.cpp baselines are unaffected.
- Power limit 350 W = default on all four GPUs. Idle gate 10/10, co-tenant sample at SSH 0.17 host cores busy over 20 s, `cuda: ok`, gpu-parity ok, comm_probe ok, baseline_modes ok (gates.json).
- Topology: GPU0-GPU1 PHB, GPU2-GPU3 PHB, every other pair NODE, one NUMA node.
- Box: 1.192 USD/h plus about 0.03 USD/h traffic, created 11:45, destroyed 12:50 local (65 min), after a verified fetch. About 1.3 USD box plus about 0.1 traffic (credit 20.12 to 18.36 USD, shared counter). First offer, no gate failure.
- Timing (box clock, one hour behind local): phase5_box.sh 11:23 to 11:48 (25 min: build 0.3 min on the cached build, gpu-parity 6.4, probe and prepass 1.7, anchors 3.3, calibrate and predict 0.1, rounds and reruns 8.6, pp-overlap 3.1, vLLM check 1.9 and skipped: no AWQ/GPTQ checkpoint).
- Not committed: `gpu-parity.log.full` (stayed on the box). `phase5_box.log` is the box log.

## Layer splits and candidates

- Qwen3-32B: single, layer-4 (16/16/16/16), tp-4, tp2-pp2. Qwen3-30B-A3B: single, layer-4 (12/12/12/12), ep-4, tp2-pp2 (tp-4 pruned: 3 blocks over 4 ranks).

## Routing record (`routing-qwen3-30b-a3b.json`, sha256 11f608d4, 30B, pp2048 and tg128 of the judged token stream)

Busiest ep-4 rank (contiguous spans of 32 experts) as a share of the layer's routed rows, 48 layers. Uniform routing gives 0.25 (aggregate) and 0.437 per decode step (top-8 Monte Carlo).

| | mean | min | max |
|---|---|---|---|
| prefill, busiest-rank share of rows | 0.391 | 0.290 (L25) | 0.544 (L12) |
| decode, busiest-rank share of the 128 tokens' counts | 0.387 | 0.288 | 0.535 |
| decode, per-step busiest-rank share (my Monte Carlo, 4000 draws of 8 distinct experts, not the planner's code) | 0.432 | 0.391 | 0.514 |

Layers above 0.45 in prefill: L6 0.47, L12 0.54, L24 0.50, L30 0.45, L35 0.51, L41 0.52, L45 0.52. The per-step decode skew is small against uniform (0.432 against 0.437 on average) because 8 distinct picks even out the ranks; the prefill skew is large (0.391 against 0.25).

## Calibration (10 anchors incl. 32B tp-4 at L16 and L32, cost model v3, one fit over both models)

- Band 0.108 (s3b: 0.102). Fitted `a` (`allreduce_latency_s`) 2.686 us, `b` (`allreduce_step_s`) 5.007 us: fitted, because the anchors now span 2 and 4 ranks. `c_range_graph_s` 3.31 us, `c_step_s` 20.8 us.
- The group-size term is real: a ring over 4 ranks costs `a + 6 b` = 32.7 us per all-reduce against 2.7 + 2 x 5.0 = 12.7 us over 2 ranks.

## Predicted vs measured (err = pred/meas - 1; `x` marks beyond 0.15; clean rounds last)

| Case | Candidate | pred pp / tg | meas pp / tg | err pp / tg | clean rounds |
|---|---|---|---|---|---|
| D4 | single | 1252 / 39.5 | 1300 / 38.6 | -0.037 / +0.024 | 2 |
| D4 | layer-4 | 2784 / 39.7 | 2934 / 38.9 | -0.051 / +0.020 | 2 |
| D4 | tp-4 | 1609 / 78.9 | 1497 / 81.6 | +0.075 / -0.033 | 2 |
| D4 | tp2-pp2 | 2064 / 58.8 | 1425 / 60.1 | +0.449 x / -0.022 | 2 |
| M4 | single | 5065 / 214.4 | 5591 / 205.3 | -0.094 / +0.045 | 2 |
| M4 | layer-4 | 11213 / 218.5 | 11918 / 186.9 | -0.059 / +0.169 x | 2 |
| M4 | ep-4 | 4801 / 152.8 | 4040 / 179.4 | +0.188 x / -0.148 | 2 |
| M4 | tp2-pp2 | 6512 / 184.6 | 5079 / 184.4 | +0.282 x / +0.001 | 2 |

Against s3b: tp-4 is now priced within 0.08 on both phases (s3b +0.09 / +0.10, tg over before the tp-4 anchor); ep-4 tg went from +0.444 to -0.148 (within the line) and ep-4 pp from +0.350 to +0.188 (routing record priced in, still over); tp2-pp2 pp is unchanged (+0.449 and +0.282, see the overlap below). New miss: 30B layer-4 tg is over-predicted by 0.169 (218 predicted against 187 measured; s3b +0.099); the measured value is with the decode toggles on, so it is not the s3b configuration. The 30B single tg is also over by 0.045 and measured 205 (s3b 196), the toggles' effect on the single GPU (+4.8 to +5.5% in `s3c-20261119-decode-decode`).

## Picks and winners

- D4: planner picks tp-4; winner set {tp-4}; `pick_ok` true. `prediction_ok` false (tp2-pp2 pp +0.449). Best baseline not established: `llama-bench -sm tensor` has 1 clean round (runs r2 and r3 had a 0x48 and a 0x20 window sample, r4 a 0x4 share 0.25 against a median 0.16, more than 0.15 apart); `-sm layer` t 3.927 s. Informal, not judged: tp-4 t 2.937 s against `-sm tensor` 3.022 s (1.03x) and `-sm layer` 3.927 s (1.34x).
- M4: planner picks layer-4; winner set {layer-4}; `pick_ok` true. Best baseline `llama-bench -sm layer` t 0.808 s (tensor 1.227 s); layer-4 t 0.857 s: baseline ratio 0.943 (needs at least 0.92), `baseline_ok` true. `prediction_ok` false (layer-4 tg +0.169, ep-4 pp +0.188, tp2-pp2 pp +0.282). A/A t 0.988 s against 0.987 s (0.09%). `clean` true, `pass` false.
- D4 A/A: 4.888 s against 4.891 s (0.06%).

## verdict.json fields

- `phase5_pass` false; M4 `{pick_ok: true, prediction_ok: false, baseline_ratio: 0.943, baseline_ok: true, clean: true, pass: false, reasons: []}`; D4 `{pick_ok: true, prediction_ok: false, baseline_ratio: null, baseline_ok: null, clean: false, pass: null, reasons: ["llama-bench -sm tensor: 1 clean round(s), need 2"]}`.

## pp-overlap (tp2-pp2, prefill_micro_batches 1 against 4, two interleaved rounds, not judged)

| Model | pp mb1 (median) | pp mb4 (median) | ratio mb4/mb1 | tg mb1 / mb4 |
|---|---|---|---|---|
| Qwen3-32B | 1306.3 | 1413.4 | 1.082 | 60.1 / 60.1 |
| Qwen3-30B-A3B | 4055.5 | 5007.8 | 1.235 | 182.8 / 184.2 |

The pipeline of two tp-2 stages overlaps, but by 8% (32B) and 23% (30B). The model's tp2-pp2 pp predictions (2064 and 6512) stay over the mb4 measurements (1413 and 5008) by 0.46 and 0.30, so the micro-batch overlap alone does not close them; no model change was made (ADR 0078 History).

## Throttle and co-tenant

- 58 co-tenant files, none with `cotenant_ok` false; largest outside-container CPU in a decode window 0.73 cores, smt pairs busy 0.
- Unclean runs (all on the 32B, all rerun): `single-r2` and `tp-4-r2` one 0x20 sample on GPU 0 each, `llama-bench-tensor-r2` one 0x48 sample on GPU 3, `-tensor-r3` one 0x20 sample on GPU 0, `-tensor-r4` share rule (0x4 share 0.25 on GPU 1 against the median 0.16). GPU 0 shows the 0x20 (SW thermal slowdown) on the 32B again, as on s3 and s3b on this host. The 30B and pp-overlap runs were clean.
- `phase5/gpu-samples.csv` (0.5 MB, 1 Hz, all GPUs) is committed.

## Failures and deviations

- None in the session. The 32B tensor baseline never reached two clean rounds after two reruns (0x20 and 0x48 sample windows, then the share rule), so D4's verdict is null.
- vLLM skipped (no AWQ/GPTQ checkpoint).
