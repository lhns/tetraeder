# Session s3f2-20261120

- Status: clean for the criteria below: P1, P2 and P3 pass, no 0x20 sample in any run. Rerun of s3f-20261120 (see `bench/results/s3f-20261120/README.md`: its gpu-parity gate failed on the host-memory timing test `save_through_pinned_staging`, now skipped in the gate by 98795ee; the criteria below are unchanged and were written before either session measured).
- Scripts: `scripts/s3f_confirm_box.sh` (stage A: `phase5_box.sh` with ROUNDS=1; stage B: default against `GGML_CUDA_DISABLE_GRAPHS=1`), `scripts/s3f_analyze.py` (`python -I scripts/s3f_analyze.py bench/results/s3f2-20261120 bench/results/s3c-20261119 bench/results/s3e-20261120` writes `analysis.txt`).
- Stage A is the s3c-style rehearsal: gates (now including the ggml-sys test `gpu_no_capture_flag_skips_the_cuda_graph` under `GGML_CUDA_GRAPH_DEBUG=1`), anchors, calibrate, predict, one round of every candidate and baseline, pp-overlap. Results in `phase5/`. A gate failure (rc=3) ends the session after stage A.
- Stage B (then part C, below): per model (30B first), tp2-pp2 and layer-4, both at `prefill_micro_batches` 4 (stage A's plans with that line set), arms `default` and `nograph` (`GGML_CUDA_DISABLE_GRAPHS=1`) in the order default, nograph, nograph, default, so two runs per arm. Each run is `bench -p 2048 -n 128 -r 5` (one warm-up and 5 timed reps per row, so `samples_ns` has 5 values for pp and for tg), no `GGML_CUDA_GRAPH_DEBUG`, no trace. 1 Hz `gpu-samples.csv` with throttle reasons, and a `<run>.window.txt` per run to cut it.
- Caveat for every number: one box, one night, clocks not lockable. Compare only the pairs run back to back here. s3c numbers on the same host are context (other build, other calibration), not a pair.

## Criteria (written before measuring)

- **P1 (gate):** `gates.json` has `gpu_parity` ok, and the ggml-sys test `gpu_no_capture_flag_skips_the_cuda_graph` ran and passed rather than being skipped: its `test ... ok` is in `phase5/gpu-parity.log` (ggml's debug lines may sit between the name and `ok`, because the log is `--nocapture`), and the log has no `skip:` line.
- **P2 (flag live):** for each model and configuration in stage B, in the default arm pp, rep 1 of each default run is within 5% of the median of reps 2 to 5 of that run, and the default pp median is within 5% of the nograph pp median (paired, same session; each arm's median is over all timed reps pooled over its two runs). Any miss, or a missing run, fails P2.
- **P3 (decode keeps replay):** for each model and configuration, tg medians pooled as in P2. 32B: default tg median > nograph tg median. 30B (MoE): ggml-cuda can disable graphs itself (`mul_mat_id_needs_sync`), so replay may not help there; it passes if default tg is not slower than nograph by more than 2% (the ratio is reported). A 32B default not faster, or a 30B default more than 2% slower, fails P3 and suggests the flag leaked onto decode. (Amended by the coordinator before the box ran.)
- **Report only:** per candidate, stage A pp and tg ratios to the session's prediction (mean over the rounds), next to s3c-20261119's ratios (`bench/results/s3c-20261119/`), with every |ratio-1| > 0.15 marked. Anchor pp and tg in stage A against s3c's anchors on the same host (expectation only, not judged). Any 0x20 throttle samples per B run (and stage A's unclean runs from `verdict-notes.json`).
- **Part C (report only; after B, 30B only, tg128 `-r 5`, no trace, 20 ms clock log and window file per run, skipped if fewer than 10 min of the B+C budget remain):** (a) layer-4 truncated to `--layer-limit 36`; (b) the full `single` plan pinned to each GPU in turn (`CUDA_VISIBLE_DEVICES` 0, 1, 2, 3). The tg of each is reported next to s3e's layer-4 L12, L24 and full-depth numbers.
- Layer-4 at mb4 follows ADR 0081's verification line ("tp2-pp2 and layer-4 at mb4"); the mb4 plan is stage A's layer-4 plan with `prefill_micro_batches = 4`, so it differs from the as-is layer-4 candidate of stage A only in that line (the script logs the old value).

## Results

- **Verdicts: P1 PASS, P2 PASS, P3 PASS.** Full numbers in `analysis.txt` (`python -I scripts/s3f_analyze.py bench/results/s3f2-20261120 bench/results/s3c-20261119 bench/results/s3e-20261120`).
- Code: branch box-s3f-20261120 at fe14d25 (98795ee plus this README; gpu-parity skips `save_through_pinned_staging`). Box 61637569, host 20210 (Finland), 4x RTX 3090, 1.203 USD/h, created 13:17, destroyed 14:07 local (50 min), after a verified fetch (560 KB). Credit 38.75 to 37.67 USD (about 1.08 USD). Gates: `cuda: ok`, power limit 350 W = default on all four, idle gate 10/10, memory.max 259 GB, CPU quota 30. Models provisioned: 8B, 32B, 30B (the gpu-parity tests read the 8B).
- Timeline (box clock, one hour behind local): launch 12:28:51, stage A done 12:57:50 (rc=0, gpu-parity ok, gates ok, summary written), stage B and part C 12:57:50 to 13:06:34, `S3F_CONFIRM_DONE rc=0`. About 38 min of script time, well under the 90 min plan.
- The first attempt, `bench/results/s3f-20261120`, stopped at gpu-parity (8B missing, then the KV save timing test); its numbers are not used.
- **P1:** gates.json gpu_parity ok; `test gpu_no_capture_flag_skips_the_cuda_graph ... ok` in `phase5/gpu-parity.log`; no `skip:` line.
- **P2:** default rep 1 is within 0.2% of the median of reps 2 to 5 in every default run (8 default runs), and default against nograph pp medians differ by at most 0.1%: 30B tp2-pp2 6549 vs 6546, 30B layer-4 11433 vs 11425, 32B tp2-pp2 2187 vs 2190, 32B layer-4 2763 vs 2763 tok/s. Against s3e (graphs off, other night): 30B 6072 and 32B 2004 for tp2-pp2 mb4, so this night is 8% and 9% higher; the paired comparison here is the point, not the cross-night one.
- **P3:** default tg median over nograph tg median: 30B tp2-pp2 156.9 vs 76.0 (+106%), 30B layer-4 177.6 vs 147.2 (+21%), 32B tp2-pp2 56.5 vs 53.6 (+5.5%), 32B layer-4 38.3 vs 36.4 (+5.1%). Decode keeps replay (the 30B MoE too: its tg more than doubles with graphs on at tp2-pp2).
- **Stage A, measured over predicted (ratio), s3f2 against s3c** (mean of the rounds; s3f2 had one round; * marks |ratio-1| > 0.15):

| Model | Candidate | s3f2 pp | s3f2 tg | s3c pp | s3c tg |
|---|---|---|---|---|---|
| 30B | ep-4 | 0.922 | 1.065 | 0.842* | 1.174* |
| 30B | layer-4 | 1.030 | 0.995 | 1.063 | 0.855 |
| 30B | single | 1.053 | 0.974 | 1.104 | 0.957 |
| 30B | tp2-pp2 | 0.918 | 0.998 | 0.780* | 0.999 |
| 32B | layer-4 | 1.072 | 1.008 | 1.054 | 0.980 |
| 32B | single | 1.080 | 0.995 | 1.038 | 0.977 |
| 32B | tp-4 | 0.903 | 1.031 | 0.932 | 1.034 |
| 32B | tp2-pp2 | 0.957 | 0.993 | 0.690* | 1.022 |

  No s3f2 candidate is past 15%; s3c had three (30B ep-4, 30B tp2-pp2, 32B tp2-pp2, all pp or tg). The tp2-pp2 prefill ratios moved from 0.78 and 0.69 to 0.92 and 0.96, the effect of patch 0006. The predictions are the session's own (recalibrated), so the ratios are not on the same prediction.
- **Anchors against s3c, same host** (expectation only): pp within -4% to +1% for single, +9% to +17% for tp-2 and tp-4 (the tp prefill gain follows the fix); tg -2% to -4% for the 32B single and tp-2, but -10% to -15% for the 30B tp-2 and single, 30B 24-layer single, and the 32B tp-4 tg (-10%). The 30B and tp-4 decode anchors are lower than s3c's; a one-night difference is not excluded, as no paired run was made.
- **Part C** (30B tg128, median of 5 reps, tok/s): layer-4 L36 231.8; single, full model, on GPU 0 to 3: 181.0, 180.9, 181.1, 181.2 (the four cards agree within 0.2%). s3e layer-4: L12 649.5, L24 384.2, full 186.6.
- **Throttle:** 0x20 samples in B and C runs: 0. Stage A unclean runs (`verdict-notes.json`): 0.
- Stage A verdict.json is a rehearsal (ROUNDS=1); its pass flag is not part of this session's criteria.
- Not committed: `gpu-parity.log.full` stayed on the box; the `phase5/plans` and `plans/` toml are in the results.
