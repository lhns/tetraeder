# Session s3f-20261120

- Status: pending (box not yet run). Confirmation of ADR 0081 (patch 0006: prefill steps run without CUDA graph capture) on 4 GPUs, host 20210 preferred for comparability with s3c and s3e.
- Scripts: `scripts/s3f_confirm_box.sh` (stage A: `phase5_box.sh` with ROUNDS=1; stage B: default against `GGML_CUDA_DISABLE_GRAPHS=1`), `scripts/s3f_analyze.py` (`python -I scripts/s3f_analyze.py bench/results/s3f-20261120 bench/results/s3c-20261119 bench/results/s3e-20261120` writes `analysis.txt`).
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

Pending.
