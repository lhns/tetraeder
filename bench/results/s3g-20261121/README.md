# Session s3g-20261121

- Status: clean for R1 and R4 (R1 passes literally but exposes a plumbing gap, see below); R2 and R3 pass at k = 1 for the 30B, so they did not test the v5 correction. No 0x20 or other slowdown bit in any timed tail used. Rehearsal (ROUNDS=1), verdict null as expected.
- Code: branch box-s3g-20261121, session ran on committed HEAD 17249fa (main f204ed4 plus this README). Box 61810757, host 20210 (Finland), 4x RTX 3090, driver 595.91.07, AMD EPYC 7282, 32 cores, memory.max 259 GB. One-off cap 1.22 USD/h for this box (user, via coordinator). Offer 57576210, 1.192 + 0.017 traffic at default download estimate; billed 1.203 USD/h.
- Created 15:09:45 local, destroyed 15:59:47 (50 min), after a verified fetch (930 KB tgz, 44 runs, every one with clocks-*.csv and window-*.txt). Credit 36.16 to 35.21 USD (0.95 USD). Image pull took 7.5 min (loading until 15:17), under the 10 min limit.
- Gates: `cuda: ok`; power limit 350 W = default on all four; idle gate 10/10 (0x1 only); gpu-parity ok, comm_probe ok, baseline_modes ok. Script timeline (box clock, one hour behind): launch 14:28:04, build 8.7 min, gpu-parity to 14:45, anchors 14:46 to 14:50, rounds 14:50 to 14:54, pp-overlap to 14:57, `PHASE5_BOX_DONE rc=0` 14:59:05.
- Analysis: `analysis.txt` (`scripts/s3f_analyze.py`; its P1 to P3 and part C lines do not apply here, and its "s3f" columns are this session, "s3c" columns are s3f2).

## Criteria (written before measuring)

- **R1 (plumbing):** the layer anchor ran (two rounds); `clocks-*.csv` and `window-*.txt` exist for every timed run of both engines; calibrate logged one `decode_clocks <model>: ...` line per model. Report k and the reason when k = 1.
- **R2 (held-out):** the layer anchor's measured tg is within 0.05 of its v5 prediction.
- **R3:** 30B layer-4 `e_tg` (pred/meas - 1) is within 0.07.
- **R4:** the 30B tp2-pp2 decode tail clock (from its clocks file, cut to the timed tail with `scripts/decode_clocks.py`) is in the busy state, above the low clock by more than 15 MHz.
- **Report only:** every candidate's pp and tg ratios against prediction, next to s3f2's.
- **Clock-limited host:** if busy_mhz is within 15 MHz of low_mhz (as s3f2 at 1590), then k = 1. R2 to R4 then test only the plumbing, and only R1 is judged.

## Results

- **Host state:** not clock-limited. 30B full-depth-class single runs sit at 1800 MHz (the withheld L12 and L24 singles, 10 to 18 tail samples), the layer anchor's GPUs at 1695 (184 tail samples each, both rounds, 0x0), so busy 1800 against low 1695 is 105 MHz, k = 1.062 if it had been used. The 32B single is power-capped (0x4) at 1605 to 1665, median 1650, below low.
- **R1: PASS as written, with a gap.** Layer anchor `layer-4@L24` (6/6/6/30 at limit 24) ran in two rounds; clocks and window files exist for all 44 timed runs (anchors, candidates, llama.cpp baselines, A/A, pp-overlap); calibrate logged two lines:
  - `decode_clocks qwen3-32b: busy_mhz=1650 low_mhz=1695 k=1.000 (busy clock within 15 MHz of the low clock or below it)`
  - `decode_clocks qwen3-30b-a3b: busy_mhz=n/a low_mhz=1695 k=1.000 (no busy clock for layout model 'Qwen3 30B Gptq Fp16'; busy_mhz keys: 'Qwen3 32B Awq Compatible Instruct')`
  - **Gap:** all four 30B single anchors were withheld by `decode_clocks.py` for under 25 tail samples at 20 ms (10, 10, 17, 18 samples; tails of 0.20 and 0.35 s at 630 and 365 tok/s for one rep of 128 tokens). So k = 1 for the 30B is a sampling artifact, not the host: the correction ADR 0082 is about was never applied to the model it was written for. The 32B k = 1 is real (single below the low clock, as the ADR predicts). Fix options: more timed reps on the 30B single anchors (the layer anchor uses -r 10), or a lower MIN_SAMPLES, or a 10 ms sampler. Not changed here.
- **R2: PASS.** Layer anchor tg measured 341.5 tok/s, held out predicted / measured - 1 = +0.019 (limit 0.05), at k = 1. Offline with the 30B busy clock set to 1800 (`scripts/costmodel_replica.py --clocks`, file not committed): -0.021.
- **R3: PASS.** 30B layer-4 tg 186.9 measured against 195.3 predicted: `e_tg` +0.045 (limit 0.07), at k = 1. Offline at k = 1.062: 186.7 predicted, -0.001.
- **R4: PASS.** 30B tp2-pp2 decode tail (0.744 s, 37 samples per GPU, 0x0): 1800 MHz on all four GPUs, 105 MHz above low (limit 15). Other tails: 30B ep-4 1800, 30B layer-4 1695 (all four, 34 samples), 30B single 1800, 32B tp2-pp2 1800, 32B tp-4 1800, 32B layer-4 1695, 32B single 1650 (0x4).
  - This is the first logged decode clock of tp2-pp2: it is in the busy state, consistent with the ADR's 0.32 threshold choice (modeled duty 0.34 to 0.44).
- **Report only, measured over predicted (ratio), s3g against s3f2** (* marks |ratio-1| > 0.15; one round each; predictions are each session's own):

| Model | Candidate | s3g pp | s3g tg | s3f2 pp | s3f2 tg |
|---|---|---|---|---|---|
| 30B | ep-4 | 0.951 | 1.062 | 0.922 | 1.065 |
| 30B | layer-4 | 1.026 | 0.957 | 1.030 | 0.995 |
| 30B | single | 1.076 | 0.975 | 1.053 | 0.974 |
| 30B | tp2-pp2 | 0.945 | 1.008 | 0.918 | 0.998 |
| 32B | layer-4 | 1.063 | 1.001 | 1.072 | 1.008 |
| 32B | single | 1.046 | 0.989 | 1.080 | 0.995 |
| 32B | tp-4 | 0.942 | 1.029 | 0.903 | 1.031 |
| 32B | tp2-pp2 | 0.968 | 0.999 | 0.957 | 0.993 |

  No candidate past 15%. The 30B layer-4 tg ratio is 0.957 against s3f2's 0.995: on this host the layer-4 GPUs sit at 1695 while single boosts to 1800, as on s3c, and the 4% shortfall is what k would price. s3f2 had every GPU at 1590 and no gap.
- Anchors against s3f2 (same host): pp +5% to +9%, tg +2% to +10% (30B +8% to +10%). llama.cpp baselines ran in layer and tensor modes (row and graph rejected in the prepass). Unclean runs: none (`verdict-notes.json`).
- Verdict: null (`phase5_pass` false, cases D4 and M4 null), rehearsal.
- pp-overlap ran for both models, report only (`pp-overlap-*.json`).
