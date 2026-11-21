# Session v4a-20261121

- Status: clean. Verdict session on four GPUs, judged under ADR 0068 (band 0.08, provisional cuda-default) with cost model 5 (ADR 0082). D4 passes and M4 passes; `phase5_pass` is false because this session holds two of the four cases (V2 supplies D2 and M2, `--merge` combines them).
- Code: branch box-v4a-20261121, session ran on committed HEAD 692947a (main 5da9273 plus two skip-line commits). Box 61850705, host 20210 (Finland), 4x RTX 3090 at 350 W default (all four), driver 595.91.07, 32 cores, memory.max 259 GB, NCCL 2.25.1 (`comm_impl` nccl). Offer 57576210, 1.192 + 0.017 USD/h at the default traffic estimate, billed 1.203 USD/h; cap 1.22 (user, 2026-11-21, host 20210 re-admitted).
- Launch: `SESSION=v4a-20261121 NGPUS=4 RERUNS=4 BUDGET_MIN=120`, default MODELS and ROUNDS (3), LAYER_ANCHOR on by default at 4 GPUs. RERUNS=4 because ADR 0068 leaves the count open; it was not needed: every configuration had three clean rounds, no unclean run (`verdict-notes.json`), no rerun.
- Timeline (local, box clock one hour behind): created 20:47, ready 20:48 (pull under 1 min), provision and 52 GB of models by 20:57 (about 75 MB/s), launched 20:58, build to 21:07, gpu-parity to 21:15, anchors 21:17 to 21:20, rounds 21:20 to 21:33, `PHASE5_BOX_DONE rc=0` 21:33. Fetch verified at 21:33 (1621 KB tgz, DONE marker).
- Gates: `cuda: ok`; power gate 350 = default on all four; idle gate 10/10 (the 0x4 bit on GPU 3 in some samples is not disqualifying); gpu-parity ok; comm_probe ok; baseline_modes ok (llama.cpp `layer` and `tensor` accepted, `row` rejected; ik_llama `layer`, `row`, `graph` prepassed, report only). No GPU had a 0x20 sample in any run. Not run: pp-overlap (ROUNDS=3), vLLM (no checkpoint).
- Earlier hosts this evening, all destroyed before any session: 208874 (second stall, 10 min loading), 685226 (all four GPUs capped, 280/280/250/300 W), 399360 (320 W against 350). They are in the box-driver skip line.
- Cost: box 1.203 USD/h from 20:47, traffic 52 GB billed 0.11 USD at 21:35. The box was kept after this session and reused for V2 (`v2c-20261121`, two of the four GPUs), so the total box cost is in that README; this session alone ran 20:47 to 21:34 (about 0.95 USD plus traffic).

## Verdict

Both cases `pass: true` in `phase5/verdict.json` (`predict_sha256` and the log precede the first candidate run).

| Case | Pick (analytic) | Winner set W (T within 8% of best) | Pick passes | Prediction within 15% | Baseline gate | A/A (T start, end) | Clean | Pass |
|---|---|---|---|---|---|---|---|---|
| D4 (32B) | tp-4 | tp-4, tp2-pp2 | yes | yes, every candidate | 1.022 (best llama.cpp: `tensor`, T 2.904 s) | 4.870, 4.879 (+0.2%) | yes | yes |
| M4 (30B-A3B) | layer-4 | layer-4 | yes | yes, every candidate | 0.987 (best llama.cpp: `layer`, T 0.841 s) | 1.003, 1.005 (+0.2%) | yes | yes |

Baseline ratio r = T_llama_best / T_pick against the gate 1 - b = 0.92. Measured T: D4 tp-4 2.840 s, tp2-pp2 3.035, layer-4 3.985, single 4.876. M4 layer-4 0.852 s, single 1.004, tp2-pp2 1.028, ep-4 1.203. M4 `tp-4` was pruned by the enumeration (3 blocks over 4 ranks, expert down rows), as in s3g.

Verdict line: "D4 passes and M4 passes at 3 clean rounds each, picks tp-4 (D4) and layer-4 (M4), every candidate within 15% on pp and tg, baseline gate 1.022 and 0.987."

## Measured over predicted (pp, tg; one ratio per candidate, medians of three clean rounds)

None is past 15% (largest: D4 tp-4 pp 0.929, M4 single pp 1.060, D4 layer-4 pp 1.063).

| Case | Candidate | pp | tg | T meas / pred (s) |
|---|---|---|---|---|
| D4 | single | 1.047 | 0.989 | 4.876 / 4.912 |
| D4 | layer-4 | 1.063 | 1.001 | 3.985 / 4.033 |
| D4 | tp-4 | 0.929 | 1.029 | 2.840 / 2.803 |
| D4 | tp2-pp2 | 0.990 | 1.008 | 3.035 / 3.044 |
| M4 | single | 1.060 | 0.981 | 1.004 / 1.012 |
| M4 | layer-4 | 1.014 | 1.007 | 0.852 / 0.859 |
| M4 | ep-4 | 0.959 | 1.057 | 1.203 / 1.230 |
| M4 | tp2-pp2 | 0.966 | 1.025 | 1.028 / 1.037 |

Against s3g (same host, one round, cost model 5 with k = 1 for the 30B): 30B layer-4 tg 0.957 there, 1.007 here, so k = 1.062 fixes the layer-4 decode that ADR 0082 was written for. The rest are within about 0.04 of s3g's.

Baselines (llama-bench, three rounds, pp2048 and tg128 medians): 30B layer 11176 and 194.7 tok/s, tensor 4513 and 164.7; 32B layer 2879 and 39.6, tensor 1688 and 75.7.

## Clean rounds per configuration

Three clean rounds for every candidate, every llama.cpp baseline (layer, tensor, both models) and the A/A run; `unclean_runs` is empty. Throttle shares per configuration are in `verdict.json` (`throttle`); the largest cap share is the 32B llama.cpp `layer` baseline on GPU 1 at about 0.14, inside the 0.15 spread rule.

## Cost model 5 (ADR 0082)

`decode_clocks` lines from the calibrate stage:
- `decode_clocks qwen3-32b: busy_mhz=1627.5 low_mhz=1695 k=1.000 (busy clock within 15 MHz of the low clock or below it)` (the 32B single is power-capped, 0x4, 1650 to 1665 in the timed tail, below the low clock; the clamp gives 1, as the ADR predicts)
- `decode_clocks qwen3-30b-a3b: busy_mhz=1800 low_mhz=1695 k=1.062`. The 10 ms sampler gave enough tail samples for the 30B single anchors (20 samples in the 0.20 s L12 tail, more at L24), so k is applied to the 30B (s3g withheld it).

Held out: the layer anchor `layer-4@L24` (6/6/6/30 at limit 24, tg128 -r 10, two rounds; 362 tail samples per GPU, all four at 1695 MHz, 0x0) measured tg 341.0 tok/s; predicted over measured minus 1 = -0.024 (limit 0.05). ADR 0082's box confirmation (done criterion item 5: single boosts, `busy_mhz > low_mhz`, held-out tg within 0.05) passes with k applied to the 30B: 1800 against 1695, -0.024.

Decode tail clocks from the 20 ms-class logs (10 ms here), cut to the timed tail with the tail rule of `scripts/decode_clocks.py`, median per GPU:
- 30B: single 1800 (GPU 0), tp2-pp2 1800 on all four (72 samples, 0x0), ep-4 1800 on all four, layer-4 1695 on all four (66 samples; GPU 2 at 1800 in round 2).
- 32B: single 1650 to 1665 (0x4), tp-4 1800, tp2-pp2 1800, layer-4 1695.
- tp2-pp2 is in the busy state on both models, consistent with ADR 0082's duty threshold 0.32 (modeled 0.34 to 0.44); this is the second logged decode of tp2-pp2 after s3g.

Calibration residuals (`calib.json`, fit): worst per-anchor residual +0.075 (32B tp-4 pp at L16 and L32), tg residuals all within 0.013.

## Notes and deviations

- Created three boxes before this one (208874, 685226, 399360), each destroyed before provisioning or at the power gate, about 0.3 USD in total; the stall rule and skip entries are in `docs/box-driver.md`.
- The image pull of 20210 took under a minute this time (cached).
- BUDGET_MIN 120 against the 3 h box cap; the session used 35 minutes of it.
- Not judged: ik_llama (prepassed only) and vLLM (skipped: no AWQ or GPTQ checkpoint on the box).
