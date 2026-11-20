# Session v2-20261120

- Status: **null** (no verdict for D2 or M2). Judged under ADR 0068 (clean rule, sampler throttle, A/A, two clean rounds per configuration). Not enough clean rounds, 34 unclean runs: 0x20 thermal samples (28 runs, both GPUs), co-tenant guard (11 runs, 19 to 27 host cores busy outside the container against a limit of 19.1 of 127), throttle-share spread (15 runs). The D2 A/A run (`qwen3-32b-single-aa`) is void (co-tenant), so D2 has no `t_end`.
- Code: branch box-v2-20261120 at 3d8a455 (main). `phase5_box.sh` with `SESSION=v2-20261120 NGPUS=2 BUDGET_MIN=150`, default MODELS and ROUNDS 3, up to two rerun rounds (RERUNS 2); `PHASE5_BOX_DONE rc=0`. Script time 30 min (box clock 18:18:58 to 18:49:02), well under the 150 min budget.
- Box 61681450, host 87485 (Spain), 2x RTX 3090 (driver 620.70.06), CPU quota 54, NCCL 2.25.1, no peer access, comm_impl nccl. Created 19:15, destroyed 19:50 local (35 min), after a verified fetch (480 KB). Power limit 350 W = default on both GPUs, idle gate 10/10, `cuda: ok`. Gates: gpu_parity, comm_probe, baseline_modes all ok.
- Cost: billed 51.4 GB traffic 0.47 USD, box 0.799 USD/h for 35 min about 0.47 USD, about 0.94 USD in total; credit 37.55 to 36.65 USD (0.90, the display may lag). One-off cap 0.92 USD/h all-in for this host (AGENTS.md); the hourly 0.799 is 0.02 above the offer's 0.779, so the all-in estimate was slightly over 0.92. Two earlier boxes (hosts 208874 and 167329, `--min-inet 500`) stalled in image pull for over 10 minutes and were destroyed (about 0.2 USD together).
- Band 0.08 (provisional `cuda-default`). Prediction sha256 in `phase5/verdict.json`.

## Clean rounds per configuration (need 2)

| Case | Configuration | Clean rounds |
|---|---|---|
| D2 | single | 2 |
| D2 | layer-2 | 2 |
| D2 | tp-2 | 1 |
| D2 | llama.cpp -sm layer, -sm tensor | 2 or more (not listed as short) |
| M2 | single | 2 |
| M2 | layer-2 | 2 |
| M2 | tp-2 | 1 |
| M2 | ep-2 | 1 |
| M2 | llama.cpp -sm layer | 1 |
| M2 | llama.cpp -sm tensor | 0 |

## Not judged: predictions against measurements

Medians over the clean rounds only. e = predicted / measured - 1. No candidate is past 15% (the largest is 4.7%). These are not verdict numbers, because the cases are null.

| Case | Candidate | e_pp | e_tg | Measured T (s) | Predicted T (s) |
|---|---|---|---|---|---|
| D2 | single | +0.2% | +1.5% | 5.002 | 4.948 |
| D2 | layer-2 | -0.9% | -4.6% | 4.222 | 4.387 |
| D2 | tp-2 | +0.9% | +1.4% | 3.659 | 3.615 |
| M2 | single | -2.4% | +4.2% | 0.997 | 0.980 |
| M2 | layer-2 | -1.8% | -2.3% | 0.852 | 0.871 |
| M2 | tp-2 | -4.7% | -0.7% | 1.132 | 1.159 |
| M2 | ep-2 | -4.6% | +3.9% | 1.151 | 1.145 |

## Not judged: pick and baseline

- D2: analytic pick tp-2 (predicted lowest T, 3.615 s). Measured lowest T is tp-2 (3.659 s); layer-2 4.222 s is 15% behind, so the pick would be in the winner set if the case were clean. Baselines: llama.cpp -sm tensor 3.696 s, -sm layer 4.231 s. T_llama_best / T_pick = 1.010 (band gate 1 - 0.08).
- M2: analytic pick layer-2 (0.871 s). Measured lowest T is layer-2 (0.852 s); single 0.997 s and tp-2 and ep-2 at about 1.13 to 1.15 s are outside the band. Baselines: -sm layer 0.866 s (1 clean round), -sm tensor has no clean round. T_llama_best / T_pick = 1.016 against -sm layer only.
- A/A (M2): 0.9954 s at start, 1.0012 s at end (+0.6%, within the band). D2 A/A: start 5.001 s, end void.

## SM clocks (median over each configuration's window samples, all runs, pp and tg together; the sampler column is the SM clock in MHz)

- Dense 32B: single GPU 0 1725; tp-2 1725 / 1732; layer-2 1920 / 1875; llama.cpp layer 1905 / 1890, tensor 1890 / 1905.
- MoE 30B: single 1725; tp-2 1725 / 1725; ep-2 1725 / 1725; layer-2 1890 / 1725; llama.cpp layer 1950 / 1747, tensor 1935 / 1950.
- The own-executor shapes (single, tp-2, ep-2) ran at 1725 MHz and llama.cpp at 1890 to 1950 MHz. 1725 is the clock the 0x200/0x400 board-limit share of the same windows follows. The sampler does not separate the pp and tg phases of one run.

## Findings

- The host is the problem: a co-tenant used 19 to 27 of 127 physical cores while runs went, and 0x20 appeared on GPU 0 or GPU 1 in 28 of 69 non-anchor runs. The script's reruns (up to 5 rounds) did not give two clean rounds for tp-2, ep-2 and the llama.cpp baselines.
- The cost model is within 5% of every clean median on 2x RTX 3090. The pp prediction error of tp-2 (the s3c miss) is gone on 2 GPUs.
- Host 87485 is added to the skip line in `docs/box-driver.md`.
- Next: V2 on another 2-GPU host. Raw data: `phase5/` (verdict.json, verdict-notes.json, gates.json, env.json, gpu-samples.csv, timeline.jsonl, predictions, plans and all run files).
