# Session v2c-20261121

- Status: clean. Verdict session on two of the four GPUs of the V4a box (GPUs 2 and 3, new `P5_DEVICES` mode), judged under ADR 0068 (band 0.08, provisional cuda-default) with cost model 5 (no layer anchor on 2 GPUs, so no decode clock factor: `low_mhz` null, predictions as cost model 4). D2 passes and M2 passes; `phase5_pass` is false because the session holds two of the four cases (V4a holds D4 and M4, `--merge` combines them).
- Code: branch box-v4a-20261121, session ran on committed HEAD 4c73db7 (origin/main 46c826c, which has `P5_DEVICES`, plus the v4a results). Box 61850705, host 20210 (Finland), 4x RTX 3090 at 350 W default, driver 595.91.07, 32 cores (CPU quota 30, `-t 30`), NCCL 2.25.1. The session saw only GPUs 2 and 3 (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3` for every process; plan line `PLAN two of four: P5_DEVICES=2,3`). `env.json` still lists the four GPUs' names and limits (nvidia-smi is not narrowed), `gpu_count` is 2, p2p `none`.
- Launch: `SESSION=v2c-20261121 NGPUS=2 P5_DEVICES=2,3 RERUNS=4 BUDGET_MIN=100`, default models and ROUNDS (3). Pair choice: `nvidia-smi topo -m` shows GPU 2 and 3 on one host bridge (PHB), every other pair NODE, so 2,3 is the best-linked pair among 1, 2, 3 and GPU 0 is left out for its 0x20 history.
- Timeline (local, box clock one hour behind): launched 21:40, build cached, gpu-parity 21:40 to 21:48, anchors and calibrate to 21:49, rounds 21:49 to 21:58, `PHASE5_BOX_DONE rc=0` 21:58. Fetch verified (2647 KB tgz). The new mode ran without a first-run failure: no unclean run (`verdict-notes.json`), three clean rounds for every configuration, no rerun.
- Gates: power 350 = default on all four, idle gate 10/10 (V4a, same box), gpu-parity ok, comm_probe ok, baseline_modes ok.
- Cost: this box (61850705) ran 20:47 to 21:59 (1h12m at 1.203 USD/h, 1.43 USD) plus 0.11 USD traffic, shared with V4a. Credit 34.81 before the create, 33.28 after the destroy (1.53 USD for the box). Destroyed 21:59 after both fetches, confirmed gone in `status`.

## Verdict

Both cases `pass: true` in `phase5/verdict.json`.

| Case | Pick (analytic) | Winner set W | Pick passes | Prediction within 15% | Baseline gate | A/A (T start, end) | Clean | Pass |
|---|---|---|---|---|---|---|---|---|
| D2 (32B) | tp-2 | tp-2 | yes | yes, every candidate | 1.011 (best llama.cpp: `tensor`, T 3.595 s) | 4.830, 4.836 (+0.1%) | yes | yes |
| M2 (30B-A3B) | layer-2 | layer-2 | yes | yes, every candidate | 1.008 (best llama.cpp: `layer`, T 0.883 s) | 1.000, 0.999 (-0.1%) | yes | yes |

Gate: r = T_llama_best / T_pick against 1 - b = 0.92. Measured T: D2 tp-2 3.556 s, layer-2 4.214, single 4.836; M2 layer-2 0.876, single 0.999, tp-2 1.154, ep-2 1.172.

Verdict line: "D2 passes and M2 passes at 3 clean rounds each, picks tp-2 (D2) and layer-2 (M2), every candidate within 15% on pp and tg, baseline gate 1.011 and 1.008."

## Measured over predicted (pp, tg)

None is past 15% (largest: M2 single pp 1.074, M2 layer-2 pp 1.051, D2 tp-2 pp 0.936).

| Case | Candidate | pp | tg | T meas / pred (s) |
|---|---|---|---|---|
| D2 | single | 1.027 | 0.996 | 4.836 / 4.864 |
| D2 | layer-2 | 1.035 | 1.017 | 4.214 / 4.304 |
| D2 | tp-2 | 0.936 | 1.000 | 3.556 / 3.467 |
| M2 | single | 1.074 | 0.984 | 0.999 / 1.015 |
| M2 | layer-2 | 1.051 | 1.015 | 0.876 / 0.897 |
| M2 | tp-2 | 0.989 | 1.019 | 1.154 / 1.163 |
| M2 | ep-2 | 0.995 | 0.973 | 1.172 / 1.149 |

llama.cpp (round 1 rates, pp2048 and tg128): 30B layer 8385 and 200.5 tok/s, tensor 4246 and 173.6; 32B layer 2075 and 39.9, tensor 1423 and 59.4.

## Notes

- This is the first judged run of `P5_DEVICES` (two of four).
- `decode-clocks.json`: `low_mhz` null (no layer anchor on 2 GPUs), busy clocks 1800 (30B) and 1680 (32B); one 30B single L12 round was withheld for 14 tail samples (limit 15). Both only affect the anchored decode clock, which cost model 5 does not use on 2 GPUs.
- Same host, same night as V4a (`bench/results/v4a-20261121/README.md`); the 2-GPU and 4-GPU sessions differ in GPU set, so absolute rates are compared only inside each session.
