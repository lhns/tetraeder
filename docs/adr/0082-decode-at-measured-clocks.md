# 0082. Cost model 5: decode priced at the clock a GPU's duty gets

Status: Proposed (2026-11-20). Phase 5. Extends ADR 0070 (ranks path) and ADR 0080 (cost model 4); nothing of either changes. Adds one anchor to ADR 0068's list (owner's call). Brief: `docs/phase5-briefs.md`, "Brief V5". Tech tree node `planner-cost-model-v5`.

Measurements: `bench/results/s3d-20261120/README.md` (part B), `bench/results/s3e-20261120/README.md` (part C), `bench/results/s3f2-20261120/README.md` (part C), `bench/results/v2-20261120/README.md` (branch `box-v2-20261120`). Numbers below that are not in those READMEs come from cutting the committed 20 ms and 1 Hz logs to the timed tail and from an offline prototype on `scripts/costmodel_replica.py` (version-4 fit, then the rule below); the implementation reproduces them in the replica and commits them to `bench/results/s3c-20261119/costmodel-v5/README.md`.

## Context

- s3c: the 30B layer-4 decode ran at 0.855 of its prediction (v4: +0.167 in `pred/meas - 1`). Single is +0.044 on the same fit, so the layer-4 excess beyond single is 1.118.
- Host 20210, 20 ms clock logs cut to the timed tail (s3d, s3e): single 1905 MHz, power-capped (0x4, 337 W); every layer-4 GPU 1695 MHz, 140 to 160 W, no throttle reason; ep-4 1965 to 1995; fenced ep-4 1695. llama.cpp shows the same: `-sm none` 1920, `-sm layer` 1695 on all four, tg 8.5% lower. The memory clock is 9501 MHz in every state. 1905/1695 = 1.124; the s3d sync trace shows layer-4 device time per token 1.098 of single's.
- 1695 is also the clock of a GPU that holds a context and does little: loading, and the idle GPUs of `-sm none`. A GPU busy about a quarter of each step stays in that state.
- The 32B shows no excess (layer-4 38.8 against single 38.6, s3d). Its full-depth single is already below 1695 (power cap, 1620 to 1635); its truncated single anchors range 1665 to 1950 on the 1 Hz samplers.
- s3f2 ran every GPU at 1590 MHz all session, single included; there layer-4 tg was 0.995 and llama.cpp's 30B tg fell 12% against s3c.

### L12 and L24 explained

s3e part C found no layer-4 loss at L12 and L24 (0.997, 1.01) and 9.5% at full depth. `--layer-limit` keeps only the stages that start below the limit (`qwen3moe_shard.rs`: `if s.layers.0 >= limit { break }`), and layer-4 cuts the 30B 12/12/12/12 (s3e `candidates-qwen3-30b-a3b.json`). So:

| Run | GPUs that compute | Modeled decode duty | Clock in the tail (20 ms) |
|---|---|---|---|
| layer-4 L12 | gpu0 only (with the head) | 0.86 | gpu0 1890; gpus 1 to 3 1695 at a flat 109 W, the open-context state |
| layer-4 L24 | gpu0, gpu1 | 0.42, 0.54 | 1965, 1995; gpus 2, 3 flat 109 W at 1695 |
| layer-4 L36 (s3f2) | gpu0 to gpu2 | 0.29, 0.29, 0.36 | 1590 everywhere (capped host); gpu3 flat 117 W |
| layer-4 full | all four | 0.22 to 0.29 | 1695 on all four |

L12 was a single-GPU run and L24 a two-GPU split; they lost nothing because their GPUs boosted. The data are consistent with one rule: a GPU busy at least about 0.4 of a decode step boosts, one busy 0.29 or less sits at the low clock. Nothing remains unexplained between the three depths. The s3e README's reading ("three GPUs at 1695 and no loss") counted idle GPUs as active. It also follows that a truncated layer-N anchor as the m2-plan sketched it cannot measure the penalty: the limit drops whole stages and never lowers a GPU's duty.

### The V2 question: does our executor run at a lower clock than llama.cpp?

V2 (host 87485, 2x RTX 3090, 1 Hz sampler) reports own single, tp-2 and ep-2 at 1725 MHz and llama.cpp at 1890 to 1950, medians over each run's whole window. Of the 533 samples at 1725 in our runs' windows, 428 are under 130 W (the open-context state of loading) and 76 at 135 W or more (7 with 0x4; transitions within a 1 s sample); llama.cpp's windows have 82 such samples, 51 under 130 W. Our windows are 13 to 17 s with timed phases under 2 s, llama-bench's about 6 s, so the whole-window median of ours is mostly load. Restricted to samples at 135 W or more, own single (30B) is 1935, tp-2 and ep-2 1950, llama.cpp 1950. The 32B single at 1650 is one power-capped GPU against llama.cpp's two-GPU modes. On host 20210 with 20 ms logs cut to the timed tail, own single is 1905 and llama.cpp `-sm none` 1920 (30B), 1620 and 1605 (32B), tg 206.5 and 210.1. The 1 Hz samplers of s3, s3b and s3c (host 20210) give the same picture with the 135 W filter: own 30B single 1950, llama.cpp `-sm layer` and `-sm tensor` 1965 to 1995 on their busy GPUs (no `-sm none` baseline there). s3f2: 1590 for both engines. s3e part C ran no llama.cpp. So no executor-specific clock loss is visible; the 1 Hz data cannot separate pp, tg and load and do not settle it either way. The step pattern does matter, but through duty: ep-4 with a fence after every op (host gaps) fell to 1695.

## Decision

### Which clock enters the price

Two states per GPU in a decode step, not a curve:

- **Duty.** At the graph's fitted rates (cost model 4), the decode step `T` and each GPU's busy time `b_r` = sum over phases of `repeat x (t(r) + the all-reduce time when r is in its group)`. `d_r = b_r / T`.
- **Low state** when `d_r < 0.32`, else **busy state**. The logged data put the switch between 0.29 (layer-4's busiest GPU, low) and 0.42 (L24's gpu0, boosted). 30B tp2-pp2 lies in that gap on some sessions (modeled 0.34 to 0.44 over s3, s3b, s3c); its decode clocks were never logged, and its tg is within 0.02 at busy rates on all three (at low rates s3 would give -0.023, so its tg does not decide the state). 0.32 is chosen so that tp2-pp2 keeps the busy state it is verified at; layer-4's GPUs (0.18 to 0.29) are low. A plan between 0.32 and 0.42 (for example layer-3, about 0.33) is priced busy, which may be optimistic.
- **Factor.** For model m, `k_m = busy_mhz(m) / low_mhz` when that exceeds 1 by more than one 15 MHz clock step, else 1. In a low-state GPU's decode part (`part_time_s`), the clock-bound terms are multiplied by `k_m`: `c0 x ops`, the routed-expert roofline `max(2 p_exp / F_exp, W_exp / B)` and the routed-slot bytes. Dense weight and KV reads stay (memory clock 9501 MHz in both states; a dense 32B shows no clock effect). Collectives and prefill steps are unchanged.
- **Why these terms.** One rule fits the three host-20210 sessions with a layer-4 run. Predicted against measured layer-4/single tg at k = 1.124: s3c 0.930 against 0.910, s3 and s3b 0.903 against 0.930. Scaling the whole part time fits s3c (0.908) but misses s3 and s3b by 0.047, and moves the 32B layer-4 by about 0.10, where its anchors can sit above 1695 (1 Hz: L16 1950 on s3 and s3b) and the data show no 32B effect. With these terms the 32B layer-4 moves by at most 0.011 at k = 1.124.
- One pass: duty comes from the unscaled step; no fixed point.

### What calibration measures

- **Clock logs.** Every anchor and the new anchor below run with `nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active --format=csv -lms 10` beside them and a window file (`date +%s.%N` before and after the process), as s3e part C. A run's clock per GPU is the median over its timed decode tail (the last `reps x n_gen / tg` seconds before the window end). Whole-window or 1 Hz medians are not used: V2 shows they measure the load.
- `busy_mhz(m)`: median of the tail clocks of model m's `single` anchors (both depths, both rounds).
- **New anchor `layer-g@L/2`** on boxes with g >= 3 GPUs, first MoE model of the session (else the first model): a layer-g plan re-cut for the truncated depth, the first g - 1 stages L/(2g) layers each and the last stage the rest, run with `--layer-limit L/2`, so every stage computes L/(2g) layers (30B on 4 GPUs: 6/6/6/30 at limit 24, modeled duties 0.21, 0.20, 0.20, 0.33). tg128 only, `-r 10`, two rounds. It is truncated, so no anchor equals a candidate (ADR 0068). `low_mhz`: median tail clock over its GPUs without the head, both rounds. Its tg is a held-out check, priced with v5 and reported beside the fit residuals; it does not enter the fit, so no other prediction moves.
- Cost: two processes of about 10 s each (6 GB of weights from page cache, 1280 timed tokens at about 400 tok/s), well under one box minute, about 0.02 USD at 1.2 USD/h. Clock logs cost nothing. On 2-GPU boxes nothing is added: layer-2 GPUs run at duty 0.5 and boost (V2: 1890 to 1950 on its active samples).

### Clock-limited hosts

- Both clocks come from the same session as the candidates. On a host like s3f2 (1590 everywhere) `busy_mhz = low_mhz`, `k = 1`, and every prediction is bit-identical to cost model 4.
- `k = 1` also when the layer anchor is missing, has fewer than 15 tail samples on a GPU (25 at 20 ms until s3g-20261121, whose 30B single anchors gave 10 to 18; the logger now samples every 10 ms), or shows 0x8, 0x20, 0x40 or 0x80 in its tail; the calibrate log names the reason. Where a model's single anchor sits below the low clock (a power-capped single, as the 32B's full-depth single at 1620 on host 20210), the clamp gives 1.
- Graphs carry `HardwareGraph::decode_clocks`; without it a graph predicts as cost model 4. `MODEL_VERSION` 5.

### The check (done criterion, replacing "within 15%, others within 0.01")

The old line passes with a 0.02 nudge from v4's +0.167. Instead, offline on committed data, with host 20210's clocks from s3e part C (`busy_mhz` 1905, `low_mhz` 1695, k = 1.124, applied to both models: s3, s3b and s3c have only 1 Hz samplers that mix pp, tg and load, and their 32B single anchors range 1665 to 1950, so 1.124 is an upper bound there):

1. 30B layer-4 on host 20210: the predicted layer-4/single tg ratio within 0.03 of measured on s3, s3b and s3c (prototype 0.903 / 0.930, 0.903 / 0.930, 0.930 / 0.910), and `e_tg` within 0.07 (prototype +0.001, -0.003, +0.067; v4 +0.093, +0.089, +0.167).
2. s3f2 with its own clocks (1590 both): every prediction bit-identical to cost model 4. Out of sample on the s3f2 fit: layer-4 L36 tg within 0.03 of 231.8 (v4 and v5: 230.9).
3. 32B layer-4 moves by at most 0.015 in `e_tg` (prototype 0.010 to 0.011); every other candidate of s2f, s2g, s3, s3b, s3c, s3f2 and v2-20261120 by at most 0.005 in `e_pp` and `e_tg` (by their modeled duties no other GPU falls below 0.32: tp2-pp2 0.34 and up, 2-GPU shapes 0.44 and up).
4. Rust and replica agree within 1% on the version-5 graphs (`crates/planner/tests/costmodel_v2.rs`, as for v4).
5. Box confirmation, not part of done: the next 4-GPU session on a host where single boosts logs `busy_mhz > low_mhz` and its layer anchor's held-out tg is within 0.05.

## Not settled by the data

- What drives the state: duty, power, or the length of the idle gaps. Every logged point fits duty, but duty, mean power and idle-gap length move together in these runs.
- The threshold between 0.29 and 0.42; nothing ran in between on a boosting host with logs, and tp2-pp2 (0.34 to 0.44) has never had its decode clocks logged. The next 4-GPU box logs tp2-pp2 decode at 20 ms.
- Which terms follow the clock. The split (per-op cost and routed experts follow it, dense and KV reads do not) fits s3, s3b, s3c and the 32B within 0.03, but the per-op cost is a fitted number and the split is not measured per kernel. s3c alone would prefer the whole part time.
- Whether 1695 and 1725 are per-host constants or follow a GPU state that changes with driver or night; only host 20210 has both a low-duty run and logs.
- Prefill at low duty (layer-N with micro-batches 1): no clock data; it stays at busy rates.
- The 4% left on s3c single tg (+0.044), which v5 does not touch.
- Why the layer-4 excess beyond single was 1.118 on s3c, 1.059 on s3 and s3b, and 1.10 to 1.11 on s3d and s3e, all host 20210. Their 1 Hz samplers cannot isolate the 0.6 s timed decode of the 30B.

## Consequences

- One new graph field and one anchor; no new fitted parameter. Clocks enter as measured numbers, never fitted.
- Layer-N decode on 3 or more GPUs is priced slower wherever single boosts, which can move the pick away from layer-N on those hosts.
- Calibration depends on 20 ms clock logs; a box without `nvidia-smi` gets cost model 4.
- ADR number: 0082 was the next free number on `main` and the fetched branches when written.

## Implementation notes

Brief V5, branch `track-cost-model-v5`. Offline numbers: `bench/results/s3c-20261119/costmodel-v5/README.md` (items 1 to 3 pass and equal the prototype's: layer-4 / single tg 0.902, 0.903, 0.930 against 0.930, 0.930, 0.910; `e_tg` +0.001, -0.003, +0.067; s3f2 bit-identical, L36 230.9 against 231.8; 32B layer-4 moves 0.010 to 0.011; every other candidate 0).

- `HardwareGraph::decode_clocks` and `RankFeatures::model` as decided; `predict_ranks` computes the duty from the unscaled step and recomputes the decode step once. `Prediction::decode_clocks` lists duty, factor and assumed MHz per GPU (the diagnostics; not in the step sum) and feeds the gauge `tetraeder_plan_decode_clock_hertz`, set when a plan is loaded or switched to. A switch to a plan priced without clocks sets the gauge of its previous devices to 0 (the series has no delete); the alert compares only plan clocks above 0.
- The key into `busy_mhz` is `layout.model`, which in session layouts is the GGUF name ("Qwen3 30B Gptq Fp16"), not the file stem. `scripts/decode_clocks.py` writes that key from `layouts/<model>.json`; the host-20210 file of the check uses it.
- "The planner search (`stage_time`) uses the same helper" has no target: `search/cost.rs::stage_time` is the older single-run search model and shares no code with `part_time_s`. Only the ranks path (`predict_ranks`, so `predict_multi` and `calibrate`) changed.
- `calibrate` fits on the graph without `decode_clocks` and writes them back after the fit; the layer anchor's residual is `held_out` in `calib.json` (`ln predicted - ln measured`, as `residuals`) and a line in the log. `tetraeder-plan calibrate --clocks FILE` copies `decode-clocks.json`; a file without `low_mhz` leaves the graph at cost model 4.
- `tetraeder-plan candidates --multi --anchor <id>` writes only the plans of named anchors (`layer-4@L24`); `phase5_box.sh` uses it for `plans-anchor/<model>-layer-4@L24.toml`. With `LAYER_ANCHOR=1` only; the clock logs and `decode_clocks.py` run on every session.
- `MODEL_VERSION` is 5. `bench/ledger.json` and the two `bench/results/20261103/*/calib.json` carry the version number 5 now (the unit test `committed_results_match_the_ledger` requires it; their entries do not change, as for v4). The brief's "result files unchanged" gave way to that test.
- Replica: `--version 5` is the default, `--clocks FILE`; without clocks version 5 is version 4 to the last bit (the committed v4 rows reproduce).
