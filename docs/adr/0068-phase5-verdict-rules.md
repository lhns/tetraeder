# 0068. Phase 5: verdict rules for the multi-GPU exit test

Status: Proposed (2026-11-15). Phase 5. Written before any Phase 5 code, as ADR 0065 did for Phase 4. Companions: ADR 0069 (own executor), ADR 0070 (parallel shapes and their cost). Briefs: `docs/phase5-briefs.md`. No verdict session yet; the S2d step-0 session (`s2d-20261118-step0`) was judged null by the throttle rule, which was then replaced (History).

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

## Context

The Phase 5 exit test (`docs/plan.md`, Roadmap): the planner picks the measured winner among layer split, tensor parallel and expert parallel; predictions within 15%. M1 lost weeks to disputes over which configuration counts, whether a baseline was tuned and whether a session was clean (`docs/m1-open.md`). So the rules below are fixed before the code.

Facts that shape them:
- A winner is only meaningful when every shape runs on the same engine. All candidates run on Tetraeder's own executor (ADR 0069); llama.cpp's split modes are baselines, never candidates.
- Only a MoE model can run all three shapes; a dense model has no expert parallel.
- ADR 0056 found single-GPU candidate errors well above the 15% line on the committed fixtures (`20261103`), against anchor bands of 9.2% and 5.8%. "Every candidate within 15%" fails on that data; it holds here only if the multi-GPU cost model is better than the single-GPU one was, or the candidate set is small and the anchors well chosen. The rule stays the literal reading (below) and the risk is named.
- No noise class for multi-GPU boxes exists in `bench/noise/`.
- RTX 3090 boxes on Vast usually have no GPU peer access; transfers stage through host memory. The box image builds CUDA for sm_86 (`docker/box/Dockerfile`, baselines sm_86 only), so Ampere cards are the default class.

## Verdict

**Cases.** A case is one (model, box). Four cases, all must pass:

| Case | Model | Box | Candidates (ids) |
|---|---|---|---|
| D2 | Qwen3-32B Q4_K_M (dense, 64 layers, about 19.8 GB) | 2x RTX 3090 | `single` if it fits, `layer-2`, `tp-2` |
| M2 | Qwen3-30B-A3B Q4_K_M (MoE, 48 layers, 128 experts, top 8, about 18.6 GB) | 2x RTX 3090 | `single`, `layer-2`, `tp-2`, `ep-2` |
| D4 | Qwen3-32B Q4_K_M | 4x RTX 3090 | `single` if it fits, `layer-4`, `tp-4`, `tp2-pp2` |
| M4 | Qwen3-30B-A3B Q4_K_M | 4x RTX 3090 | `single`, `layer-4`, `tp-4` (if ADR 0070's block rule allows it), `ep-4`, `tp2-pp2` |

Shapes are ADR 0070's. The candidate set of a case is exactly the output of `tetraeder-plan candidates --multi` on the box's hardware graph before calibration (ADR 0040: enumeration fixed before scoring); a shape that its memory check or ADR 0070's divisibility rules prune is listed as pruned with the reason and is not run. Knobs are equal for all candidates: CUDA graphs on, `ubatch 512`, F16 KV, flash attention, context 4096, threads from the CPU quota.

**Hardware.** Linux, PCIe, no NVLink required. 2x box: two RTX 3090 (or RTX A6000 / A40 as fallback class), 64 GB RAM or more, CPU quota 8 or more, 150 GB disk. 4x box: four RTX 3090, 96 GB RAM or more, same CPU and disk. Each session records in `env.json`: host id, GPU names, driver, clocks and throttle reasons, `nvidia-smi topo -m`, the peer-access matrix (`nvidia-smi topo -p2p r`), PCIe gen and width per GPU, NCCL version, and which all-reduce path ggml chose (`nccl`, `internal`, `fallback`, read from the executor's startup log, ADR 0069).

**Workload.** One sequence: prompt 2048 tokens, then 128 generated (`tetraeder bench`-style pp2048 and tg128, `-r 1` per call, the harness repeats). Rates: `pp_tps` and `tg_tps`. Score: request time `T = 2048 / pp_tps + 128 / tg_tps` (ADR 0040's score). Batched decode is not judged in Phase 5 (the analytic model is batch 1; the simulator covers load).

**Calibration, separate from candidates.** Per box, before any candidate run, in its own stage (ADR 0040):
- Collective probe (ADR 0070): all-reduce over every group the candidates use and a copy per stage hop, sizes 4 KiB to 64 MiB, written to `calib-comm.json`. These are measured curves, not fitted.
- Anchors per model: the model truncated to its first L/4 and L/2 layers (`--layer-limit`, ADR 0069), each as `single` and as `tp-2`, pp2048 and tg128, two rounds. They fit ADR 0070's per-device parameters. No anchor equals a candidate (anchors are truncated).
- Then `tetraeder-plan predict` writes `predict-<model>-<box>.json` (predicted `pp_tps`, `tg_tps`, `T` per candidate, git commit, `sha256` of `hw.json` and `calib-comm.json`) before the first candidate runs. `verdict.json` records its `sha256`; a prediction file written after a candidate run makes the case invalid.

**Rounds.** Candidates and baselines of one model run in rounds on one box in one session: round k runs every configuration once, in the case's order rotated by k (round 2 reversed). Three rounds. A configuration's rate is the median over its clean rounds.

**Band b.** `band` of `bench/noise/<n>x-<gpu-class>.json` (new class names, for example `2x-rtx-3090`) if its `n` is at least 3, else the spec band 0.08 (`bench/spec/phase0.toml`, class `cuda-default`). Recorded as `band` and `band_source`. Until three sessions exist per class, 0.08 applies.

**Clean.** A run is clean when it exits 0 with its JSON, its GPU throttle record passes on every GPU (below), and `scripts/cotenant_guard.py` does not say false. The idle gate at SSH is ADR 0065's (`scripts/idle_gate.sh`, all GPUs). **Power limit:** at SSH every GPU's `power.limit` must equal its `power.default_limit` (both recorded in `env.json` with `power.max_limit`); a box with a lowered limit is not used for judged or rehearsal rounds, because a cap that one shape hits and another does not changes their ranking.

**Throttle** (as `scripts/phase5_summary.py` enforces it; History has how it got here):
- A 1 Hz sampler (`scripts/gpu_sampler.sh`, every GPU, `gpu-samples.csv`) runs for the whole session. A run's window is its process start to end (`window-<run>.json`), narrowed to the last `2 * sum(avg_ns) + 10 s` when its result rows carry `avg_ns` (cuts the model load); vLLM runs take theirs from gateway_bench `finished` and `wall_s`.
- Any window sample with a bit outside 0x1, 0x2, 0x4, 0x100, 0x200, 0x400 makes the run unclean.
- The shares of window samples with 0x200/0x400 (BoardLimit, Reliability) and with 0x4 (SW power cap) are recorded per run and GPU and compared only among the runs of one configuration (its rounds and its A/A run), never across configurations. Anchor runs are left out of the sampler rule and judged by the single-sample rule. When a share on a GPU spreads by more than 0.15 among them, the run furthest from the configuration's median is unclean (a tie goes to the later round). A configuration with one run is judged by the bit rule alone.
- A run without window samples falls back to the single-sample rule: before the run only idle and applications-clock bits, after it also SW power cap 0x4 (ADR 0065 addendum 2026-11-16). With window samples the before and after samples are recorded, not judged.
- `verdict.json` and `step0.json` carry a `throttle` block: the rule used, shares, counts and configuration medians per run.

The session is clean when the first candidate of the case, repeated as the last run of the session (A/A), agrees within the band on `T`: `|T_end / T_start - 1| <= b`. An unclean run is rerun in a later round, never dropped to make a pass. A case with fewer than two clean rounds for any configuration has no verdict.

**Winner.** `T_best` is the smallest measured `T` among the case's candidates. The winner set W holds every candidate with `T <= T_best * (1 + b)`. The planner's pick is its analytic pick: smallest predicted `T` (before ADR 0040's close-call benchmark, which is reported but not judged). **Pick passes** when the pick is in W.

**Within 15%.** Per candidate that ran: `e_pp = pp_pred / pp_meas - 1` and `e_tg = tg_pred / tg_meas - 1`, with the medians above. **Prediction passes** when `|e_pp| <= 0.15` and `|e_tg| <= 0.15` for every candidate of the case. A candidate predicted feasible that fails on the box (out of memory, error) fails the case; one predicted infeasible is not run.

**Baselines.** Same box, same session, same rounds, same `-t`, same workload:
- llama.cpp baseline build (`/opt/llama.cpp-baseline`, `vendor/PIN` `baseline_commit` (fc504f91, b12426, since 2026-11-16; was 202c1a22)): `llama-bench -m F -ngl 99 -fa 1 -p 2048 -n 128 -r 1 -t T -sm S` for S in `layer`, `row`, `tensor`. Which of them the build accepts, and whether `-sm tensor` uses NCCL or the internal all-reduce, is read on the box and recorded (`-sm row` uses the split buffer type the header calls old; it may be slow or unsupported for some types). **Gated:** the pick's `T` against the fastest llama.cpp split mode, `r = T_llama_best / T_pick >= 1 - b` (ADR 0040's rule that the planner's pick is at least as fast as the baselines; a tie inside the band passes).
- ik_llama.cpp at `ik_commit`: `-sm layer`, `-sm row`, and its graph or tensor split mode if its `llama-bench --help` lists one. Reported, not gated.
- vLLM with tensor parallel on an AWQ or GPTQ int4 checkpoint of the same model, if it installs and loads within 15 minutes: reported only, never gated (other weights, other kernels), and skipped first when the session runs short.

**Pass.** A case passes when the session is clean, the pick passes, the prediction passes and the baseline gate passes. `phase5_pass` is true when all four cases pass.

**Files** (`bench/results/<session>/phase5/`): `env.json`, `hw.json` (profiler hardware graph after the probe), `calib-comm.json`, `anchors-<model>-r<k>.json`, `candidates-<model>.json` (enumeration with pruned shapes and reasons), `plans/<model>-<id>.toml`, `predict-<model>-<box>.json`, `bench-<model>-<id>-r<k>.json` (llama-bench JSON rows from `tetraeder bench`), `baseline-<model>-<engine>-<sm>-r<k>.json`, `gpu-<run>.txt`, `*.log`, and `verdict.json` from `scripts/phase5_summary.py`:

```text
{band, band_source, predict_sha256: {<model>: ...},
 cases: [{case, model, box_gpus, gpu_class, p2p, comm_impl,
          candidates: [{id, shape, pruned, reason, predicted: {pp_tps, tg_tps, t_s},
                        measured: {pp_tps, tg_tps, t_s, rounds_clean}, err: {pp, tg}, within: bool}],
          pick, winner_set, pick_ok, prediction_ok,
          baselines: [{engine, sm, t_s, comm}], best_baseline, baseline_ratio, baseline_ok,
          aa: {t_start, t_end}, clean, reasons, pass}],
 phase5_pass}
```

Engine names in files: `tetraeder-exec`, `llama-bench`, `ik-llama-bench`, `vllm`.

**Settled now.**
- Batch-1 request time is the score; batched throughput is not judged in Phase 5.
- The judged pick is the analytic one; the close-call benchmark is reported.
- Every candidate within 15% on both rates is the literal exit test. Owner question: if one candidate misses while the median error is under 10%, does the case fail? Default: it fails.
- The 4x cases need a 4x box, which may cost more than the 0.90 USD/h box cap. Without it the verdict is partial and `phase5_pass` stays false.
- One case per model per box session is allowed (D4 and M4 may run on two 4x sessions) to stay inside the 3 h box cap; each case needs its own clean A/A then.

## Consequences

- A verdict is a file: disputes are settled by `verdict.json` and the committed predictions, not by prose.
- Predictions are committed before measurements, so the cost model cannot be fitted to its own answers.
- The baseline gate keeps a planner that picks the best of slow shapes from passing.
- The 30B MoE fits one 3090, so `single` is a real competitor on M2; the planner has to predict when splitting does not pay.

## Settled 2026-11-15 (owner delegated these to the advisor)
- **Within 15%: strict, every candidate,** as written above; the median rule is not adopted. ADR 0056's largest candidate errors (Context) are to be checked for whether they came from streamed or tiered plans, which Phase 5 does not judge. The judged set is fixed now as one candidate per shape per case (layer split, tensor parallel, expert parallel where it applies, tensor times pipeline where it applies) at the planner's own settings; a wider grid is calibration, not judged. Reopened only with the S2 session's measured errors, before the verdict session, as the owner's call.
- **Baseline gate: kept,** in the Phase 4 form: the pick is not slower than the best llama.cpp split mode (`-sm layer`, `row`, `tensor`) beyond the band. The M1 goal defined "met" as a paired same-box run against the strongest open tool, and without the gate Phase 5 could not meet it. vLLM and ik_llama.cpp stay report-only: vLLM runs other weights (AWQ or GPTQ checkpoints) and other kernels, ik_llama.cpp does not run the split modes on these models; the same reasoning as KTransformers' GPU experts under M1 point 3 (not runnable on the same weights in the regime).
- **Compute device per shard narrowed to experts on single-GPU plans:** the design's default, accepted.


## As built (E8, 2026-11-16)
- `scripts/phase5_box.sh <session> <2|4>` and `scripts/phase5_summary.py`, spec `bench/spec/phase5.toml`. Stages: gpus (the `nvidia-smi -L` count must match, else rc 1), env, build, models, gpu-parity, discover, enumerate, probe, prepass, anchors, calibrate, predict, rounds, vllm, summary. A failed gate (parity, comm probe, or no accepted llama-bench split mode) writes the verdict and ends with rc 3 before the anchors.
- Extra files: `run-<run>.rc`, `cotenant-<run>.json`, `prepass-<model>-<engine>-<sm>.json`, `gates.json`, `timeline.jsonl`, `verdict-notes.json`, `predict-*.sha256`. `hw.json` is not rewritten with the probe; `calib-comm.json` goes to `calibrate` and `predict` with `--comm`, which hash it. The A/A run is `bench-<model>-<first candidate>-aa.json`; the session budget defaults to 150 minutes.
- Prediction validity: the first `timeline.jsonl` event naming the prediction file's sha256 must precede the first candidate or A/A run (mtimes as fallback); the prediction travels in the fetched results directory rather than a git commit.
- A candidate that exits non-zero in at least two runs with no success fails the case; one merely unclean, with fewer than two clean rounds, gives no verdict. A single session's `phase5_pass` is false unless it holds all four cases; `--merge` combines V2, V4a and V4b.
- `tetraeder plan hw --out hw.json` and `tetraeder bench --engine dense|moe|llama --plan <plan.toml> --layer-limit <n>` exist (E6, E8 prep). `plan hw` is `profiler::discover` plus `to_json`; it reads `CUDA_VISIBLE_DEVICES` (index lists), numbers the visible GPUs 0..n as ggml does and skips the PCIe lane-group read while GPUs are hidden. A `cuda` build of `tetraeder` now enables NVML, without which the graph had no GPUs. Bench rows add `plan_id` and, when limited, `layer_limit`. Tests: `crates/tetraeder/tests/phase5_cli.rs`. The harness is still DRY_RUN-tested only until S2 runs on a box.
- The throttle check is the Verdict's "Throttle" rule; at E8 it was phase4_summary's single-sample check (History).

## As built (2026-11-17, sampler and step-0)
- `scripts/phase5_box.sh` starts `scripts/gpu_sampler.sh` at session start (not with `SKIP_GPU`/`DRY_RUN`) and stops it at the end and in `die`. The sampler CSV gains a last column `index` (the GPU number); Phase 4 reads only the first seven columns. `run_one` writes `window-<run>.json` (box-local start and end of the process). The summary applies the Verdict's "Throttle" rule.
- Step-0 check (ADR 0069): `ONLY="tp-2 llama-bench:tensor"` restricts the rounds (two configurations alternate their order each round), skips anchors, calibrate, predict, vllm, reruns and A/A, and the summary writes `step0.json` (ratios own / baseline per model, round and rate against `1 - band`). S2 order: `SESSION=<s>-step0 ROUNDS=2 ONLY=... phase5_box.sh <s>-step0 2`, then `ROUNDS=1` for the rehearsal.

## History
- 2026-11-16, As built (E8): throttle judged by phase4_summary's single before and after samples, SW power cap (0x4) benign after a run (ADR 0065 addendum 2026-11-16).
- 2026-11-17, addendum "throttle judged as in Phase 4" (before S2): a 1 Hz session sampler judges each run's window; disqualifying bits as now; each run's 0x200/0x400 and 0x4 shares on a GPU had to stay within 0.10 of the session's median share on that GPU; single samples recorded, not judged. Why: on drivers 615 and later single samples fail hosts whose measured windows are clean (`20261117e`, `20261117g`).
- 2026-11-18, addendum "throttle shares are compared per configuration across rounds" (before the next session): shares compared only between the rounds of one configuration, more than 0.15 apart makes the round furthest from the median unclean; with one round only the bit rule. Why: the bits follow load level, and Phase 5 candidates load the GPUs differently by design, so the cross-configuration rule measured the shape, not cleanliness; it marked every S2d step-0 round unclean (`s2d-20261118-step0`, which stays judged null). The co-tenant guard is unchanged.
- 2026-11-18: power-limit gate at SSH, decided before the next session: on `s2f-20261118` both GPUs were limited to about 250 W (default 350 W on the 3090), the single-GPU 32B ran at a median 1350 MHz against 1740 to 1830 MHz for the split shapes, and layer-2 decoded faster than one GPU, which batch 1 rules out at full power.
- 2026-11-18 (owner): the 15% rule keeps its meaning: the judged prediction is the planner's analytic prediction made before the candidates run, not an extrapolation from runs of each candidate (ADR 0076, not adopted). Next: the cost model's known errors are fixed (ADR 0070) and one rehearsal runs on a host at default power with the per-shape lines (`scripts/costmodel_lines.py`) reported beside the predictions. What reopens the question, fixed now: if the anchored shapes (single, tp-2) still miss 15%, the calibration is too thin and the fallback is to judge measured lines for anchored shapes only, analytic predictions for the rest; if unanchored shapes (layer-2, ep-2) miss, the model lacks a physical term and the rule goes back to the owner.
