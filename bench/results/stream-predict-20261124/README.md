# Session stream-predict-20261124

- Status: clean measurement, bad outcome. On one box the planner's frozen, uncalibrated prediction of the single-node Qwen3-235B-A22B is outside 15% in both regimes, for pp2048 and tg128: arm R (all experts in RAM, 32 VRAM slots) e_pp +0.350, e_tg -0.448; arm D (64 GB budget, experts streamed from NVMe, the re-priced arm D2) e_pp +1.320, e_tg -0.948 (uniform routing prior; the recorded routing trace gives +0.337, -0.333 and +1.316, -0.709). Not a verdict on M3.
- Rule: ADR 0091 Case S, `e = pred / meas - 1` on the medians of three rounds, pass when `|e_pp| <= 0.15` and `|e_tg| <= 0.15`; node `bench-235b-stream-predict` is successful if both arms pass, mixed if one, bad if neither. Predictions written and committed before the first measured run of each arm (below).
- Commit: code `fdb66019` (tetraeder-plan `node-predict`, `TETRAEDER_EXIT_PREFILL_WARM`, session scripts) on branch `bench-235b-stream-predict`; the box ran the pushed HEAD `f31672e3` (arms D and R) and `f58b4405` plus later script-only commits (arm D2, gate, disk).
- Box 62274134, host 682584 (Belgium), 1x NVIDIA A40 (46068 MiB, compute 8.6), driver 595.99.02, power limit 300 W = default (power gate ok), AMD EPYC 7662 (cgroup quota 61 cores, 61 threads in every run), host RAM 2 TB with a container limit of 1032772911104 B, 230 GB overlay disk, measured NVMe 4.0 GB/s at depth 8 (`predictions/disk-curve.json`). 0.771 USD/h all-in with disk, traffic 0.0039 USD/GB. Gates at SSH: `cuda: ok`, power gate, idle gate 10/10, co-tenant 0.46 busy host cores over 20 s. Life 02:47 to 04:02 (box age 1h18m, local clock; the box clock runs one hour behind).
- Spend: credit 28.06 to 26.22 USD for the whole session (1.84 USD), including the four boxes that never reached work (see Deviations).

## Frozen predictions (written and committed before the measured runs)

`tetraeder-plan node-predict` (new, `crates/planner/src/cluster.rs` `single_node_runs`) prices the whole model on one node as one builder placement, as a segment of `enumerate_cluster` is priced (reader path, the node's disk, cost model 5, uncalibrated). Hardware graph: the RTX 3090 spec-sheet prior (`crates/planner/fixtures/rtx3090_prior.toml`) with only box facts changed (`scripts/sp_make_hw.py`): RAM budget of the arm, GPU memory, thread quota 61 and the NVMe queue curve measured on this box. The planner picks the slot pair and ubatch; the pick is run as the `exit_regime` environment in each file.

| Freeze | Commit | `FROZEN.sha256` (sha256 of `SHA256SUMS`, which lists layout, disk curve, graphs, predictions, replays, trace hash) |
|---|---|---|
| 1: arms D and R | `fc2807b1` | `93b2fc46bd667e7ccef9dbad7093b31954dca4555066c9b21a5e13d78c01730f` |
| 2: arm D2 | `29663e1e` | `86b54209424dc19375dccc9a18c33eb73a0150d247fb2e7d7ed2608435862691` |

`scripts/sp_arms.sh` and `scripts/sp_arms_d2.sh` refuse to start unless the frozen files still match `SHA256SUMS` and `FROZEN_SHA` (passed after the commit) equals the recorded hash; `arms/freeze-check.txt` and `arms-d2/freeze-check.txt` hold the check.

Routing trace: `bench/results/20261114` run of the 235B at 32K prefill and 256 decode tokens, 10 VRAM slots, hot set from the first prefill (`routing-235b-20261114-041805.trace`, sha256 in `predictions/trace.sha256`, not committed: 74 MB). `tetraeder-bench replay` replays it for exactly the picked slot pair (`policy lru`, prefetch on). Decode is priced from that outcome; prefill keeps the uniform prior, because a 32K prefill does not describe a 2048-token one.

| Arm | Pick (VRAM slots, RAM slots, ubatch) | Prior | pp pred (tok/s) | tg pred (tok/s) | `T` (s) | File |
|---|---|---|---|---|---|---|
| D (64 GB) | 16, 56, 2048 (RAM tier 60.07 GB) | uniform | 85.74 | 0.145 | 903.8 | `predictions/pred-D-uniform.json` |
| D (64 GB) | same | trace | 85.60 | 0.908 | 164.9 | `predictions/pred-D-trace.json` |
| D (64 GB), disk at the prior 5.2 GB/s | same | uniform | 100.42 | 0.150 | 872.4 | `predictions/pred-D-uniform-diskprior.json` |
| D2 (64 GB) | 16, 52, 2048 (RAM tier 55.77 GB) | uniform | 82.05 | 0.136 | 965.8 | `predictions/d2/pred-D2-uniform.json` |
| D2 (64 GB) | same | trace | 81.92 | 0.766 | 192.0 | `predictions/d2/pred-D2-trace.json` |
| R (no balloon) | 32, 96, 2048 (VRAM 34.3 GB, RAM tier 103.0 GB) | uniform | 257.78 | 5.448 | 31.4 | `predictions/pred-R-uniform.json` |
| R (no balloon) | same | trace | 255.42 | 6.585 | 27.5 | `predictions/pred-R-trace.json` |
| R, disk at the prior | same | uniform | 257.78 | 5.448 | 31.4 | `predictions/pred-R-uniform-diskprior.json` |

The History step 2 reference of ADR 0091 (0.49 tok/s decode, 100 tok/s prefill, uniform prior, 64 GB) was priced on another graph (the cluster fixtures: 12 threads, the 5 GB/s curve, the `moe_235b` fixture layout); this graph has 61 threads and this box's disk curve, and decode comes out at 0.145. (Not checked: no test compares `single_node_runs` with `single_node_references` yet, so the difference to the History step 2 number may also hide a difference between the two code paths.)

## Measured (three rounds per arm, medians; each round a separate process)

Workload as Case S: the frozen exit prompt (`bench/prompts/exit_prompt.txt`, fnv1a64 in the result JSON), pp2048 then tg128, greedy, `exit_regime` with the env of the frozen pick, 61 threads, `TETRAEDER_EXIT_OP_OFFLOAD=1`, prefetch on at depth 1, the reader path (no I/O service, no overlap), as the planner prices it. Sources: `arms/exit-R-r<k>.json`, `arms-d2/exit-D2-r<k>.json` (`runs[0].prefill_tok_s`, `runs[0].decode_tok_s`); the table is `scripts/sp_table.py results` (`errors.json`).

| Arm | pp rounds (tok/s) | pp median | tg rounds (tok/s) | tg median |
|---|---|---|---|---|
| R | 192.73, 190.97, 185.54 | 190.97 | 9.869, 9.941, 9.674 | 9.869 |
| D2 | 35.37, 36.32, 33.93 | 35.37 | 2.728, 2.636, 2.542 | 2.636 |
| D (the first pick) | no round ran | | | |

**Arm D's first pick (16 VRAM and 56 RAM slots, 60.07 GB RAM tier) was not measured.** All three rounds were OOM-killed (SIGKILL, rc 101) after 64 to 95 s, before the first token (`arms/exit-D-r<k>.log`). The cgroup memory series (`arms/oom-D-cgroup-memory.json`, from the 1 Hz sampler) shows the container at its limit of 1032.8 GB with the balloon at 968.8 GB: the rest of the container peaked at 60.7 to 61.5 GB anon at the last samples before the kill. That is not evidence that the planner's footprint is wrong: from arms R and D2 the process peaks at its tier plus 2.3 to 2.4 GB (62.4 GB for this pick), under the planner's own budget of 63.4 GB. The harness delivered less than the nominal 64 GB; the balloon's page tables (about 1.9 GB for 969 GB) and the container's other processes share the limit. So arm D's nominal 64 GB regime ran with about 62 GB. The balloon took 8 to 9 minutes to fill (`scripts/sp_arms.sh` waits until it holds 90%).

**Arm D2** is arm D re-priced after that failure, a second freeze made after the failure and before any D2 run (so the first pick's failure is known to the second): the failed pick's VRAM slots (16) and four fewer RAM slots (52, a 55.77 GB tier), same budget (64 GB), graph, disk curve and trace (`scripts/sp_predict_d2.sh`, `PIN_D2=16,52`). It is my pin, not the planner's own pick: lowering the planner's RAM budget to 59 GB drops its pick to the next option of its grid (quarter, half or all of the experts outside VRAM: 28 slots, 30 GB, predicted tg 0.098 uniform and 0.266 trace; that file was written and discarded before the freeze, not run). Peak RSS was 58.2 GB, `balloon_ok` true on all three rounds.

## Errors, e = pred / meas - 1, against 0.15

| Arm | Prior | pp pred | pp meas | e_pp | tg pred | tg meas | e_tg | Both within 0.15 |
|---|---|---|---|---|---|---|---|---|
| R | uniform | 257.78 | 190.97 | +0.350 | 5.448 | 9.869 | -0.448 | no |
| R | trace (decode) | 255.42 | 190.97 | +0.337 | 6.585 | 9.869 | -0.333 | no |
| D2 | uniform | 82.05 | 35.37 | +1.320 | 0.136 | 2.636 | -0.948 | no |
| D2 | trace (decode) | 81.92 | 35.37 | +1.316 | 0.766 | 2.636 | -0.709 | no |
| D | first pick | 85.74 | none | | 0.145 | none | | does not run |

Neither arm is within 15%, so the node's outcome is **bad**. The direction differs by phase: both arms over-predict prefill (+0.34 to +1.32) and under-predict decode (-0.33 to -0.95).

### Where the model is off (measured profile against the frozen breakdown)

From `decode_profile_per_token`, `prefill_profile_per_ubatch`, `decode_io`, `prefill_io` (medians of the rounds) and the `tg_breakdown`, `pp_breakdown` of the frozen files; the replay bytes are `predictions/replay-*.json`, `predictions/d2/replay-D2.json`.

- **Arm R decode**, measured 100.8 ms per token (CPU expert compute 82.5 ms, GPU 5.3 ms, nothing waited on reads: 6.2 MB per token, 0.8 GB in 128 tokens) against 5.448 tok/s predicted uniform (183 ms: CPU 163 ms, GPU 18 ms, switches 2 ms) and 6.585 trace (152 ms: CPU 128 ms). The CPU term is the error: it is priced 163 ms against 82.5 ms measured (2.0x), and 128 ms (1.55x) with the trace prior, which moves the VRAM share of reads from the slot share 0.25 to 0.41 and closes part of it.
- **Arm R prefill**, measured 10.7 s per 2048 tokens (CPU segments 9.6 s, GPU segments 6 ms, streamer 1.0 s) against 7.9 s predicted (5.15 s offload copy, 1.17 s offload compute, 1.38 s GPU). The run did not offload expert matmuls to the GPU (`seg_compute_gpu_ms` 6 against 142 in arm D2) and still read 21.6 GB (1884 experts) from disk in the timed prefill although a warm pass had run first, so the RAM tier of 96 slots did not hold every expert outside VRAM at that point. The cold first prefill of the same prompt (the warm pass) ran at 23.4 tok/s, 126 GB read (`runs[0].warm_prefill`).
- **Arm D2 decode**, measured 378.6 ms per token (compute 168 ms, streamer 208 ms of which 41 ms waited, 682.5 MB read per token, RAM hit rate 0.874) against 7.34 s (uniform, 7.14 s of disk) and 1.30 s (trace, 1.12 s of disk). The bytes are close once the trace is used (replay 630.7 MB per token, measured 682.5, +8%); the uniform prior's bytes are about 6x too high (it prices 0.46 of the outside experts as held). The time is the large error: the model charges the 631 MB at 0.56 GB/s (1.12 s) while the builder read 682 MB in 208 ms (3.3 GB/s with prefetch, close to the box's 4.0 GB/s at depth 8), so the disk term is priced at a read rate and without the overlap with compute that the run achieves.
- **Arm D2 prefill**, measured 58 s (compute 16.1 s, streamer 36.0 s, 137.8 GB read, `read_s` 19.7) against 25 s predicted (disk 16.1 s). The layer-sequential sweep of a 2048-token ubatch reads the whole file through a RAM tier of 52 of 128 slots (nothing is reused before it is evicted), so the bytes are the file, not the uniform prior's share.

### Prior inputs against structure

The predictions are uncalibrated (ADR 0091 step 2: the spec-sheet prior with this box's thread count and disk curve; the exit's anchor calibration is not run here). Part of the error is therefore an input the exit's anchors fit: arm R's decode CPU rate (163 ms against 82.5 ms measured) and the GPU terms. The rest is structure that calibrating the same terms does not change: the decode disk term of arm D2 (implied 0.56 GB/s against 3.3 GB/s achieved, with the disk curve already measured), the prefill bytes of arm D2 (the whole file against a share), and arm R's prefill (expert matmuls run on the CPU, 21.6 GB still read).

## Does the planner predict from the recorded routing trace?

Yes, with limits. `tetraeder-plan node-predict --pin V,R --tier-replay replay.json` prices decode from the outcome of `tetraeder-bench replay` for that slot pair (ADR 0050 c, the path `serve --tier-replay` uses for builder candidates); `scripts/sp_predict.sh` runs the replay on the trace for the picked pair (seconds). Limits: the committed replay files cover 10 to 13 VRAM slots only, so each pair needs its own replay; the trace is a 32K prefill with 10 hot slots, so prefill stays on the uniform prior; and it does not change the disk rate or the CPU term, the two errors above. The trace prediction is closer in decode in both arms (R -0.333 against -0.448, D2 -0.709 against -0.948) and does not reach 0.15.

## serving-cuda-parity (node `serving-cuda-parity`, correctness only, `parity/`)

llama-server built from the patched `vendor/llama.cpp` (fc504f91, CUDA, arch 86, `parity/llama-server-version.txt`), `target/release/tetraeder` with CUDA, the parity suite of `regression-gate-20261123` (145 cases), `TETRAEDER_REQUIRE_PARITY=1`. Summary: `parity/summary.txt`, logs `parity/a1.log` and so on.

| Run | Result | repeat-penalty | seeded-sampler-names | logprobs | logprobs-shape |
|---|---|---|---|---|---|
| (a) suite as run, first | 4 failed, 135 passed, 6 xfailed | fail | fail | fail | fail |
| (a) suite as run, second | the same 4 fail | fail | fail | fail | fail |
| (b) suite with `--no-prefix-reuse --cache-ram 0` on tetraeder and `--no-cache-prompt --cache-ram 0` on llama-server | 139 passed, 6 xfailed | pass | pass | pass | pass |
| (c) each case alone on fresh servers | each passes | pass | pass | pass | pass |

The failure reproduces exactly as on the A100 (tetraeder -1.8332 and -1.8114, llama-server -1.8150 in both cases) and is deterministic across suite runs. llama-server on its own (`parity/llama-stability.json`, two fresh servers per case, each sent the case twice): across fresh servers it is stable on all four cases; on one server the second identical request differs on three of them. `logprobs` and `logprobs-shape`: first request -1.833250 on a fresh server, second request -1.811400 (the two values tetraeder shows; inside the suite llama-server answered -1.8150 for the same prompt, a third value, so the number depends on what that server evaluated before, not on two states); `repeat-penalty`: the generated text of the second request differs from the first; `seeded-sampler-names`: stable.

Finding: on CUDA the same prompt gives different logits (third digit) when it is evaluated from scratch and when it is evaluated after prefix or prompt-cache reuse, in both engines; the batch shape most likely changes the float sums (not isolated further). The suite sends the same prompt in several cases, so which value tetraeder returns depends on the earlier cases on that server; with reuse off (b) or one case per server (c) the engines agree on all four cases. CPU CI is clean. The behaviour is shared with llama-server, so the conformance suite needs either reuse off or a stated CUDA tolerance for the logprob cases and a non-sampled text case for the cases that compare generated text of a random-weight fixture (ADR 0083 decision 2). Not changed here.

Device proof for `--engine dense` and `--engine moe` (`parity/device-dense.txt`, `parity/device-moe.txt`): the server logs `qwen3 dense loaded backend="CUDA0"` and `qwen3moe loaded backend="CUDA0"` (debug level), ggml_cuda_init finds the A40, and GPU memory used goes from 0 to 277 and 273 MiB with the server idle and 309 and 319 MiB after one request.

## Qwen3-8B c4 gate pair (node `bench-regression-gate`, `ab/`)

A = `f68d2a0`, B = `41dcedd2` (both built from `git archive` on the box, CUDA release, same `vendor/PIN` and patch queue), `--engine dense --ngl -1 --parallel 4 --ctx 32768 -t 61 --accept-any-model`, `scripts/ab_moe_box.sh` as `regression-gate-20261123` used it (`MODELS=q8b TTFT=0`, three rounds A B, B A, A B, c4 `--prompt-tokens 512 --max-tokens 128 --concurrency 4 --requests 16`, c1 8 requests, one fresh server per run). All three rounds clean (no co-tenant flag, no disqualifying throttle bit). Source `ab/ab-summary-q8b.json`.

| Round | c4 A (tok/s) | c4 B | c4 B/A | c1 A | c1 B | c1 B/A |
|---|---|---|---|---|---|---|
| 1 (A B) | 227.54 | 243.13 | 1.068 | 102.70 | 102.24 | 0.996 |
| 2 (B A) | 244.75 | 237.47 | 0.970 | 102.46 | 101.84 | 0.994 |
| 3 (A B) | 245.48 | 243.60 | 0.992 | 102.62 | 102.04 | 0.994 |
| median | | | **0.992** | | | **0.994** |

By the rule (median at least 0.98 is no regression evidence): the Qwen3-8B c4 metric reads **no regression evidence** with three clean rounds; the earlier inconclusive reading came from the host, not a direction. c1 reads no regression evidence too. Absolute tok/s are for orientation (A40 here, A100 there).

## Prefix disk tier (report only, `disk/`, `disk-b/`, `disk-c/`)

Synthetic multi-session replay (`scripts/disk_tier_log.py`: round-robin, each session its own long system prompt, turn n repeats the earlier questions; `scripts/gateway_replay.py --speed 0 --concurrency 1`), Qwen3-8B `--engine dense --ngl -1 --parallel 4 --ctx 40960`, `RUST_LOG=tetraeder::step=debug`; one run with `--cache-ram` only (recompute) and one with `--cache-disk` as well, same requests; restores from the `admit` lines (`restore_ms`, `source`). TTFT of the later turns (turn 2 on, the ones that can restore) from `disk-summary.json`. One pass per mode, so noise is not measured; the RAM-only runs of `disk-b` and `disk-c` have the same prompts and settings apart from the cache size and differ in median TTFT by 1.37 s against 1.00 s, which is the host's spread.

| Run | Sessions x turns, prompt | `--cache-ram` / `--cache-disk-mib` | Later-turn TTFT median / p90, recompute | Later-turn TTFT median / p90, disk tier | Admits by source (disk run) | Restore from disk p50 / p90 |
|---|---|---|---|---|---|---|
| `disk` | 12 x 4, about 2150 tokens (common 2111) | 512 MiB / 8192 | 0.468 / 0.483 s | 0.369 / 0.388 s | disk 34, ram 1, live 11, copy 3 | 37.3 / 39.3 ms (34 restores); the one RAM restore 23.0 ms |
| `disk-c` | 24 x 4, about 4180 tokens (common 4144) | 768 MiB / 16384 | 0.995 / 1.008 s | 0.734 / 0.759 s | disk 71, live 23, copy 3 | 85.9 / 88.3 ms (71 restores) |
| `disk-b` | 24 x 4, about 4180 tokens | 256 MiB / 8192 | 1.365 / 1.382 s | 1.328 / 1.343 s | live 94, copy 3 (no disk admit) | none |

The tier restores a saved prefix in 37 to 86 ms against 0.47 to 1.0 s of recompute (about 20 to 26% lower later-turn TTFT end to end: the request still pays its own new tokens and decode). In `disk-b` the RAM budget (256 MiB) is below one saved sequence (about 590 MB of KV for 4000 tokens), nothing enters the RAM cache and so nothing spills: the tier never ran and the 3% TTFT difference is noise. Sequences larger than `--cache-ram` bypass the disk tier; `--cache-ram` has to hold at least one sequence for `--cache-disk` to do anything. (A first pass of the regression gate had this: its 2048 and 4096 MiB budgets never evicted.)

## Deviations

- **Box.** No 3090 box with at least 180 GB RAM passed the gates in the first hour, so the arms ran on an A40 (48 GB, Ampere, 696 GB/s) instead of an RTX 3090. Arm D plans against the 3090 prior's 24 GiB (`GPU_D_BYTES`, so the engine only gets the planned slot bytes and dense weights, 17.2 GB of slots); arm R plans against the A40's 48 GB, which holds the brief's 32 VRAM slots per layer (34.3 GB of slots; a whole 94-layer model at 32 slots does not fit a 24 GB card: one slot row is 1.07 GB). The GPU terms of the prior (936 GB/s) are slightly off for the A40.
- **Boxes that did not reach work (all destroyed, 4 attempts of the rule "destroy and re-create once", then the A40):** 62267263 (RTX 4090, host 254084) destroyed at the co-tenant gate after a sample that counted iowait as busy (60 busy cores, the corrected count on the next box gave 0.46); 62268446 (RTX 3090, host 570868) rejected the SSH key for 10 minutes; 62270376 and 62271718 (L40, host 169171) stayed `loading` 10 minutes twice; 62272858 (A100, host 458442) stayed `loading` 10 minutes. About 0.6 USD of box time at the listed prices. Hosts 570868 and 169171 stalled; whether the 4090's host was quiet is unknown.
- **Arm D re-priced** (D2, a second freeze), above, and D2 is a pin of mine, not the planner's pick. The planner's own pick for 64 GB was not measured because the harness gave the process less than 64 GB; the measured arm D regime is a 55.8 GB tier in about 62 GB.
- **Arm R's timed prefill is not read-free**: 21.6 GB were read from disk in it (above), decode read 6.2 MB per token. Arm R is the regime "experts resident in RAM", not "nothing read".
- **Planner knobs vs the harness.** The planner prices `ReadPath::Reader` with prefetch depth 1 and op offload; the runs use the same (`io_service` false, `overlap` false, `prefetch_depth` 1, `op_offload` true, `cpu_poll` 0 by the quota rule), not the I/O-service-with-overlap setup of the earlier 235B stages (`main-overlap`).
- **Co-tenant.** The guard passed every round (`cotenant_ok` true; outside cores 12.5 to 17.3 against a limit of 19.2, host 29 to 44 busy cores), so the host was busier than on the A100 gate but under the limit. Throttle sampler: no disqualifying bit in any window (bits 0x1 and 0x4 only), power at default.
- **Trace upload.** The routing trace (74 MB) is not in the repository's pushed paths; it was `taxameter put --binary` to the box (a text-mode put would have changed it). Its hash is in `predictions/trace.sha256`.
- **Disk-tier runs** `disk-b` and `disk-c` were added beyond the brief's single example after the first run, to see the tier at a longer prompt; the 8B gate ran three rounds only (all clean, no extra rounds).
- Frozen files ending in `.toml`, `SHA256SUMS` and `FROZEN.sha256` are not in taxameter's fetch patterns; they were read over SSH (`cat`) and checked with `sha256sum -c` locally.

## Files

`predictions/` (freeze 1, `d2/` freeze 2), `arms/` (arm R results; arm D logs), `arms-d2/`, `parity/`, `ab/`, `disk/`, `disk-b/`, `disk-c/`, `env/`, `llama-build/`, `build-*.log`, `errors.json`. Scripts: `scripts/sp_prep.sh`, `sp_predict.sh`, `sp_predict_d2.sh`, `sp_arms.sh`, `sp_arms_d2.sh`, `sp_abbuild.sh`, `sp_post.sh`, `sp_make_hw.py`, `sp_table.py`, `parity_llama_stability.py`, `disk_tier_log.py`, `disk_tier_summary.py`.
