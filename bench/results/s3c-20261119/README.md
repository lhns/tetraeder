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

## Diagnosis of the misses (offline, ADR 0080)

Replica `scripts/costmodel_replica.py`, refit on each session's own anchors, recorded routing where the session has it (s3c only). Errors are pred/meas - 1.

**tp2-pp2 pp (+0.449 on 32B, +0.282 on 30B): the pipeline overlap, not the stage price.**
- The judged tp2-pp2 runs use `prefill_micro_batches` 4 (candidates file, both models); the model prices them as ADR 0077's GPipe over two stages, which credits 1.56x (32B) and 1.57x (30B) over the stages run in turn.
- The stage price is right: the model at micro-batches 1 gives 1325.7 tok/s (32B) and 4145 (30B, version 3; 4037 with version 4) against the pp-overlap stage's measured mb1 medians 1306.3 and 4055.5 (+0.015, +0.022; version 4 -0.005 on the 30B, a run no anchor fits).
- Measured mb4/mb1 is 1.082 and 1.235: 24% (32B) and 53% (30B) of the modeled saving. layer-4, the same pipeline without all-reduces, overlaps as modeled (-0.05, -0.04).
- Contention of the two groups' all-reduces does not explain it: one 2-rank all-reduce moves 3.4 GB/s per GPU link (gen4 x8 now), the 4-rank probe carries about 2.7x a pair's traffic, and with the all-reduces of both stages fully serialized (compute overlapping) the 32B step would still take about 920 ms against 1435 measured. ggml's NCCL path does not block the host (`ggml_backend_cuda_comm_allreduce_nccl`). What serializes the tp stages is not identified offline; no model change.
- Code read (`track-tp2pp2-overlap`, no box): no suspect is confirmed in the code.
  - Not a shared comm or stream: each tp group gets its own comm (`executor/src/exec.rs:284-298`, `find_comm` `:381`), a 2-device `ncclCommInitAll` (ggml-cuda.cu:1234), each rank's all-reduce on its own backend stream (:1076, :1104). The all-rank comm of `Ranks::open` (`ranks.rs:30`) is built but tp2-pp2 never uses it.
  - Not the issue order: in `ranked.rs:589-608` and `qwen3moe_shard.rs:1084-1105` a stage's copies close its (stage, micro-batch) group, and `gpipe_sort` issues (1, m-1) before (0, m), so the hop's `cudaStreamWaitEvent` (ggml-cuda.cu:2740-2743) lands after stage 1's previous micro-batch, as ADR 0077 History intends.
  - Not tp-specific in the copy: the hop is `cudaMemcpyPeerAsync` on the source stream (ggml-cuda.cu:2730); the synchronous fallback (ggml-backend.cpp:665-669) needs `GGML_CUDA_NO_PEER_COPY`, which our build does not set. layer-4 uses the same call and overlaps.
  - No host block in NCCL 2.25.1 under defaults: work goes in kernel args when it fits (`enqueue.cc:168`), the FIFO spin (`:1029`) only otherwise; host-stream callbacks only for plans with proxy ops (`:1396-1403`), which SHM has only with `NCCL_SHM_USE_CUDA_MEMCPY=1` (`shm.cc:501-509`); the cross-comm barrier only with `NCCL_LAUNCH_MODE=GROUP` (`group.cc:199`, default PARALLEL `init.cc:1221-1225`). This holds only if the comms run SHM/direct: on NET or SHM with CE memcpy every plan has proxy ops, so each all-reduce adds a host-stream callback (`cudaLaunchHostFunc`) and proxy-thread work, a host path both groups could queue on. The transport line below decides.
  - Arithmetic: the all-reduces are about half the stage time on both models (probe curve: about 0.8 ms per 32B micro-batch all-reduce, 128 per stage; 0.35 ms per 30B one, 96 per stage), so the share does not explain 1.08 against 1.24. A host that waits at every all-reduce gives about 1.0; all-reduces exclusive across both stages gives about 1.23 at that share (the 30B, not the 32B).
- Zero-code A/Bs for the next 4-GPU box, after the two traces below: `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,P2P,SHM,NET` (transport per channel, launch mode); `GGML_CUDA_ALLREDUCE=internal` (2-device groups use ggml's pipeline; the all-rank comm drops to none, unused here); `NCCL_LAUNCH_MODE=PARALLEL` set explicitly.
- Box measurement that decides it: tp2-pp2 pp2048 at mb 4 with `TETRAEDER_EXEC_TRACE` (host time per op, sync off) and once with `TETRAEDER_EXEC_TRACE_SYNC=1` (device time per op), 32B and 30B. If an issue call (an all-reduce or a range) holds the host for about its device time, the one issuing thread serializes the stages and the fix is in the executor (per-stage issue), not the model; if every issue is short, per-op device times show which op slows down when both stages run (then nsys on the four streams).

**ep-4 pp (+0.188): the routed slots' activation traffic, fixed (ADR 0080).**
- With the recorded routing the busiest ep-4 rank holds 0.391 of a layer's routed rows on average; version 3 scales the whole expert time by that share. The measured step implies about 0.55.
- Each rank of an expert-parallel layer computes all k slots of every token (ids of experts elsewhere are -1, patch 0004): gate and up outputs, swiglu, quantized inputs, the down output, its weighting and the sum over slots are full size on every rank, about 576 KB per token and layer on the 30B (MMQ reads a broadcast gate and up input once per token and writes its Q8_1 copy per slot, `mmq.cu` `dedup_bcast`). Version 3 has no activation term at all, so these bytes sat inside `F_exp` on the single and tp-2 anchors and were scaled down with the rows on ep.
- Version 4 prices them at the memory rate, apart from the matmuls (`DevicePart::expert_slot_bytes_per_token`). The fit moves `F_exp` from 24.5 to 31.5 TFLOP/s (the slots take about 68 of the single 30B's 400 ms prefill), and ep-4 pp goes to +0.088. Out of sample, the 30B tp2-pp2 at micro-batches 1 (tp-2 ranks paying their unsplit slots) moves from +0.022 to -0.005.

**30B layer-4 tg (+0.169): an excess per stage on the MoE decode, cause open.**
- Measured layer-4 against single: +0.48 ms per token for three boundaries (s3c), +0.39 (s3b), +0.38 (s3); the model gives -0.09, +0.07, +0.07. Excess per boundary 0.19, 0.11, 0.10 ms. llama.cpp `-sm layer` on the decode session lost a similar 0.15 ms per boundary against `-sm none` on this host.
- Not a hop: tp2-pp2 has the same cross-NODE hop and its tg is exact (+0.002). Not dense: 32B layer-4 tg has no excess (-0.03 ms per boundary). Not on the 2-GPU host of s2g (layer-2 30B -0.17 ms).
- The sync-mode trace of the decode session (toggles off) gives 1.27 to 1.30 ms of device time per 12-layer rank against 1.02 to 1.04 predicted, while the 48-layer single is about 0.2 ms over: a fixed device cost of about 0.2 ms per rank per token, which the model does not have and which a constant per boundary would not fit across hosts and models (it would cost the 32B and s2g). Suspected: the clock or wake state of a GPU idle three quarters of each token (a 30B rank works 1.3 ms of 5.3, a 32B rank 6 of 25, where a fixed wake cost hides in 2%).
- Box measurement: 30B layer-4 and single tg128 with both toggles on and `TETRAEDER_EXEC_TRACE_SYNC=1`, with `nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate --format=csv -lms 20` alongside. A per-rank excess that is constant in the rank's layer count and comes with a clock or pstate dip at each rank's start confirms the wake cost; then the model needs a measured per-rank decode term, fitted from a layer-N anchor.

**Other.** ep-4 tg goes from -0.148 to -0.1496 with the slot bytes (0.03 ms per token): inside the line by 0.0004, so marginal; a Rust fit within 1% of the replica can land on either side. Why ep-4 decode is over-predicted (6.5 ms predicted, 5.6 measured; about half of it is the 96 four-rank all-reduces at 33 us each): see "ep-4 tg" below. tp2-pp2 30B pp improves from +0.282 to +0.249, still over for the overlap reason above. s3 and s3b ep-4 tg stay at +0.39 and +0.50: those sessions have no routing record.

**ep-4 tg (-0.150): the probe's 4-rank head point (ADR 0080 History).**
- Per decode all-reduce, (measured step - predicted step without all-reduces) / count, version 4 before the fix: 30B tp-2 anchors 16.7 and 14.7 us (predicted 15.5), 32B tp-2 23.1 and 22.4 (23.6), 32B tp-4 anchors 35.3 and 32.8 (34.4), 32B tp-4 31.3, 30B tp2-pp2 15.5 (15.4), 30B ep-4 22.6 (32.8). At 2 ranks the 32B's 20 KiB all-reduce costs about 8 us more than the 30B's 8 KiB, as the 2-rank curve says (28.5 to 37.2 us from 4 to 16 KiB). At 4 ranks the curve says nothing: 77.7 us at 4 KiB, 77.2 at 16 KiB. s3 and s3b on the same host had 63.7 and 75.8, 49.4 and 58.6.
- So the 4-rank byte part of a decode all-reduce is 0 at 8 KiB and 1.6 us at 20 KiB, `a + 6 b` (32.8 us) is fitted on the 32B's 20 KiB, and ep-4 pays it on 8 KiB: about 10 us too much on each of 96 all-reduces, the 1 ms of the miss. The 4 KiB point is the probe's first timed op (`comm_probe.rs`: first group, smallest size, two warm-ups, median of 20).
- Check without a rule: s3c refit with s3's 4-rank curve gives ep-4 tg -0.024, with s3b's -0.055; 32B tp-4 tg stays -0.032.
- Not the routing: uniform routing gives ep-4 tg -0.151 (recorded -0.150). The model prices the per-step busiest rank (Monte Carlo, 0.432 of 8 slots); the expert bytes are about 0.5 ms of the 6.5 ms step, so even the aggregate share (0.387) would move it by about 0.05 ms. The s3c-20261119-decode-decode traces cover single and layer-4 only, no ep or all-reduce.
- Fix (cost model 4): a curve head not below its successor takes the next segment extended back (77.7 to 70.7 us here). Dropping the point alone leaves -0.142, extending every curve to zero bytes alone -0.150; the rule gives -0.084; s2f, s2g, s3 and s3b keep every number, the other s3c rows move by at most 0.002 through the refit (table). `a` 2.52 to 3.59 us, `b` 5.05 to 3.29 us.
- Open: -0.084 is a whole-step residual. ep-4 tg was 7.0 and 7.2 ms on s3 and s3b, 5.57 on s3c with the decode toggles on, so per-range host cost is in it. Box measurements: the comm probe with the 4-rank sweep timed again last and sizes 8192 and 20480 added (the rule fires on a 0.5 us margin); 30B ep-4 tg128 with `TETRAEDER_EXEC_TRACE`, then `TETRAEDER_EXEC_TRACE_SYNC=1`. s3 and s3b ep-4 tg (+0.39, +0.50) are the other sign of the same gap: no tp-4 anchor, `b` not fitted, a 4-rank all-reduce priced at the 2-rank latency.

| Session | Model | Candidate | v4 | fix |
|---|---|---|---|---|
| s3c | 30B | ep-4 | +0.088 / -0.150 | +0.088 / -0.084 |
| s3c | 30B | layer-4 | -0.047 / +0.168 x | -0.047 / +0.167 x |
| s3c | 32B | layer-4 | -0.051 / +0.019 | -0.051 / +0.019 |
| every other row | | | as below | unchanged |

### Before and after (version 3 against version 4, err pp / tg, `x` beyond 0.15)

| Session | Model | Candidate | v3 | v4 |
|---|---|---|---|---|
| s2f | 30B | single | -0.061 / +0.008 | -0.043 / +0.007 |
| s2f | 30B | layer-2 | -0.139 / -0.104 | -0.122 / -0.105 |
| s2f | 30B | tp-2 | -0.002 / +0.009 | -0.036 / +0.010 |
| s2f | 30B | ep-2 | +0.096 / +0.057 | -0.003 / +0.058 |
| s2f | 32B | single | -0.002 / +0.087 | -0.002 / +0.086 |
| s2f | 32B | layer-2 | -0.025 / -0.221 x | -0.026 / -0.222 x |
| s2f | 32B | tp-2 | +0.009 / -0.072 | +0.009 / -0.071 |
| s2g | 30B | single | -0.096 / +0.018 | -0.085 / +0.018 |
| s2g | 30B | layer-2 | +0.009 / -0.016 | +0.021 / -0.016 |
| s2g | 30B | tp-2 | +0.033 / -0.023 | +0.006 / -0.022 |
| s2g | 30B | ep-2 | +0.111 / +0.013 | +0.039 / +0.014 |
| s2g | 32B | single | -0.018 / +0.006 | -0.018 / +0.005 |
| s2g | 32B | layer-2 | +0.145 / -0.035 | +0.144 / -0.037 |
| s2g | 32B | tp-2 | +0.039 / -0.001 | +0.038 / -0.001 |
| s3 | 30B | single | -0.091 / +0.032 | -0.080 / +0.032 |
| s3 | 30B | layer-4 | +0.269 x / +0.094 | +0.285 x / +0.093 |
| s3 | 30B | ep-4 | +0.377 x / +0.391 x | +0.214 x / +0.392 x |
| s3 | 30B | tp2-pp2 | +0.329 x / +0.014 | +0.294 x / +0.015 |
| s3 | 32B | single | -0.024 / +0.014 | -0.025 / +0.013 |
| s3 | 32B | layer-4 | +0.350 x / +0.004 | +0.350 x / +0.003 |
| s3 | 32B | tp-4 | +0.093 / +0.066 | +0.093 / +0.068 |
| s3 | 32B | tp2-pp2 | +0.511 x / -0.010 | +0.511 x / -0.010 |
| s3b | 30B | single | -0.117 / +0.027 | -0.107 / +0.026 |
| s3b | 30B | layer-4 | -0.065 / +0.090 | -0.053 / +0.089 |
| s3b | 30B | ep-4 | +0.353 x / +0.495 x | +0.197 x / +0.496 x |
| s3b | 30B | tp2-pp2 | +0.279 x / +0.019 | +0.247 x / +0.020 |
| s3b | 32B | single | -0.017 / +0.017 | -0.017 / +0.016 |
| s3b | 32B | layer-4 | -0.027 / +0.007 | -0.027 / +0.005 |
| s3b | 32B | tp-4 | +0.093 / +0.121 | +0.093 / +0.123 |
| s3b | 32B | tp2-pp2 | +0.483 x / -0.010 | +0.483 x / -0.010 |
| s3c | 30B | single | -0.094 / +0.045 | -0.083 / +0.044 |
| s3c | 30B | layer-4 | -0.059 / +0.169 x | -0.047 / +0.168 x |
| s3c | 30B | ep-4 | +0.188 x / -0.148 | +0.088 / -0.150 |
| s3c | 30B | tp2-pp2 | +0.282 x / +0.001 | +0.249 x / +0.002 |
| s3c | 32B | single | -0.037 / +0.024 | -0.037 / +0.022 |
| s3c | 32B | layer-4 | -0.051 / +0.020 | -0.051 / +0.019 |
| s3c | 32B | tp-4 | +0.070 / -0.032 | +0.070 / -0.032 |
| s3c | 32B | tp2-pp2 | +0.449 x / -0.022 | +0.448 x / -0.021 |

- s3 ran before the pipeline issue-order fix (its layer-4 and tp2-pp2 pp understate the engine); s3 and s3b have no routing record (uniform ep-4). s2f 32B layer-2 tg is an old miss of that session (v3 the same).
- 30B anchors (pp) on s3c: single L12 -0.007 to +0.009, L24 -0.057 to -0.045, tp-2 L12 +0.093 to +0.069, L24 +0.028 to +0.002.
- Graph and rows: `costmodel-v4/fit-s3c.json` (`--json`, with the routing record).
