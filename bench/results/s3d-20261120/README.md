# Session s3d-20261120

- Status: diagnostic, no verdict (ADR 0080 "Not decided here"; the tp2-pp2 overlap miss of ADR 0077/0078 and the decode excess). All stages ran, every run rc 0, no gate failure, no rerun.
- Code: e6100d1 (branch box-s3d-20261119: probe script 3d898f7 plus origin/main merged). Box 61538386, host 20210 (Finland), 4x RTX 3090 (same host as s3c), driver 595.91.07 (CUDA driver 13.2), AMD EPYC (32 threads, CPU quota 30), memory.max 259 GB, NCCL 2.25.1+cuda12.8, llama.cpp baseline fc504f9.
- Topology: GPU0-GPU1 PHB, GPU2-GPU3 PHB, every other pair NODE, one NUMA node; PCIe links gen4 x8 at the idle read.
- Gates: `cuda: ok`, power limit 350 W = default on all four, idle gate 10/10, co-tenant sample: `top` 99.4% idle over 10 s, baseline binary at fc504f9.
- Libnccl (`env/nccl-lib.txt`): the binary loads `/usr/lib/x86_64-linux-gnu/libnccl.so.2` (2.25.1, the system package; `ldd` and `LD_DEBUG=libs` agree; no pip or bundled copy, no LD_PRELOAD). The `/proc/<pid>/maps` section is empty (the probe process had exited).
- Box cost: 1.192 USD/h plus about 0.03 traffic. Created 00:54, destroyed 01:28 local (35 min), after a verified fetch (147 files, tgz 2.9 MB). About 0.77 USD all in (credit 16.94 to 16.17). Models 37 GB, provision 4m49s.
- Timeline (local; the box clock is one hour behind): provision done 01:05, gates 01:06, launch 01:07:58, `S3D_DIAG_DONE rc=0` 01:26 (18 min: build 3, part A about 10, part B about 8; budget was 65), fetch 01:26, destroy 01:28.
- Not committed: `qwen3-30b-a3b-ep4-tg-trace.trace.jsonl` and `qwen3-30b-a3b-ep4-tg-sync.trace.jsonl` (9.0 and 8.7 MB, 768 steps each; their `.trace.summary.txt` are committed). All other traces are under 0.3 MB. `analysis.txt` is the output of `python scripts/s3d_analyze.py bench/results/s3d-20261120`. No `.rc` files came back with the fetch; the box log (`runs/vast/s3d_diag.log`) shows rc=0 for every run.
- Caveat for every number: one box, one night, `-r 2` timed reps for pp2048 (plus one warmup), `-r 5` for tg128. Clocks cannot be locked. Pairs below were run back to back on this box.

## Part A: tp2-pp2, pp2048, prefill micro-batches 4 vs 1 (tok/s)

The two numbers of mb4 are the two timed reps (rep 1, rep 2). The step list of the trace has 3 steps for mb4: the warmup, rep 1, rep 2.

| Run | 30B mb4 | 30B mb1 | 32B mb4 | 32B mb1 |
|---|---|---|---|---|
| plain (NCCL, default) | 5187, 6693 | 4098, 4100 | 1410, 2030 | 1304, 1301 |
| host trace | 5222, 6757 | 4111, 4112 | 1403, 2033 | 1298, 1299 |
| sync trace (device time per op, fenced) | 2228, 2774 | 2564, 2568 | 841, 895 | 872, 875 |
| `GGML_CUDA_ALLREDUCE=internal` plain | 5657, 5847 | 5283, 5277 | 1711, 1723 | 1666, 1655 |
| internal, host trace | 5705, 5839 | 5271, 5279 | 1710, 1723 | 1653, 1662 |
| `NCCL_LAUNCH_MODE=PARALLEL` | 5170, 6666 | | 1417, 2047 | |
| `NCCL_DEBUG=INFO` | 5201, 6758 | | 1411, 2042 | |

Reference points from s3c (ADR 0078 History): measured tp2-pp2 pp 5079 (30B) and 1425 (32B), predicted 6512 and 2064.

### Findings

1. Tracing does not perturb the run: host trace within 1% of plain at every point (mb4 rep for rep).
2. The mb4 NCCL result is two different speeds. Rep 1 is 1410 (32B) and 5187 (30B), the s3c measured level. Rep 2 is 2030 and 6693, within 2% (32B, 2064) and 3% (30B, 6512) of the s3c prediction. Rep 2 is 44% and 29% faster than rep 1. mb1 does not do this (4098/4100, 1304/1301), nor does internal all-reduce (1711/1723, 5657/5847). s3c ran `-r 1` per call (ADR 0068), one fresh process per judged run, so each judged s3c number is a first step in the slow state (s3c JSONs: one sample each, 1422 and 1428 on the 32B, 4991 and 5168 on the 30B), not a mean of the two states.
3. What the host trace shows about the two states (32B mb4, one issuing thread for all ranks, `Exec::issue`): in the warmup and rep 1 the thread spends 806 ms (warmup) and 753 ms (rep 1) of a 1403 and 1458 ms step inside the rank-2 `compute` enqueues (rank 0: 76 and 563 ms), with `sync2` 0 to 32 ms at the end. In rep 2 every op kind sums under 10 ms on every rank (the whole step is issued in a few ms), then `sync2` waits 983 ms of the 1006 ms step for the device. 30B: step 864, 391, 302 ms (warmup, rep 1, rep 2), rank-2 compute 217, 157, 14 ms. So the slow state is the issuing thread blocked in `compute_async` of the later stage's ranks; the fast state is fully asynchronous issue and a device-bound step. This is the "one issuing thread serializes the stages" shape of ADR 0080, but it happens only in the first two steps here.
4. mb4 vs mb1, each at its own level: NCCL rep 1 / mb1 = 1.26 (30B) and 1.08 (32B); NCCL rep 2 / mb1 = 1.63 and 1.56; internal / internal = 1.09 and 1.04. The harness-mean ratio is 1.45 and 1.32.
5. Internal all-reduce at mb1 is faster than NCCL at mb1 by 29% (30B, 5280 vs 4099) and 27% (32B, 1661 vs 1303). At mb4 internal sits between NCCL rep 1 and rep 2 and is steady (spread 3% and 1%). So at mb1 the NCCL path costs more than the internal one; at mb4 rep 2 NCCL is 16% (30B) and 18% (32B) faster than internal. With the sync trace both are serial: sync mb4/mb1 = 0.975 and 0.993 (reps 2228/2774 against 2564/2568 and 841/895 against 872/875).
6. NCCL transport (`*.nccl.log`, 4 ranks, comm init `Init COMPLETE` on both models): every data channel is `SHM/direct/direct` (8 channel lines per model: 2 channels per direction for the pairs 0-1 and 2-3, nothing else). `Check P2P Type intraNodeP2pSupport 0 directMode 1`: no peer-to-peer. `NET/Socket` (eth0) and `NET/IB: no device` appear only as the bootstrap network; no data channel uses NET, no P2P, no CE memcpy line. NVLS unavailable, 2 coll channels, 2 p2p channels per peer, P2P chunk size 131072. The default transport is host shared memory.
7. `NCCL_LAUNCH_MODE=PARALLEL` changes nothing (both reps equal the plain run within 1%, 1417/2047 vs 1410/2030).
8. Fenced device time per op (sync trace, rank 0): all-reduce per call 707 us (30B, 192 calls per step) and 1654 us (32B, 256 calls), 15% and 17% of the fenced step (918 and 2434 ms); internal all-reduce in the same runs shows 663 and 1872 us per call but as host-blocking time. At mb1 the all-reduce enqueue is about 30 us per call with NCCL, so the stage cost shows up in `sync2` (30B 45.7 ms, 32B 366.6 ms per step), the final wait.

### What this says about the tp2-pp2 overlap miss, and what it does not settle

- Says: the executor can reach the modeled overlap on this box (rep 2 within 3% of the prediction on both models), so the GPipe schedule and the all-reduce cost are not what keeps tp2-pp2 prefill 0.28 to 0.45 under the model. The miss is in how the first steps run: the single issuing thread blocks for most of the step inside the second stage's compute enqueues, and every judged run (`-r 1`, a fresh process) is such a step.
- Says: the allreduce transport is not the cause of the miss through contention: it is host SHM on both NCCL and the internal path, the PARALLEL launch mode changes nothing, and the internal path gives a steady mb4 at 1.04 to 1.09 of its mb1.
- Does not settle why the first two steps block. Candidates the data cannot separate: lazy per-shape work in the first uses of each stream or comm (NCCL SHM channel buffers, graph or range instantiation), or launch-queue backpressure from NCCL kernels. It also does not show that rep 3 and later are steady: 30B kept falling (864, 391, 302 ms) and only two timed reps exist.
- s3c's judged values (one sample per process) match rep 1 here, so s3c measured the slow state. More reps per call alone would not change a judged number under `-r 1`.
- Does not give a verdict: no candidate was ranked, nothing was judged under ADR 0068.

## Part B: decode excess, tg128 (tok/s, `-r 5`, decode toggles on)

| Model | single | layer-4 (traced) | layer-4 (sync trace) | llama -sm none | llama -sm layer | ep-4 plain | ep-4 traced | ep-4 sync |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 206.5 | 186.4 | 185.2 | 210.1 | 192.3 | 179.4 | 175.6 | 44.1 |
| Qwen3-32B | 38.6 | 38.8 | 38.8 | 39.2 | 39.5 | | | |

(single: traced 206.5, sync 206.4; 32B single traced 38.6, sync 38.6.)

Median SM clock per GPU during the run (samples above 1000 MHz only, 20 ms nvidia-smi log beside each run; idle GPUs sit at 210 MHz P8):

| Run | GPU0 | GPU1 | GPU2 | GPU3 |
|---|---|---|---|---|
| 30B single | 1905 | (idle) | (idle) | (idle) |
| 30B layer-4 | 1695 | 1695 | 1695 | 1695 |
| 30B llama -sm none | 1935 | 1695 | 1695 | 1695 (idle most of the run) |
| 30B llama -sm layer | 1695 | 1695 | 1695 | 1695 |
| 30B ep-4 plain | 1950 | 1695 | 1695 | 1695 |
| 32B single | 1635 | (idle) | (idle) | (idle) |
| 32B layer-4 | 1695 | 1695 | 1695 | 1695 |
| 32B llama -sm none | 1605 | | | |

### Findings

- 30B layer-4 is 9.7% under single (186.4 vs 206.5); llama -sm layer is 8.5% under -sm none (192.3 vs 210.1). Same sign and size in the baseline, so the excess is not specific to Tetraeder. The 32B has none (38.8 vs 38.6; llama 39.5 vs 39.2).
- Where it sits (30B, sync trace, ms per token): single step 4.83, layer-4 step 5.34 (+0.51, +10.6%). Layer-4 device compute per rank 1.24 + 1.21 + 1.21 + 1.53 = 5.19 against 4.72 single (+9.8%); copies total about 0.05, logits 0.09. The stage chain is serial and the sum of its device time explains the step; no bubble between stages is visible. The excess is kernel time per layer, not communication or scheduling.
- Clocks: GPU0 runs at 1905 MHz alone and at 1695 MHz when all four GPUs are active, in both engines. 1905/1695 = 1.124 against a measured 1.108 (single/layer-4) and 1.093 (llama). The 32B is power-bound alone (1635) and shows no change at 1695, no excess. So the 30B excess matches the clock drop one for one on this box.
- ep-4 30B: 179.4 plain, 175.6 traced (-2.1%), 44.1 with a fence after every op (96 all-reduces and 97 compute ranges per token). Traced step 5.56 ms: host all-reduce 2.5 ms (96 calls, 26 us each), final sync 1.4 ms on rank 0, compute enqueue 0.35 ms. Fenced device time per op: all-reduce 44 us per call (4.25 ms per token on rank 0), compute 46 to 49 us per range (4.4 to 4.6 ms per rank). The fenced per-op times include the fence itself, so they are an upper bound; the layer-4 and single steps (5.34, 4.83 ms) bracket the ep-4 step (5.57 ms) closely.

### What this says about the decode excess, and what it does not settle

- Says: on this box the 30B layer-4 excess is kernel time scaled by the clock plateau (1695 vs about 1900 MHz), present in llama.cpp too, absent for the 32B. It is not a Tetraeder pipeline cost.
- Correction (offline, 2026-11-20, 20 ms logs cut to the timed tail): the 1695 MHz state follows each GPU's duty within a step, not the number of GPUs loaded. Plain ep-4 runs 1965 to 1995 on all four; fenced ep-4 (GPUs mostly waiting) and layer-4 (each GPU busy about a quarter of a step) sit at 1695. The ep-4 row in the clock table mixes the load phase with the run.
- Does not settle: that the clock is the cause. Clocks cannot be locked in the container and the single run leaves three GPUs idle (more power and thermal headroom), so single and layer-4 are not at equal conditions. The 1695 MHz plateau on every active GPU may be a power or thermal governor, a boost limit with several GPUs loaded, or the host. Not shown: the same comparison on another box or night.
- Does not settle the ep-4 decode residual of ADR 0080: ep-4 is 5.56 ms per token here, 0.2 ms above layer-4, with 96 all-reduces per token; the fenced numbers cannot be split into kernel, launch and wait time without an unfenced device timeline.

## Files

`<model>-<layout>-tg-{trace,sync}.json|.log|.trace.jsonl|.trace.summary.txt|.clocks.csv`, `<model>-tp2pp2-mb{4,1}-pp-{plain,trace,sync,internal-plain,internal-trace}`, `-mb4-pp-{launchparallel,nccldebug}` (+ `.nccl.log`), `env/nccl-lib.txt`, `env/ld-debug.log`, `env.txt`, `hw.json`, `layouts/`, `candidates-*.json`, `gpu-samples.csv` (1 Hz), `analysis.txt`.
