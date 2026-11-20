# Session s3e-20261120

- Status: diagnostic, no verdict (Phase 5 tp2-pp2 prefill states plus cost model v5 calibration data; follows s3d-20261120 "Does not settle"). All 40 runs rc 0 (34 arm runs, 6 part C), no gate failure, no rerun of a run. One restart of the script (see timeline).
- Code: 3ada436 (branch box-s3e-20261120, `scripts/s3e_diag_box.sh`). Box 61604497, host 20210 (Finland), 4x RTX 3090, driver 595.91.07, AMD EPYC (32 threads, CPU quota 30), memory.max 259 GB, llama.cpp baseline fc504f91 (tag b12426). Topology as s3d (GPU0-GPU1 and GPU2-GPU3 PHB, others NODE).
- Gates: `cuda: ok`, power limit 350 W = default on all four, idle gate 10/10 (0x1 only), co-tenant sample 99.2% idle (`top`, 10 s), baseline at fc504f91, memory.max 259 GB.
- Cost: 1.192 USD/h plus traffic. Created 09:12:24, destroyed 09:44:21 local (32 min), after a verified fetch (168 files, tgz 1.3 MB). Credit 16.15 to 15.39, so 0.76 USD all in.
- Timeline (local; the box clock is one hour behind): create 09:12, provision (8B and 30B, the default models) done 09:19, gates 09:19, first launch 09:20:33 died at 09:29 (rc=1: `qwen3-32b missing`, the default provision does not fetch the 32B; the build took 9 min that time), `provision --models Qwen/Qwen3-32B-GGUF:Qwen3-32B-Q4_K_M.gguf` done 09:34, relaunch 09:34:45, `nograph check: ok` 09:35:12, `S3E_DIAG_DONE rc=0` 09:43:32 (9 min: build 14 s from cache, plans 2 s, arms about 8 min, part C 1 min), fetch 09:44, destroy 09:44:21 (gone: `status` shows 0 instances). The local taxameter venv had a damaged httpx/certifi install and was repaired with `uv sync --reinstall` before the first call.
- Not committed (over 1 MB): none; the largest file is `gpu-samples.csv`. No `.rc` files came back with the fetch (as in s3d); the box log shows 40 `done rc=0` lines.
- Caveat for every number: one box, one night, `-r 5` timed reps plus the bench warmup, clocks not lockable. Compare only the pairs run back to back here.
- `analysis.txt` is `python scripts/s3e_analyze.py bench/results/s3e-20261120` (per-rep samples, per-step per-op host sums of every trace, then the s3d analysis). The clock cut for part C is in the table below (script not committed; the rule is in the text).

## Part A: tp2-pp2, pp2048, tok/s per timed rep (rep 1 to 5; the bench warmup is step 0 of the trace)

| Model | mb | allreduce | arm | rep 1 | rep 2 | rep 3 | rep 4 | rep 5 | median reps 2-5 |
|---|---|---|---|---|---|---|---|---|---|
| 30B | 4 | nccl | default | 5138 | 6646 | 6632 | 6633 | 6628 | 6633 |
| 30B | 4 | nccl | nograph | 6068 | 6075 | 6069 | 6068 | 6081 | 6072 |
| 30B | 4 | nccl | graphdebug | 5114 | 6587 | 6578 | 6588 | 6593 | 6588 |
| 30B | 4 | nccl | eager | 5129 | 6595 | 6601 | 6588 | 6602 | 6598 |
| 30B | 4 | internal | default | 5640 | 5850 | 5851 | 5847 | 5844 | 5848 |
| 30B | 4 | internal | nograph | 5684 | 5772 | 5762 | 5765 | 5763 | 5764 |
| 30B | 1 | nccl | default | 4042 | 4067 | 4066 | 4057 | 4067 | 4066 |
| 30B | 1 | nccl | nograph | 4081 | 4121 | 4107 | 4122 | 4135 | 4121 |
| 30B | 1 | nccl | graphdebug | 4047 | 4038 | 3977 | 4038 | 4039 | 4038 |
| 30B | 1 | nccl | eager | 4074 | 3999 | 4072 | 4073 | 4074 | 4073 |
| 30B | 1 | internal | default | 5285 | 5290 | 5286 | 5286 | 5274 | 5286 |
| 30B | 1 | internal | nograph | 5328 | 5353 | 5392 | 5379 | 5372 | 5375 |
| 32B | 4 | nccl | default | 1394 | 2008 | 2007 | 2006 | 2008 | 2007 |
| 32B | 4 | nccl | nograph | 2004 | 2003 | 2005 | 2005 | 2003 | 2004 |
| 32B | 4 | nccl | graphdebug | 1399 | 2016 | 2014 | 2016 | 2016 | 2016 |
| 32B | 4 | nccl | eager | 1394 | 2006 | 2006 | 2009 | 2005 | 2006 |
| 32B | 4 | internal | default | 1706 | 1717 | 1719 | 1719 | 1719 | 1719 |
| 32B | 4 | internal | nograph | 1715 | 1716 | 1715 | 1713 | 1713 | 1714 |
| 32B | 1 | nccl | default | 1286 | 1286 | 1285 | 1282 | 1285 | 1285 |
| 32B | 1 | nccl | nograph | 1287 | 1290 | 1290 | 1288 | 1288 | 1289 |
| 32B | 1 | nccl | graphdebug | 1282 | 1281 | 1280 | 1280 | 1282 | 1281 |
| 32B | 1 | nccl | eager | 1287 | 1284 | 1285 | 1283 | 1285 | 1284 |
| 32B | 1 | internal | default | 1659 | 1659 | 1658 | 1659 | 1659 | 1659 |
| 32B | 1 | internal | nograph | 1666 | 1665 | 1662 | 1664 | 1663 | 1663 |

varykey (mb4, NCCL, one process per length, -r 1; one timed rep each):

| Model | -p | rep tok/s | timed-step ms (trace step 1) | warmup step ms (trace step 0) |
|---|---|---|---|---|
| 30B | 2048 | 5145 | 397.2 | 853.3 |
| 30B | 2040 | 5173 | 393.6 | 900.6 |
| 30B | 2032 | 4839 | 419.1 | 860.4 |
| 30B | 2024 | 5143 | 392.7 | 866.5 |
| 30B | 2016 | 5077 | 396.3 | 851.6 |
| 32B | 2048 | 1404 | 1456.4 | 1465.2 |
| 32B | 2040 | 1396 | 1459.1 | 1394.3 |
| 32B | 2032 | 1399 | 1451.0 | 1381.8 |
| 32B | 2024 | 1387 | 1457.7 | 1419.8 |
| 32B | 2016 | 1386 | 1452.5 | 1387.7 |

## Part C: tg128, -r 5 (tok/s per rep, median clock per GPU in the timed tail)

Tail = last 5*128/tok_s seconds before the window end (`date +%s.%N` after the process). Clock columns: median SM MHz of the GPU over the tail (all samples), and in brackets the median of samples above 1000 MHz when the GPU was active at all.

| Run | tok/s reps 1-5 | tail s | GPU0 | GPU1 | GPU2 | GPU3 |
|---|---|---|---|---|---|---|
| single L12 | 651 651 651 651 650 | 0.98 | 1890 (1890, n 49) | 210 (idle) | 315 (idle) | 210 (idle) |
| layer-4 L12 | 636 650 649 649 649 | 0.99 | 1890 (1890, n 48) | 1695 (1695, n 48) | 1695 (1695, n 48) | 1695 (1695, n 48) |
| single L24 | 380 380 377 380 380 | 1.69 | 1905 (1905, n 83) | 210 (idle) | 210 (idle) | 210 (idle) |
| layer-4 L24 | 381 384 384 384 384 | 1.67 | 1965 (1965, n 82) | 1995 (1995, n 82) | 1695 (1695, n 82) | 1695 (1695, n 82) |
| single full | 207 206 206 207 206 | 3.10 | 1905 (1905, n 153) | 210 (idle) | 210 (idle) | 210 (idle) |
| layer-4 full | 187 187 187 187 187 | 3.43 | 1695 (1695, n 169) | 1695 (1695, n 169) | 1695 (1695, n 169) | 1695 (1695, n 169) |

## Per-step host blocking (trace, ms; enqueue sums on the issuing thread per op kind and rank, `sync[2]` the final wait)

Step 0 is the bench warmup (call 1), step 1 is timed rep 1, steps 2 to 5 reps 2 to 5. All steps of every run are in `analysis.txt`.

| Run (mb4, NCCL) | step | wall | all_reduce[0] / [2] | compute[0] / [1] / [2] / [3] | sync[2] |
|---|---|---|---|---|---|
| 30B default | 0 | 854 | 98.5 / 169.9 | 155.9 / 55.1 / 219.3 / 60.4 | 0 |
| 30B default | 1 | 398 | 7.1 / 6.9 | 120.1 / 44.9 / 161.5 / 44.7 | 10.1 |
| 30B default | 2 | 307 | 6.1 / 14.2 | 7.3 / 2.3 / 13.8 / 84.8 | 176.8 |
| 30B default | 3 | 308 | 6.1 / 14.2 | 7.3 / 2.3 / 13.8 / 84.9 | 177.3 |
| 30B nograph | 0 | 891 | 105.7 / 178.2 | 156.6 / 50.5 / 227.5 / 65.5 | 0 |
| 30B nograph | 1 | 337 | 5.9 / 102.7 | 22.3 / 19.7 / 106.4 / 30.5 | 47.1 |
| 30B nograph | 2 | 336 | 5.9 / 102.6 | 22.4 / 19.6 / 106.5 / 30.5 | 46.9 |
| 30B eager | 0 | 3678 | 19.6 / 136.2 | 2928.3 / 46.9 / 119.1 / 58.1 | 47.4 |
| 30B eager | 1 | 399 | 7.2 / 7.1 | 120.1 / 46.3 / 160.0 / 45.5 | 9.8 |
| 30B eager | 2 | 310 | 6.2 / 14.4 | 7.3 / 2.3 / 13.8 / 85.2 | 178.4 |
| 32B default | 0 | 1440 | 134.6 / 256.3 | 78.7 / 30.3 / 816.5 / 38.4 | 0 |
| 32B default | 1 | 1468 | 8.9 / 8.9 | 573.8 / 37.3 / 765.6 / 36.2 | 33.7 |
| 32B default | 2 | 1018 | 7.2 / 7.1 | 1.5 / 1.4 / 1.4 / 1.4 | 995.7 |
| 32B nograph | 0 | 1425 | 141.1 / 240.1 | 85.7 / 28.5 / 812.4 / 26.7 | 0 |
| 32B nograph | 1 | 1020 | 134.3 / 162.7 | 85.3 / 15.9 / 374.3 / 16.0 | 228.5 |
| 32B nograph | 2 | 1022 | 136.2 / 163.4 | 85.3 / 16.0 / 374.1 / 15.9 | 227.8 |
| 32B eager | 0 | 1350 | 51.0 / 342.8 | 38.5 / 24.9 / 367.9 / 19.7 | 226.8 |
| 32B eager | 1 | 1468 | 9.0 / 9.0 | 576.9 / 37.7 / 761.1 / 37.5 | 33.2 |
| 32B eager | 2 | 1019 | 7.1 / 7.1 | 1.4 / 1.4 / 1.4 / 1.4 | 996.7 |

## Answers

1. **nograph-check: `ok`, but unproven.** The first nograph run (30B, mb4, NCCL, `GGML_CUDA_GRAPH_DEBUG=1`) exited 0 and logged no capture or reuse line. The positive control failed: the default-graphs run with `GGML_CUDA_GRAPH_DEBUG=1` (all four `*.graphdebug.log`, 446 to 448 bytes) logged no such line either, so the silence of the nograph run proves nothing by the log. Likely cause (read in `crates/ggml-sys/src/llama/mod.rs`, not tested): ggml-sys installs a log callback that counts these lines (`CudaGraphLogCounts`), so they never reach stderr in the bench path. What does point to graphs off is timing: with `GGML_CUDA_DISABLE_GRAPHS=1` the mb4 NCCL run has no slow first rep and no fast later reps, and every step shows the issuing thread blocked in rank 2 and 3 (30B all_reduce[2] 102.7 ms, compute[2] 106.4 ms in steps 1 to 5), where default steps 2 to 5 show enqueue sums under 15 ms except rank 3. That is the signature of direct eval in every step; indirect evidence, not a log proof.
   - Warm direct-eval mb4 (nograph, NCCL, reps 2 to 5): 30B 6075, 6069, 6068, 6081 (median 6072); 32B 2003, 2005, 2005, 2003 (2004). Against the prediction (30B 6512, 32B 2064): 6.8% and 2.9% under. Reps 1 to 5 are equal within 0.2%.
   - Against mb1 nograph (reps 2 to 5 median 4121 and 1289): 30B 1.47, 32B 1.55.
   - Default graphs for reference (reps 2 to 5, replay): 30B 6633 (1.9% above the prediction), 32B 2007 (2.8% under). Rep 1 (capture call) 5138 and 1394: 15% (30B) and 30% (32B) under warm direct. Replay is 9.2% faster than direct on the 30B and 0.2% on the 32B. mb1 and the internal path barely move: mb1 nograph is 0 to 1.4% faster than default; internal mb4 default 5848 vs nograph 5764 (30B), 1719 vs 1714 (32B).
   - Consequence: a judged `-r 1` process runs the warmup (call 1 direct) and then times call 2, the capture. So every judged default value (5138, 1394) is the capture-call speed, not warm direct (6072, 2004) and not replay (6633, 2007).
2. **Graph debug: no log, so no step list from it.** The graphdebug runs repeat the default timing (30B rep 1 5114 then 6587 to 6593; 32B 1399 then 2014 to 2016) and their logs hold no graph line (see 1). From timing and traces only: step 0 (warmup, call 1) is direct and cold (854 ms 30B, 1440 ms 32B, with lazy init); step 1 (rep 1, call 2) is the capture, the issuing thread blocked in compute enqueues (30B rank 2 161.5 ms, three single enqueues of 38 to 41 ms at event index 48, 112, 176; 32B rank 0 574 ms and rank 2 766 ms, four enqueues of 172 to 194 ms on rank 2); steps 2 to 5 (call 3+) replay: enqueue sums a few ms and a device-bound `sync[2]` (30B 177 ms, 32B 996 ms). 30B ranges that do not reach a clean replay: in every replay step rank 3 still blocks 84.8 ms in compute enqueues (one enqueue of 46.8 ms at event index 137 of its 190 compute events, plus 11 of about 1.6 ms at index 150 to 190), rank 2 sums 13.8 ms; the 32B has 1.4 ms on every rank. The log gives no reason and which ranges these are is not identified.
3. **Eager: the first-step excess does not disappear.** `CUDA_MODULE_LOADING=EAGER NCCL_RUNTIME_CONNECT=0` leaves rep 1 where it was (30B 5129 vs 5138; 32B 1394 vs 1394) and reps 2 to 5 equal (30B 6588 to 6602 vs 6628 to 6646; 32B 2005 to 2009 vs 2006 to 2008). It moves cost into the warmup step: 30B step 0 3678 ms vs 854 (compute[0] 2928 vs 156 ms: module loading), 32B step 0 1350 vs 1440. Step 1 equals default (30B compute[2] 160.0 vs 161.5; 32B 761 vs 766). So lazy module or NCCL-connect work is not what slows rep 1; with graphs off the same step is fast (30B 6068, 32B 2004), so rep 1 is slow because it is the capture call.
4. **varykey (mb4, NCCL, five processes, -r 1; step 0 = warmup = call 1 direct at that length in a fresh process, step 1 = the timed capture call).**
   - 30B step 0 / step 1 (ms): 2048 853.3 / 397.2; 2040 900.6 / 393.6; 2032 860.4 / 419.1; 2024 866.5 / 392.7; 2016 851.6 / 396.3 (5145, 5173, 4839, 5143, 5077 tok/s).
   - 32B step 0 / step 1 (ms): 2048 1465.2 / 1456.4; 2040 1394.3 / 1459.1; 2032 1381.8 / 1451.0; 2024 1419.8 / 1457.7; 2016 1387.7 / 1452.5 (1404, 1396, 1399, 1387, 1386 tok/s).
   - Step 1 at every length equals default rep 1 (398 ms, 1468 ms) within 5% (30B 2032 is the outlier: rank 2 and 3 compute 169.0 and 58.3 ms against about 160 and 45). Step 0 is a cold process, so it does not show call 1 at a new shape on a warm process. Does not settle: whether a warm call 1 at a new shape is slow; that needs a bench change (a `-p` list in one process). Inference only: nograph steps (all direct, warm) take 337 ms (30B) and 1020 ms (32B), under the capture call (398, 1468), so a warm call 1 should sit near the nograph value if direct eval does not depend on the length.
5. **Part C: layer-4 with `--layer-limit` ran** (rc 0, five clean reps each, all six runs). The limited layer-4 layout was not dumped, so whether layers sit on all four ranks was not checked; all four GPUs show clocks of 1695 MHz or more in the layer-4 tails, so each was active.

   | 30B tg128 tok/s | single | layer-4 | layer-4 / single |
   |---|---|---|---|
   | L12 | 651 | 649 (rep 1 636) | 0.997 |
   | L24 | 380 | 384 | 1.01 |
   | full (48) | 206.5 | 187 | 0.905 |

   Clocks (tail table above): full single GPU0 1905 MHz alone, full layer-4 1695 on all four (ratio 1.124 against measured 1.104). L12: single GPU0 1890, layer-4 1890/1695/1695/1695. L24: single 1905, layer-4 1965/1995/1695/1695. The layer-4 deficit shows at the full model only (-9.5%), not at L12 or L24, although layer-4 L12 has three GPUs at 1695 and no throughput loss. That weakens "the 1695 MHz plateau explains the excess one for one" (s3d): the full-model gap and the clock gap agree in size, the limited runs do not follow the clock. Does not settle: the limited layer-4 plans may place layers differently (not dumped); single leaves three GPUs idle, so conditions are not equal. A calibration from L12 and L24 would see no layer-4 penalty, from the full model 9.5%; the data are given, not a fit.

## What this does not settle

- That graphs were off in the nograph runs, by the ggml log (the debug line does not reach stderr). Only timing and the host-blocking pattern say so. A proof needs `CudaGraphLogCounts` printed at process end.
- Why the capture call is slower than direct (30B 398 vs 337 ms, 32B 1468 vs 1020) nor why replay is 9% faster than direct on the 30B only; the 30B rank 3 tail that still blocks in replay steps is unexplained.
- Call 1 at a new shape on a warm process (see 4). Judged `-r 1` runs measure the capture call.
- One box, one night; nothing was judged under ADR 0068; no verdict.

## Files

`<model>-tp2pp2-mb{4,1}-pp-{nccl,internal}-{default,nograph}` (plus `-pp-nccl-{graphdebug,eager}` and mb4 `-pp-nccl-varykey-<p>`), each `.json|.log|.trace.jsonl|.trace.summary.txt` (graphdebug also `.graphdebug.log`); part C `qwen3-30b-a3b-{single,layer-4}[-L12|-L24]-tg128.{json,log,clocks.csv,window.txt}`; `nograph-check.txt`, `env.txt`, `plans/`, `layouts/`, `gpu-samples.csv` (1 Hz), `analysis.txt`.
