# Diagnosis: the single-node 235B prediction errors (ADR 0091 Case S risk)

Date 2026-11-24. Phase 6, M3. Input: session `bench/results/stream-predict-20261124/` (README, `errors.json`, `predictions/`, `arms/exit-R-r*.json`, `arms-d2/exit-D2-r*.json`, `arms/oom-D-cgroup-memory.json`), code at `fdb66019`. Diagnosis only: nothing was built or run. Numbers below are medians of the three rounds unless a round is named; derived numbers say how they were derived.

## Summary

- The two code paths price the same plan the same way. 0.49 (ADR 0091 step 2) against 0.145 (`node-predict`) is the hardware graph, not the code: the reader's copy rate is the CPU bandwidth divided by the thread count (40 GB/s / 61 = 0.66 GB/s here, / 12 = 3.3 GB/s on the cluster fixture). Neither number is right; the run read at about 2.8 GB/s.
- Three of the README's readings do not hold. CPU expert compute is not priced 2x too slow: the model sends 75% of decode pairs to the CPU and the run sent 30%, so the bytes are off, not the rate. Arm R's prefill expert matmuls did not run on the CPU: op offload ran them on the GPU, as the cost model assumes. Arm D2's whole-file prefill read is a cold first request, not a property of the 52-slot tier.
- The judged plan's segments are arm R's regime (32 VRAM and 96 RAM slots, every expert resident). There, the decode error (-0.45, -0.33 with the trace) is structural and survives any calibration of the fitted rates: the routing input (VRAM pair share 0.25 uniform, 0.41 trace, 0.70 measured). ADR 0091's "the uniform routing prior does not enter" is wrong for these segments.
- Arm R's prefill error (+0.35) is mostly fitted (offload copy at about 15 GB/s against the 20 GB/s PCIe prior). About +0.11 is structural: 21.6 GB still read after one warm pass, because the RAM tier keeps copies of the VRAM hot set.
- Arm D2's errors are mostly structural: the reader copy-rate artifact in decode; in prefill, the cold first request, the hot-set refresh inside the timed call, and the offload copy of whole RAM-slot tensors per chunk.
- Arm D OOMed because the planner was given the nominal 64 GB, not the 61 GB the balloon leaves. The footprint was right.
- The exit's two-node segments (about 54 GB host, 22.6 GB GPU per node) fit the usable memory with 6 to 8 GB to spare.
- Fixes: three engine or harness steps, four cost-model steps, one budget rule, and one-box confirmation of a 47-layer segment on a 3090 before the rehearsal.

## 1. Tool check: `single_node_runs` against `single_node_references`

**Same pricing.** Both reach the same features and the same predictor (`crates/planner/src/cluster.rs`):

| Step | `single_node_references` (step 2) | `single_node_runs` (`node-predict`) |
|---|---|---|
| Node | `usable_nodes` -> `node_of(hw, name)` | `node_of(hw, None)` |
| Layout | `layout` (whole), and in `cluster_features` `segment_layout(layout, 0..n, embed, head)` | `segment_layout(layout, 0..n, true, true)` |
| Budget and grid | `choose`: 99% of GPU and RAM, `feasible_vram_tiers` x `ram_slot_options`, `footprint` | the same three calls, same 99% |
| Features | `run_features` -> `builder_features_tier(sub, v, uniform_tier(sub, [(v, r)]), (threads, ub, true), Reader, (16, 16), nvme0)` | the same call; `--tier-replay` replaces only `tier.decode` |
| Threads | `plan.knobs.threads.unwrap_or(node.threads)`; `assemble` leaves it `None` | `node.threads` |
| Predictor | `predict_cluster`, which for one segment returns `predict(node_graph, run)` unchanged (`costmodel/src/cluster.rs` line 185) | `predict(node.graph, feat)` |

The pick can differ only on a tie of `request_s` (references pick per ubatch, then across ubatches; `node-predict` sorts all). The numbers for one pair are bit-identical by construction.

**The difference is the graph.** Both planned (16, 56) under the uniform prior, which sends `(1 - 16/128)(1 - 56/112) = 0.4375` of the active expert bytes (8 x 94 x 11.41 MB = 8.58 GB) through the disk: 3.75 GB per token. The reader term is `bytes / B_eff` with `1/B_eff = 1/B_disk(16) + 1/B_copy`, and `B_copy = cpu_rates(1)`. A CPU device without a thread curve scales linearly, so `B_copy = B_cpu / threads` (`profiler/src/graph.rs` `cpu_rates`):

| Graph | Threads | `B_copy` | `B_disk(16)` | `B_eff` | Read term | tg |
|---|---|---|---|---|---|---|
| Cluster fixture (`cluster_two_3090_lan.toml`) | 12 | 3.33 GB/s | 5.2 GB/s | 2.03 GB/s | 1.85 s | 0.49 (plus 0.19 s compute) |
| Session graph `hw-D.toml` | 61 | 0.66 GB/s | 3.99 GB/s | 0.563 GB/s | 6.67 s | 0.145 |
| Session graph with the prior disk (`pred-D-uniform-diskprior.json`) | 61 | 0.66 GB/s | 5.2 GB/s | 0.58 GB/s | | 0.150 |

The prefill gap (100 against 85.7 tok/s) is the disk curve: prefill reads have no copy term, and the disk-prior variant gives 100.4.

**Which is right:** neither. One thread's copy rate does not shrink when the container gets more cores. Arm D2 read 682 MB per token in 242 ms of reader time, copies included (2.8 GB/s), on 61 threads. The 12-thread number is closer by accident. The fix is in section 7 (C2).

**Equivalence test (for a Sonnet worker)**, `crates/planner/tests/cluster.rs`, next to `a_reference_predicts_as_the_single_node_path_does`:

```
#[test]
fn a_reference_equals_its_node_run() {
    let layout = fixtures::moe_235b();
    for (name, hw) in graphs() {
        for r in single_node_references(&hw, &layout, &Hints::default()) {
            let seg = segments(&r.plan, 0, &hw).unwrap().remove(0);
            let st = &segment_plan(&r.plan, 0, &seg).replicas[0].stages[0];
            if st.expert_cache.is_empty() { continue; } // CPU-only node: node_run needs a GPU
            let slots = |id| st.expert_cache.iter().find(|c| c.device == id).unwrap().slots;
            let (v, ram, ub) = (slots("gpu0"), slots("cpu0"), r.plan.knobs.ubatch.unwrap());
            let one = hw.node_graph(seg.node.as_deref());
            let (pinned, _) = single_node_runs(&one, &layout, &W, Some((v, ram)), None).unwrap();
            let run = pinned.iter().find(|x| x.ubatch == ub).unwrap();
            assert_eq!(run.pp_tps.to_bits(), r.prediction.pp_tps.to_bits(), "{name}");
            assert_eq!(run.tg_tps.to_bits(), r.prediction.tg_tps.to_bits(), "{name}");
            assert_eq!(run.request_s.to_bits(), r.prediction.request_s.to_bits(), "{name}");
            let (all, _) = single_node_runs(&one, &layout, &W, None, None).unwrap();
            assert_eq!(all[0].request_s.to_bits(), r.prediction.request_s.to_bits(), "{name}: same pick");
        }
    }
}
```

`W` is the file's `Workload { prompt: 2048, gen: 128 }`. Nodes without `nvme0` are skipped by `single_node_runs` returning `Err`; skip on `Err` only when `seg`'s node has no disk. A second test pins the explanation and fails until C2 lands: in `costmodel/src/predict.rs` tests, `the_reader_copy_rate_does_not_depend_on_the_thread_quota` takes `hw_disk()` with `cpu0.threads` 12 and then 61 (no thread curve), the same `TierFeatures`, and asserts equal `read:nvme0` terms.

## 2. Term table

Measured terms come from `decode_profile_per_token` and `prefill_profile_per_ubatch`, from `*_io` and `*_prefetch`, and from the host sampler block (`host.prefill`, `host.decode`). The profile's `seg_compute_cpu_ms` is the time of segments that hold a RAM-tier group (`qwen3moe.rs` line 1765), not CPU time. Within one such segment, GPU work and CPU groups run serially. "Fitted" means a parameter that ADR 0091's anchor fit sets: the ADR 0040 fit in `costmodel/src/calib.rs` covers GPU F, B and c0, CPU F and B per thread count plus c0, and PCIe alpha and beta. Disk curves are measured, not fitted. "Structural" means calibration does not change it: the bytes, the routing input, the formula's shape, or a run state the model does not represent.

### Arm R (32 VRAM + 96 RAM slots, every expert resident; the exit's segment regime)

| Phase | Term | Predicted (uniform; trace) | Measured | Class |
|---|---|---|---|---|
| tg | CPU experts `compute:cpu0` | 163.0 ms (6.52 GB at 40 GB/s); 128.1 ms | about 54 ms: 82.5 ms of RAM-group segments minus about 29 ms of their GPU share (0.36 ms per GPU-only segment). 2.56 GB per token: 224.6 CPU pairs x 11.41 MB, about 47 GB/s | Bytes structural (the routing input: VRAM pair share 0.25 uniform, 0.41 trace, 0.70 measured). Rate fitted (40 against about 47 GB/s) |
| tg | GPU `compute:gpu0` | 18.3 ms; 20.6 ms | about 34 ms (95 segments at about 0.36 ms) | Fitted (GPU F, B, c0) |
| tg | Hops `switches` | 2.2 ms; 2.9 ms | not separable (inside the segments). Full-hit layers 0.148 measured against 0.02 trace | Fitted (alpha); the full-hit share is a routing input |
| tg | Host: alloc, build, readback, streamer | none, except c0 x ops | 12.8 ms (5.4 + 2.4 + 2.2 + 2.8) | Fitted through c0 (per op, not per segment: approximately the right shape) |
| tg | Disk | 0; 0.15 ms | 6.2 MB per token, nothing waited | Correct |
| tg | Total | 183.6 ms; 151.9 ms | 101.3 ms | e_tg -0.45; -0.33 |
| pp | Offload copy of the RAM tier (`offload_copy:cpu0`) | 5.15 s (103 GB at 20 GB/s) | 9.59 s of RAM-group segments hold the copy plus all GPU compute: about 15 GB/s if the GPU terms are as priced, at least 10.7 GB/s | Fitted (PCIe beta; the RAM tier is pageable host memory) |
| pp | GPU compute (`compute:gpu0` + `offload_compute:cpu0`) | 1.38 + 1.17 s | inside the 9.59 s; segment 0 alone 6 ms | Fitted |
| pp | Hops | 0.24 s | inside the segments | Fitted |
| pp | Disk | 0 (a covering tier) | 21.6 GB, 1884 experts, about 20 per layer; streamer 1.04 s, 97.5% prefetched | Structural (section 3) |
| pp | Total | 7.94 s; 8.02 s | 10.72 s | e_pp +0.35; +0.34 |

### Arm D2 (16 VRAM + 52 RAM slots, 64 GB balloon, streamed)

| Phase | Term | Predicted (uniform; trace) | Measured | Class |
|---|---|---|---|---|
| tg | Disk `read:nvme0` | 7.14 s (4.02 GB at 0.563 GB/s); 1.12 s (631 MB at 0.563 GB/s) | streamer 208 ms: 682 MB read in 242 ms of reader time, 139 ms of it waited. RAM hit 0.874 against the prior's 0.464 | Bytes: uniform structural (6x too many), trace right (+8%). Rate structural: `B_copy = B_cpu / threads` (section 1), which calibration keeps, since the fit has one thread point and scales down from it |
| tg | CPU experts | 190.2 ms; 163.9 ms | 167.5 ms of RAM-group segments (4.58 GB, 401.7 pairs; VRAM pair share 0.466 against 0.125 uniform and 0.246 trace) | Rate fitted; bytes structural (routing) |
| tg | GPU and hops | 19.1 ms; 21.3 ms | 0.5 ms GPU-only segments | Fitted |
| tg | Host | none | 16.3 ms | Fitted through c0 |
| tg | Total | 7.35 s; 1.30 s | 379 ms | e_tg -0.95; -0.71 |
| pp | Disk | 16.1 s (64.4 GB at 4.0 GB/s, the uniform steady-state share) | streamer 36.0 s for 137.8 GB: 12073 experts, every one once plus 41 | Structural, a protocol mismatch: the timed prefill was the process's first (no warm pass), so it is a cold first touch. A warm 52-slot tier reads about 60 of 112 per layer, about 64 GB, which the uniform prior gets right |
| pp | Hot-set refresh | none | 7.7 s (`prefill_s` 57.9 against profile total 50.2; 1496 uploads; `finish_step` runs `refresh_hot_set` after the first prefill) | Structural (a first-request cost) |
| pp | Offload copy | 6.01 s (120 GB at 20 GB/s) | in `compute_ms` 16.1 s: 3 chunks per layer, each copying its whole 52-slot tensor, so 156 slot rows (167 GB) instead of 112 | Rate fitted. Bytes structural (copies per chunk) |
| pp | GPU compute and hops | 1.19 + 1.37 + 0.28 s | in the 16.1 s; segment 0 took 142 ms (cold CUDA kernels, no warm pass) | Fitted |
| pp | Total | 25.0 s | 57.9 s | e_pp +1.32 |

### What remains after an ideal calibration of the fitted terms

Ideal: every fitted parameter set to the value that makes its term equal the measured physical time, with the model's own bytes kept. That is calibration that "fits the rates" rather than "absorbs the errors".

| Arm, phase | Uncalibrated | After ideal calibration | What is left |
|---|---|---|---|
| R tg | -0.45 (uniform), -0.33 (trace) | -0.45 (uniform: 6.52 GB at 47 GB/s, about 184 ms), -0.35 (trace: about 155 ms) | All of it: the routing input. With the measured share the model lands at about 101 ms (e about 0) |
| R pp | +0.35 | about +0.11 | The 1.04 s first-touch read after one warm pass |
| D2 tg | -0.95, -0.71 | -0.94, -0.67 (fitting B_cpu to 47 GB/s lifts `B_copy` only to 0.78 GB/s) | The copy-rate artifact and the uniform bytes. With the trace and a real copy rate (8 GB/s; `B_eff` 2.66 GB/s, 237 ms) about -0.10 |
| D2 pp | +1.32 | about +1.14 (copy at 15 GB/s) | Cold first touch (+73 GB, about 18 s), refresh (7.7 s), per-chunk copies (+47 GB, about 3 s). With all three priced, about +0.03 |

**A caveat that matters for the exit.** A real fit does not reach the "ideal" column: it fits the rates through the model's features. If the anchors' routing matched the judged run, B_cpu would absorb the share error: 6.52 GB per 54 ms gives an unphysical 120 GB/s, and the judged prediction would land close for the wrong reason. ADR 0091's anchors take a fixed-seed hidden tensor as input, not the exit prompt's hidden states, so their routing locality is unknown and need not match. Arm R's share went from 0.25 (slots) to 0.70 because the hot set came from the same prompt's prefill. The model needs the routing as an input (section 7, C1), and the anchors must be fitted with their own measured shares.

## 3. The 21.6 GB in arm R's prefill

Mechanism, from `crates/model/src/experts.rs` and `crates/memory/src/slots.rs`:

1. The VRAM compute tier and the RAM tier are independent LRU tables (`SlotTable::lookup`), not one exclusive split. `plan_cpu` sends every routed expert that is not in a compute slot to the RAM tier, in chunks of the RAM slot count (`fill_ram`).
2. The warm pass (the process's first prefill) starts with empty compute slots. The about 116 distinct experts per layer that the prompt routes all go through RAM (96 slots, LRU). 11071 experts were read, about 117.8 per layer.
3. At the end of that first prefill, `finish_step` calls `refresh_hot_set`. The top 32 experts by count are made resident through `load_via_ram`, which fills RAM first and then copies to VRAM, so all 32 keep a RAM slot. After the refresh, RAM holds the 32 hot experts plus 64 others.
4. The timed prefill routes the same about 116 experts. The 32 hot ones run from VRAM, and about 84 go to RAM, which holds only 64 of them. About 20 per layer are read (1884 measured), and they evict the LRU-oldest entries: the hot copies, untouched since the refresh. After this second prefill, RAM holds exactly the non-hot experts, and decode reads 69 experts in 128 tokens (RAM hit 0.998).

The routing replay of `predictions/replay-R.json` shows the same thing: it mirrors the engine's LRU and gives 5.07 MB per token of prefill disk reads for (32, 96). ADR 0056 rule 5 anticipated it ("if the chunked `fill_ram` breaks inclusion ... RAM = all experts outside VRAM giving no decode misses after warm-up").

**Does it break the exit's premise?** Yes, in two ways. (a) ADR 0091's "nothing comes from disk" holds only from the third request of a process. The first request reads the whole segment, about 68.6 GB of experts at 47 layers, plus the refresh (arm R's warm pass ran at 23.4 tok/s). The second reads about 16% of the RAM tier. (b) Case S takes ADR 0068's workload and rounds, and each `exit_regime`-style round is a fresh process, so a judged pp2048 would be the cold first request unless the protocol says otherwise. ADR 0091 does not say.

**Fix: engine-side, plus an explicit protocol.** When a layer's active compute slots plus RAM slots cover every expert, `refresh_hot_set` should leave RAM holding exactly the complement of the hot set: free the RAM slots of the hot experts and read the missing non-hot ones. Then a covering plan is read-free from the second request on, for any routing, and the one-time cost falls on the first request. The model should not price first touch. The exit judges a serving rate, and first touch is a per-process transient whose size depends on the warm-up, not on the plan. Pricing it would make the prediction a prediction of the protocol. The protocol has to make the steady state explicit: warm requests until a prefill reads 0 bytes (covering plans) or one full request (others), and a covering plan whose judged prefill reads more than 0 bytes is unclean.

## 4. Prefill expert matmuls in arm R: on the GPU, not the CPU

They ran on the GPU. With `op_offload` and a step of at least 32 tokens (`offloads`, ADR 0044), RAM-group matmuls go to ggml's scheduler as ops on host weights, which ggml runs on CUDA after copying the RAM-slot tensors (`group_cap(Ram, true, true)` is `Some`, the CUDA row cap). The profile counts these segments as "cpu" because they hold a RAM group. The evidence:

- The engine used 3.8 cores during arm R's prefill and 26.9 in decode (host sampler `engine_cores`), with GPU busy at 65%.
- 366,635 pairs at 2048 rows per layer in 9.5 s is not a 61-thread CPU K-quant rate.
- Both arms have exactly one GPU-only segment per ubatch (`segs_gpu` 1): segment 0, which holds no expert group. Its 142 ms in D2 against 6 ms in R is cold CUDA kernels: D2 had no warm pass.

ADR 0030 has no rule on this (its decision list has no point 6 about prefill placement). The rule is ADR 0044. The cost model knows it: `predict.rs` charges `offload_copy` plus `offload_compute` for CPU parts at M >= 32, with the same threshold. Arm R's prefill error is therefore the copy rate (fitted) and the first-touch reads (section 3), not placement. Aside (not a fix here): the hot set served 76% of prefill pairs (CPU pairs 23.8%), yet op offload copies the full 103 GB RAM tier per 2048-token ubatch. Whether computing the remaining 24% on the CPU beats the copy is a separate measurement.

## 5. Budget rule under `EMULATE_RAM_GB`

**Why the 64 GB pick OOMed.** `arms/oom-D-cgroup-memory.json`: limit 1032.77 GB, balloon 968.77 GB. At the last sample before the kill, anon was 1029.71 GB (60.94 GB outside the balloon, the engine's RSS 61.2 GB including file-backed pages) and file 0.52 GB. That leaves 2.55 GB of kernel memory: the balloon's page tables (968.77 GB / 4096 x 8 B = 1.89 GB) plus the engine's page tables and slab. So the engine could use about 61 GB, and the planner's pick needed tier + 2.4 GB = 62.4 GB. The planner's own footprint (`footprint`: tier + embed 0.35 GB + `BUILDER_IO` 0.27 GB + `HOST_RESERVE` 2.15 GB = 62.9 GB) is slightly conservative against the measured peak (tier + 2.29 to 2.45 GB in R and D2) and is not the problem. The problem is the budget: the planner was given 64 GB (63.4 GB after its 1% margin), and the container left about 61. One composition detail: the measured `io` buffer is 1.6 GB, not `BUILDER_IO`'s 256 MiB. `HOST_RESERVE` covers it today; worth correcting when the footprint is next touched.

**Rule (proposed for ADR 0092).** The planner's RAM budget under the balloon is measured, not nominal. `exit_run.sh`, after the balloon holds and with the samplers running, before the engine starts, writes into the node graph's `cpu0.capacity_bytes`:

```
ram_budget = memory.max - memory.current + inactive_file - 1 GiB
```

All values are from cgroup v2 `memory.max`, `memory.current` and `memory.stat`. Reclaimable page cache counts as free. The 1 GiB reserve covers the engine's own page tables and slab, allocator growth, and the reclaim floor. The planner's 1% margin stays on top. For cross-checking: `ram_budget` is about `EMULATE_RAM_GB - balloon / 512 - 1 GiB - other processes`. That is about 59.9 GB on this box, where the rule accepts D2's 58.6 GB footprint (it ran, peak 58.2) and rejects D's 62.9 GB. On a 3090 box with a 128 GB limit it is about 61.7 GB (balloon page tables 0.13 GB). `env.json` records `ram_budget` and its inputs. A run whose engine peak exceeds `ram_budget` fails the budget, not the balloon.

**ADR 0091's two-node footprint.** From `predictions/layout.json` (slot row 0.536 GB per 47 layers):

| Segment | RAM tier (96 slots) | Host footprint (planner) | Host, measured style (tier + 2.45 GB) | GPU footprint (32 slots, dense, KV, ubatch 2048 buffers, 600 MiB) |
|---|---|---|---|---|
| Leader, layers 0 to 46 (ADR 0091's span 0..47), embed and head | 51.5 GB | 54.3 GB | 53.9 GB | 22.6 GB |
| Node b, layers 47 to 93 | 51.5 GB | 53.9 GB | 53.9 GB | 22.0 GB |

Against about 61 to 62 GB usable (the rule above) this leaves 6.6 to 8 GB of host margin. Against a 24 GiB 3090 (25.77 GB, minus about 0.3 GB for the driver) it leaves about 2.8 GB of GPU margin. The GPU footprint was conservative in arm R (41.9 GB planned, 41.6 GB peak). The exit plan fits. The leader's report-only single-node reference, the (16, 56) pick, does not fit and must be re-picked under the rule.

## 6. Hardware confound (A40 and 61-thread EPYC here, 3090 and 8 or more threads at the exit)

| Error | Different on the exit's boxes? | Why |
|---|---|---|
| Routing input (VRAM pair share, full-hit share, RAM hit) | No | A property of model, prompt and the first-prefill hot-set rule. It weighs at least as much with fewer threads, because the CPU term takes a larger share of decode |
| First-touch reads after one warm pass; the cold first request; the refresh inside it | No | Engine logic and protocol |
| Offload copy per chunk (`n_chunks x ram_slots` slot rows) | No | Engine logic. Covering segments have one chunk, so the exit is unaffected |
| Reader `B_copy = B_cpu / threads` | Yes, in size | At 8 to 12 threads it is 3 to 5 GB/s, about the disk rate, so the error shrinks from 7x to about 2x. It matters only where disk bytes move: the report-only reference, not covering segments |
| CPU expert rate, GPU rates, c0 | Yes | Fitted per box by the anchors (A40 696 GB/s against the 3090's 936; consumer CPUs have fewer memory channels) |
| Offload copy rate (PCIe, pageable host memory) | Yes | Fitted. It depends on host memcpy and the PCIe generation (some Vast 3090 boxes run Gen 3) |
| Disk curve | Yes | Measured per box by `disk-curve` |
| Balloon overhead | Yes | Page tables scale with `memory.max - 64 GB`. 1.9 GB here, 0.1 to 0.4 GB on a 128 to 256 GB container. The measured rule absorbs it |

So the errors that survive calibration also survive the change of hardware. The hardware-dependent ones are exactly the fitted ones, plus the copy artifact, whose effect shrinks at the exit's thread counts but not to zero.

## 7. Recommendation

### Build steps (each with its check)

Engine and harness:

- **E1. Complement fill of a covering RAM tier.** In `Streamer::refresh_hot_set`: when `active + ram.slots >= n_expert` for a layer, free the RAM slots of the hot experts and read the missing non-hot ones. Check: a model test on the tiny MoE fixture (n_expert 8, 2 VRAM + 6 RAM slots, prefetch off): after one prefill and its refresh, a second prefill and a decode read 0 bytes (`prefill_io.bytes == 0`), and the logits are bitwise equal to the run without E1. The `replay` crate's LRU mirrors the change (`tetraeder-bench replay` gives 0 prefill disk bytes for (32, 96) on the committed traces).
- **E2. Exit warm protocol.** `exit_regime` gains `TETRAEDER_EXIT_PREFILL_WARM=n` (n untimed full requests, prefill and decode). The result records `prefill_io.bytes` of the timed prefill. The Case S harness and the rehearsal use warm-up until a covering segment's prefill reads 0 bytes, and a covering plan with timed prefill bytes above 0 is unclean. Check: CI test of the flag on the tiny fixture. On the box, the confirmation run's JSON shows 0.
- **E3. Segment exit regime.** `exit_regime` runs a layer span (`TETRAEDER_EXIT_LAYERS=a..b`) with no head, with hidden-state input from a file (`TETRAEDER_EXIT_HIDDEN_IN`) or a fixed seed, and can dump the hidden rows at a layer boundary from a whole-model run (`TETRAEDER_EXIT_HIDDEN_OUT=layer,file`; prefill rows and the 128 greedy decode rows). It shares the builder's hidden-row entry point with `cluster-pipeline-nodes`. Check: CI parity of a two-span run of the tiny fixture (span 1 dumps, span 2 reads) against the whole-model run within `tol_batch`.

Cost model and planner (one ADR, 0092; `MODEL_VERSION` 6; the ledger ratchet applies):

- **C1. Routing as an input.** The decode split of a builder segment (VRAM pair share, full-hit share, RAM hit, disk bytes) comes from a replay of a routing trace of the judged workload: the frozen exit prompt, pp2048 plus greedy tg128, recorded once with `TETRAEDER_EXIT_ROUTING_OUT` and hash-frozen before any judged run. It no longer comes from the uniform prior. `predict --cluster` and `node-predict` take `--tier-replay` per segment. Without a trace, the prediction is marked `routing: uniform` and is not judgeable. Anchor features are built from each anchor's own measured `decode_layers` shares and `cpu_pairs`, not from a prior, so the fit sets physical rates. Check: in sample, arm R and D2 re-priced with their measured shares and in-sample fitted rates are within 0.15 on tg (in-sample only, recorded as such). Planner rule-4 tests run on every fixture with a synthetic trace.
- **C2. Reader copy rate.** `B_copy` becomes the CPU device's measured single-thread copy rate (`copy_bytes_per_s`; prior 8 GB/s, measured by `tetraeder-bench disk-curve --copy` or a memcpy probe in the profile), not `cpu_rates(1)`. Check: the failing test of section 1 passes. D2 decode re-priced with the trace replay lands within 0.15 in sample (estimate -0.10).
- **C3. Offload copy per chunk.** The prefill offload copy of a builder segment is `ceil(outside / ram_slots) x ram_slots` slot rows per layer, not `(1 - f)` of the expert bytes. Covering plans are unchanged (one chunk). Check: unit test (16, 52) gives 3 chunks and 156 rows. D2's in-sample prefill compute term within 0.15 of `compute_ms` (16.1 s).
- **C4. Budget rule.** Section 5's `ram_budget` in `exit_run.sh` and in the node graph. `ram_slot_options` gains the largest RAM slot count that fits the budget, so the pick is not limited to quarter, half and all of the experts outside VRAM. Check: rule-5 property "footprint never exceeds budget" on every fixture, plus a test that the session's graph with `ram_budget` 59.9 GB rejects (16, 56) and accepts (16, 52).

Not changed: the model keeps pricing steady state. First touch and the refresh are handled by E1 and E2, not priced.

### ADR outline: 0092, "Cost model 6: routing input, copy rate and chunked offload for streamed MoE segments"

- **Status.** Proposed. Phase 6, M3. Supersedes nothing. Amends ADR 0091 Case S calibration and protocol, with the 15% rule unchanged.
- **Context.** `stream-predict-20261124` (this note): the four error sources, which of them calibration fits, and the 64 GB OOM.
- **Decision.** C1 to C4 and E1 to E3 as above. The exit's judged prediction uses the exit prompt's routing trace (a workload input recorded before the judged runs, not a timing anchor, so ADR 0068's "no anchor equals a candidate" holds). Anchors stay truncated segments and are fitted with their own measured routing shares. Judged rounds start from a warm, read-free state.
- **Fitted on / confirmed on.** Forms and constants are chosen on this session only (in-sample numbers labelled as such). Confirmation is a fresh one-box session with predictions frozen before its first measured run (below). If it misses 15% on either rate, the exit does not run and the miss goes back to this ADR.
- **Consequences.** The ledger re-predicts every committed builder result under version 6 (ratchet, `--accept-worse` with reasons). `node-predict` and `predict --cluster` refuse a judged prediction without a trace. One more box stage (trace record, about 3 minutes on the 235B).
- **Owner question.** Is the exit prompt's routing trace an admissible input to the judged prediction? Recommended yes: it fixes the workload, not the plan's timing. The alternative is an anchor input from real hidden states plus a share-transfer rule, which is weaker.

### One-box confirmation run (gates the Phase 6 rehearsal)

One RTX 3090 box (24 GB, `cuda: ok`, 230 GB disk, CPU quota 8 or more, the exit's box spec), `EMULATE_RAM_GB=64` with the section 5 budget, the 235B Q4_K_M.

1. `disk-curve`; `env.json` with `ram_budget`.
2. Whole-model greedy run of the frozen exit prompt (streamed, report only). It dumps the hidden rows at layer 47 (E3) and records the routing trace (C1). Hash both. About 3 minutes.
3. Anchors the exit's way: layers 47 to 58 and 47 to 70 (quarter and half of the span), fixed-seed hidden input, no head, 32 VRAM + 96 RAM slots, pp2048 tg128, two rounds, warm (E2). Fit.
4. Freeze `predict` of the segment layers 47 to 93 (32 + 96 slots, ubatch 2048), with decode from the trace replay of step 2. Commit the hash before step 5.
5. Judged: the 47-layer segment, input from step 2's hidden rows, warm until prefill reads 0 bytes, three rounds. Pass: `|e_pp| <= 0.15` and `|e_tg| <= 0.15` on the medians.
6. Report only: the same segment with fixed-seed input (the routing gap between anchor input and real input), and step 3's fit redone on real-hidden anchors.

About 1.5 h of box time, dominated by the download.

### ADR 0091 "Expected numbers": reliability

| Number | Reliable? | Why |
|---|---|---|
| Hand estimate, one node 64 GB: tg 2.0 to 2.7, pp about 30 | Roughly | Rests on measured runs. D2 (52 RAM slots, A40) measured 2.64 and a cold 35.4 |
| Hand estimate, two nodes: tg about 4 (LAN), 2.9 (WAN); pp about 100, 95 | Unreliable | "Nothing comes from disk" (false before the third request, section 3), and 250 ms per token of CPU expert work at a 27% VRAM share against the measured 70% pair share |
| Planner step 2, two nodes: tg 5.43, 3.62; pp 298, 294, 274 | Unreliable | Uniform VRAM share in a covering segment (arm R: -0.45 on tg), the 20 GB/s PCIe prior for the offload copy (arm R: +0.35 on pp). By arm R's analogy decode is under-predicted and prefill over-predicted |
| Planner step 2, one node: tg 0.49, pp 100 | Unreliable | The `B_copy` thread artifact (section 1) and the uniform prior. Its plan (16, 56) also does not fit the balloon (section 5) |
| "Each spanning segment ... nothing is read from disk and the uniform routing prior does not enter" | Wrong | Routing enters through the VRAM and CPU split of compute. The disk is read until the third request |

## Proposed tree changes (for the coordinator; `docs/roadmap/nodes.yaml` not edited)

```yaml
- id: planner-node-predict-check
  title: Node prediction equals the cluster reference
  track: planner
  kind: tooling
  state: next
  depends_on: [bench-235b-stream-predict]
  lane: ci
  plain: 'A test pins that the single-node prediction tool and the cluster reference price the same plan bit for bit, and a second test pins that the reader copy rate does not depend on the thread count (fails until the cost-model fix).'
- id: engine-ram-tier-complement
  title: Read-free covering expert tiers
  track: memory
  kind: defect
  state: next
  depends_on: [bench-235b-stream-predict]
  lane: ci
  plain: 'When VRAM and RAM slots together hold every expert, the hot-set refresh leaves the RAM tier holding exactly the other experts, so from the second request on nothing is read from disk (today about 16% of the tier is read again).'
- id: exit-segment-harness
  title: Run one layer span with hidden-state input
  track: engine
  kind: tooling
  state: next
  depends_on: [bench-235b-stream-predict]
  lane: ci
  plain: 'The exit harness runs a span of layers without the head, takes hidden rows from a file or a fixed seed, dumps them at a layer boundary from a whole-model run, and warms up until a prefill reads nothing from disk.'
- id: cluster-owner-routing-input
  title: Owner admits the exit prompt routing trace
  track: cluster
  kind: owner-decision
  state: next
  depends_on: [bench-235b-stream-predict]
  lane: owner
  plain: 'The owner decides whether the routing trace of the frozen exit prompt, recorded before the judged runs, may price the judged prediction (recommended, ADR 0092 draft).'
- id: costmodel-stream-terms
  title: Cost model 6 for streamed MoE segments
  track: planner
  kind: feature
  state: next
  depends_on: [bench-235b-stream-predict, planner-node-predict-check, cluster-owner-routing-input]
  lane: ci
  plain: 'ADR 0092 makes routing an input from a trace replay, fits anchors with their own measured shares, prices the reader copy at a measured single-thread rate and the prefill offload copy per chunk, and gives the planner the RAM budget the balloon really leaves.'
- id: bench-235b-segment-confirm
  title: Confirm one exit segment on a 3090
  track: bench
  kind: experiment
  state: blocked
  depends_on: [costmodel-stream-terms, engine-ram-tier-complement, exit-segment-harness]
  lane: box
  plain: 'One 3090 box under the 64 GB balloon runs the 47-layer exit segment with 32 VRAM and 96 RAM slots on real hidden-state input, calibrated the exit way and predicted before measuring, and must land within 15% on both rates before the two-box sessions.'
```

Changes to existing nodes: `bench-phase6-rehearsal` gains `bench-235b-segment-confirm` in `depends_on`. The `notes` of `bench-235b-stream-predict` should drop "CPU expert compute is priced 2.0x too slow ... a prior the anchor calibration fits" (it is the routing share, which is structural) and point to this note.
