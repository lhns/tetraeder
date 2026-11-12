# 0064. Decode dispatch: a scheduler sized for one segment, then one graph per decode token with a host callback per layer

Status: Proposed (2026-11-12). Phase 2 exit follow-up (ADR 0045, m1-open). Builds on ADR 0035, 0036, 0038, 0054 and 0061. Step 2, if it ships, supersedes for streamed one-token decode steps the part of ADR 0036 that keeps one graph segment per MoE layer, and the ADR 0035 clause "work is exactly the routed pairs" for the GPU group of such steps (parked rows, below).

## Context

A streamed decode token of Qwen3-235B-A22B (94 layers, 8 of 128 experts) runs 95 segment graphs (`run` in `crates/model/src/qwen3moe.rs`, one `build_segment` per layer boundary) plus the overlap group graphs of ADR 0054 (`run_overlap_layer`, `run_cpu_group` on `sched_cpu`). A fit over the 16 decode windows of `20261112c` (EPYC 7302P, 30 threads) gave 0.88 ms fixed per CPU-carrying segment plus 0.063 ms per CPU pair, so about 70 ms of fixed cost against 40 ms of expert work per token, and 0.44 ms per GPU-only segment. llama.cpp's expert cache (#30966) decoded 1.35x to 1.54x faster on the 7642 hosts (`20261112a`). This ADR counts both engines' dispatch per token from source, says where the fixed term goes, and designs the change that cuts it.

Sources: llama.cpp b12274 (202c1a229) as `docs/sources.md` names it. The cache runs of `20261112a` and `c` record b12211 (7d7f028); between the two, `ggml-backend.cpp`, `llama-context.cpp` and `llama-graph.cpp` only add `hv_tensor_requires_copy` (skip copies for uncomputed select branches) and scheduler reuse across reserves, so the decode structure below holds for both. Our side: the pinned ggml (2e0d86be) and the files named.

### Per decode token, from source

llama.cpp (`-cmoe --moe-cache-mib 10458 --moe-cache-cpu-share 0.95`, one GPU, OpenMP off as in `docker/box/Dockerfile`):

- One graph compute. `llama_context::process_ubatch` reuses the previous graph when `res->can_reuse(gparams)` holds (no build, no `ggml_backend_sched_alloc_graph`), and `graph_compute` calls `ggml_backend_sched_graph_compute_async` once. The `20261112c` log: `graph (pp bs=2048, tg bs=1): nodes = 5944 / 7153, splits = 284 / 284` and `graphs reused = 69` of about 72 decode tokens.
- 284 splits, 3 per layer plus 2. `build_moe_ffn` (`src/llama-graph.cpp`) puts the router, `cur` and then `build_moe_cache_slots`: a `get_rows` of the host slot map, pinned to the GPU with `ggml_backend_sched_set_tensor_backend`. A host weight read by a GPU node starts a new split (`ggml_backend_sched_split_graph`, "check if a weight is on a different and incompatible backend"). Per layer: GPU split (attention, router), GPU split (slot lookup, cached experts, which come first in the graph "so that they are computed while the misses are"), CPU split (`get_rows` of the host `cpu_map`, the three `mul_mat_id` on the host experts with `-1` for experts not on the CPU), and the merge `add` opens the next GPU split.
- One host callback per layer. Copying the slot map into its GPU split calls `llama_context::sched_copy_experts`, which calls `llama_moe_cache::impl::copy` (`src/llama-moe-cache.cpp`): `ggml_backend_tensor_get_async` of the ids plus `ggml_backend_synchronize` (one GPU sync), the LRU plan, async uploads of the misses, the CPU map written in host memory, the slot map set async. `ggml_backend_sched_compute_splits` copies host weights last so the callback sees the split's other inputs.
- CPU and GPU overlap inside a layer. `ggml_backend_sched_prefetch_inputs` (added by #30966, not in our pin) starts the D2H copy of the CPU split's inputs on the GPU stream before the cached-expert split, so the CPU split waits on that copy only and runs while the GPU computes the cached experts.
- About 95 CPU threadpool kickoffs (94 CPU expert splits plus the embedding `get_rows`; `tok_embd` stays on the CPU). Workers spin between them: llama-server's default poll 50 is `1024 * 128 * 50` rounds of `ggml_thread_cpu_relax` in `ggml_graph_compute_poll_for_work` before sleeping, longer than a layer's GPU work.
- About 2 to 3 blocking device operations per layer: the callback's sync and ids read, the merge split's copy of the CPU rows (no events: `parallel` is false with one GPU, so `ggml_backend_sched_copy_input` falls back to `ggml_backend_synchronize` and a blocking copy).
- CPU pairs: 32,623 CPU misses of 54,629 lookups (59.7%, warm-up tokens included), about 449 of 752 pairs per token; 35.3% cache hits.

The builder (`main-overlap`, `20261112c`, per token):

- 133.5 graph builds, scheduler allocations and computes: 95 segments plus 38.5 CPU group graphs. Build 3.2 ms, alloc 12.8 ms. Every `Sched::alloc_graph` (`crates/ggml-sys/src/ggml/sched.rs`) resets the scheduler, splits and allocates.
- Splits: a CPU segment has 2 (CPU: `get_rows` of `xn`, the RAM group's `mul_mat_id` and `swiglu`; GPU: `set_rows` into `acc`, the compute-tier group, weighting, attention, router), a GPU-only segment 1, a group graph 1: about 78 x 2 + 17 + 38.5 = 212. (`max_splits` 4 is over the run, prefill included.)
- Blocking device operations per CPU segment, from `ggml_backend_sched_copy_input` without events: the CPU split's `xn` input (GPU sync, D2H), the GPU split's four `set_input` index tensors (allocated on the CPU, each a GPU sync and a blocking H2D) and the RAM rows (sync, H2D), the closing `ggml_backend_sched_synchronize`, then the readback of `ids`, `pred` and `pred2` (three blocking D2H, 3.0 ms per token): about 7 syncs and 9 copies.
- About 117 CPU kickoffs (78 CPU segments, 38.5 group graphs). The CPU group runs before the GPU work of its segment, and the GPU waits for it.
- CPU pairs: 628 of 752 (83.5%); VRAM pair share 16.5% at a similar budget (10.97 GB of slots against the cache's 10,458 MiB).
- The scheduler is sized for the whole-step graph: `step_nodes(h) * 4` = 128 x 95 x 4 = 48,640 nodes, so its hash set has 65,537 entries (`ggml_hash_size`). `ggml_backend_sched_reset` memsets `hv_tensor_backend_ids` (262 KB) and `hv_tensor_copies` (2 backends x 8 B, 1.05 MB) on every segment: about 1.3 MB, 124 MB per token. At 15 to 30 GB/s that is 4 to 8 ms of the 12.8 ms alloc. Alternating segment topologies also make `ggml_gallocr_alloc_graph` fail and re-reserve, which synchronizes every backend first (`ggml_backend_sched_alloc_splits`).

### Where the 0.88 ms goes

The fit's intercept is mostly expert bytes, not dispatch:

- The slope, 0.063 ms per pair for a 12.24 MB expert (Q4_K gate and up, Q6_K down), implies 194 GB/s, 2.9x the box's measured 66.5 GB/s. `cpu_pairs` also counts about 110 pairs per token computed outside segments in overlap group graphs, and pairs per CPU segment hardly vary, so the intercept absorbs the bytes.
- The unthrottled `exit-trace` run of `20261112c` (no overlap, every segment a CPU segment, `throttled_share` 0.00): 127.4 ms over 94 segments, 1.36 ms each, with 6.6 pairs. 6.6 x 12.24 MB = 80 MB at 66.5 GB/s is 1.21 ms, leaving about 0.15 ms for the segment's GPU part and all dispatch.
- The fit's windows ran at `throttled_share` 0.5 to 0.68 (poll 50, before ADR 0061), so quota stalls land in the terms too.
- Upper bound: if the whole difference to a GPU-only segment were dispatch, the CPU-specific fixed part is (0.88 - 0.44) x 78 = 34 ms per token.

Bottom-up, the dispatch a CPU segment pays beyond llama.cpp's per-layer structure is the build and alloc above (0.12 ms per graph) and about 4 extra syncs and 6 extra small blocking copies, roughly 10 to 15 us each: 0.1 to 0.15 ms. Whatever of the 34 ms bound is real beyond that would be the threadpool: each CPU segment wakes the pool (cond broadcast when the workers sleep under poll 0 or after a pause, ADR 0061) and runs about 6 barriers across 30 threads. One graph per token does not change that count: llama.cpp also kicks the pool once per layer.

On the 7642 hosts the per-segment cost follows the CPU quota, not dispatch: 2.74 ms per CPU segment at `throttled_share` 0.05 (`20261112e` poll 0), 3.5 to 5.8 ms at 0.80 to 0.98, 4.9 and 8.5 ms at 0.30 and 0.60 (`20261112f`). Memory bandwidth there is 24 to 53 GB/s, so 80 MB per segment alone is 1.5 to 3.3 ms. The cache's lead of 1.35x to 1.54x tracks its CPU pair count: 628 / 449 = 1.40x more CPU expert bytes per token for us. That is a placement question (ADR 0059: `main-overlap-adapt` reached a 37.2% VRAM share but decoded slower, 2.47 and 2.65 against 4.77 and 4.33 tok/s, because of admission cost), not one this ADR answers.

So cheaper dispatch is worth about 7 to 28 ms of the 202 ms per token on `20261112c` (3% to 14%, steps 1 and 2 below), with 50 ms (16 ms build and alloc plus 34 ms) as the bound if the intercept were all dispatch. Not the main lever, but the cheapest part of it is nearly free.

## Decision

### Step 0: measure the split before changing it

- `StepProfile` gains `seg_pairs`: the RAM-tier pairs computed inside segment graphs (today's `cpu_pairs` minus the overlap groups' pairs). The exit JSON reports it per window, so the fit regresses on the pairs a segment actually computes.
- The offline micro-benchmark below runs before and after each step.

### Step 1: a scheduler sized for one segment (smallest change)

- When a streamer exists, `sched` and `sched_small` are created with `SEGMENT_NODES * 4` nodes (4,099 hash entries, 82 KB of reset memsets) instead of `step_nodes(h) * 4`. The whole-step graph of ADR 0036 only runs without a streamer, so it keeps the large size there. No numeric change.
- Expected: 4 to 8 ms of the 12.8 ms alloc per token on `20261112c` (2% to 4%). The micro-benchmark measures it on a CPU before any box run.

### Step 2: one graph per decode token, a host callback per layer

Scope: one-token decode steps with a streamer, `MissPolicy::Cpu` and no op offload (decode never offloads, ADR 0044). `LoadOptions::fused_decode: Option<bool>`, `None` meaning on with a GPU primary; a CPU primary keeps segments in production (below) and can force it for tests. Prefill, multi-token steps and `MissPolicy::Upload` keep segments.

Routing of layer L+1 depends on layer L's MoE output, so it cannot be computed before the graph runs; the ADR 0038 predictions are right 85% (depth 1) and 78% (depth 2) of the time on `20261112c`, so steering compute by them would break parity. The host must see each layer's ids mid-graph, as llama.cpp's callback does.

1. **Graph.** `build_decode_step` builds the whole step once per key `(n_kv, last_only, record)`, kept allocated like ADR 0036's `StepGraph`, so it is rebuilt every `KV_PAD` (256) tokens. It reuses `attention_and_router`, `expert_rows` and the state slots. Per layer L, with P = `n_used` fixed rows (row i is pair i, token-major as the resident `mul_mat_id` visits them):
   - attention, router and predictions as in `build_segment`, ending with the copies of `resid`, `xn`, `w`, `pred`, `pred2` and, last, `ids` into slot `L % 2`;
   - the copied group (ADR 0054's `add_copied`) at fixed P rows: `set_rows(acc, cpu_out, cpu_dst)`, where `cpu_out` is a host tensor written by the callback;
   - the compute-tier group at fixed P rows: `get_rows(xn, tok)`, `mul_mat_id(vram slots, ., gpu_ids)`, `set_rows(acc, ., gpu_dst)`;
   - weighting and sum of `acc` with the resident ops (ADR 0034), the residual.
   `acc` gets one dump row after its P rows. `gpu_dst` and `cpu_dst` send each pair to its own row in exactly one of the two scatters and every other row to the dump row, which nothing reads. Rows of non-GPU pairs in the GPU group are parked on a slot this layer's GPU pairs use (else slot 0); one id per row, so duplicates across rows are allowed. CUDA `mul_mat_id` does not skip negative ids (`mmvq.cu` indexes `ids[...]` directly), hence parking. The index tensors live in one I32 host tensor per slot set, marked as weights and copied whole (one H2D), with views taken on the device; `cpu_out` is a host weight too.
2. **Hook.** `ggml_backend_sched_set_eval_callback`, wrapped as `Sched::compute_observed(&mut self, graph, observed: &[Tensor], f: impl FnMut(usize) -> bool) -> Result<()>` in `ggml-sys` (trampoline with `catch_unwind`, the only new `unsafe`, Rule 7). The scheduler computes up to an observed node, synchronizes that backend and calls back; false stops the graph. Unlike the copy callback of the pin, it also fires on a CPU-only scheduler, where the fixtures and CI run. The observed node per layer is the `ids` copy. The inputs it writes are consumed in a later split: a host weight in a plain (not pinned) CPU buffer read by a GPU node starts a new split when the current one has inputs (`ggml_backend_sched_split_graph`), and every attention split after the first reads the previous layer's `cpu_out` and index tensor; layer 0's split gets one by taking the token ids as a graph input. On a GPU primary each layer is then one GPU split ending at its observed node, so the scheduler's sync after the range is the callback's only sync. A test asserts the split count. On a CPU primary the CPU nodes read host memory when they run, after the callback.
3. **Callback.** For layer L: read `ids`, `pred`, `pred2` (after the scheduler's sync), score predictions, then the body of today's per-layer loop, factored into one function both paths call: the gate and `finish_prefetch`, `plan_cpu`, `stage_ram` or `stage_overlap`, `admit`, `prefetch_ahead`, `set_compute_threads`. The CPU experts run here, never in the step graph: `run_overlap_layer` for a layer with reads in flight (ready experts, then arrival order, ADR 0054 unchanged), and the same group graph (`run_cpu_group` on `sched_cpu`, which the fused path always creates) for the others; more than one RAM chunk runs as more group graphs into the same rows. The callback copies `xn` to the host, writes the outputs into `cpu_out` and the indices, and returns. Predictions are read and prefetches started exactly where the segments do (after layer L's routing), so ADR 0038's timing holds.
4. **Parity.** CPU pairs go through the ADR 0054 group graph whose outputs the segments already scatter; GPU pairs through ADR 0035's gather form; both land in `acc` by `set_rows` and the weighting and sum are the resident ops, so the result is bitwise equal to the segments and to the resident graph by the ADR 0034 and 0035 arguments. Parked rows are computed and dropped. No `x + 0` merge (llama.cpp adds the two parts), so no sign-of-zero case.
5. **What stays.** One CPU kickoff per CPU layer, as today and as llama.cpp. On a CPU primary the step graph's ranges add a kickoff per layer to the group graphs, so segments stay the production path there.
6. **Metrics (Rule 6).** Counter `tetraeder.step.graph.builds` with `tetraeder.step.kind` (`resident`, `segment`, `fused`); `tetraeder.step.segments` records 1 plus the group graphs for a fused step. Overview panel "Graph builds per decode token". Alert `TetraederFusedDecodeRebuilding`: fused decode on and builds per decode token above 0.05 for 15 min (expected 1/256). The exit JSON adds `fused` and `graph_builds` per window.

Not in this ADR: backporting `ggml_backend_sched_prefetch_inputs` (#30966, MIT) or making CUDA `mul_mat_id` skip negative ids. Each is a ggml patch (Rule 8) with its own ADR, only if the box shows the parked rows or the serial CPU-then-GPU order cost more than the micro-benchmark leaves for them.

### Interfaces touched

- `crates/ggml-sys/src/ggml/sched.rs`: `Sched::compute_observed`.
- `crates/model/src/qwen3moe.rs`: scheduler sizing (step 1); `build_decode_step`, `DecodeGraph`, the per-layer host step factored out of `run`, `Slot.acc` with a dump row, host index and `cpu_out` tensors per slot set, `alloc_state` host buffers whenever fused, `StepProfile::seg_pairs`.
- `LoadOptions::fused_decode`; `crates/telemetry` (`step.graph.builds`); `deploy/grafana/dashboards/overview.json`; `deploy/alerts`; the exit JSON in `crates/model/tests/exit_regime.rs`.

### Expected savings (`20261112c`, 202 ms per token)

| Item | Today | Step 1 | Step 2 |
| --- | --- | --- | --- |
| Build and alloc | 16.0 ms | 8 to 12 ms | about 2 to 4 ms (group graphs only; the step graph once per 256 tokens) |
| Syncs and small blocking copies per CPU layer (readbacks included) | about 7 and 9 | same | about 3 and 6 (callback sync, `xn` out, three readbacks, index tensor and `cpu_out` in): 7 fewer x 10 to 15 us x 95, 7 to 10 ms less |
| Readback (`ids`, `pred`, `pred2`) | 3.0 ms | same | same, inside the callback |
| Parked GPU rows | 0 | 0 | 0 to 8 ms: up to 6.7 extra rows x 12.24 MB / 936 GB/s = 88 us per layer, less if duplicates hit L2 |
| Empty CPU work in GPU-only layers | 0 | 0 | 0 (no CPU node in the step graph on a GPU primary) |

Step 1: 4 to 8 ms (2% to 4%). Step 2 on top: build and alloc 4 to 10 ms, syncs 7 to 10 ms, parking 0 to -8 ms, so 3 to 20 ms (2% to 10%); up to 34 ms more only if the fit's intercept were dispatch, which the bandwidth arithmetic above contradicts. Neither closes the 1.35x to 1.54x gap on the 7642, which follows CPU pairs and the quota.

### Risks

- The new split after the observed node rests on the attention split having inputs. Guarded by the split-count test and a debug assertion that the host index tensor is an input of a split after the one holding the observed node.
- `sched_cpu` computes inside the outer scheduler's callback. On a GPU primary the step graph has no CPU node; on a CPU primary the CPU backend is idle between ranges. Tested on both.
- A streamer error or panic inside the callback stops the graph; the step returns the error and leaves the state as a failed segmented step does (`begin_step` drops a leftover prefetch).
- Parked rows can cost up to 8 ms per token on a 3090; measured by the GPU test before step 2 is turned on by default.
- One more code path for decode. The per-layer host step is shared, so only graph construction differs.

### Tests first (failing before the code)

- `ggml-sys`, `cpu`: `compute_observed` calls back once per observed node, in order, after the node is computed; a host tensor the closure writes is seen by a later node; false stops the graph with an error; a panic becomes an error.
- `ggml-sys`, `gpu`: a host weight read after the observed node in a later split carries the callback's write.
- `model`, `cpu`, synthetic MoE fixture (`common::synth`, `MissPolicy::Cpu`, `expert_slots` below and `ram_slots` below the expert count, 64 decode tokens, prefetch depth 1 and 2): fused logits bitwise equal to segments and to the resident graph; greedy tokens equal; prefetch issued and used counts and prediction scores equal.
- `model`, `cpu`, `overlap.rs`: with reads held back, fused decode computes CPU experts in arrival order and matches segments bitwise.
- `model`, `cpu`: one step-graph build over 255 decode tokens and two after crossing 256; `segments_run` per decode token 1 plus group graphs.
- `model`, `cpu`: an injected read fault mid-step returns an error; after `rewind` the next tokens match a clean run.
- Property (Rule 5 style, beside `overlap_groups_partition_all_pairs`): for random routing and tier membership, `gpu_dst` and `cpu_dst` put each pair in its own row exactly once and everything else in the dump row.
- `model`, `gpu` (`gpu_graphs.rs`): the fused graph has the expected split count; logits bitwise equal to segments on CUDA; a different parking slot gives the same logits.
- Step 1: the existing parity and step tests pass unchanged with the smaller scheduler.

### Offline micro-benchmark

`segment_overhead` in `crates/model/tests/cpu_bench.rs`, label `cpu`, `#[ignore]`, release, minutes on a CI runner or a laptop, JSON out (Rule 3):

1. **Dispatch floor.** A CPU-only `Sched` computes a 1-node graph and a 6-node chain shaped like a CPU group (`get_rows`, three `mul_mat_id`, `swiglu`, `cpy`) on tiny tensors, 2,000 times, with a main-thread spin gap of 0, 100 and 500 us between computes (the GPU part of a segment), for threads 1, n/2 and n, poll 0 and 50, pause on and off. Reports us per compute and per barrier: the kickoff and barrier cost that no graph count change removes.
2. **Allocation.** `Sched::alloc_graph` of a segment-sized graph on schedulers of 4,096 and 48,640 nodes, one topology repeated and two alternating. Reports us per alloc: the reset memsets (step 1) apart from gallocr re-reserves.
3. **End to end.** The `builder_cpupolicy` decode of the synthetic 48-layer Q4_K_M MoE already in `cpu_bench.rs` (32 experts, top 8, `expert_slots` n/4, all experts in RAM slots): per token graphs, build, alloc, segment compute, `seg_pairs`, and bytes per pair from the fixture, so compute splits into bandwidth and the rest. Variants: segments, step 1, step 2 forced on the CPU primary. Greedy tokens equal across variants.

It cannot see CUDA syncs, H2D and D2H copies, kernel launches or parked rows; those come from the GPU test and a paired box run (ADR 0021 pairing rules) after step 2.

## Consequences

- Step 1 is a sizing change with no numeric effect and can ship on the micro-benchmark alone.
- Step 2 replaces 95 segment graphs per decode token by one kept graph plus the CPU group graphs, removes the per-segment allocations and about 40% of the blocking device operations, and keeps the CPU kickoff count, ADR 0054's arrival order and ADR 0038's prediction timing.
- The llama.cpp gap on the 7642 is not a dispatch problem by these numbers: it follows CPU pairs per token (1.40x) and the quota (ADR 0061, 0062). Those stay the levers to pursue first.
