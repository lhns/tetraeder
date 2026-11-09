# Implementation briefs for ADR 0050 and 0054

Three self-contained briefs, one per lever, plus two follow-ups that run only when their gate passes. Every design decision is in ADR 0050 (Brief A: ADR 0054, which supersedes ADR 0050 (a) stage 1); read it first, then AGENTS.md. Do not change the prefetch levers of ADR 0038 (depth 2, reader chunking, `prefetch_ahead` placement). Box runs are on hold: no brief creates a box; box stages are written and left for the next session.

All briefs: failing tests first, then code. Every local cargo command goes through the shared lock with one job: `L=/c/Users/pierr/AppData/Local/Temp/claude/cargo-host.lock; until mkdir "$L" 2>/dev/null; do sleep 20; done; CARGO_BUILD_JOBS=1 cargo <args>; rmdir "$L"` (the local VM has no AVX and no GPU; AVX2 paths run on CI). Done means `cargo build --workspace`, `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`, `cargo fmt --check` pass (Rule 11; use CI on the `track-*` branch when the shared cargo lock starves local builds), `python -m pytest tests/test_alert_rules.py` passes, and the ADR gets an "Implementation notes" section with what was built and any deviation. Commit only if your brief says so.

## Order and parallelism

| Brief | Touches | Can run in parallel with |
|---|---|---|
| C1 recorder and replay | `memory` (new `replay/`, `slots.rs` untouched), `experts.rs` (one refactor), `qwen3moe.rs` run loop (about 15 lines near the routing log), `exit_regime.rs`, `bench` | B |
| B low-bit KV | `plan`, `planner`, `ggml-sys`, `qwen3moe.rs` (`alloc_kv`, `attention_and_router`, `LoadOptions`, `memory()`), `exit_regime.rs`, telemetry | C1 or A |
| A I/O service and arrival order (ADR 0054) | `memory` (new `io.rs`), `experts.rs` (prefetch engine, new methods), `qwen3moe.rs` run loop, `State`, `build_segment`; telemetry; `exit_run.sh` | B (C1 has merged) |
| A2 GPU group async | after A, only if A's profiling shows the VRAM group at 5 ms per token or more | |
| C2 RAM tier policy | after C1 and a box trace, only if the replay passes the threshold | |

So: C1 and B in two worktrees now; A after C1 merges (or in parallel with a planned rebase onto C1). B and A conflict only textually (different functions of `qwen3moe.rs`, the env var list of `exit_regime.rs`, `telemetry/src/metrics.rs`, `deploy/alerts/engine.yml`, `deploy/grafana/dashboards/overview.json`): merge, do not redesign. Three workers on one machine share the cargo lock; prefer CI for compile loops.

---

## Brief A: I/O service and experts computed in arrival order (ADR 0054; replaces ADR 0050 (a) stage 1)

**Goal.** (1) A persistent I/O service in `memory` that reads expert chunks on 16 workers, each copying its own chunk, earliest deadline first, with completion per expert; prefetch goes through it. (2) In decode, a layer stops waiting for its prefetch before it computes: the resident experts compute first, the in-flight and missed ones compute as they land. (3) Box stages for the paired run, including a smaller RAM reserve. Logits bitwise unchanged. Build part 1 first and get its tests green before part 2.

**Read first.** ADR 0054 (all of it), ADR 0035 (groups, exactness, implementation notes), ADR 0038 (reader, pins, ordering, "Levers" and "Reads that hide"); `crates/memory/src/reader.rs` (the `Reader` this replaces, `HostDst`, `copy_sink`, `tensor_dst`); `crates/memory/src/dio/mod.rs` (`Options`, `DirectFile::read_batch_with` at 338, `Engine` at 145); `crates/model/src/experts.rs` (`Streamer` fields 188-222, `Streamer::new` 241, `start_prefetch` 540, `finish_prefetch` 659, `settle` 705, `stage_ram` 713, `fill_ram` 739, `read_experts` 972, the `READER_*` constants 181-186); `crates/model/src/qwen3moe.rs` (`StepProfile` 60, `run` 828, the layer loop 1008-1089, `group_pairs` 1207, `run_group_graph` 1292, `add_group` 1342, `prefetch_ahead` 1455, `set_compute_threads` 1473, `alloc_state` 1521, `build_segment` 1555); `crates/model/tests/prefetch.rs`, `replay_fidelity.rs`, `common/synth.rs`.

**Decisions already made (do not revisit).**

Part 1, `memory::io::IoService` (new file `crates/memory/src/io.rs`, exported from `lib.rs`; behind the `ggml` feature only where it needs `tensor_dst`):
- `IoService::new(paths: &[PathBuf], opts: ServiceOptions) -> io::Result<Self>`; `ServiceOptions { workers: usize (0: 16), chunk_bytes: usize (0: 1 MiB), buffered: bool }`. (The test-only `delay` of the first version is gone: ADR 0056 puts a virtual-clock `SimIo` behind the `Io` trait and `LoadOptions::io_factory` in its place.)
- Each worker thread owns one `DirectFile` per shard, opened with `dio::Options { queue_depth: 1, bounce_bytes: chunk_bytes + 2 * ALIGN, chunk_bytes, buffered }` (inline read, no thread spawn; one handle per thread on Windows). It reads one chunk with `read_batch_with(&[range], ..)` and copies it into the chunk's `HostDst` in the sink.
- `HostDst::sub(&self, offset, len) -> io::Result<HostDst>` (new, in `reader.rs` or moved to `io.rs` with `HostDst`): a bounds-checked sub-range, so a slab's destination splits into chunk destinations. Keep the safety contract of `HostDst::new` and write the safety comment.
- `submit(&self, deadline: Deadline, parts: Vec<Part>) -> UnitId`, where `Part { shard: usize, range: Range, dst: HostDst }` (the three slabs of one expert) and `Deadline { order: u64, urgent: bool }`. The service splits every part into chunks of at most `chunk_bytes` and queues them. Workers take chunks from a `BinaryHeap` of `Reverse` keys ordered by `(order, !urgent, submit sequence, chunk index)`, smallest first (one `Mutex` plus `Condvar`).
- `wait_any(&self, units: &[UnitId]) -> Vec<(UnitId, io::Result<UnitDone>)>`: blocks until at least one of `units` has completed, returns every completed one among them and forgets them. `wait_all(&self, units) -> Vec<..>` blocks for all. `poll(&self, units) -> Vec<..>` never blocks. `UnitDone { requested: u64, device: u64, took: Duration }` (`took`: submit to last chunk). A unit fails with the first chunk error; its other chunks still run or are dropped from the queue, but the unit only completes when no chunk of it is running.
- `inflight(&self) -> usize` (chunks queued or running), and `chunks_in_flight` sampled at each `submit` into `tetraeder.io.inflight` (see Metrics).
- `Drop`: closes the queue, workers finish every queued chunk, joins them (destinations stay valid until then, as with `Reader`).
- `bounce_bytes(&self) -> u64` for `memory()` (`io` kind).

Part 1, streamer and options:
- `IoOptions` (the `tetraeder_memory::IoOptions` used by `LoadOptions::io`) gains `service: bool` (default `false`) and `workers: usize` (0: 16). `chunk_bytes` keeps its meaning; when the service is on and it is 0, use 1 MiB.
- In `Streamer`, the prefetch engine becomes a private enum `Prefetcher { Reader { reader, reader_dio, inflight: VecDeque<InFlight> }, Service { io: IoService, inflight: Vec<InFlightUnit> } }`, with `InFlightUnit { il: usize, expert: u32, unit: UnitId, prefetch: bool, routed: bool }`. Declared before `ram` (drop order). `start_prefetch(il)` first waits for leftover units of `il` from earlier steps (mispredicted ones kept by part 2; long landed, so the wait is free), so `prefetch_plan` sees the same in-flight set as today; then it builds the same `missing` list and pins (`prefetch_plan`, unchanged) and submits one unit per expert with `Deadline { order: step * n_layer + il, urgent: false }`; the streamer keeps a `step` counter, incremented at the start of `run` (add `Streamer::begin_step`). `finish_prefetch(il)` on the service waits for all units of `il` (`wait_all`) and does the same accounting as today: `prefetch_read_time` from `took`, `prefetch_wait_time` from the time blocked, `batches += 1` per `finish_prefetch` call that had units, `bytes_read`, metrics, unpin, clear the layer on error. `settle` waits for everything. With `service = false` behavior is today's, byte for byte.

Part 2, arrival order (decode):
- `LoadOptions::overlap: bool` (default `false`; load error if set without `io.service`). Gate per layer step: `overlap && policy == Cpu && ram tier && !offloads(..) && plan.cpu.len() == 1`. Otherwise today's code path.
- Overlap path at layer `seg`, replacing lines 1041-1088 for that case: readback as today; do **not** call `finish_prefetch(seg)` first; `plan_cpu`; `prefetch_ahead` (unchanged, same place); `let st = streamer.stage_overlap(&file, seg, &plan.cpu[0])?`.
- `Streamer::stage_overlap(file, il, experts) -> Result<StagedLayer>`, `StagedLayer { slots: Vec<u32> (per expert, input order), ready: Vec<u32>, pending: Vec<u32> }`: first reap the landed units of every layer with `poll` (accounting as `finish_prefetch`), then `set_hot_set(experts of il still in flight)`, which drops the resident predicted pins `start_prefetch` left (today `finish_prefetch` clears them before the lookups; keeping them would change LRU victims); then the RAM lookups exactly as `fill_ram` makes them (same order: resident first, an expert with a unit in flight for `il` counts as resident because it has its slot; the same counters, `prefetched` flags, `note_wasted`, `metrics.lookup`); misses are pinned as they get their slot and submitted as units with `Deadline { order: step * n_layer + il, urgent: true }`, counted in `batches`, `experts_read`, `bytes_read` when they land (as `read_experts` counts them, `read_time` from `took`). In-flight units of `il` whose expert is routed are marked `routed`. `ready` = resident experts with no unit in flight plus those whose unit `poll` reports landed; `pending` = the rest. On any error: wait for the layer's units, clear the layer (as `stage_ram`), return it.
- `Streamer::wait_landed(il, pending: &[u32]) -> Result<Vec<u32>>`: `wait_any` on their units, accounting as above, plus `urgent_wait_time` (new `IoCounters` field) for the time blocked; returns the experts that landed.
- `Streamer::finish_overlap(il)`: unpins the routed experts of `il`; mispredicted in-flight units stay pinned and in `inflight`, and are reaped without blocking (`poll`) at every later `stage_overlap` or `finish_prefetch` of any layer, and waited for by the next `finish_prefetch(il)` (still called at the top of the non-overlap path) or `settle`.
- If `st.pending` is empty: today's path with `st.slots` (the CPU group goes into segment L+1 as a `PendingGroup { src: Ram }`).
- Else: read back `xn` of the slot into `xn_host` (`set_bytes` from `slot.xn.get_f32()` is fine); if `ready` is not empty, run one group graph over `ready` on `sched_cpu`; then `while !pending.is_empty() { let landed = streamer.wait_landed(seg, &pending)?; run one group graph over landed; remove them }`; then one `cpu_out.set_bytes(&cpu_out_host rows 0..p)`; `streamer.finish_overlap(seg)`; push `PendingGroup { src: ExpertSrc::Copied, pairs }` with `pairs` the concatenation of the groups' pairs in the order their rows were written; the compute group stays a pending group as today.
- Groups per expert subset with `group_pairs(&chosen, nu, subset, subset_slots)`, never by splitting an expert's pairs.
- State: `xn_host` (`F32 [n_embd, n_tokens]`) and `cpu_out_host` (`F32 [n_embd, n_used * n_tokens]`) in one context allocated on the CPU backend; `cpu_out` (`F32 [n_embd, n_used * n_tokens]`) on the primary, per `Slot` parity like `acc`. `alloc_state` takes the CPU backend. `sched_cpu: Sched::new(&[cpu.clone()], SEGMENT_NODES * 4, false)`, created at load when `overlap` is on.
- The CPU group graph: `get_rows(xn_host, tokens)`, the three `mul_mat_id` over `streamer.ram_exps(seg)` exactly as `add_group` builds them, the output reshaped to `[n_embd, p]` and `cpy` into a view of `cpu_out_host` at row offset `r0`. Index tensors written after `alloc_graph`, as `GroupInputs::write` does. Times go to `StepProfile::cpu_ready` (first graph) and `cpu_landed` (the others).
- In `build_segment`, a `Copied` group adds `acc = set_rows(acc, view(cpu_out, [n_embd, p]), pairs_index)` before the other groups; the pairs index is an input written after allocation.
- `set_compute_threads` is called after `stage_overlap` as after `prefetch_ahead` today (the service counts as a prefetch in flight while it has chunks).

**Tests to write first.**
1. `crates/memory/src/io.rs` unit tests (buffered and direct temp files, as `reader.rs` tests): a unit of three ranges lands byte-exact across chunk boundaries (chunk 4 KiB, unaligned offsets); `wait_any` returns landed units only and never blocks when one has landed; deadline order: with one worker and `delay`, units submitted as (order 5, prefetch), (order 3, prefetch), (order 5, urgent) complete in the order 3, 5 urgent, 5 prefetch; a read past the end fails its unit only and the service keeps serving; `Drop` waits for queued units; `HostDst::sub` rejects out-of-range.
2. `crates/model/tests/overlap.rs` (label `cpu`, synthetic Q4_K_M MoE from `tests/common/synth.rs`, small VRAM and RAM tiers so most decode steps have hits and misses, `delay` of 2 ms so units are in flight at plan time):
   - `service_decode_is_bitwise_equal_to_reader`: prefill 64 then 16 greedy decode steps; logits bitwise equal with `service` off and on, prefetch depth 1, 2 and off.
   - `overlap_decode_is_bitwise_equal_to_serial`: the same with `overlap` on against everything off.
   - `overlap_keeps_cache_behaviour`: per layer, RAM hits, misses, `prefetch_issued`, `_used`, `_wasted` and `experts_read` equal with overlap on and off (so the replay of ADR 0050 (c) stays valid).
   - `overlap_runs_pending_groups`: with `delay`, `StepProfile::read_wait > 0` and `cpu_landed > 0` in decode; without `delay` the counts still match the serial run.
   - `overlap_read_error_fails_the_step_and_recovers`: cut the model file as `tests/prefetch.rs` does; the step errors, the next step after restoring the file gives logits equal to a fresh load.
3. Unit proptest in `qwen3moe.rs` (Rule 5, every unit computed exactly once): for random routings, resident sets and random splits of the CPU experts into ready and arrival groups, the pairs of the compute group and the CPU groups partition all pairs, each exactly once, each expert's pairs all in one group.
4. `crates/model/tests/replay_fidelity.rs`: one more case with `service` and `overlap` on; the replay still equals `ExpertStats` exactly.
5. GPU (label `gpu`, skips without CUDA, add to the box `gpu-tests` list, do not run): `gpu_overlap_logits_equal_serial` on the synthetic model with `BackendChoice::Gpu`.

**Metrics, panel, alert (Rule 6, same PR).**
- `crates/telemetry/src/metrics.rs`: keys `tetraeder.io.class` and `tetraeder.part`; histograms `tetraeder.io.inflight` (chunks, by class `urgent`/`prefetch`), `tetraeder.moe.read.hidden` (ratio, one sample per decode layer step with pending experts: `1 - read_wait / read_span`, span from `stage_overlap` to the last landing), `tetraeder.moe.layer.duration` (s, by part `cpu_ready`, `read_wait`, `cpu_landed`, `copy`). Thin wrappers in `crates/memory/src/metrics.rs` like `prediction_accuracy`.
- `StepProfile`: `cpu_ready`, `read_wait`, `cpu_landed`, `copy`, counted inside `streamer` so `total` still adds up; document on the struct.
- `deploy/grafana/dashboards/overview.json`: panel "Expert reads under compute" in the "Experts and cache" row (p50 of chunks in flight by class, p50 hidden share, `rate(tetraeder_tier_read_bytes_total{tier="nvme"})` over the decode token rate); unique id.
- `deploy/alerts/engine.yml`: `TetraederExpertReadsNotHidden` as in ADR 0054, with a comment line like the existing rules; `python -m pytest tests/test_alert_rules.py` must pass.
- Harness `crates/model/tests/exit_regime.rs`: env `TETRAEDER_EXIT_IO_SERVICE`, `TETRAEDER_EXIT_OVERLAP` (`0`/`1`, unset follows `LoadOptions`), `TETRAEDER_EXIT_IO_WORKERS`, recorded in JSON `options`; the profile fields in `decode_profile_per_token`; a `decode_overlap` block (pending experts per token, `read_wait` ms per token, hidden share mean, `urgent_wait_time`); document in the file header.

**Box stages (write, do not run).** In `scripts/exit_run.sh`, right after `main-prefetch` (line 260), paired with it:
```
exit_regime main-iosvc 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_IO_SERVICE=1 || rc=30
exit_regime main-overlap 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_IO_SERVICE=1 TETRAEDER_EXIT_OVERLAP=1 || rc=30
exit_regime main-overlap-d2 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_PREFETCH_DEPTH=2 TETRAEDER_EXIT_IO_SERVICE=1 TETRAEDER_EXIT_OVERLAP=1 || rc=30
exit_regime main-overlap-ram 60 "${deep[@]}" TETRAEDER_EXIT_PREFETCH=1 TETRAEDER_EXIT_IO_SERVICE=1 TETRAEDER_EXIT_OVERLAP=1 TETRAEDER_EXIT_RAM_SLOT_BYTES=$((mem - 4000000000)) || rc=30
```
In the `trace` stage's replay line (line 268) change `--ram 48,64,77,96,110` to `--ram 48,64,77,79,81,83,96,110`, so the replay prices the smaller reserve (ADR 0054 section 3). And after `main-repeat` (line 297) add `main-overlap-repeat` (same args as `main-overlap`), so the pair interleaves. Add the five names to `STAGES` in `scripts/exit_session.sh` (line 27) after `main-prefetch` and `main-repeat`, and a comment line in both scripts naming ADR 0054. Box acceptance (for the report after the run, not for this brief): greedy tokens equal to `main-prefetch`; `main-overlap` decode at least 15% above `main-prefetch` in both pairs with prefill within 3%; hidden share at least 0.5; `main-overlap-ram` peak RSS under 90 GB. If it holds, `service` and `overlap` default on and the `Reader` path is deleted in a follow-up.

**Done.** Tests 1 to 4 pass locally and in CI on Windows and Linux; test 5 compiles and skips; the Rule 11 set (`cargo build --workspace`, `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`, `cargo fmt --check`) and `python -m pytest tests/test_alert_rules.py` pass; ADR 0054 gets "Implementation notes" (what was built, every deviation). Report: files changed, test names, deviations, and the local synthetic exit regime (`cargo test --release -p tetraeder-model --test exit_regime -- --ignored`) with service off/on and overlap off/on: decode ms per token, `read_wait`, hidden share.

**Out of scope.** Native AIO or io_uring for the service; a memory-mapped tier (ADR 0054 option B); any change to the prefetch selection (`prefetch_plan`), the hot set or RAM-tier policy; prefill pipelining; op offload paths; ADR 0050 (a) stage 2 (A2).

---

## Brief B: low-bit KV per attention unit (ADR 0050 (b))

**Goal.** The plan and the builder carry a KV format (K and V separately); the builder runs Q8_0/Q8_0, Q8_0/Q4_0 and Q4_0/Q4_0 KV with libllama parity; the planner's footprint uses the format.

**Read first.** ADR 0050 (b); ADR 0027, 0031, 0037, 0047 (role formats pattern), 0049 (prefix cache); `crates/plan/src/{plan.rs,model.rs,format.rs,validate.rs}`; `crates/planner/src/{candidate.rs,layout.rs,builder.rs}`; `crates/model/src/qwen3moe.rs` (`LoadOptions`, `alloc_kv`, `attention_and_router`, `memory`, `kv_snapshot`); `crates/ggml-sys/build.rs`; `crates/ggml-sys/src/llama/context.rs`; `crates/model/tests/parity.rs`.

**Decisions already made.**
- `tetraeder_plan::KvFormat { k: Format, v: Format }` (serde, `Default` F16/F16), `KvFormat::ALLOWED = [F16/F16, BF16/BF16, Q8_0/Q8_0, Q8_0/Q4_0, Q4_0/Q4_0]`, `KvFormat::bytes(n_embd_kv) -> u64`, `Display` as `q8_0/q4_0` (lower case, the metric label).
- `Assign.kv_format: Option<KvFormat>` (`skip_serializing_if = "Option::is_none"`); absent means F16/F16 for the units it selects. `Plan::kv_formats(desc, replica) -> BTreeMap<u32, KvFormat>` (layer to format, first matching rule with attention units in the layer's stage, like `unit_formats`). `Plan::kv_tokens(desc, replica, stage, device) -> u64`: the stage's KV bytes on the device divided by the per-token bytes of its layers at their formats.
- `ModelDesc::kv_bytes_per_token_per_layer_as(KvFormat) -> u64`, with `n_embd_kv = kv_bytes_per_token_per_layer / 4` (the field stays F16 bytes of K plus V; document it on the field).
- Validator: `Issue::KvFormat { at, rule, detail }` when the pair is not in `ALLOWED`, or the rule's `units` cannot select attention units.
- Planner: `footprint` and `build_plan` take a `KvFormat` (callers pass `KvFormat::default()`); `ModelLayout::kv_bytes_per_token_per_layer_as(KvFormat)`. No new candidates enumerated.
- Builder: `LoadOptions::kv: KvFormat` (default F16/F16). `alloc_kv` allocates K and V by type (`Type` from `Format`; add the conversion if missing). In `attention_and_router` the K and V views become `ctx.view(kc, &[hd, n_kv, nkv], &[row_bytes, row_size(type, hd)], 0)` with `row_bytes = kc.nb()[1]` (today the second stride is `hd * 2`); reject at load a format whose block does not divide `head_dim`. `memory()` reports KV capacity from the real buffer (already) and the format.
- Prefix cache: wherever ADR 0049's entries are keyed (`crates/scheduler/src/prefix_cache.rs`), add the KV format to the key so snapshots of different formats never mix.
- `ggml-sys/build.rs`: `.define("GGML_CUDA_FA_QUANTS", "q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16;q8_0-q4_0")` when CUDA is on. `ContextParams` gains `type_k`/`type_v` (`Type`, default F16) mapped to `llama_context_params.type_k`/`type_v`.
- Harness: `exit_regime.rs` env `TETRAEDER_EXIT_KV` (`f16`, `q8_0`, `q8_0/q4_0`, `q4_0`), recorded in JSON `options`, applied to both engines (builder `LoadOptions::kv`, libllama `type_k`/`type_v`).

**Tests to write first.**
1. `crates/plan`: `KvFormat` serde roundtrip and `Display`; `kv_bytes_per_token_per_layer_as` for the 235B shape (`n_embd_kv = 512`): F16/F16 2048, Q8_0/Q8_0 1088, Q8_0/Q4_0 832; validator rejects `Q4_K/Q4_K` and a `kv_format` on an expert-only rule; old plan files without the field parse and validate unchanged (`crates/plan/tests/validate.rs`).
2. `crates/plan/tests/props.rs` (Rule 5, "memory never exceeded"): for generated plans with random allowed KV formats, `kv_tokens * per-token bytes <= KV budget` per stage and device, and tokens at Q8_0/Q8_0 at least 1.88x tokens at F16.
3. `crates/planner`: footprint with Q8_0/Q8_0 KV has `kv_gpu` 0.53125x of F16 and fits at least as many VRAM slots under `feasible_vram_tiers`.
4. `crates/model/tests/parity.rs` (label `cpu`): builder with KV Q8_0/Q8_0 and Q8_0/Q4_0 against libllama with the same `type_k`/`type_v`, same tolerance and greedy check as the F16 case, resident and streamed (`MissPolicy::Cpu`) configurations.
5. `crates/model/tests` (label `cpu`): `kv_snapshot`/`kv_restore` roundtrip with Q8_0 KV gives identical next-step logits; `memory()` KV capacity equals `n_layer * n_ctx * kv_bytes_per_token_per_layer_as(fmt)`.
6. `crates/model/tests/kv_quality.rs` (label `gpu`, ignored, box only, do not run): the KL acceptance of ADR 0050 (b) on Qwen3-30B-A3B Q4_K_M, writing JSON to `TETRAEDER_KV_QUALITY_OUT`; asserts the stated bounds.

**Metrics, panel, alert.** `tetraeder.format` attribute on the `kind = kv` memory gauges; gauges `tetraeder.kv.tokens` and `tetraeder.kv.capacity` set in `finish_step`; panel "KV cache" (bytes by format, tokens used of capacity); alert `TetraederKvNearFull` in `deploy/alerts/engine.yml`; `tests/test_alert_rules.py` passes.

**Box stages (write, do not run).** In `scripts/exit_session.sh`: `kv-q8` and `kv-q8q4` (decode and prefill, same VRAM slot bytes as `main`, then with `VRAM_SLOT_BYTES` raised by the freed KV bytes), paired with `main` and interleaved; `kv-quality` running test 6.

**Done.** Tests 1 to 5 pass locally and in CI on both OSes; test 6 compiles and skips; ADR 0050 implementation notes. Report: files, tests, measured local KL on the synthetic model at 2K (informative only), deviations.

**Out of scope.** Enumerating KV formats in the planner search; KV partly in RAM; per-layer KV formats in the builder (refuse them at load with a clear error).

---

## Brief C1: routing trace recorder and replay scorer (ADR 0050 (c), first half)

**Goal.** Record the builder's routing with its predictions to a file, and score RAM-tier policies offline with the engine's own slot cache and prefetch selection, so the box trace decides whether C2 is built.

**Read first.** ADR 0050 (c); ADR 0035 (hot set), 0038 (prefetch order, pins, counters); `crates/memory/src/slots.rs`; `crates/model/src/experts.rs` (`start_prefetch`, `plan_cpu`, `fill_ram`, `refresh_hot_set`, `RoutingStep`); `crates/model/src/qwen3moe.rs` run loop lines 974-1050 and `finish_step` (hot set after the first multi-token step); `crates/model/tests/exit_regime.rs`; `crates/bench/src/main.rs`.

**Decisions already made.**
- `RoutingStep` gains `n_tokens: usize`, `predicted: Option<Vec<i32>>`, `predicted2: Option<Vec<i32>>`, filled in the run loop from the readbacks that already happen (`slot.pred`, `slot.pred2`) when routing is recorded. The resident whole-step path sets them to `None`.
- A hot-set refresh is visible to the recorder: `Qwen3Moe` pushes a marker (a `RoutingStep`-adjacent enum or a field `refresh: bool`, your choice, documented) when `refresh_hot_set` runs.
- Trace format `tetraeder-routing` v1 exactly as ADR 0050 (c), in `crates/memory/src/replay/trace.rs`: `TraceHeader` (serde JSON line), `TraceWriter::{create, layer_step, hot_refresh, finish}`, `TraceReader` (iterator of `Record`). `memory` has no serde today: add `serde` and `serde_json` as dependencies of `tetraeder-memory` (both already in the workspace lock), or write the header line by hand and parse it with a minimal parser; prefer serde.
- Prefetch selection becomes a pure function in `memory` (`replay` or `slots` module): `prefetch_plan(predicted: &[i32], n_expert: u32, in_compute: &LayerSlots, ram: &LayerSlots, inflight: &[u32], ram_slots: usize) -> (pins: Vec<u32>, missing: Vec<u32>)`, reproducing `start_prefetch` lines 518-545 exactly. `Streamer::start_prefetch` calls it; behavior unchanged (existing `tests/prefetch.rs` must stay green).
- Simulator `crates/memory/src/replay/sim.rs`: per tier a `SlotCache`; per layer step the engine's sequence: complete the layer's prefetch (unpin), VRAM lookups for routed experts (hits only, as `plan_cpu`), prefetch top-up of `il + 1` from `predicted` and of `il + 2` from the depth-2 prediction recorded two layers back, then RAM lookups of the CPU experts hits first (as `fill_ram`). Prefetched reads land before the target layer is planned (the engine waits); count their bytes as prefetch bytes. The VRAM hot set comes from the routing counts at each hot-refresh marker (`hot_from_counts`), pinned with `set_hot_set` on the VRAM cache. The refresh also touches the RAM tier, as the engine does (`refresh_hot_set` -> `make_resident` -> `load_via_ram` -> `fill_ram`, `experts.rs:823-833, 889`): for each layer, replay the RAM lookups of the new hot set's missing experts in hits-first chunks of the RAM slot count, counting their lookups and reads.
- Policies (RAM tier): `Lru`; `Freq { share }` with frequency pins = top `ceil(share * slots)` experts by routing count so far (causal), refreshed every 16 decode steps and at each marker, applied by unioning them into the pins the simulator passes to `set_hot_set` (C1 does not change `LayerSlots`; the simulator owns the union); `Belady` (no prefetch; evict the resident expert whose next use is farthest, computed from the trace). Pair counts per expert per byte use the per-layer expert bytes from the header (add `expert_bytes: Vec<u64>` per layer to the header).
- Output JSON per (VRAM slots, RAM slots, policy, prefetch on/off): decode and prefill separately: RAM hit rate, disk bytes per token, synchronous miss bytes per token, prefetch bytes and wasted bytes per token, VRAM share of routed pairs.
- CLI: `tetraeder-bench replay --trace F --vram 10,12,13 --ram 48,64,77,96,110 [--policies lru,freq:0.25,freq:0.5,freq:0.75,freq:1,belady] --out F`. `tetraeder-bench` depends on `tetraeder-memory` with `default-features = false`.
- Harness: `exit_regime.rs` env `TETRAEDER_EXIT_ROUTING_OUT=<path>` sets `record_routing` and writes the trace, draining `take_routing_log` after every `eval`; header filled from the run's options and `ModelFile`.
- Box stage (write, do not run): `scripts/exit_session.sh` stage `trace`: the `main` configuration with 256 decode tokens and `TETRAEDER_EXIT_ROUTING_OUT`, then `tetraeder-bench replay` on the box, result JSON fetched to `bench/results/<date>/replay-235b.json`.

**Tests to write first.**
1. `crates/memory/src/replay/trace.rs` unit tests: write then read roundtrip (u8 and u16 ids, with and without predictions, a marker record); a truncated file is an error, not a panic.
2. `crates/memory/tests/replay.rs`:
   - `belady_never_loses_to_lru` (proptest, Rule 5 style): for random traces without prefetch, Belady hits >= LRU hits at every slot count.
   - `frequency_beats_lru_on_skewed_and_ties_on_uniform`: a synthetic Zipf(1.1) trace with periodic scans: `freq:0.5` hit rate exceeds `lru` by at least 2 points; a uniform trace: within 1 point.
   - `pins_never_exceed_slots`: for random traces and shares, pinned plus transient pins never exceed the active slots and no pinned expert is evicted.
   - `prefetch_plan_matches_streamer`: the pure function gives the same `missing` list as the current `start_prefetch` on fixed inputs (copy the inputs from a `tests/prefetch.rs` scenario).
3. Fidelity (label `cpu`, `crates/model/tests/replay_fidelity.rs`): run the synthetic exit regime (`MissPolicy::Cpu`, small RAM tier, prefetch depth 1 and 2) with a trace, replay `lru` with the same configuration: per-layer RAM hits and misses equal the engine's `ExpertStats` exactly (the synthetic run is deterministic). This is the local form of the 0.5-point gate.

**Metrics.** The recorder and replay are bench tooling: results JSON (Rule 3), no runtime metric. C2 ships the runtime metrics.

**Done.** Tests pass locally and in CI on both OSes; `tests/prefetch.rs` unchanged and green; the replay on the local synthetic trace runs end to end and its JSON is committed under `bench/results/<date>-local/replay-synthetic.json`; ADR 0050 implementation notes. Report: files, tests, the synthetic replay table, deviations.

**Out of scope.** Any change to `LayerSlots` or the engine's tier policy (C2); changing prefetch behavior.

---

## Follow-up A2: GPU expert group asynchronous (only if gated in)

Gate: the box run of Brief A's path with the profiling switch (`TETRAEDER_EXIT_TIME_GPU_GROUP=1`) shows the VRAM group at 5 ms per decode token or more. Then: `Sched::compute_async` and `Sched::synchronize` in `crates/ggml-sys/src/ggml/sched.rs` (`ggml_backend_sched_graph_compute_async`, `ggml_backend_sched_synchronize`, safety comments); a third scheduler for the VRAM group graph; launch it before the hit group; assert one split, else fall back to stage 1 for the layer and count `tetraeder.moe.layer.duration{part="gpu_group"}` fallbacks; synchronize before the next segment allocates. Same tests as A with the async path on; bitwise equality holds (same kernels, same inputs).

## Follow-up C2: RAM tier by routing frequency (only if gated in)

Gate: ADR 0050 (c) threshold on the box trace. Then: `LayerSlots::set_frequency_pins(&[u32])` with a second pin set, eviction skipping the union, `pinned_count` the union (property tests in `crates/memory/tests/slot_props.rs`: union never exceeds active slots, pinned experts never evicted, `set_hot_set` keeps frequency pins); `LoadOptions::ram_pin_share: f32` (default 0); `Streamer` refreshes frequency pins from `counts` where it refreshes the VRAM hot set and every 16 decode steps; prefetch budget `slots - union` through `prefetch_plan`. Metrics `tetraeder.expert.pinned` (tier, source), panel "RAM tier: hit rate and pinned slots", alert `TetraederRamTierHitDrop`. The replay with the chosen share must predict the box run's RAM hit rate within 0.5 points.
