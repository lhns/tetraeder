# Implementation briefs for ADR 0050

Three self-contained briefs, one per lever, plus two follow-ups that run only when their gate passes. Every design decision is in ADR 0050; read it first, then AGENTS.md. Do not change the prefetch levers of ADR 0038 (depth 2, reader chunking, `prefetch_ahead` placement). Box runs are on hold: no brief creates a box; box stages are written and left for the next session.

All briefs: failing tests first, then code. Done means `cargo build --workspace`, `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`, `cargo fmt --check` pass (Rule 11; use CI on the `track-*` branch when the shared cargo lock starves local builds), `python -m pytest tests/test_alert_rules.py` passes, and the ADR gets an "Implementation notes" section with what was built and any deviation. Commit only if your brief says so.

## Order and parallelism

| Brief | Touches | Can run in parallel with |
|---|---|---|
| C1 recorder and replay | `memory` (new `replay/`, `slots.rs` untouched), `experts.rs` (one refactor), `qwen3moe.rs` run loop (about 15 lines near the routing log), `exit_regime.rs`, `bench` | B |
| B low-bit KV | `plan`, `planner`, `ggml-sys`, `qwen3moe.rs` (`alloc_kv`, `attention_and_router`, `LoadOptions`, `memory()`), `exit_regime.rs`, telemetry | C1 or A |
| A overlap stage 1 | `qwen3moe.rs` run loop, `State`, `build_segment`; `experts.rs` (new methods); telemetry | B; after C1 merges (both edit the run loop around `qwen3moe.rs:974-1050` and `experts.rs`) |
| A2 GPU group async | after A, only if A's profiling shows the VRAM group at 5 ms per token or more | |
| C2 RAM tier policy | after C1 and a box trace, only if the replay passes the threshold | |

So: C1 and B in two worktrees now; A after C1 merges (or in parallel with a planned rebase onto C1). B and A conflict only textually (different functions of `qwen3moe.rs`, the env var list of `exit_regime.rs`, `telemetry/src/metrics.rs`, `deploy/alerts/engine.yml`, `deploy/grafana/dashboards/overview.json`): merge, do not redesign. Three workers on one machine share the cargo lock; prefer CI for compile loops.

---

## Brief A: miss reads under the CPU hit group (ADR 0050 (a), stage 1)

**Goal.** For decode steps under `MissPolicy::Cpu`, read the layer's RAM misses on a second reader thread while the CPU computes the layer's RAM-hit experts, then compute the misses. Logits bitwise unchanged.

**Read first.** ADR 0050 (a); ADR 0035 (groups, exactness); ADR 0038 (reader, pins, ordering); `crates/model/src/qwen3moe.rs` `run` (lines 911-1089), `build_segment`, `add_group`, `run_group_graph`, `alloc_state`; `crates/model/src/experts.rs` `start_prefetch`, `finish_prefetch`, `settle`, `fill_ram`, `stage_ram`; `crates/memory/src/reader.rs`.

**Decisions already made (do not revisit).**
- Gate: `LoadOptions::overlap: bool`, default `false`. Overlap applies when `overlap && policy == Cpu && ram tier && !offloads(...)` (so not with op offload at large steps) and `plan.cpu.len() == 1`. Otherwise the existing path runs unchanged.
- A second `Reader` in `Streamer` (`urgent`), created when the RAM tier exists and `overlap` is on, reading through the main thread's `dio` files (`Arc<DirectFile>` clones). Declared before the RAM tier like `reader`, so it drops first.
- Order at layer `seg` (overlap path): readback of ids as today, plus `xn` into `xn_host`; `finish_prefetch(seg)`; `plan_cpu`; `prefetch_ahead` (unchanged, same place); split `plan.cpu[0]` into RAM hits and misses (`ram.cache.layer(seg).slot_of`); `streamer.read_misses_async(file, seg, &misses)`; build and compute the hit group on `sched_cpu`; `streamer.wait_misses(seg)`; build and compute the miss group on `sched_cpu`; one `set_bytes` of `cpu_out_host` rows `0..p` into `cpu_out`; push `PendingGroup { src: ExpertSrc::Copied, pairs }` where `pairs` is the concatenation (hit pairs then miss pairs) in the order the rows were written; the compute group stays a pending group as today.
- Groups are formed per expert with `group_pairs` (hits as one expert list, misses as another), never by splitting an expert's pairs.
- `read_misses_async` does the routed RAM lookups exactly as `fill_ram` does for misses (counters, `prefetched` flags, `note_wasted`, `metrics.lookup`), pins hits and misses with the layer's other pins (`set_hot_set` union with in-flight prefetch pins of that layer), submits one job per shard with `copy_sink(tensor_dst(...))`, and records the ticket. `wait_misses` waits, adds to `IoCounters` (`batches`, `experts_read`, `bytes_read`, `read_time` from the ticket's `took`, new `urgent_wait_time`), and restores the layer's pins to the in-flight prefetch pins. On error: clear the layer (as `stage_ram` does) and return it. `settle` also waits for the urgent reader.
- New state tensors: `xn_host` (`F32 [n_embd, n_tokens]`) and `cpu_out_host` (`F32 [n_embd, n_used * n_tokens]`) in one context allocated on the CPU backend; `cpu_out` (`F32 [n_embd, n_used * n_tokens]`) on the primary, per `Slot` parity like `acc`. `alloc_state` takes the CPU backend.
- `sched_cpu: Sched::new(&[cpu.clone()], SEGMENT_NODES * 4, false)`, created at load when `overlap` is possible.
- The CPU group graph: `get_rows(xn_host, tokens)`, the three `mul_mat_id` over `streamer.ram_exps(seg)` as in `add_group`, output reshaped to `[n_embd, p]` and copied with `cpy` into a view of `cpu_out_host` at row offset `r0` (hits at 0, misses at `p_hits`). Write the index tensors after `alloc_graph`, as `GroupInputs::write` does.
- In `build_segment`, a `Copied` group adds `acc = set_rows(acc, view(cpu_out, [n_embd, p]), pairs_index)` before the other groups; the pairs index is an input tensor written after allocation.

**Tests to write first** (`crates/model/tests/overlap.rs`, hardware label `cpu`, synthetic Q4_K_M MoE from `tests/common/synth.rs`, small VRAM and RAM tiers so most decode steps have hits and misses):
1. `overlap_decode_is_bitwise_equal_to_serial`: prefill 64 then 16 greedy decode steps; all logits bitwise equal with `overlap` off and on, at prefetch depth 1 and 2 and with prefetch off.
2. `overlap_counts_reads_and_hits`: with overlap on, `StepProfile::cpu_hits > 0`, `read_wait` recorded, `ExpertStats.io.experts_read` equal to the serial run's (same misses, same bytes).
3. `overlap_read_error_fails_the_step_and_recovers`: cut the model file as `tests/prefetch.rs` does; the step errors, the next step after restoring the file is correct (logits equal to a fresh load).
4. Unit test in `qwen3moe.rs` (proptest, Rule 5 "every unit computed exactly once"): for random routings and random resident sets, the pairs of the compute group, the hit group and the miss group partition all pairs, each exactly once, and each expert's pairs are all in one group.
5. GPU (label `gpu`, skips without CUDA, add to the box test list, do not run): `gpu_overlap_logits_equal_serial` on the synthetic model with `BackendChoice::Gpu`.

**Metrics, panel, alert (Rule 6, same PR).**
- `crates/telemetry/src/metrics.rs`: key `tetraeder.part`; histograms `tetraeder.moe.layer.duration` (s, by part: `segment`, `cpu_hits`, `read_wait`, `cpu_misses`, `copy`) and `tetraeder.moe.read.hidden` (ratio). Thin wrappers in `crates/memory/src/metrics.rs` like `prediction_accuracy`.
- `StepProfile`: `cpu_hits`, `read_wait`, `cpu_misses`, `copy` (durations), included in `streamer` today's accounting so `total` still adds up; document.
- `deploy/grafana/dashboards/overview.json`: panel "MoE layer critical path" in the "Experts and cache" row (stacked rate of the duration sums by part, and p50 of the hidden ratio); unique id.
- `deploy/alerts/engine.yml`: `TetraederMissReadsNotHidden` as specified in ADR 0050, with a comment line like the existing rules; `tests/test_alert_rules.py` must pass (its series check needs the metric names emitted).
- Harness: `crates/model/tests/exit_regime.rs` env `TETRAEDER_EXIT_OVERLAP` (`0`/`1`, unset follows `LoadOptions`) and the new profile fields in the JSON; document in the file header. Add a stage `main-overlap` to `scripts/exit_session.sh` (pair with `main`, same box, interleaved), not run.

**Done.** Tests 1 to 4 pass locally and in CI on Windows and Linux; test 5 compiles and skips; checks of the header pass; ADR 0050 gets implementation notes. Report: files changed, test names, any deviation from the decisions above, local step profile of the synthetic exit regime (`cargo test --release -p tetraeder-model --test exit_regime -- --ignored`, overlap off and on).

**Out of scope.** Stage 2 (A2), any change to prefetch selection or order, prefill pipelining, op offload paths.

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
- Builder: `LoadOptions::kv: KvFormat` (default F16/F16). `alloc_kv` allocates K and V by type (`Type` from `Format`; add the conversion if missing). In `attention_and_router` the K and V views use `nb1 = row_size(type, head_dim)` for the head stride and the cache row (`kc.nb()[1]`) for the position stride; reject at load a format whose block does not divide `head_dim`. `memory()` reports KV capacity from the real buffer (already) and the format.
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
- Simulator `crates/memory/src/replay/sim.rs`: per tier a `SlotCache`; per layer step the engine's sequence: complete the layer's prefetch (unpin), VRAM lookups for routed experts (hits only, as `plan_cpu`), prefetch top-up of `il + 1` from `predicted` and of `il + 2` from the depth-2 prediction recorded two layers back, then RAM lookups of the CPU experts hits first (as `fill_ram`). Prefetched reads land before the target layer is planned (the engine waits); count their bytes as prefetch bytes. The VRAM hot set comes from the routing counts at each hot-refresh marker (`hot_from_counts`), pinned with `set_hot_set` on the VRAM cache.
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

Gate: A's profiling run on the box (`TETRAEDER_EXIT_TIME_GPU_GROUP=1`) shows the VRAM group at 5 ms per decode token or more. Then: `Sched::compute_async` and `Sched::synchronize` in `crates/ggml-sys/src/ggml/sched.rs` (`ggml_backend_sched_graph_compute_async`, `ggml_backend_sched_synchronize`, safety comments); a third scheduler for the VRAM group graph; launch it before the hit group; assert one split, else fall back to stage 1 for the layer and count `tetraeder.moe.layer.duration{part="gpu_group"}` fallbacks; synchronize before the next segment allocates. Same tests as A with the async path on; bitwise equality holds (same kernels, same inputs).

## Follow-up C2: RAM tier by routing frequency (only if gated in)

Gate: ADR 0050 (c) threshold on the box trace. Then: `LayerSlots::set_frequency_pins(&[u32])` with a second pin set, eviction skipping the union, `pinned_count` the union (property tests in `crates/memory/tests/slot_props.rs`: union never exceeds active slots, pinned experts never evicted, `set_hot_set` keeps frequency pins); `LoadOptions::ram_pin_share: f32` (default 0); `Streamer` refreshes frequency pins from `counts` where it refreshes the VRAM hot set and every 16 decode steps; prefetch budget `slots - union` through `prefetch_plan`. Metrics `tetraeder.expert.pinned` (tier, source), panel "RAM tier: hit rate and pinned slots", alert `TetraederRamTierHitDrop`. The replay with the chosen share must predict the box run's RAM hit rate within 0.5 points.
