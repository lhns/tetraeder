# 0056. Cost model: a RAM-tier and disk term from replayed routing, a virtual clock for I/O tests, a calibration ledger in CI

Status: Proposed (Phase 3a planner exit on the MoE builder fixture; Phase 2 levers). Amends ADR 0040 "Builder scoring" (the RAM tier becomes a cost-model feature). Extends ADR 0050 (c) (the replay) and ADR 0054 (the I/O service). Brief: `docs/costmodel-brief.md`.

## Context

- **Builder candidates tie.** `score::builder_features` has no RAM tier and no disk (ADR 0040, "Builder scoring"; `costmodel/src/predict.rs`, `sim.rs` model devices and links only). `enumerate_builder` gives each VRAM tier up to three RAM options (a quarter, half, all of the rest), so up to two thirds of the 24 candidates sit in groups the model predicts equal. The sort is stable, so a tie goes to the first in id order, the smallest RAM tier; the close-call rule then benchmarks the two smallest RAM options of the group and never the largest.
- **The ties are not small.** `bench/results/20261102/builder-real-202732.json` (Qwen3-30B-A3B Q4_K_M, 5 threads, 32 VRAM slots, CPU miss policy, same box): decode 19.4 tok/s with 32 RAM slots, 33.8 with all 128, 1.74x apart, predicted equal. The builder fixture (`planner-moe-builder`, `docs/m1-open.md` item 6) has never run; it is the open part of the Phase 3a planner exit test.
- **The fit absorbs disk time today.** Builder anchors with a small RAM tier read from disk; with no disk term the fit charges that time to the CPU and PCIe rates.
- **Hit rates are interpolated.** `HitModel` is linear in the VRAM slot share between two anchors. The routing replay (`memory::replay::sim`) runs the engine's own slot caches on a recorded trace and gives VRAM share, RAM hit rate and disk bytes per token for any slot counts, but counts bytes, not time.
- **I/O tests sleep.** The I/O service (ADR 0054, branch `track-w-iosvc`) is threads plus `ServiceOptions::delay` (`thread::sleep` per chunk); its ordering tests and the model crate's overlap tests depend on wall time. The replan engine loops on `track-y-replan` (`wait_drained`, `restart`, `healthy`) poll with `Instant::now` and `sleep`.
- **No drift guard, no provenance.** Nothing re-predicts committed box results after the cost model changes. `planner-moe` records `git_commit: "unknown"`; `exit-*.json` records three `env` fields and no engine commit.
- **The anchor bands are not candidate bands.** ADR 0040's 9.2% (MoE) and 5.8% (dense) are `Fit::band`, the largest anchor residual of the fit, in sample. On the committed fixtures the candidates' request-time errors are: MoE median 4.3%, 5 of 24 above 9.2%, max 23.1% (c08); dense median 2.1%, 4 of 11 above 5.8%, max 27.8%. A per-result gate at the anchor band would fail on the committed data.

## Decision

### 1. Tier and disk term

**Owners.** `predict.rs` owns the formula (analytic, fast, used by the fit and the search). The replay owns bytes: it turns slot counts and a trace into bytes per token. The replay's virtual clock (section 2) is a check and a pricing tool, not an input to `predict`. Nothing is counted twice.

**Disk curve.** `profiler::Device` gains `queue_curve: Vec<QueuePoint { depth: u32, bytes_per_s: f64 }>` and `queue_chunk_bytes: Option<u64>` (Disk devices). `Device::disk_rate(depth)` interpolates like `cpu_rates` (piecewise linear, proportional below the first point, flat above the last); without a curve it falls back to `bandwidth_bytes_per_s`. Latency is not a separate field: at a fixed chunk size, mean chunk latency is `depth * chunk / bytes_per_s` (Little's law). The curve is measured on the box (section 4), never fitted: anchors cannot separate disk from CPU time.

**Features.** `RunFeatures` gains `tier: Option<TierFeatures>` (serde default `None`):

```text
TierFeatures { disk: String, path: ReadPath, depth: u32, sync_depth: u32,
               decode_sync_bytes_per_token, decode_prefetch_bytes_per_token,
               prefill_bytes_per_token }
ReadPath = Reader | Service | ServiceOverlap
```

`path` comes from the measured configuration: `Reader` is today's `memory::reader` path (copy serial on the reader thread); `Service` is ADR 0054 section 1; `ServiceOverlap` adds section 2 (arrival-order compute). `depth`: the reader's queue depth for `Reader`, `min(workers, ceil(read bytes per layer step / chunk))` for the service. `sync_depth`: the direct-I/O queue depth of the synchronous reads (`read_experts`).

**Formula** (decode, seconds per token; `B = disk_rate(depth)`; `C_rest` = every part except the CPU expert part plus switches; `C_exp` = the CPU part, at `threads - 1` for `ServiceOverlap` when reads occur, as ADR 0054 runs compute one thread fewer; `t_s = sync / B`, `t_p = prefetch / B`, `D = t_s + t_p`):

```text
Reader:          B_eff = 1 / (1/B + 1/B_copy)   (B_copy = cpu bytes/s at 1 thread)
                 t_tok = C_rest + C_exp + (sync + prefetch) / B_eff
Service:         t_tok = max(C_rest + C_exp + t_s, D)
ServiceOverlap:  t_tok = max(C_rest + max(C_exp, t_s), D)
```

The reader path is the serial sum because box C shows its prefetch barely hides (95 ms of 257 waiting with the reader busy 131 ms): the sum gives 291 ms there, 13% over. The service formulas are "critical chain or disk busy, whichever is longer", a lower bound until the paired `main-iosvc` and `main-overlap` runs (ADR 0054) measure them. Prefill: misses are read synchronously (ADR 0054 keeps `read_experts` for prefill), so `t_pp` adds `prefill_bytes / disk_rate(sync_depth)` per ubatch, serial. Box C's prefill read 62 s of about 450 s, so the term is not optional.

Breakdown label `read:<disk>` (seconds per token the disk adds over `tier = None`); `binding` can name it. With `tier: None`, or all byte counts 0, every prediction is bit-identical to today.

**Bytes from the replay.** `PhaseOutcome` gains `full_hit_share` (layer steps with every routed expert in VRAM). A builder session records a routing trace on the fixture's prompt and runs the existing `tetraeder-bench replay` over the VRAM and RAM slot counts of its candidates and anchors (`--policies lru`, prefetch as configured) into `tier-replay.json`. The file keeps today's format (`outcomes[]` with `vram_slots`, `ram_slots`, `policy`, `prefetch` and the `prefill` and `decode` counters) and gains `full_hit_share` per phase, the trace's `fnv1a64` and a `replay_version`. The planner reads the fields it needs from that JSON into its own `TierOutcome`; it does not depend on `memory` (ADR 0040 keeps it free of ggml, and feature unification would link ggml). The trace is committed next to it when it is under 20 MB compressed, and a `bench` test (which already depends on `memory`) checks the committed `tier-replay.json` against a fresh replay of it.

**Builder features.** `builder_features_tier(layout, vram_slots, &TierOutcome, knobs, path, depth, disk)`: GPU expert bytes at the decode `vram_share` (replaces the interpolated hit), CPU at `1 - vram_share`, switches from `full_hit_share`, `tier` from the byte counts. `builder_features(layout, vram_slots, hit, knobs)` stays with its signature and today's output (it is `builder_features_tier` with `vram_share = full_hit_share = hit` and no tier), so `track-y-replan`'s `serve.rs` merges unchanged. `score_builder` uses `tier-replay.json` when the fixture has it, else today's `HitModel`. Before scoring, the replay must pass its fidelity gate on the anchors: LRU replay within 0.5 points of each anchor's measured `ram_hit_rate_mean` (ADR 0054 section 3); otherwise the scorer falls back to `HitModel` and records `tier_source: "hitmodel (fidelity gate failed: <anchor>, <delta>)"` in `score.json` and the ledger, so the miss is visible and the fixture stays scoreable.

**Planner.** Candidates and their enumeration do not change (ADR 0040: fixed before scoring). Only the scores change: a RAM option now costs disk bytes, so RAM-only candidates differ by `read:<disk>` and the close-call rule compares real neighbours. `serve --plan auto` (ADR 0055) uses the uniform prior (no trace) and stays as it is; with `--tier-replay <file>` it uses the replayed bytes. `correct_for_drift` learns the `read:<disk>` binding: it scales the disk's `queue_curve`, so a slow disk re-picks toward a larger RAM tier.

**Not modelled here.** The page cache (direct reads leave it empty, ADR 0054), multi-GPU tiers, batching. If the builder fixture's residuals on disk-reading anchors exceed the band, the next step is a per-run `hidden_share` input taken from the timed replay; until then the replay checks the formula and does not feed it.

### 2. Virtual clock

**memory** (minimum change, after `track-w-iosvc` merges):
- `io::Queue<D>`: the service's earliest-deadline-first heap and unit bookkeeping, moved out of `IoService` without threads or time. `submit(now, deadline, parts) -> UnitId`, `pop() -> Option<Chunk<D>>`, `complete(now, chunk, io::Result<u64>) -> Option<UnitId>` (the unit that landed), `collect(units)`, `inflight()`. `now` is seconds from the caller's clock, so `UnitDone::took` works for both drivers.
- `io::Disk`: a processor-sharing disk over `queue_curve`. Up to `workers` chunks in service; with `n` in service each progresses at `disk_rate(n) / n`. `next_completion() -> Option<f64>`, `advance_to(t) -> Vec<chunk>`. Deterministic: ties land in submit order.
- `IoService` becomes the threaded driver of `Queue<HostDst>` (behaviour and public API unchanged). `ServiceOptions::delay` is deleted.
- `io::Io` trait with the five methods the streamer calls (`submit`, `poll`, `wait_any`, `wait_all`, `inflight`), `Send`. The streamer holds `Box<dyn Io>`. Implemented by `IoService` and by `io::SimIo` (feature `sim`, test only): `Queue<HostDst>` + `Disk` + `SimClock`. `SimClock` is a shared virtual time (`Arc<AtomicU64>` of f64 bits) the test advances; `poll` lands what completes by now, `wait_any`/`wait_all` advance the clock to the next landing. A landing copies the bytes from the shard file (buffered read, instant in real time) into the destination, so logits stay exact.
- `replay::timed::replay_timed(trace, cfg, &TimeModel) -> TimedOutcome`: the existing `Sim` call order with a timing hook at its call sites; urgent misses and prefetches go through `Queue<()>` and `Disk` with `(step, layer)` deadlines. `TimeModel { gpu_layer_s, cpu_expert_s, cpu_group_s, curve, chunk_bytes, workers, path, sync_depth }` is plain numbers (the planner writes it from a calibrated prediction with `tetraeder-plan explain --time-model`). `TimedOutcome` adds per decode token: step time, `cpu_ready`, `read_wait`, `cpu_landed`, read span and hidden share, with the names and definitions of ADR 0054's `StepProfile` and `tetraeder.moe.read.hidden`. Bytes equal `replay`'s exactly.

**control** (after `track-y-replan` merges): `Clock` gains `fn sleep(&self, s: f64)` with a default of `std::thread::sleep`; `FakeClock::sleep` advances the fake time and `FakeClock` moves to `Arc` so it can cross to the control thread. The binary's `wait_drained`, the restart cut wait and the health deadline use `clock.now_s()` and `clock.sleep()`. `ResidualMonitor::tick` is already driven by its caller and stays.

**What the clock tests, in milliseconds:** EDF and urgent-before-prefetch order and failure isolation on `Queue`; arrival-order groups in the model crate's overlap tests (ready group first, then landing order, logits bitwise equal) on `SimIo`; prefetch depth 1 against 2, tier sizing and worker counts on the timed replay; a drift loop: timed-replay step samples with a disk slowed 4x at t = 300 s into the step ring, the monitor fires after 5 ticks, `correct_for_drift` scales the curve, the replanner switches to a larger RAM tier after its dwell.

### 3. Calibration ledger

**Provenance, required on every result the ledger reads.** A `provenance` block: `engine_commit` (40 hex; `git rev-parse HEAD`, else `TETRAEDER_GIT_COMMIT`, which `session.sh` exports), `engine_dirty`, `session` (`<date>-<host_id>-<HHMMSS>`), `host_id`, `gpu`, `cpu_quota`, `threads`, `cpu_variant` (ADR 0052; it changes CPU rates), and for baselines the baseline engine's `build_commit`. Plus the measured configuration as already written (`options` of `exit_regime`, the variant of the harness) and, for builder runs, the read path, prefetch depth, workers and chunk size. The harness and `exit_regime` write `provenance`; a result without `engine_commit` is not admitted.

**Calibration = a session's anchors under the current code.** A session that wants its results checked runs the anchor set of its engine (ADR 0040; a short builder anchor stage for exit sessions, section 4). The calibration is the deterministic fit of those committed anchors, recomputed in CI, never trusted as stored. `tetraeder-plan ledger --write` exports it as `calib.json` in the session directory (`cost_model_version`, the session key, anchors, fitted graph with the disk curve, band) for `serve --hw` and `what-if`; CI checks the export is current.

**Like with like.** A result is re-predicted only with the calibration of its own session key `(engine_commit, host_id, cpu_variant)`. Without one it is listed as `uncalibrated`, not failed. A new engine build is a new key, so its calibration never re-evaluates old results.

**Model version.** `costmodel::MODEL_VERSION: u32` bumps when `predict` changes any output for a configuration that already exists. Every term is gated by the configuration that enables it: a result whose configuration lacks a lever gets exactly the old prediction (no overlap flag, no overlap term; no `tier`, today's numbers). A configuration with a lever the model does not know (an `options` field the feature builder does not map, at a non-default value) is `excluded: unmodelled <lever>`, listed, not failed. So a new lever turns nothing red by design; modelling it later brings those results in.

**Snapshot.** `bench/ledger.json`: per result `{path, id, session, class (moe | dense), status, predicted_s, measured_s, err, cost_model_version}`. The CI test `committed_results_match_the_ledger` (planner crate, beside `committed_fixtures_meet_the_exit_test`):
1. the snapshot's `cost_model_version` equals `MODEL_VERSION`, else "run `tetraeder-plan ledger --write`";
2. at the same version every prediction equals the snapshot within 1e-9 relative (catches a silent model change); if Linux and Windows fits differ by more, the tolerance becomes the measured difference, recorded as a constant;
3. per session, the median error of its non-anchor results is at most the class band (MoE 9.2%, dense 5.8%, ADR 0040); today 4.3% and 2.1%;
4. every exported `calib.json` equals the recomputed fit (same tolerance as 2);
5. a ratchet: `ledger --write` refuses to raise any result's error by more than `max(0.02, its repeat spread)` over the previous snapshot unless `--accept-worse <id>=<reason>`, and the reason is stored in the snapshot.

**Superseding.** A new model version supersedes the old by `ledger --write` under the ratchet; the PR diff shows every changed prediction and calibration, git keeps the old snapshot. A new calibration of the same box class is a new session with its own `calib.json`; old sessions stay in the ledger. Picking the newest session per GPU class for serving (`calib export --latest`) waits until two sessions of one class exist.

**Legacy.** `planner-moe` and `partial-planner-dense` are single campaigns (anchors and candidates from one build in one file), so they get a hand-pinned `provenance.json` (session from the fixture, `engine_commit: "pre-0056"`). `exit-*.json` without anchors are `uncalibrated`. Box disk points (sampler `read_mb_s` at `disk_queue_depth_mean`) against the session's curve are reported, not gated, until two sessions exist.

### 4. Box stages (when the hold lifts)

- **`disk-curve`**, every builder session, about 2 minutes: `tetraeder-bench disk-curve --file <largest shard> --chunk-bytes 1048576 --depths 1,2,4,8,16,32 --seconds 4`, random aligned chunks through `IoService` with `workers = depth`, direct I/O; once idle, once beside `quota - 1` spinning threads (ADR 0054's starvation case). Writes bytes/s and p50/p99 chunk latency per depth plus `provenance`; the loaded curve goes into the session's graph.
- **`trace`** in `scripts/planner_builder_campaign.sh`: one builder run on the fixture prompt with the recorder on, then the replay into `tier-replay.json`. About 5 minutes on the 30B.
- **`anchors-builder-short`** for exit sessions that want ledger coverage: the four builder anchors at pp512 tg32. About 12 minutes on the 235B.

### Rules

- **Rule 4.** Planner tests run the builder enumeration and tier scoring on every simulated fixture: the RTX 3090 prior and `costmodel/fixtures/two_gpu_cpu_nvme.toml` gain a prior `queue_curve`; `wan_two_node.toml` has no Disk device, so a candidate with disk bytes fails to predict with `MissingSpec("disk")` and is dropped with a note. Faults from `analysis::simfault`: a throttled NVMe (curve x0.25) moves the pick to fewer disk bytes per token; a CPU co-tenant (half the threads) is scored without a crash and the pick fits the reduced quota.
- **Rule 5** (proptest): `tier = None` and zero bytes predict bit-identically to today; predicted time is non-increasing in fewer disk bytes, a faster curve, and VRAM share; the builder footprint including the service's bounce (`workers * shards * chunk`) never exceeds the budget; `Queue` pops in key order and serves every chunk once; the virtual clock lands every unit exactly once, never before its curve time; timed and untimed replay give identical bytes; with LRU and prefetch off, replayed disk bytes are non-increasing in RAM slots (LRU is a stack algorithm; if the chunked `fill_ram` breaks inclusion the counterexample is reported as an engine finding and the property narrows to RAM = all experts outside VRAM giving no decode misses after warm-up).
- **Rule 6.** No runtime component is added: the curve is measured by a box stage, `SimIo` is test only, the ledger is CI. The runtime counterparts already exist (`tetraeder.moe.read.hidden`, `tetraeder.io.inflight`, `tetraeder.plan.residual`).

### Expected value

A prediction, to be checked by the `b3-planner` session on the builder fixture: today the analytic pick inside a RAM tie group is the smallest RAM tier, and on the 30B that tier decodes up to 1.74x slower than the largest at the same VRAM slots. With the term, RAM-only candidates are ranked by their disk bytes, the close call compares real neighbours, and regret stays within the 10% bar with Spearman at least 0.8. If it does not, the ledger shows which anchors miss and whether the reader formula or the bytes are wrong.

## Consequences

- The builder fixture needs a trace and a disk curve from its own session; without them it scores as today.
- Two analytic read formulas are lower bounds until the service runs are measured; the ledger and the timed replay make that visible.
- `memory` gains a pure queue, a disk model and a test driver; the streamer depends on a trait instead of `IoService`; `ServiceOptions::delay` goes.
- `control::Clock` gains one method; the binary's three poll loops change.
- Every box script exports `TETRAEDER_GIT_COMMIT` and `TETRAEDER_SESSION`; results without them never enter the ledger.
- A cost-model change that moves any committed prediction needs a version bump and a reviewed snapshot.
