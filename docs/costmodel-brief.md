# Implementation brief for ADR 0056 (tier and disk term, virtual clock, calibration ledger)

Six parts; each merges alone in the order below. Every design decision is in ADR 0056; read it first, then AGENTS.md, ADR 0040 ("Cost model", "Calibration", "Builder candidates", "Builder scoring"), ADR 0054 and ADR 0050 (c). Do not revisit the decisions; if one turns out wrong, stop and report the evidence. No box runs until the hold lifts (part F only).

All parts: failing tests first, then code. Every local cargo command goes through the shared lock with one job: `L=/c/Users/pierr/AppData/Local/Temp/claude/cargo-host.lock; until mkdir "$L" 2>/dev/null; do sleep 20; done; CARGO_BUILD_JOBS=1 cargo <args>; rmdir "$L"`. Prefer `cargo test -p <crate>` while iterating; CI on the pushed `track-*` branch is the compile loop when the lock starves. At most one `gh` call per 15 minutes. Done means `cargo build --workspace`, `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`, `cargo fmt --check` pass on Windows and Linux (Rule 11), and ADR 0056 gets an "Implementation notes" line for the part. No em dashes in docs or comments. Commit only if your brief says so.

## Order

| Part | Touches | When | Effort |
|---|---|---|---|
| A. Provenance and ledger | `bench` (env), `model` (exit_regime JSON), `costmodel` (`MODEL_VERSION`), `planner` (`ledger.rs`, CLI, CI test), `scripts/session.sh`, two `provenance.json`, `bench/ledger.json` | now | 1 to 1.5 d |
| B. Tier term | `profiler` (`queue_curve`), `costmodel` (`predict.rs`), `planner` (`score.rs`, priors, synthetic fixture), cost model fixtures | now, after or beside A | 1.5 to 2 d |
| C. Replay output for the planner | `memory` (`replay::sim`), `bench` (`replay.rs`, a snapshot test) | now | 0.5 d |
| D. Virtual clock in `memory` | `memory` (`io.rs` split, `SimIo`, `replay::timed`), `model` (streamer on `dyn Io`, overlap tests) | after `track-w-iosvc` is on main (`git log origin/main --oneline \| grep -i iosvc`) | 2 to 2.5 d |
| E. Virtual clock in `control`, drift loop | `control` (`Clock::sleep`), `tetraeder` (replan loops), `planner` (`correct_for_drift`, one test) | after B and after `track-y-replan` is on main | 1 d |
| F. Box stages | `bench` (`disk-curve`), `scripts/planner_builder_campaign.sh`, `scripts/exit_run.sh`, `docs/sessions.md` | after A, C and `track-w-iosvc`; runs when the hold lifts | 0.5 to 1 d, then box time |

B and C are independent: the file between them is the existing `tetraeder-bench replay` output (ADR 0056, "Bytes from the replay"); B tests with a hand-written sample, C makes the real one.

---

## Part A. Provenance and ledger

**Goal.** Every new result names the engine build and session it came from, and CI re-predicts the committed planner fixtures with their own session's calibration and fails on drift (ADR 0056 section 3).

**Read first.** `crates/bench/src/env.rs`, `crates/bench/src/run.rs` (where `env` enters the result), `crates/model/tests/exit_regime.rs` around line 596 (the `env` block), `crates/planner/src/score.rs` (`score_dir`, `score_with`, `committed_fixtures_meet_the_exit_test`), `crates/costmodel/src/calib.rs` (`Fit`), `scripts/session.sh`, `bench/results/20261103/planner-moe/` and `partial-planner-dense/`.

**Changes.**
- `costmodel/src/lib.rs`: `pub const MODEL_VERSION: u32 = 1;` with a doc comment: bump when `predict` changes any output for a configuration that already exists (ADR 0056).
- `bench/src/env.rs`: `EnvInfo` gains `session: Option<String>` (from `TETRAEDER_SESSION`) and keeps `git_commit`, but prefer `TETRAEDER_GIT_COMMIT` when set (the box checkout may not be a git repo). Serde defaults so old files still parse.
- `exit_regime.rs`: the result gains a `provenance` object: `engine_commit` (`TETRAEDER_GIT_COMMIT`, else `git rev-parse HEAD`, else `null`), `engine_dirty`, `session`, `host_id` (`CONTAINER_ID`, else hostname), `cpu_quota`, `threads`, `cpu_variant` (the ADR 0052 variant the run loaded; `null` if the API is not there). Also write the read path into `options`: `io_service`, `overlap`, `workers`, `chunk_bytes` when those options exist on main, else omit them.
- `scripts/session.sh`: export `TETRAEDER_GIT_COMMIT` (the commit taxameter pushed: `git rev-parse HEAD` in the checkout, set once at start) and `TETRAEDER_SESSION=<date>-<host_id>-<HHMMSS>`.
- `planner/src/ledger.rs` (new, re-exported):
  - `SessionKey { engine_commit, host_id, cpu_variant }`. A planner fixture directory is one session: its key comes from the harness file's `env` (`git_commit`, `host_id`, `cpu_backend`), or from a `provenance.json` beside `candidates.json` when present (legacy override).
  - `pub fn evaluate(results_root: &Path) -> Ledger`: for every fixture directory `score_dir` already accepts, run the scorer (it fits on the anchors only) and turn each non-anchor variant with a measurement into a `LedgerEntry { path, id, session, class, status, predicted_s, measured_s, err, cost_model_version }`. `class` is `moe` when `layout.n_expert > 0`, else `dense`. `status` is `checked`, `uncalibrated` (no anchors measured), or `excluded: <reason>` (failed on the box; no prediction). Entries sorted by path then id, so the JSON is stable.
  - `pub fn check(current: &Ledger, snapshot: &Ledger) -> Vec<String>` (empty means pass), the four gates of ADR 0056 section 3: version equal; same-version predictions equal within 1e-9 relative; per session, median `err` of `checked` entries at most the class band (`MOE_BAND = 0.092`, `DENSE_BAND = 0.058`, constants citing ADR 0040); entries in the snapshot missing from the current ledger are reported.
  - `pub fn write(current: &Ledger, previous: Option<&Ledger>, accept_worse: &[(String, String)]) -> Result<Ledger, String>`: the ratchet. Refuses when any entry's `err` exceeds the previous one's by more than `max(0.02, spread)` (`spread` = (max - min) / median of that variant's repeats when there are two or more, else 0) unless its id is in `accept_worse`; the reasons are stored in the snapshot's `accepted` list.
- CLI: `tetraeder-plan ledger [--root bench/results] [--write] [--accept-worse id=reason]...` prints a table (session, class, median err, worst entry) and, with `--write`, rewrites `bench/ledger.json`.
- `bench/results/20261103/planner-moe/provenance.json` and `partial-planner-dense/provenance.json`: `{"engine_commit": "pre-0056", "host_id": <from the harness env>, "cpu_variant": null, "session": "20261103-<host_id>-planner"}`.
- `bench/ledger.json`: generated by `--write` on the final code, committed.
- Test `committed_results_match_the_ledger` in `score.rs`'s test module beside `committed_fixtures_meet_the_exit_test`: `check(&evaluate(root), &snapshot)` is empty. Skips cleanly when `bench/ledger.json` is absent.

**Tests first.** `check` on hand-built ledgers: a version mismatch, a 1e-6 prediction change, a session median over its band, a missing entry, each produce one message; equal ledgers produce none. `write` refuses a 0.03 increase, accepts it with `accept_worse`, accepts a 0.01 increase. `evaluate` on the committed fixtures: MoE median about 0.043, dense about 0.021 (from `score.json`), both `checked`. `EnvInfo` parses an old `env` without `session`.

**Report.** The ledger table on the committed fixtures, and the exact `--write` command for later version bumps.

---

## Part B. Tier term

**Goal.** RAM-only builder candidates no longer tie: `predict` charges disk reads from a measured queue-depth curve, and the builder scorer takes VRAM share and disk bytes from a replay file (ADR 0056 section 1). With no tier, every prediction is bit-identical to today.

**Read first.** `crates/costmodel/src/predict.rs` (module doc and `predict`), `crates/profiler/src/graph.rs` (`Device`, `cpu_rates`), `crates/planner/src/score.rs` (`builder_features`, `HitModel`, `score_builder`, `write_synthetic_builder`), `crates/planner/src/builder.rs`, `crates/planner/fixtures/rtx3090_prior.toml`, `crates/costmodel/fixtures/*.toml`, `crates/analysis/src/simfault.rs` (fault classes), `bench/results/20261109-local/replay-synthetic.json` (the replay file format).

**Changes.**
- `profiler::Device`: `queue_curve: Vec<QueuePoint>` (`QueuePoint { depth: u32, bytes_per_s: f64 }`, serde default empty, skip if empty) and `queue_chunk_bytes: Option<u64>`. `pub fn disk_rate(&self, depth: u32) -> Option<f64>`: same interpolation as `cpu_rates`; empty curve falls back to `bandwidth_bytes_per_s`; `None` when neither exists.
- `costmodel::predict`: `RunFeatures.tier: Option<TierFeatures>` (`#[serde(default)]`), `TierFeatures` and `ReadPath` exactly as ADR 0056. In `predict`: compute today's parts first (unchanged code path), then apply the ADR's decode formula for the path and add the prefill term per ubatch (`M * prefill_bytes_per_token / disk_rate(sync_depth)`; `sync_depth` is the run's direct-I/O queue depth from its options). `C_exp` is the sum of the CPU parts' compute terms (the part whose device is a CPU); for `ServiceOverlap` with `sync > 0` recompute it at `threads - 1` (at least 1). `B_copy` = the CPU's `cpu_rates(1).1`. Add breakdown `read:<disk>` = `t_tok - t_tok_without_tier` (and the prefill equivalent in `pp_breakdown`); `binding` may become it. New `PredictError::MissingSpec("disk <id>")` when the tier names a device without a rate. Update the module doc with the formulas.
- `planner::score`:
  - `TierOutcome` (planner-side mirror of one replay outcome: `vram_slots`, `ram_slots`, `policy`, `prefetch`, and from `decode`/`prefill`: `vram_share`, `full_hit_share` (default to `vram_share` when absent, for files from before part C), `sync_miss_bytes_per_token`, `prefetch_bytes_per_token`, `disk_bytes_per_token`). `pub fn read_tier_replay(path) -> Result<Vec<TierOutcome>, String>` ignores unknown fields.
  - `pub fn builder_features_tier(layout, vram_slots, tier: &TierOutcome, knobs, path: ReadPath, depth: u32, disk: &str) -> RunFeatures`. Refactor `builder_features` to call it with `vram_share = full_hit_share = hit` and no tier; its output must stay bit-identical (existing tests prove it).
  - `score_builder`: when the fixture dir has `tier-replay.json`, take each variant's outcome by `(vram_slots, ram_slots, "lru", prefetch from the variant's env)`; read path from the env (`TETRAEDER_EXIT_IO_SERVICE`, `TETRAEDER_EXIT_OVERLAP`, absent means `Reader`), `depth` and `sync_depth` from the env per the ADR, disk id `nvme0`. Fidelity gate first: for each anchor, `|replayed ram_hit_rate_engine_mean - measured decode_layers.ram_hit_rate_mean| <= 0.005`, else `Err` naming the anchor. Without the file: today's `HitModel` path, unchanged.
- Priors: `rtx3090_prior.toml` gains `nvme0` (kind disk, tier nvme) with a prior curve (depth 1: 1.0e9, 4: 3.0e9, 16: 5.2e9, 32: 5.4e9 bytes/s, chunk 1 MiB; comment: box C measured 5.0 to 5.2 GB/s at depth 16, ADR 0054) and a storage link to `cpu0`; `two_gpu_cpu_nvme.toml` gains a curve on its `nvme0` with the same shape scaled to its bandwidth. `wan_two_node.toml` stays without a disk.
- Synthetic builder fixture: `write_synthetic_builder` also writes `tier-replay.json` from a truth function (disk bytes per token falling linearly to 0 as `ram_slots` reaches all experts outside VRAM; `vram_share` as the current `truth_hit`), and the truth graph includes the prior disk curve; regenerate the committed `crates/planner/fixtures/builder-synthetic/` with the ignored test and keep `check_builder_score` passing.

**Tests first.**
- Golden: every existing predict and score test passes unchanged; a test predicts every variant of both committed fixtures and the synthetic fixture with `tier: None` and with a zero-byte tier and asserts bitwise equality.
- Formula: hand-computed `t_tok` for each `ReadPath` on the `predict.rs` test graph plus a disk; prefill term per ubatch.
- Rule 5 proptests (costmodel): fewer disk bytes, a curve scaled up, or a higher `vram_share` never predict slower; `disk_rate` is non-decreasing in depth on a non-decreasing curve.
- Rule 5 proptest (planner): every `enumerate_builder` candidate's footprint plus the service bounce (`16 * shards * 1 MiB`, shards from the layout or 1) stays within the budget, over random budgets.
- Rule 4: for the RTX 3090 prior and `two_gpu_cpu_nvme`: score the synthetic builder candidates with a synthetic tier file; no two candidates that differ only in `ram_slots` predict equal; the pick has the most RAM slots among its VRAM tier when disk bytes fall with RAM. `wan_two_node`: candidates with disk bytes fail with `MissingSpec`, are dropped with a note, the rest score. Faults: with the disk curve times 0.25 (throttled NVMe) the pick's decode disk bytes per token are at most the no-fault pick's; with half the threads (CPU co-tenant) scoring succeeds and the pick's threads fit the quota.
- Synthetic fixture: regret and Spearman still meet the exit bars with the tier file; and a variant of the test with the tier file removed reproduces today's numbers.

**Report.** The synthetic fixture's regret and Spearman with and without the tier file; the number of RAM tie groups before and after on the synthetic enumeration.

---

## Part C. Replay output for the planner

**Goal.** The existing replay output carries what the planner reads (ADR 0056, "Bytes from the replay").

**Read first.** `crates/memory/src/replay/sim.rs` (`PhaseOutcome`, `Acc`, `layer_step`, `outcome`), `crates/bench/src/replay.rs`, `crates/memory/tests/replay.rs`, `crates/model/tests/replay_fidelity.rs`.

**Changes.**
- `PhaseOutcome.full_hit_share`: layer steps where every routed expert of every token was in VRAM, over layer steps.
- `bench/src/replay.rs`: the output gains `"trace": {..., "fnv1a64": <of the trace file bytes>}` and `"replay_version": 1` (a constant in `memory::replay`, bumped when replay counters change for an existing trace).
- A test in `crates/bench/tests/` (`committed_tier_replays_are_current`): for every `bench/results/*/*/tier-replay.json` whose directory also holds the trace it names, replay it again and compare the outcomes exactly; skips cleanly when none is committed.

**Tests first.** `full_hit_share` on a hand-built two-step trace (one step all in VRAM, one not: 0.5); 1.0 with VRAM slots equal to the expert count; the replay output of a small synthetic trace round-trips through the planner's `read_tier_replay` (copy the planner struct's expectations as a JSON sample into the bench test; no planner dependency).

---

## Part D. Virtual clock in `memory`

**Precondition.** `track-w-iosvc` is on main; otherwise stop and report.

**Goal.** No sleeps in I/O tests: one queue type driven by threads in production and by a virtual clock in tests and in the timed replay (ADR 0056 section 2).

**Read first.** `crates/memory/src/io.rs` (whole file), `crates/memory/src/reader.rs` (`HostDst`), `crates/model/src/experts.rs` (the streamer's use of `IoService`), `crates/model/tests/overlap.rs`, `crates/memory/src/replay/sim.rs`.

**Changes.**
- `io.rs` split into `io/queue.rs` (`Queue<D>`: heap, units, `submit(now, deadline, parts)`, `pop`, `complete(now, chunk, result)`, `collect`, `inflight`; generic over the destination with a small `Dst` trait providing `len` and `sub`, implemented for `HostDst` and `()`), `io/disk.rs` (`Disk`: processor sharing over a curve of `(depth, bytes_per_s)`, `workers`, `submit(chunk_bytes)`, `next_completion`, `advance_to`), `io/service.rs` (`IoService` on `Mutex<Queue<HostDst>>`, unchanged API, `Instant` turned into seconds for `now`), `io/sim.rs` (feature `sim`: `SimClock`, `SimIo`), and the `Io` trait in `io/mod.rs`. Delete `ServiceOptions::delay` and its sleep.
- The streamer in `model` holds `Box<dyn Io>`; construction stays `IoService::new` in production. `model`'s dev-dependency on `memory` enables `sim`.
- `replay/timed.rs`: `TimeModel`, `TimedOutcome`, `replay_timed` per ADR 0056. Add one optional timing hook to `Sim` (a `&mut dyn Timing` called at layer begin, prefetch submit, miss submit, CPU group, layer end); `replay` passes none, so its counters cannot change.
- `tetraeder-bench replay --timed <time-model.json>` writes `TimedOutcome` beside the byte outcomes.

**Tests first (no sleeps anywhere).**
- `Queue`: the existing ordering, failure and drop tests ported from timing to explicit `pop`/`complete` sequences; proptest: for any submit sequence, pop order is sorted by key and every chunk pops exactly once.
- `Disk`: one chunk at depth 1 takes `chunk / rate(1)`; two equal chunks at depth 2 both finish at `2 * chunk / rate(2)`; proptest: no completion before `chunk / rate(1)` after its start; total bytes served equals submitted.
- `SimIo`: lands byte-exact; `poll` at time 0 lands nothing; `wait_any` lands earliest-deadline first; urgent before prefetch at equal deadline.
- `model/tests/overlap.rs` on `SimIo`: the ready group is computed first, then landed groups in landing order; logits bitwise equal with overlap on and off (the existing assertion). Remove every `delay`.
- `replay_timed`: bytes equal `replay` on the synthetic traces (proptest over slot counts); every submitted unit lands exactly once; exposed wait at most the read span; with zero disk bytes, step time equals the compute sum; prefetch depth 2 has no more `read_wait` than depth 1 on a fixed synthetic trace where the disk is not saturated (an example test, not a property).
- Timing: the whole `memory` and `model` overlap test set runs under 2 s locally (print the time in the report).

**Report.** Test wall time before and after; any engine behaviour the queue split changed (expected none).

---

## Part E. Virtual clock in `control`, drift loop

**Precondition.** Part B and `track-y-replan` on main; otherwise stop and report.

**Read first.** `crates/control/src/replan.rs`, `crates/tetraeder/src/replan/{engine.rs,monitor.rs,control.rs,source.rs}`, `crates/planner/src/serve.rs` (`correct_for_drift`), `crates/planner/tests/serve_replan_sim.rs`.

**Changes.**
- `control::Clock::sleep(&self, s: f64)` with a default `std::thread::sleep`; `FakeClock` on `Arc<Mutex<f64>>` (Send) with `sleep` advancing it. `SystemClock` keeps the default.
- The binary's `wait_drained`, restart cut wait and health deadline take the replanner's clock (pass an `Arc<dyn Clock + Send + Sync>` into the engine wrapper) instead of `Instant` and `thread::sleep`.
- `correct_for_drift`: a `read:<disk>` binding scales that device's `queue_curve` points (or its `bandwidth_bytes_per_s` without a curve) by `1 / k`, same clamp.

**Tests first.** The replan engine's drain and health tests run with `FakeClock` and finish in milliseconds; `correct_for_drift` on a `read:nvme0` binding scales the curve; a closed-loop test in `serve_replan_sim.rs`: a builder plan on the RTX 3090 prior with a synthetic tier file, step samples from `predict` on the graph with the disk curve times 0.25 from t = 300 s, `ResidualMonitor` ticks every 60 fake seconds, `Trigger::Drift` after 5 ticks, the proposal has more RAM slots or more VRAM slots and fewer disk bytes, the replanner switches after its dwell.

---

## Part F. Box stages

**Precondition.** Parts A and C and `track-w-iosvc` on main. Write and test locally now; the stages run when the hold lifts and the coordinator assigns a box.

**Changes.**
- `tetraeder-bench disk-curve --file F --chunk-bytes 1048576 --depths 1,2,4,8,16,32 --seconds 4 [--load-threads N] --out F.json`: random 4 KiB aligned chunks of `F` through `IoService` with `workers = depth`, direct I/O; per depth bytes/s, p50 and p99 chunk latency; with `--load-threads` the same beside N spinning threads. The output has `provenance` and a `queue_curve` ready to paste into a graph. Local test on a temp file with buffered fallback: the JSON has one point per depth and bytes/s is positive.
- `scripts/planner_builder_campaign.sh`: stages `disk-curve` (idle and with `quota - 1` load threads), `trace` (one builder run with the recorder on the fixture prompt, then `tetraeder-bench replay` over the candidates' and anchors' slot counts into `tier-replay.json`; keep the trace when under 20 MB compressed). The loaded curve goes into the fixture's graph as `nvme0`.
- `scripts/exit_run.sh`: `disk-curve` for every builder session (2 min), and an optional `anchors-builder-short` stage (the four builder anchors at pp512 tg32) that writes a planner-format directory `ledger-<session>/` (layout, a `candidates.json` with the anchors and the session's builder stages as variants with their env) so the ledger reads it like a fixture.
- `docs/sessions.md`: the three stages with minutes and what they decide.
- Verify after fetch: the sampler's decode `read_mb_s` at `disk_queue_depth_mean` lies within 20% of the loaded curve at that depth; report it (not gated).

**Report.** The local disk-curve output, the stage minutes added per session, and what each session must export.
