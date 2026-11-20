# Phase 5 implementation briefs

Briefs for ADR 0068 (verdict), ADR 0069 (own executor) and ADR 0070 (parallel shapes and cost). Every design decision is in those ADRs; read the ones your brief names, then AGENTS.md. A brief is executable without design decisions; if a premise looks wrong, stop and report with evidence instead of redesigning.

**For every brief.** Failing tests first, then code. Every local cargo command goes through the shared lock with one job: `L=/c/Users/pierr/AppData/Local/Temp/claude/cargo-host.lock; until mkdir "$L" 2>/dev/null; do sleep 20; done; CARGO_BUILD_JOBS=1 cargo <args>; rmdir "$L"`. Done means `cargo build --workspace`, `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`, `cargo fmt --check` pass (Rule 11; use CI on your `track-*` branch when the lock starves local builds), plus the Python tests your brief names, and the ADR your step implements gets an "Implementation notes" paragraph (what was built, any deviation). Tests that need GPUs are labelled `gpu_` (one GPU), `gpu2_` or `gpu4_` (two or four) and skip cleanly with `skip: ...` unless `TETRAEDER_REQUIRE_GPUS=<n>` (new, E1) asks for at least n GPUs (Rule 9). `unsafe` only in `ggml-sys` and `memory`, each block with a safety comment (Rule 7). No brief provisions a box; boxes are the coordinator's (table at the end). Commit only if your brief says so; push only your own branch.

## Order and parallelism

| Wave | Steps | Where | Shared files |
|---|---|---|---|
| 1 | **E1, E2 start now** (CI only); **E4 starts now** (CPU half on CI, CUDA half on session S1, one GPU); E9 (CI) | CI | E9 alone touches `tetraeder/src/serve.rs` in wave 1; E1 and E4 both edit `ggml-sys/build.rs` (separate blocks, append) |
| 2 | E3 (needs E2's `plan::Topology`), E5 (needs E1) | CI | E5 and E9 append to `telemetry/src/metrics.rs`, `overview.json`, `engine.yml`: append, do not reorder, merge textually |
| 3 | E6, E7, E8 in parallel (need E2, E3, E5; E7's CUDA expert parallel also needs E4) | CI, then S2 and S3 | E6 and E7 both add `--plan` handling to `tetraeder/src/bench.rs` and backends in `scheduler/src/{dense_backend.rs,backend.rs}` (separate files); append only. Both read, never change, `executor` |
| 4 | Verdict sessions V2, V4 (ADR 0068), run by the coordinator | boxes | |

Status (2026-11-19): waves 1 to 3 are built and on main (E1 to E9; the S2 and S3 rehearsals ran every stage). Wave 4 is open; what is left is in `docs/m2-plan.md`, section "Phase 5: where it stands".

---

## Brief E1: ggml-sys bindings for the executor (start now)

**Goal.** Expose what ADR 0069 needs from the pinned ggml, all public API: range graphs, async compute, a graph allocator, events, the comm API; build NCCL on Linux.

**Read first.** ADR 0069 (Context, Ranks and the step program, Collectives); `crates/ggml-sys/src/ggml/{sched.rs,backend.rs,context.rs,mod.rs}`, `crates/ggml-sys/build.rs`; in the vendored tree `ggml/include/ggml.h` (`ggml_graph_add_node`, `ggml_new_graph_custom`), `ggml-alloc.h` (`ggml_gallocr_*`), `ggml-backend.h` (`ggml_backend_graph_compute_async`, `ggml_backend_event_*`, `ggml_backend_tensor_copy_async`, the comm typedefs at line 209), `ggml/src/ggml-cuda/ggml-cuda.cu` 1203 to 1300 (comm init and selection).

**Decisions.**
```rust
// context.rs
impl<'a> Graph<'a> {
    /// A new graph in `ctx` with nodes [i0, i1) of `self` (same tensors, `ggml_graph_add_node`).
    /// Allocation stays with the graph `self` was allocated as.
    pub fn range(&self, ctx: &'a Context, i0: usize, i1: usize) -> Graph<'a>;
}
// backend.rs
impl Backend {
    /// `ggml_backend_graph_compute_async`: returns once enqueued on CUDA; the CPU computes now.
    pub fn compute_async(&self, g: &Graph<'_>) -> Result<()>;
    /// CUDA device index of a GPU backend, None for the CPU.
    pub fn gpu_index(&self) -> Option<usize>;
}
// alloc.rs (new): ggml_gallocr on the backend's default buffer type
pub struct Gallocr { /* ptr, backend, allocated: Option<u64> (graph id, as Sched) */ }
impl Gallocr {
    pub fn new(backend: &Backend) -> Result<Gallocr>;
    pub fn reserve(&mut self, g: &Graph<'_>) -> Result<()>;
    pub fn alloc_graph(&mut self, g: &Graph<'_>) -> Result<()>;
    pub fn is_allocated(&self, g: &Graph<'_>) -> bool;
    pub fn buffer_size(&self) -> usize;
}
// event.rs (new)
pub struct Event { /* ggml_backend_event_t */ }
impl Event {
    pub fn new(backend: &Backend) -> Result<Event>;
    pub fn record(&self, backend: &Backend);
    pub fn wait(&self, backend: &Backend);
    pub fn synchronize(&self);
}
// comm.rs (new)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommPath { Nccl, Internal, None }
pub struct Comm { /* ctx, fn pointers, backends kept alive */ }
impl Comm {
    /// None when the first backend's registry has no `ggml_backend_comm_init` (CPU) or it returns null.
    pub fn new(backends: &[Backend]) -> Option<Comm>;
    /// In-place sum of one tensor per backend, in backend order. False: ggml declined.
    pub fn all_reduce(&self, tensors: &[Tensor<'_>]) -> bool;
    /// What ggml will use, derived: `Nccl` when built with NCCL (cfg `tetraeder_nccl`), on Linux,
    /// and `GGML_CUDA_ALLREDUCE` unset or `nccl`; `Internal` with exactly 2 backends otherwise
    /// (unless the variable says `none`); else `None`.
    pub fn path(&self) -> CommPath;
}
```
- `mod.rs` re-exports `Gallocr`, `Event`, `Comm`, `CommPath`. `Tensor::copy_to_async` already exists; check it wraps `ggml_backend_tensor_copy_async(src_backend, dst_backend, ..)` and fix its doc if not.
- `Backend::gpu_available()` learns `TETRAEDER_REQUIRE_GPUS=<n>`: a helper `pub fn require_gpus(n: usize) -> bool` returns true (run) when `gpu_count() >= n`, panics when the variable asks for at least n and fewer exist, else prints `skip: needs n GPUs` and returns false.
- `build.rs`: `GGML_CUDA_NCCL=ON` on Linux, `OFF` elsewhere. After cmake, if `CMakeCache.txt` has `NCCL_FOUND:BOOL=TRUE` (or `NCCL_LIBRARY` set), emit `cargo:rustc-link-lib=dylib=nccl`, its directory as a link search path, and `cargo:rustc-cfg=tetraeder_nccl`; declare the cfg with `cargo::rustc-check-cfg`. Everything that changes cmake config is already in the build tree name; add NCCL to it.

**Tests first** (`crates/ggml-sys/tests/exec_bindings.rs`, label `cpu` unless named):
- A 6-node chain (`add`, `mul`, `scale`, `rms_norm`, `add`, `cpy`) allocated once with `Gallocr`, computed as one graph and as three `range` graphs via `compute_async` plus `synchronize`: outputs bit-equal.
- `Gallocr` allocates once; a second compute with new input data gives the new result without a new allocation (`is_allocated`).
- `copy_to_async` between two CPU backends.
- `Comm::new` on CPU backends is `None`.
- `gpu_`: `Comm::new` on two CUDA backends of one GPU returns some path or None without crashing; `Event` record and wait across two CUDA backends.
- `gpu2_`: `Comm::all_reduce` on two GPUs sums `[1..n]` and `[n..1]` to equal values on both; `path()` printed.

**Done.** Workspace checks on Windows and Linux; the CUDA compile CI jobs still link.

**Must not change.** `Sched` and every existing binding's behaviour.

---

## Brief E2: plan rules, phase 5 capabilities, executor topology (start now)

**Goal.** The plan IR, validator and capability table say what Phase 5 executes (ADR 0069 capability table, ADR 0070 "IR change, one rule"), and `plan::Topology` turns a valid plan into ranks and stages. It lives in `plan` (no ggml) so the planner, the executor and the builders share one slicing rule.

**Read first.** ADR 0027; ADR 0069 Scope, Interfaces, Capability table; ADR 0070 Shard conventions; `crates/plan/src/{plan.rs,expand.rs,validate.rs,phase.rs,testing.rs}`; `crates/profiler/src/graph.rs` (`Device::index`).

**Decisions.**
- `expand`: in a stage with `ep = 1` and `tp > 1`, each expert unit has `tp` shards on the group's devices in order; with `ep = tp`, one shard. Shard bytes as today (divided by the shard count, rounded up).
- `validate`: under phase 5 caps reject `ep > 1` with `tp != ep`; reject a stage with more than one device whose compute devices include a CPU or whose experts have a home on RAM or NVMe or slot caches; reject compute device other than home for non-expert units. Messages name the stage and the rule. Phase 4 and lower behave as today.
- `phase.rs`: `Caps` gains `multi_gpu_cpu_compute: false` for 5 (and 6), with the existing `tp_ep: p >= 5`. One test per row of ADR 0069's table.
- New `crates/plan/src/topology.rs`, re-exported from `plan` (the crate already depends on `tetraeder-profiler`):
  ```rust
  pub type RankId = usize;
  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct RankSpec { pub device: String, pub gpu_index: Option<usize> }
  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct StageSpec { pub layers: Span, pub ranks: Vec<RankId>, pub tp: u32, pub ep: u32,
                         pub embed: bool, pub head: bool }
  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct Topology { pub ranks: Vec<RankSpec>, pub stages: Vec<StageSpec>, pub n_layer: u32 }
  impl Topology {
      /// One replica only. Ranks in stage order, then `Stage::devices` order; a device in two
      /// stages is one rank. Collects every reason it cannot run.
      pub fn from_plan(plan: &Plan, x: &Expanded, hw: &HardwareGraph) -> Result<Topology, Vec<String>>;
      pub fn single(device: &str, gpu_index: Option<usize>, n_layer: u32) -> Topology;
      pub fn stage_of_layer(&self, il: u32) -> &StageSpec;
      pub fn is_single(&self) -> bool;
  }
  /// Range of `len` elements for slice `i` of `n`, whole blocks of `block`, extra blocks
  /// to ranks `(rot + j) % n` for j in 0..extra (ADR 0070). Errors if any slice is empty or
  /// `len` is not a multiple of `block`.
  pub fn block_slice(len: u64, block: u64, n: u32, i: u32, rot: u32) -> Result<std::ops::Range<u64>, String>;
  ```

**Tests first.**
- `plan`: expand of `tp = 2, ep = 1` gives 2 shards per expert, `tp = ep = 2` one; validator rejects `tp = 1, ep = 2` and `tp = 4, ep = 2` in phase 5, and in phases 1 to 4 rejects any `tp` or `ep` above 1 as today; rejects a 2-GPU stage with CPU compute or a RAM expert home. Existing property tests (Rule 5) run over generated plans that include tp and ep stages: memory never exceeded, every unit-shard computed exactly once.
- `plan::Topology`: `from_plan` on plans for `single`, `layer-2`, `tp-2`, `ep-2`, `tp2-pp2` (write them with `plan::testing` helpers over the 2x and 4x devices of a literal `HardwareGraph`): ranks, stages, embed and head flags as ADR 0070 says; every error case collects all reasons. Property test: for random stage layouts every layer maps to exactly one stage and every rank belongs to at least one stage.
- `block_slice`: 768 / 256 over 2 gives 512 and 256, rotation swaps them; over 4 errors; 25600 / 256 over 4 gives 6400 each; slices tile `[0, len)` exactly (property).

**Done.** Workspace checks; `cargo test -p tetraeder-plan`.

**Must not change.** Phase 1 to 4 validation results, the TOML format of existing plans, `expand` output for stages with `tp = ep = 1`.

---

## Brief E3: links, collective curves, multi-rank cost model, `enumerate_multi`, calibration (after E2)

**Goal.** The profiler, cost model and planner price every Phase 5 shape (ADR 0070 "Cost model", "Planner"), on simulated 2x and 4x fixtures, in CI.

**Read first.** ADR 0070 Cost model and Planner; ADR 0040, 0056 section 1 (how a feature is added with `None` bit-identical); `crates/profiler/src/{graph.rs,discover.rs,pcie.rs}`; `crates/costmodel/src/{lib.rs,predict.rs,calib.rs}`; `crates/planner/src/search/{mod.rs,cost.rs,prep.rs,solve.rs}`, `candidate.rs`, `cli.rs`; `crates/costmodel/fixtures/two_gpu_cpu_nvme.toml`.

**Decisions.**
- `profiler::LinkKind::HostStaged`; `discover` adds a GPU pair link of that kind whenever peer access is off, with the prior of ADR 0070.
- `HardwareGraph::collectives: Vec<CollectiveCurve>` (serde default empty):
  ```rust
  pub struct CollectiveCurve { pub op: CollectiveOp, pub ranks: Vec<String>, pub comm: String,
                               pub points: Vec<(f64 /*bytes*/, f64 /*seconds*/)> }
  pub enum CollectiveOp { AllReduce, Copy }
  impl HardwareGraph { pub fn collective_time(&self, op: CollectiveOp, ranks: &[String], bytes: f64) -> Option<f64>; }
  ```
  Interpolation as ADR 0070; `ranks` match as a set for all-reduce, as an ordered pair for copy.
- `costmodel`: `RunFeatures::ranks: Option<RankFeatures>` and the types and formula of ADR 0070, plus `c_range` per `GraphMode` and `c_step` as new fit parameters in `calib.rs` (fitted only when an anchor has `ranks`; absent otherwise, so committed fits are unchanged). Prior when no curve: `allreduce_time` and `transfer_time` with the link's alpha and beta. Breakdown labels `allreduce:<group>`, `copy:<a>-<b>`, `ranges`.
- `planner`:
  - `pub fn enumerate_multi(hw: &HardwareGraph, layout: &ModelLayout, hints: &Hints, phase: Phase) -> MultiCandidates` with `MultiCandidates { candidates: Vec<Candidate>, pruned: Vec<(String /*id*/, String /*reason*/)> }`, ids `single`, `layer-<g>`, `tp-<g>`, `ep-<g>`, `tp2-pp2` (ADR 0068 table), knobs: graphs on, ubatch 512, threads from the CPU quota as `planner::serve::budget_of` reads it, F16 KV, context 4096. Layer spans from the existing dynamic program. Block pruning from the layout's tensor types and shapes with `ggml` block sizes from `plan::format`.
  - `pub fn features_multi(layout: &ModelLayout, topo: &plan::Topology, knobs: &Knobs, layer_limit: Option<u32>, routing: Option<&[Vec<f64>]>) -> Result<RunFeatures, String>` builds `ranks` per shape with `plan::block_slice` for the shares (ADR 0070; EP Monte Carlo with seed 20261115 over `routing`, per layer per expert frequencies, uniform when `None`). `layer_limit` gives an anchor's features (first N layers plus the head, ADR 0068). A one-rank `Topology` gives `ranks: None` and today's features.
  - `tetraeder-plan calibrate --hw hw.json --layout-dir layouts/ --anchors anchors/ --out calib.json`: reads the anchor runs (`anchors-<model>-r<k>.json`, llama-bench rows with `plan_id`, `layer_limit`), builds their features with `features_multi`, fits with `calib.rs` (existing parameters plus `c_range`, `c_step`), writes the fit and its residuals.
  - search skeletons: `P2p` or `HostStaged` links; EP group `tp = ep = g`; the existing Linux-only prune stays.
  - CLI: `tetraeder-plan candidates --multi --hw hw.json --layout l.json --out candidates.json --plans-dir plans/` and `tetraeder-plan predict --hw hw.json --layout l.json --calib calib.json --candidates candidates.json --out predict.json` (fields of ADR 0068's predict file; `sha256` of the `hw.json` bytes).
- Fixtures `crates/costmodel/fixtures/two_3090_hoststaged.toml` and `four_3090_hoststaged.toml`: RTX 3090 priors as the existing 3090 calibration (`bench/results/20261103/planner-moe/`), host links gen4 x16 (x8 for two GPUs of the 4x fixture behind one lane group), prior collective curves from the ring formula at 10 us and 12 GB/s.

**Tests first.**
- `calibrate`: on synthetic anchors generated from known parameters on `two_3090_hoststaged`, the fit recovers `c_range` and `c_step` within 5%.
- `profiler`: a discovered topology without peer access yields `HostStaged` links with the prior; with peer access `P2p`.
- `costmodel`: `ranks: None` predictions bit-identical on every existing fixture and committed fit (compare against values recorded before the change); a two-phase hand-built `RankFeatures` gives the formula's value; uneven parts take the max; `collective_time` interpolates and extrapolates as specified.
- `planner`, on both new fixtures and the existing ones (Rule 4): `enumerate_multi` for the Qwen3-32B and Qwen3-30B-A3B layouts (write literal layouts in `layout.rs` test helpers from the GGUF facts in ADR 0070) gives exactly ADR 0068's candidate ids, with `tp-4` pruned for the 30B ("3 blocks over 4 ranks"); deterministic across runs; predictions finite for every candidate; the predicted `pp_tps`, `tg_tps` per candidate and fixture are committed as a snapshot test (a change must update the snapshot on purpose).
- EP Monte Carlo: deterministic, and `E[max]` for uniform frequencies at top 8 over 2 ranks lies between 4 and 8.

**Done.** Workspace checks; `cargo test -p tetraeder-profiler -p tetraeder-costmodel -p tetraeder-planner`; the planner exit tests over committed results (ADR 0040, 0056) unchanged.

**Must not change.** Any prediction with `ranks: None`; enumerations of `enumerate` and `enumerate_builder`; committed result files.

---

## Brief E4: patch 0004, CUDA `mul_mat_id` skips id -1

**Goal.** ADR 0070 "Patch 0004": every CUDA `mul_mat_id` path treats id -1 as no expert, reads no weights for it and writes a zero row, as the CPU kernel does.

**Read first.** ADR 0070 Patch 0004; ADR 0011; `patches/README.md`, `patches/series`; vendored `ggml/src/ggml-cpu/ggml-cpu.c` (`ggml_compute_forward_mul_mat_id`, the `i02 == -1` branch), `ggml/src/ggml-cuda/{ggml-cuda.cu (ggml_cuda_mul_mat_id),mmvq.cu,mmq.cu,mmid.cu,mmf.cu}`. First check upstream llama.cpp master and b12274 for the same change; if present, backport it and say so in `docs/sources.md`.

**Decisions.**
- `patches/0004-cuda-mul-mat-id-skip-negative.patch`, appended to `series`, made per `patches/README.md` and applied from a clean pin by `apply.sh` and `apply.ps1`.
- Semantics: for each (expert slot, token) with id -1, the output row is zeros; no work is scheduled for it in MMVQ, MMQ (`mm_ids_helper` excludes it, and the zero rows are written), MMF and the generic path that gathers rows per expert.
- `docs/sources.md` row (own code or backport with upstream commit and MIT license).
- `crates/ggml-sys/build.rs` emits `cargo:rustc-cfg=tetraeder_patch_0004` when `patches/series` lists the patch (declared with `rustc-check-cfg`), so E7 can refuse CUDA expert parallel without it.

**Tests first** (`crates/ggml-sys/tests/mul_mat_id_skip.rs`):
- `cpu`: random ids with 30% -1 at 1, 8, 64, 512 tokens, weights Q4_K, Q6_K, Q8_0, F16: rows with -1 are exactly 0, others equal a reference built from per-expert `mul_mat`.
- `gpu_`: the same on CUDA against the CPU within `TOL_GPU`-equivalent (`5e-3` relative), zero rows exact; timing check at 512 tokens with 0%, 50% and 90% -1 ids (median of 20): time decreases monotonically, recorded to JSON.

**Done.** CPU test on CI; CUDA compile jobs green; the `gpu_` tests pass on session S1 (the coordinator runs it; report the JSON).

**Must not change.** Results for ids without -1 (bit-equal on CUDA, checked by the existing `gpu_` parity tests on S1).

---

## Brief E5: executor core (after E1)

**Goal.** `crates/executor` runs a step program (ADR 0069 "Ranks and the step program", "Collectives", metrics).

**Read first.** ADR 0069 in full; E1's bindings; `crates/telemetry/src/metrics.rs` (pattern for a new instrument), `deploy/grafana/dashboards/overview.json`, `deploy/alerts/engine.yml`, `tests/test_alert_rules.py`.

**Decisions.**
```rust
pub enum RankDevice { Gpu(usize), Cpu { threads: i32 } }
pub struct Ranks { /* backends, comm: Option<Comm>, comm_path: &'static str */ }
impl Ranks {
    pub fn open(devices: &[RankDevice]) -> Result<Ranks, ExecError>;
    pub fn backend(&self, r: RankId) -> &Backend;
    pub fn len(&self) -> usize;
    /// "nccl", "internal" or "fallback"; logged once at open.
    pub fn comm_path(&self) -> &'static str;
}
pub enum Op<'a> {
    Compute { rank: RankId, range: usize },
    AllReduce { tensors: Vec<(RankId, Tensor<'a>)> },
    Copy { src: (RankId, Tensor<'a>), dst: (RankId, Tensor<'a>) },
}
pub struct RankGraph<'a> { pub rank: RankId, pub graph: Graph<'a>, /// node index ending each range, last = n_nodes
                           pub cuts: Vec<usize> }
pub struct Program<'a> { pub graphs: Vec<RankGraph<'a>>, pub ops: Vec<Op<'a>>, pub outputs: Vec<RankId> }
pub fn validate(p: &Program<'_>, n_ranks: usize) -> Result<(), Vec<String>>;
pub struct Executor { /* ranks, one Gallocr per rank, cached range graphs per (graph id, cuts), counters */ }
impl Executor {
    pub fn new(ranks: Ranks) -> Executor;
    pub fn ranks(&self) -> &Ranks;
    /// Allocates rank graphs not allocated yet (kept while the same graphs come back, as Sched).
    pub fn prepare(&mut self, p: &Program<'_>) -> Result<(), ExecError>;
    /// Validates (debug builds), issues every op in order, synchronizes the output ranks.
    pub fn run(&mut self, p: &Program<'_>) -> Result<(), ExecError>;
    pub fn take_counters(&mut self) -> ExecCounters; // ranges per rank, collective bytes by kind and path
}
pub enum ExecError { Invalid(Vec<String>), Ggml(GgmlError) }
```
- CPU ranks: each `Backend::cpu(threads)`; tests use 1 or 2 threads per rank.
- Fallback all-reduce (ADR 0069): star through rank 0 with a scratch tensor per rank allocated once per program in its own buffer, one-node `add` graphs; F32 only (error otherwise).
- `AllReduce` calls `Comm::all_reduce` first when `comm` exists and every tensor is on a CUDA rank.
- Metrics (Rule 6): `tetraeder.executor.ranges` (counter, `tetraeder.rank`), `tetraeder.executor.collective.bytes` (counter, `tetraeder.kind`, `tetraeder.comm`); `tetraeder.executor.comm` (gauge, 1, `tetraeder.comm`); panel "Executor: ranges per token, collective bytes, worker restarts" in `overview.json` (the restarts query is E9's metric; add the panel here with both queries); alert `TetraederCollectiveFallback` in `deploy/alerts/engine.yml` (ADR 0069).
- `tetraeder comm-probe --gpus 0,1[,2,3] --out calib-comm.json`: in `crates/tetraeder/src/comm_probe.rs`, all-reduce over every group the ADR 0068 candidates use (all GPUs; on 4, also each pair {0,1}, {2,3}) and copies per pair, sizes 4 KiB to 64 MiB in powers of 4, 20 repeats, median, with `Backend::synchronize` around each timed op; output matches `CollectiveCurve` (E3) plus `comm_path`.

**Tests first** (`crates/executor/tests/`, label `cpu` unless named):
- `validate`: property tests over random programs (Rule 5): each node in exactly one range, ranges of a rank in order, a collective's tensors are boundaries of their ranks, a copy's source computed before and its destination read after; every violation reported.
- One rank, one range: bit-equal to `Gallocr` plus `compute` of the same graph.
- 2 and 4 CPU ranks: all-reduce of distinct inputs equals the host sum exactly (small integers in F32); a pipeline program (rank 0 range, copy, rank 1 range) equals one graph on one backend bit for bit.
- Reuse: a second `run` with new inputs and the same program allocates nothing (`prepare` counter).
- `gpu_`: the same programs on two CUDA backends of one GPU; CUDA graph replay of kept range graphs as ADR 0069 Tests first says (`GGML_CUDA_GRAPH_DEBUG=1`, count captures and replays from ggml's log as `crates/model/tests/gpu_graphs.rs` does): one capture per range, a replay of every range on each later step. If replays do not happen, report it with the log; do not change ggml.
- `gpu2_`, `gpu4_`: all-reduce through `Comm`; `comm-probe` writes a curve with increasing times.

**Done.** Workspace checks; `uv run pytest tests/test_alert_rules.py` if dashboards or alerts changed.

**Must not change.** `ggml-sys` public behaviour beyond E1.

---

## Brief E6: sharded dense builder (after E2, E3, E5)

**Goal.** `Qwen3` runs `single`, `layer-g`, `tp-g` and `tp2-pp2` plans on the executor (ADR 0070 shapes and role table), and `tetraeder bench --engine dense --plan` measures them.

**Read first.** ADR 0066, 0069 (Interfaces, Scope), 0070 (Shard conventions, Tensor parallel, Layer split, Tensor times pipeline, Numerics); `crates/model/src/{qwen3.rs,attn.rs,gguf.rs}`, `crates/scheduler/src/dense_backend.rs`, `crates/tetraeder/src/{bench.rs,serve.rs}`, `crates/model/tests/{dense_parity.rs,multi_seq.rs,common/mod.rs}`.

**Decisions.**
- `attn.rs`: `pub(crate) fn attention_out(..) -> Tensor` returns `wo(attn)` without the residual and takes the rank's local head counts from the weights' shapes; `attention` becomes `attention_out(..) + x`, op order unchanged (bit-equal, the existing tests are the check).
- `Qwen3::load_sharded(path: impl AsRef<Path>, topo: &Topology, opts: &DenseOptions) -> Result<Qwen3>`: one `Ranks` and `Executor` inside; per rank its weights (column slices read with strided reads, row slices with `plan::block_slice`), its KV tensors for its layers and KV heads; one `KvPool`. `Qwen3::load` stays and becomes `load_sharded` with `Topology::single` when the backend is a GPU; the CPU backend keeps `Sched`. `DenseOptions::force_sched: bool` (default false, for tests) keeps the old GPU path so the bit-equal check can run on a box.
- Step: per rank one graph per key (n_tokens, n_kv, outputs), boundaries after each `wo` and `ffn_down` partial (all-reduce), after a stage's last layer (copy to the next stage, one hop tensor per in-flight micro-batch), head by vocab slice per rank, logits read back per output rank and concatenated in vocab order. Kept graphs as `reuse_step_graph`.
- `DenseOptions::layer_limit: Option<u32>` (ADR 0069, calibration anchors).
- `DenseBackend::with_topology`; `tetraeder bench --engine dense --plan p.toml` and `serve --engine dense --plan p.toml` build the topology from the plan, the hardware graph (`profiler::discover`) and `expand`; llama-bench JSON rows as today plus `plan_id`, `ranks`, `comm_path`, `ranges_per_token`.

**Tests first.**
- `cpu`, tiny dense fixture (`tests/fixtures/tiny-qwen3-f16.gguf`, 2 layers, 4 heads, 2 KV heads): `load_sharded` with one CPU rank bit-equal to `Qwen3::load` on the CPU backend through `Sched` (logits of every position and 12 greedy steps); `tp-2` (2 CPU ranks), `layer-2` (split after layer 1), and on a 4-layer variant of the fixture (`cargo xtask tiny-dense --layers 4`, add the flag) `tp2-pp2` and layer splits at layers 1, 2, 3 (Rule 2 split points): top-5 within `tol_batch`, greedy equal or ADR 0043 near-tie. Three sequences batched in one step on `tp-2` against each alone. Snapshots per ADR 0069 "KV snapshots and copies": a `tp-2` and a `layer-2` snapshot equal the one-rank layout byte for byte on layer 0 (whose inputs are identical) and in length and layout everywhere; restoring across shapes then decoding matches within `tol_batch`; `seq_cp` under `tp-2` equals two separate prefills of the prefix.
- `cpu`: Qwen3-0.6B Q8_0 golden on `tp-2` CPU ranks: tokens equal to the golden file except reported near-ties (skips without the model unless `TETRAEDER_REQUIRE_MODELS=1`).
- `gpu_`: one GPU rank bit-equal to `force_sched` on Qwen3-8B Q4_K_M.
- `gpu2_`, `gpu4_` (`crates/model/tests/gpu_sharded.rs`): every shape against `single` on Qwen3-8B Q4_K_M within `TOL_GPU`, 64 greedy tokens equal or near-tie; ranges per decode token equal ADR 0070's count (`2 * layers + head` for tp); `layer_limit` runs.

**Done.** Workspace checks; GPU tests pass on S2 and S3 (coordinator runs them; report numbers).

**Must not change.** Single-sequence and batched numerics of the CPU path; `BatchBackend`; `DenseOptions` defaults.

---

## Brief E7: sharded MoE builder, resident (after E2, E3, E5; CUDA EP after E4)

**Goal.** `Qwen3Moe` runs `single`, `layer-g`, `tp-g`, `ep-g`, `tp2-pp2` plans with every expert resident on the executor (ADR 0070), and `tetraeder bench --engine moe --plan` measures them.

**Read first.** ADR 0033, 0036 (whole-step graph), 0065 "Backends as built", 0069, 0070 in full; `crates/model/src/{qwen3moe.rs (load 543, build_step 2413),experts.rs,attn.rs}`, `crates/scheduler/src/backend.rs` (`MoeBackend`); E6's diff for the dense pattern (attention sharding, head, pipeline hop: reuse its helpers, do not copy them).

**Decisions.**
- `Qwen3Moe::load_sharded(path, topo: &Topology, opts: &LoadOptions) -> Result<Qwen3Moe>`; allowed only without a streamer, RAM or NVMe tiers, op offload or CPU experts (error naming the option otherwise). Single-GPU resident loads go through it with `Topology::single`, bit-equal; every streamed or CPU path keeps `Sched`.
- TP: experts cut per ADR 0070 (gate and up columns, down rows with `block_slice` and rotation `rot = layer`); router and its top-k on every rank.
- EP (`tp = ep = g`): rank i loads experts `[i*E/g, (i+1)*E/g)` as its slot tensors; a per-rank I32 input `map_i[E]` (set once at load), `local = get_rows(map_i, ids)`, `mul_mat_id` with `local`, weighting and sum as the resident path, then the all-reduce. Keep the topk-moe fusion order of ADR 0036 (router weights expanded before the expert FFN); `graph_order`-style test per rank.
- On CUDA, EP refuses to load until patch 0004 is applied (check the patch series marker the build exposes; E4 adds `cfg(tetraeder_patch_0004)` from `build.rs` reading `patches/series`).
- `LoadOptions::layer_limit: Option<u32>`; `MoeBackend::with_topology`; bench and serve `--plan` as E6.

**Tests first.**
- `cpu`, tiny MoE fixture and the synthetic Q8_0 MoE (`tests/common/synth.rs`): one rank bit-equal to the current resident whole-step graph; `tp-2`, `ep-2`, `layer-2`, and `ep-4`, `tp2-pp2` on a 4-layer synthetic: within `tol_batch`, greedy equal or near-tie; EP with skewed routing (all top-k on rank 0's experts) still correct; batched 3 sequences on `ep-2`; snapshot and restore across `ep-2`, `tp-2` and one rank as E6's test.
- `cpu`: every rank's step graph keeps the CUDA fusion order (`SOFT_MAX, RESHAPE, ARGSORT, VIEW, GET_ROWS, ...` before the expert FFN).
- `gpu2_`, `gpu4_` on Qwen3-30B-A3B Q4_K_M: every shape against `single` within `TOL_GPU`, 64 greedy tokens; `tp-4` refused at load with the block reason.

**Done.** Workspace checks; GPU tests on S2, S3.

**Must not change.** Every streamed, segmented and CPU-expert path and its tests; `extract_bits`.

---

## Brief E8: verdict harness (after E3; the box script needs E5 to E7 to run)

**Goal.** The session of ADR 0068 runs from one script and ends in `verdict.json`.

**Read first.** ADR 0068 in full; ADR 0065 Files and `scripts/{phase4_box.sh,phase4_summary.py}`, `tests/test_phase4_summary.py`, `tests/test_phase4_box.py`; `scripts/cotenant_guard.py`; AGENTS.md "GPU work on Vast.ai" (flock guard, detached runs, incremental results).

**Decisions.**
- `bench/spec/phase5.toml`: models (`hf` for Qwen3-32B Q4_K_M and Qwen3-30B-A3B Q4_K_M), workload, engines `tetraeder-exec`, `llama-bench` (baseline build), `ik-llama-bench`, band class pattern `<n>x-<gpu-class>`.
- `scripts/phase5_box.sh <session> <2|4>`: guarded by `exec 8> /root/phase5.lock; flock -n 8 || exit 0`; stages in order, each skipped when its output exists (resume): `env` (ADR 0068 env fields), `build`, `models`, `discover` (`hw.json`), `enumerate` (`tetraeder-plan candidates --multi`), `probe` (`tetraeder comm-probe`, merged into `hw.json`), `anchors` (truncated `single` and `tp-2`, two rounds), `calibrate`, `predict` (writes `predict-*.json` and its `sha256` before any candidate), `rounds` (three rounds, rotation per ADR 0068, candidates and baselines, A/A last), `vllm` (optional, skipped when under 25 minutes of the 3 h budget remain), `summary`. Throttle reasons and `nvidia-smi` per run to `gpu-<run>.txt`. Results written after every run.
- `scripts/phase5_summary.py <dir>` writes `verdict.json` with exactly ADR 0068's fields and rules (winner set, within 15%, baseline gate, clean rules, A/A).
- `tests/test_phase5_summary.py`: synthetic result directories for: all pass; pick outside W; one candidate at 16% error; baseline faster beyond the band; an unclean run rerun and passing; A/A outside the band (no verdict); prediction file newer than a candidate run (invalid); pruned candidates listed and not counted.
- `tests/test_phase5_box.py`: stage order, resume skipping, lock guard present, predict before rounds (pattern of `test_phase4_box.py`).

**Done.** `uv run pytest tests/test_phase5_summary.py tests/test_phase5_box.py tests/test_stage_names.py tests/test_taxameter_config.py`; `shellcheck` clean if the repo uses it.

**Must not change.** Phase 4 scripts and their tests.

---

## Brief E9: worker process, one per box (independent)

**Goal.** ADR 0069 "Worker process, one per box" and "Failure handling": `serve --worker` runs the backend in a child process behind `RemoteBackend: BatchBackend`, restarts it on a sticky error or a missed deadline.

**Read first.** ADR 0065 Backend trait; ADR 0069 Worker process, Failure handling, Metrics; `crates/scheduler/src/{backend.rs,scheduler.rs}`, `crates/scheduler/tests/fake/mod.rs` (`FakeKv`), `crates/tetraeder/src/{main.rs,serve.rs}`.

**Decisions.**
- `crates/tetraeder/src/worker.rs`: subcommand `tetraeder worker` with the same model, engine and plan arguments as `serve`, serving one `BatchBackend` over stdin and stdout. Frame: u32 little-endian length, a `serde_json` header (call name and arguments, or the result or `BackendError`), then for `decode` replies the logits as raw little-endian f32 (`Outputs` spans in the header).
- `crates/tetraeder/src/remote_backend.rs`: `pub struct RemoteBackend` implements `BatchBackend` by forwarding each call; it spawns the child, keeps its last answer for `n_vocab`, `n_seq_max`, `kv_capacity` and the like.
- Supervision in `RemoteBackend`: a call that gets EOF, a broken pipe, a reply `Failed` that the child marks `sticky: true` (any CUDA error after which the next call also fails), or no reply within the deadline (`max(5 s, 10 x predicted step time)`, predicted from `--plan`'s prediction when given, else 5 s) kills the child, returns `BackendError::Failed`, and restarts it with the same arguments. Sequences are gone after a restart: the backend reports every live sequence as length 0, and the scheduler's existing `Failed` path fails their requests; the queue survives. Restart backoff 1 s, 2 s, 4 s, then give up and exit non-zero after 5 restarts in 10 minutes.
- `serve --worker` (default off).
- Metrics: `tetraeder.executor.ipc.duration` (histogram, per call), `tetraeder.executor.worker.restarts` (counter); alert `TetraederWorkerRestarting`: more than 2 restarts in 30 minutes, in `deploy/alerts/engine.yml`.

**Tests first** (`crates/tetraeder/tests/worker.rs`, label `cpu`): `tetraeder worker --fake` (a `FakeKv` backend, exposed behind a hidden `--fake` flag) behind `RemoteBackend`: the scheduler's batching tests P1 and P4 (ADR 0065) pass through the pipe; killing the child mid-step fails that step's requests, later requests succeed after the restart; a child that hangs is killed at the deadline; the restart cap ends the process.

**Done.** Workspace checks on Windows and Linux; `uv run pytest tests/test_alert_rules.py`.

**Must not change.** `serve` without `--worker`; `BatchBackend`.

---

## Brief V5: cost model 5, decode at measured clocks (ADR 0082)

**Goal.** Layer-N decode on 3 or more GPUs is priced at the low clock its GPUs get where single boosts; every other prediction stays as cost model 4. Tech tree node `planner-cost-model-v5`; branch `track-cost-model-v5`.

**Read first.** ADR 0082 (all of it; it holds every decision and number), ADR 0080 (how v4 was gated by a graph field), ADR 0070 "Cost model" and History; `crates/costmodel/src/{predict.rs,calib.rs,lib.rs}` (`predict_ranks`, `part_time_s`, `coll_time`), `crates/profiler/src/graph.rs` (`RangeCosts`, `HardwareGraph`), `crates/planner/src/{multi.rs,multi_cli.rs}` (`candidate_for`, `features_multi`, `calibrate`), `crates/planner/tests/costmodel_v2.rs`, `scripts/costmodel_replica.py`, `scripts/phase5_box.sh` (stage anchors, `run_one`), `scripts/s3e_diag_box.sh` (part C: 20 ms clock log and window file per run).

**Decisions (from ADR 0082; do not reopen).**
- `HardwareGraph::decode_clocks: Option<DecodeClocks>` (serde default `None`, skipped when `None`):
  ```rust
  pub struct DecodeClocks {
      pub low_mhz: f64,                     // layer anchor, head-free GPUs, timed-tail median
      pub busy_mhz: BTreeMap<String, f64>,  // per layout.model: its single anchors' tail median
      pub duty_threshold: f64,              // 0.32
  }
  ```
  `None`, a missing model key, or `busy_mhz <= low_mhz + 15` gives `k = 1`: predictions bit-identical to cost model 4.
- `RankFeatures::model: Option<String>` (serde default `None`), set by `features_multi` to `layout.model`. `None` gives `k = 1`.
- In `predict_ranks`, decode step only: compute the step at today's rates and each GPU's busy time `b_r` = sum over phases of `repeat x (part_time_s(r) + coll_time(after) when r is in the all-reduce group)` (copies do not count); `d_r = b_r / T`. For every GPU with `d_r < duty_threshold`, multiply the clock-bound terms of its decode `part_time_s` by `k = busy_mhz[model] / low_mhz`: `c0 x ops`, the routed-expert roofline `max(2 p_exp m / F_exp, W_exp / B)` and the slot bytes `S m / B`; the dense and KV rooflines stay. Then recompute the decode step once (no fixed point). Collectives, `c_range`, `c_step` and prefill are unchanged. Report the factor per GPU in the prediction's diagnostics (not in the step sum).
- The planner search (`stage_time`) uses the same helper; no other search change.
- `MODEL_VERSION` 5; the replica gets `--version 5` (default) and `--clocks FILE` (`decode-clocks.json` below); versions 1 to 4 stay reproducible.
- New anchor id `layer-<g>@L<n>`: a layer-g plan whose first g-1 stages hold `n/g` layers each and the last stage the rest of the model, run with `--layer-limit n`, n = L/2 (30B on 4 GPUs: 6/6/6/30, limit 24). `candidate_for` parses the `@L<n>` suffix and builds this cut. Only on boxes with g >= 3, for the session's first MoE model (else its first model); tg128, `-r 10`, two rounds, merged into `anchors-<model>-r<k>.json` like the others. `calibrate` never fits on it: it reports its v5 tg residual as `held_out` in `calib.json` and the log.
- Box script: every anchor run (old and new) gets `nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active --format=csv -lms 20` in the background and a window file (`date +%s.%N` before and after), as `s3e_diag_box.sh` part C; files `phase5/clocks-<run>.csv`, `phase5/window-<run>.txt`. Also log candidate runs the same way (tp2-pp2 decode clocks are an open question of ADR 0082).
- `scripts/decode_clocks.py` (standard library, `python -I`): reads those files and the anchor JSONs, cuts each run to its timed decode tail (the last `reps x n_gen / tg_tok_s` seconds before the window end), takes per-GPU medians, and writes `decode-clocks.json` in the `DecodeClocks` shape, with per-run medians and the reason when it withholds a value (under 25 tail samples on a GPU, or 0x8, 0x20, 0x40, 0x80 in a tail). `tetraeder-plan calibrate --clocks decode-clocks.json` copies it into the graph; without the flag or the file, `decode_clocks` stays `None`.
- Rule 6: gauge `tetraeder_plan_decode_clock_hertz{device}` (the clock v5 assumes for each GPU of the loaded plan; absent when `decode_clocks` is `None`), exported where `tetraeder_plan_residual_ratio` is; a line beside `tetraeder_host_gpu_clock_hertz` in the GPU clocks panel of `deploy/grafana/dashboards/overview.json`; alert `TetraederDecodeClockOffPlan` in `deploy/alerts/slo.yml`: measured over assumed clock off by more than 5% for 15 minutes while the plan decodes; covered in `tests/test_alert_rules.py`.

**Tests first.**
- `costmodel`: a graph without `decode_clocks` and features without `model` predict bit-identically to today on every fixture and the committed v4 graphs; hand-built 4-stage features with duties 0.2 to 0.3 get `k` on decode only (prefill identical); a GPU at `d_r = 0.33` gets none; `busy_mhz < low_mhz` and a missing model key give `k = 1`.
- Fixtures (Rule 4): `four_3090_hoststaged` with a `decode_clocks` block (busy 1905 for the 30B layout and 1620 for the 32B, low 1695; the 32B gets k = 1 by the clamp) and without; snapshot both. `two_3090_hoststaged` with the block: every prediction equal to without.
- `planner`: `candidate_for(.., "layer-4@L24")` gives spans 6/6/6/30 on the 30B layout and is rejected on 2 GPUs; `calibrate` gives the same fit with and without the layer anchor present.
- Replica and Rust within 1% on committed version-5 graphs (extend `costmodel_v2.rs`); `tests/test_costmodel_replica.py` keeps the version 1 to 4 numbers.
- `decode_clocks.py` on the `bench/results/s3e-20261120` part C files (20 ms logs and windows): single full 1905 on gpu0, layer-4 full 1695 on all four, L12 and L24 as in ADR 0082's table; a unit test on a small synthetic CSV for the tail cut and the withholding rules.

**Offline check (the done criterion, ADR 0082 "The check").** Write `bench/results/s3c-20261119/costmodel-v5/` with `decode-clocks-host20210.json` (busy 1905 for both models and low 1695, from s3e part C; ADR 0082 explains why the 32B gets the same upper-bound factor) and a README table: v4 and v5 `e_pp`, `e_tg` for every candidate of s2f, s2g, s3, s3b, s3c (host-20210 clocks for the 4-GPU sessions, none for the 2-GPU ones), s3f2 (1590 and 1590, from its `gpu-samples.csv`) and v2-20261120 (branch `box-v2-20261120`, no clocks needed on 2 GPUs). Pass: items 1 to 4 of the check. The ADR's prototype numbers are what to expect; a larger gap is a finding to report, not to tune away.

**Done.** Workspace checks (Rule 11), the Python tests named above, the offline table committed, "Implementation notes" in ADR 0082, tech tree node `planner-cost-model-v5` updated with the table as evidence (run the tree's build and check). Report the table and any deviation.

**Must not change.** Any v4 prediction on graphs without `decode_clocks`; the fit (the layer anchor is held out); committed result files other than the new `costmodel-v5/` directory. Until the owner accepts ADR 0082 the new anchor runs only behind `LAYER_ANCHOR=1` (default off) in `phase5_box.sh`; the clock logs are on by default.

---

## Box sessions

All on Vast through taxameter (ADR 0021, 0029), Linux, RTX 3090 class (the image's CUDA builds are sm_86), commit before provisioning, results fetched and verified before destroy. Prices are estimates for Vast RTX 3090 offers, not quotes.

**Prerequisite (done 2026-11-17).** taxameter 0.3.3 has `--num-gpus N` on `offers` and `create` (its ADR 0011); a 2x or 4x box is rented with it.

| Session | When | Box | What | Hours | USD (box) | Ran (sessions in `bench/results/`) |
|---|---|---|---|---|---|---|
| S0 | now, no Tetraeder code | 2x RTX 3090 | baselines only: `llama-bench -sm` layer, row and tensor of the baseline build on both models, pp2048 and tg128, two interleaved rounds; record the P2P matrix and the comm path `-sm tensor` logs. Tells before E5 to E7 whether tensor parallel pays on PCIe 3090s | 1 | 0.4 to 0.6 | Not run as its own session; the S2 step-0 checks measured `-sm` modes beside own `tp-2` |
| S1 | after E4 and E5 | 1x RTX 3090 | E4 `gpu_` tests and timing JSON; E1, E5 `gpu_` tests (two backends on one GPU); existing `gpu_` parity tests unchanged with patch 0004 | 1 | 0.2 to 0.3 | Done: [`20261116gpu3`](../bench/results/20261116gpu3/README.md) (0004 timing flat, test replaced, ADR 0070), [`20261116gpu5`](../bench/results/20261116gpu5/README.md) (one-GPU parity clean); earlier [`20261115gpu`](../bench/results/20261115gpu/README.md) led to ADR 0071 |
| S2 | after E6, E7, E8 | 2x RTX 3090 | `gpu2_` tests; `comm-probe`; ADR 0069 step-0 check (own `tp-2` against llama.cpp `-sm tensor`, both models, two pairs); `phase5_box.sh` rehearsal with one round | 2.5 | 0.9 to 1.5 | Done: step 0 passes in [`s2f-20261118-step0`](../bench/results/s2f-20261118-step0/README.md); rehearsals [`s2f-20261118`](../bench/results/s2f-20261118/README.md) and [`s2g-20261118`](../bench/results/s2g-20261118/README.md) (verdict null); earlier `s2-20261117-step0`, `s2b`, `s2c`, `s2d`, `s2e` stopped at gates or calibrate |
| S3 | after S2's fixes | 4x RTX 3090 | `gpu4_` tests; probe; rehearsal of D4 and M4, one round | 2 | 1.4 to 2.4 | Done: [`s3-20261119`](../bench/results/s3-20261119/README.md) (gates pass, verdict null), [`s3b-20261119`](../bench/results/s3b-20261119/README.md) (two rounds: D4 null, M4 fails the baseline gate and prediction) |
| V2 | after S3 | 2x RTX 3090 | verdict cases D2, M2 (ADR 0068) | 2.5 | 0.9 to 1.5 | Not yet |
| V4a | after V2 | 4x RTX 3090 | verdict case D4 | 2 | 1.4 to 2.4 | Not yet |
| V4b | after V2, may run beside V4a on a second box | 4x RTX 3090 | verdict case M4 | 2 | 1.4 to 2.4 | Not yet |

Traffic: about 45 GB of models per box (Qwen3-32B and Qwen3-30B-A3B, plus Qwen3-8B for S2 and S3 tests), 0.6 to 1.2 USD per box at 0.013 to 0.026 USD/GB. Total with one rerun of S2 or a verdict session: about 15 to 18 box hours, 11 to 20 USD. 2x RTX 3090 boxes fit the 0.90 USD/h cap; 4x RTX 3090 offers run about 0.70 to 1.20 USD/h, and the owner allowed up to 1.20 USD/h all-in for a 4-GPU box (2026-11-15, `AGENTS.md`, `--max-price 1.20`); V4a and V4b side by side use both box slots and must stay within the 1.80 USD/h account cap together. Verdict and rehearsal boxes run nothing else (a quiet box); S1 can run beside any other project's box.
