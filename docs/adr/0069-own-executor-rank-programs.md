# 0069. Own executor: per-rank graphs cut at collectives, issued async from one thread, one worker process per box

Status: Proposed (2026-11-15). Phase 5. Implements ADR 0016. Supersedes in part ADR 0017 (one worker process per device becomes one worker process per box, below) and amends the Phase 5 column of the capability table (ADR 0006, 0027). Companions: ADR 0068 (verdict), ADR 0070 (shapes and cost). Briefs: `docs/phase5-briefs.md`.

## Context

ADR 0016 wants an own executor because ggml's backend scheduler issues splits one after another from one host thread, while tensor parallel needs concurrent per-device subgraphs with no host sync between ranks. Today every builder computes through `ggml_sys::Sched` (`crates/ggml-sys/src/ggml/sched.rs`) on one GPU plus the CPU; the dense builder puts every weight on the primary (ADR 0066); `crates/executor` is an empty stub.

What the pin (b11976, 2e0d86be) already has, read from the vendored tree:
- **A comm API.** `ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init" | "..._comm_free" | "..._comm_allreduce_tensor")` (`ggml-backend.h:209`). CUDA implements it (`ggml-cuda.cu:1257`): NCCL via `ncclCommInitAll` (Linux default, when built with `GGML_CUDA_NCCL`), an internal all-reduce for exactly two GPUs that stages through pinned host memory with in-kernel flags or the copy engine (`allreduce.cu`, Windows default), or none, picked by `GGML_CUDA_ALLREDUCE`. All of them take every rank's tensor in one call from one thread in one process.
- **A meta backend** (`ggml-backend-meta.cpp`, `ggml_backend_meta_device`) that splits tensors per device by a callback, cuts the graph where a partial sum must be reduced, issues each device's subgraph with `ggml_backend_graph_compute_async` from one thread, then all-reduces through the comm API or a butterfly of async copies. llama.cpp's `-sm tensor` uses it; `qwen3` and `qwen3moe` are supported (`llm_arch_supports_sm_tensor`). Experts are split inside (columns of gate and up, rows of down), not placed per device.
- **Async device copies**: `ggml_backend_tensor_copy_async` on CUDA is `cudaMemcpyPeerAsync` on the source stream plus an event the destination stream waits on (`ggml-cuda.cu:2692`); without peer access the driver stages through host memory.
- `ggml_graph_add_node`, `ggml_gallocr_*`, `ggml_backend_event_*` are public.

So the plan's "all-reduce as a graph op" patch is not needed: the collective exists, and the executor calls it between subgraphs. `crates/ggml-sys/build.rs` builds with `GGML_CUDA_NCCL=OFF` today.

## Decision

### Scope

The executor runs every plan whose stages compute only on GPUs with every weight resident in VRAM, with one rank or many. Plans with CPU compute, a RAM or NVMe tier, expert streaming or op offload keep today's builders on `Sched` unchanged (single GPU plus CPU, and CPU only). A one-rank executor run is bit-equal to today's resident whole-step graph (test below), so the single-GPU resident path moves to the executor without a numeric change.

### Ranks and the step program

- **Rank** = one GPU backend, numbered in plan order (stage order, then `Stage::devices` order). All ranks live in one process and one host thread. CPU ranks exist for tests: several CPU backends with their own small threadpools stand in for GPUs in CI.
- **Rank graph.** Per step key the builder builds one ggml graph per rank holding all of that rank's nodes for the step, in one context per rank. It is allocated once with a `ggml_gallocr` on the rank's device and kept while the key repeats (ADR 0036's reuse: same tensors, same addresses, so CUDA graphs replay).
- **Boundaries.** The builder marks nodes after which a collective or a copy runs. The executor cuts each rank graph at its boundaries into **ranges** (a new `ggml_cgraph` per range, filled with `ggml_graph_add_node`; allocation stays the whole graph's, so lifetimes across ranges are correct). Each range is one `ggml_backend_graph_compute_async` call and, on CUDA, one captured CUDA graph.
- **Program**: a list of ops issued in order:
  - `Compute { rank, range }`
  - `AllReduce { tensors: [(rank, tensor)] }`: in-place sum, F32, same shape on every rank of the group.
  - `Copy { src: (rank, tensor), dst: (rank, tensor) }`: a pipeline hop.
- **Issue.** One thread walks the program and enqueues; nothing waits on the host until the step's end, where the executor synchronizes the ranks that hold outputs and the builder reads logits. Concurrency comes from the device streams, not from host threads. A CPU rank computes synchronously when issued, which serializes CI runs but keeps their results right.
- **No blocking ranges.** One issuing thread is enough only while no range blocks the host. At the pin CUDA `mul_mat_id` synchronizes its stream (reads ids to the host) only for non-F32 activations, or float expert weights outside MMVQ, MMQ and MMF (`ggml_cuda_mul_mat_id_needs_sync`, `ggml-cuda.cu:2078`); the judged models' K-quant experts take MMVQ or MMQ at every batch size, so no range blocks. A model whose ranges would block is out of scope until per-rank issue threads are added.
- **Validation** (property-tested, Rule 5): every node of every rank graph is in exactly one range; ranges of one rank appear in node order; an `AllReduce` names one tensor per rank of a group and is the boundary of each of those ranks; a `Copy` source is computed before it and its destination is read only after it.
- **Hazards.** A copy's destination is written on the source stream; the destination stream waits on the copy's event (ggml does that). A destination buffer that a still-running range of an earlier micro-batch may read gets its own tensor per in-flight micro-batch, so no write-after-read event is needed.

### Collectives

- `Comm::new(&[Backend])` asks the backend registry for the comm functions (CUDA ranks only). `AllReduce` calls `comm_allreduce_tensor`; when it returns false or there is no comm, the executor's fallback runs: copy every rank's tensor to rank 0, add there, copy the sum back (a star over `ggml_backend_tensor_copy_async` and one-node add graphs). The fallback serves CPU ranks in CI, two CUDA backends on one GPU, and any group the CUDA comm declines. Which path ran is logged once at startup and recorded per session (ADR 0068 `comm_impl`).
- `ggml-sys/build.rs` sets `GGML_CUDA_NCCL=ON` on Linux and links `nccl` dynamically when cmake found it; elsewhere OFF. Without NCCL ggml falls back to its internal path for two GPUs and the executor's fallback for more.

### Interfaces

- `plan::Topology::from_plan(&Plan, &Expanded, &HardwareGraph) -> Result<Topology, Vec<String>>`, in `plan` (no ggml) so the planner's cost features, the executor and the builders use one slicing rule (`plan::block_slice`, ADR 0070): ranks with their GPU index, stages with layer span, rank group, `tp`, `ep`, embed and head. It rejects what the executor cannot run (below) with every reason. Slices follow from ADR 0070's role table; block counts are checked where tensor shapes are known (the planner's enumeration from the model layout, the builder at load).
- Builders implement a sharded load and step: `Qwen3::load_sharded(path, &Topology, &DenseOptions)` and `Qwen3Moe::load_sharded(path, &Topology, &LoadOptions)` read only their slices from the GGUF (strided reads for column slices), allocate per-rank KV for their layers and KV heads, and build per-rank graphs plus the program for a step. The KV row allocator (`kvcache::KvPool`) stays one per model: every rank writes the same rows, so `seq_*` bookkeeping is unchanged and copies apply per rank.
- **KV snapshots and copies.** `seq_snapshot` returns the one-rank layout (per layer K then V, all KV heads in head order, the rows of the sequence), whatever the shape: a sharded builder reads each rank's head slice or layer span and assembles it on the host; `seq_restore` scatters it back. So the RAM prefix cache (ADR 0049) and its bytes do not depend on the plan, and a snapshot taken under one shape restores under another. `seq_cp` copies rows on every rank that holds KV.
- `scheduler::BatchBackend` (ADR 0065) is unchanged; the dense and MoE backends take a `Topology`. `serve --plan p.toml` and `tetraeder bench --engine dense|moe --plan p.toml` run a plan through the executor.
- `layer_limit` in `DenseOptions` and `LoadOptions` runs only the first N layers plus the head (calibration anchors, ADR 0068); logits are meaningless, timing is real.

### Capability table, Phase 5 column

| Field | Phase 5 |
| --- | --- |
| Layer split, homes on CPU or GPU | yes |
| `tp > 1` | yes, GPU stages only, `tp` divides heads and KV heads, row slices in whole quant blocks (ADR 0070) |
| `ep > 1` | yes, only with `tp = ep` (ADR 0070) |
| Stages of more than one GPU with CPU compute, tiers or streaming | no |
| Compute device other than home | experts only, single-GPU plans (as Phases 2 to 4) |

`plan::phase` and the validator get these rows with one test each; `docs/plan.md` "What each phase can execute" is updated to match.

### Worker process, one per box

The comm API, NCCL's `ncclCommInitAll`, the internal all-reduce and peer copies with stream events all need every rank in one process. Ranks in separate processes would need NCCL's per-rank init, which ggml does not expose, CUDA IPC for copies, and a ggml patch for the stream handle; on Windows there would be no collective at all. A tensor-parallel group also fails as a unit: one dead rank stops the group. So:

- `tetraeder serve` stays the parent (gateway, scheduler, prefix cache). With `--worker` it starts `tetraeder worker --plan p.toml` as a child that owns every GPU rank of the plan. The parent's backend is `RemoteBackend: BatchBackend`, which forwards each call over the child's stdin and stdout as length-prefixed frames: a `serde_json` header (already a dependency) and, for logits, raw little-endian f32 after it. No new crate.
- Pipes first: if `tetraeder.executor.ipc.duration` exceeds 2% of the decode step on the verdict box, shared memory replaces the logits frame, in `memory` (Rule 7).
- Without `--worker` the executor runs in process, as every engine does today. The verdict runs in process; the worker is about fault isolation, not speed.
- Phase 6 makes stages on other machines separate processes anyway; one process per box is the unit there too.

### Failure handling

- A failed `compute_async`, collective or copy fails the step: `BackendError::Failed`, every input back at its length and reservation (ADR 0065's rollback contract). KV rows the step wrote are overwritten later because lengths did not advance.
- A sticky CUDA error (any later call fails) or a step that exceeds its deadline (10 times the predicted step time, at least 5 s, checked by a parent-side watchdog) ends the worker: the parent fails the in-flight requests with `GenEvent::Error`, keeps the queue, restarts the worker with the same plan and readmits. Without `--worker` the process exits with an error, as today.
- Load failures (out of memory, a slice that does not divide) are startup errors naming the rank and tensor; there is no silent fallback to another plan.

### Windows

The executor and its CPU-rank tests build and run on Windows (Rule 11). ggml's internal all-reduce runs on Windows for two GPUs; nothing in Tetraeder has run on a Windows GPU (ADR 0021). ADR 0008 stands: the planner proposes tensor parallel only on Linux, and proposes expert parallel on Windows only after a Windows multi-GPU run passes the gpu2 tests.

### What stays in ggml, what changes

- Stays: kernels, `ggml_gallocr`, streams and events, the comm API and NCCL, CUDA graph capture.
- Patch queue: no all-reduce patch. One patch for expert parallel: CUDA `mul_mat_id` treats an expert id of -1 as "no expert" and writes a zero row, as the CPU kernel already does at the pin (`ggml-cpu.c`, `i02 == -1`). ADR 0070 holds its decision; `patches/0004-cuda-mul-mat-id-skip-negative.patch`, CPU and CUDA tests, a `docs/sources.md` row (own code, or a backport if b12274 or later upstream has it).

### Alternative: drive ggml's meta backend from the builders

A meta device per tensor-parallel group with a split callback built from the plan would give tensor parallel with less code. Not chosen because it gives only tensor parallel: no experts placed per device, no pipeline stages inside one device, plan-driven uneven splits only through its rounding, and the number of subgraphs and collectives (what ADR 0070 prices) is internal to it. It stays the tensor-parallel baseline (`-sm tensor`, ADR 0068) and the step-0 check below.

**Step-0 check, thresholds fixed now.** On the first 2x box (session S2 in the briefs), before the verdict harness: `tp-2` on the own executor against llama.cpp `-sm tensor` (libllama, same comm path) on both models, tg128 and pp2048, two interleaved pairs. If the own `tp-2` reaches at least `1 - b` of `-sm tensor` on both rates and models, proceed. If not, the gap is profiled (ranges per token, collective path, kernel shapes) and an addendum here decides, before the verdict session, whether tensor-parallel groups become meta devices inside the executor.

### Metrics, panel, alert (Rule 6)

- `tetraeder.executor.ranges` (counter, attribute `tetraeder.rank`): ranges launched; per decode token it should equal ADR 0070's count.
- `tetraeder.executor.collective.bytes` (counter, attributes `tetraeder.kind` = all_reduce, copy and `tetraeder.comm` = nccl, internal, fallback).
- `tetraeder.executor.ipc.duration` (histogram, worker mode) and `tetraeder.executor.worker.restarts` (counter).
- Panel "Executor: ranges per token, collective bytes, worker restarts" in `deploy/grafana/dashboards/overview.json`.
- Alert `TetraederWorkerRestarting` (`deploy/alerts/engine.yml`): more than 2 worker restarts in 30 minutes.
- `tetraeder.executor.comm` (gauge, 1, attribute `tetraeder.comm`): the all-reduce path in use. Alert `TetraederCollectiveFallback`: a plan with two or more CUDA ranks on the executor's fallback path for 10 minutes (NCCL missing or declined; tensor parallel is then far slower than predicted).
- No per-collective timing in production (ADR 0020); the probe and calibration runs time them.

### Tests first

- `ggml-sys`, `cpu`: a 6-node chain computed as three range graphs equals one compute bit for bit; `Gallocr` allocates once and recomputes with new inputs; `copy_to_async` between two CPU backends.
- `executor`, `cpu`: program validation property tests; all-reduce fallback over 2 and 4 CPU ranks equals the host sum; a copy hop between ranks; a one-rank program equals the plain graph compute bit for bit.
- `model`, `cpu`: a snapshot taken under `tp-2` equals the one-rank snapshot's layout and restores under one rank and under `layer-2`, after which decoding matches within `tol_batch`; a round trip under the same shape is bit-equal.
- `model`, `cpu`: one-rank executor logits bit-equal to today's resident whole-step graph (dense and MoE, tiny fixtures); multi-rank shapes on CPU ranks against one rank within `tol_batch` (ADR 0070).
- `executor`, `gpu` (`gpu_`, one GPU): two CUDA backends on one GPU run the fallback and the copy hop; with `GGML_CUDA_GRAPH_DEBUG=1`, a kept two-rank program captures one CUDA graph per range and replays every range on every later step (the pattern of `crates/model/tests/gpu_graphs.rs`, ADR 0036), because range graphs bypass the scheduler's `uid`; `gpu2_` and `gpu4_` tests (two and four GPUs, `TETRAEDER_REQUIRE_GPUS=<n>` makes a missing GPU fail, otherwise skip) run the comm path and every shape against one GPU within `TOL_GPU`.
- `tetraeder`, `cpu`: worker mode with a fake backend, a killed worker fails its in-flight requests, restarts, and later requests succeed.

## Consequences

- One mechanism (rank graphs cut at boundaries, three program ops) runs single GPU, layer split, tensor, expert and tensor-times-pipeline parallel; the builders decide the shape, the executor only issues.
- No ggml patch for collectives; one small patch for expert parallel. NCCL becomes a Linux build dependency when present.
- Fault isolation is per box, not per device: the serving process survives a sticky CUDA error, a single GPU's fault takes down all GPUs of the box until the restart.
- CI tests every shape's numerics on CPU ranks; only speed and the CUDA comm need boxes.
- The MoE builder's streaming paths are untouched; multi-GPU MoE is resident only in Phase 5.
