# 0055. Replanning wired into serve: plan from file or the executable enumerators, lowering from the plan, decode-step residual from the step ring

Status: Accepted (Phase 3a); part A implemented, part B open. Implements ADR 0042's replanner in `tetraeder serve`; builds on ADR 0040 (candidates, cost model) and ADR 0046 (scheduler).

## Context

`Replanner`, `PlanSource`, `EngineControl` and `Clock` (crates/control/src/replan.rs), the replan and plan metrics, panels and alerts exist, but nothing calls them: `serve` builds the scheduler from CLI flags and has no plan, no hardware graph and no prediction. ADR 0040 to 0042 leave three questions open: (1) how `serve` gets a `Plan` and a `HardwareGraph`; (2) how a `Plan` lowers to the load options of the libllama engine (`Placement`, `ContextConfig`) and the MoE builder (`LoadOptions`), when `planner::lower` only writes llama-bench arguments and `exit_regime` environment; (3) where the predicted step time behind `tetraeder.plan.residual` comes from and what compares it with the measured one. The scheduler also has no admission gate or in-flight count. Other tracks are editing `scheduler`, `engine` and `model`, so the wiring must not depend on changing them first.

## Decision

**(1) Plan and hardware graph.** `serve --plan <file>|auto` (absent: today's flag path, no replanner; `--plan` conflicts with the placement flags).
- Hardware: `--hw <file>` (a `HardwareGraph` TOML or JSON, typically the calibrated graph `tetraeder-plan score` writes, ADR 0040), else `profiler::discover` with GPU compute and bandwidth filled from `whatif::Priors` by name. Budget: device capacities (overridable by `--plan-vram`, `--plan-ram`), threads `min(cpu0.threads, available_parallelism)`.
- `auto` searches only the executable candidates the regret exit test measures: `enumerate` (libllama) or `enumerate_builder` (MoE builder), plus a new `enumerate_cpu` (placement `Cpu`, every unit on `cpu0`) when the graph has no GPU. Memory is sized for `--ctx`; candidates are scored with `costmodel::predict` for the `--plan-workload P:N` shape (default 2048:128, the ADR 0040 fixture shape); the lowest predicted request time wins. If prediction fails (a device without rates), the first candidate (most GPU weights, most threads) runs and replanning stays off with a warning; replanning needs a predictable graph.
- Reason: these are the only plans one box can execute, and serving from the set the exit test scores means the served prediction is the one whose regret was measured. The full multi-device search has no lowering yet.

**(2) Lowering.** `planner::lower` gets engine-neutral types (the planner stays free of ggml, engine and model): `recognize(&Plan, &ModelLayout) -> Result<(Placement, Knobs), String>` inverts `build_plan` for the four executable shapes (`Dense`, `Moe`, `Builder`, `Cpu`) and refuses anything else by naming the rule; `lower(&Plan, &ModelLayout) -> Result<Lowered, String>` gives `Lowered::Llama { n_gpu_layers, cpu_patterns, threads, ubatch, batch, op_offload }` or `Lowered::Builder { gpu: Option<u32>, expert_slots, ram_slots, threads, ubatch, op_offload }`. `to_variant_args` renders `Lowered::Llama`, so llama-bench arguments and serve options come from one function. The binary maps `Lowered` to `Placement` + `ContextConfig` or `LoadOptions` (`miss = Cpu` with a RAM tier, as in the campaign).
- `Applies` lists the knobs the serving engines honour. Today libllama has no op offload switch and the builder's serve path chunks prefill at a fixed 512, so `Applies::today().project(&Plan) -> Plan` rewrites those knobs to what runs (op offload on for libllama; builder ubatch 512). `auto` projects every candidate and deduplicates before scoring, and a `--plan` file is projected with a warning per changed knob, so the prediction describes what runs. The additive engine and scheduler fields of brief part B widen `Applies`.
- Non-default KV formats (ADR 0050) and per-role formats (ADR 0047) are refused by `lower` until their engines take them from a plan.
- The engine follows the placement: `Builder` runs the builder, `Dense` and `Moe` run libllama, `Cpu` follows `--engine` (auto: builder for `qwen3moe`).

**(3) Prediction and residual.** The predicted decode step is `1 / tg_tok_s` of the running plan's prediction (batch 1, what the one-sequence scheduler runs), recorded as `tetraeder.plan.predicted_step`. A `ResidualMonitor` in the binary reads `Telemetry::ring()` (both engines already push `StepSample` there) every `--replan-tick` (default 60 s), takes the median of new decode samples with one token, needs at least 16, and records `tetraeder.plan.residual = |measured - predicted| / predicted`. Above `max(0.2, 2 * band)` for 5 consecutive ticks it sends `Trigger::Drift`. Decode only: prefill steps are chunked and depend on context. `tetraeder.plan.regret` is the running plan's predicted time over the best candidate's on the current graph, minus 1.
- `PlannerSource` (the `PlanSource`) keeps the current graph. On `Drift` it scales the rates of the device the prediction's `binding` term names so the running plan's predicted step equals the measured one (one factor, the other terms kept), then re-picks. On other triggers it re-reads the budget (`--hw` file or discovery) and re-picks. A proposal is made only if the pick differs; gain is `1 - t_new / t_cur` (a current plan that no longer fits, by memory, thread quota or a missing device, counts as infinitely slow, so it switches whenever dwell, rate and freeze allow); switch cost is `planner::diff::DRAIN_S` plus the measured duration of the last engine load.
- SLO burn, load shift and device change come from outside the process (the SLOs live in vmalert): an admin listener (`--admin`, default `127.0.0.1:8081` with `--plan`) serves `POST /api/v2/alerts` (the Alertmanager v2 format vmalert's notifier sends; firing `TetraederSloBurnFast` or `TetraederSloBurnSlow` become `SloBurn { sli }`), `POST /admin/replan {"trigger": ...}` and `GET /admin/plan`. The same route is the fake trigger of the end-to-end tests.

**Engine control.** A `SwitchableEngine` in the binary implements the gateway's `Engine` over an `RwLock<Option<Arc<Scheduler>>>`, an admitting flag and an in-flight counter (each `generate` forwards the scheduler's channel through a task that decrements on `Finished`, `Error` or a closed receiver). While not admitting, `generate` returns `Overloaded` (429/529 with Retry-After, ADR 0028). `restart` drops the old scheduler and starts a new one from the lowered plan, checking that model id and context length are unchanged; `healthy` generates one greedy token with a timeout; `supports_knobs` is false, so every switch drains and restarts. A drain timeout cancels the remaining requests. The replanner, monitor and `PlannerSource` live on one control thread fed by a channel. Freeze windows are read in UTC.

**Metrics, panel, alert (Rule 6).** New: `tetraeder.engine.load.duration` (histogram, `outcome`) and `tetraeder.replan.admitting` (gauge, 0 or 1); panel "Admission and engine loads" in the "Planner and detector" row; alert `TetraederAdmissionStopped` (admitting 0 for 15 minutes). Residual, regret, predicted step and replans use the existing metrics, panels and alerts.

## Consequences

- Replanning only works with a predictable graph; on a new box that means a calibration campaign first, or priors with a wider residual band.
- The drift correction is one factor on one device: enough to move work off a slowed device, not a recalibration. Repeated drift on several devices still needs the campaign.
- Residuals are against the plan's workload shape, not each request's context; long contexts read as drift. The band and the 5-tick dwell absorb part of it; per-context prediction is a follow-up.
- One forwarding hop per token and a dropped prefix cache per switch.
- Until `Scheduler::shutdown` joins the worker (part B), a GPU restart can race the old backend's memory release; the rollback path covers a failed load.
- Benchmark (after the box hold lifts): switch time and requests refused per switch, paired against restarting llama-server with the equivalent flags on the same box.

## Implementation notes (part A)

Where: `planner` (`Placement::Cpu`, `enumerate_cpu`, `lower::{recognize, lower, Applies, Lowered}`, `serve` for pick, drift correction, propose and regret, `fill_from_priors` in `whatif`), `control` (`SystemClock`), `telemetry` (`StepRing::pushed`, `engine.load.duration`, `replan.admitting`), `tetraeder` (`serve` flags and `replan/`: `engine`, `monitor`, `source`, `control`, `admin`, `glue`, `start`). Panel id 70 and alert `TetraederAdmissionStopped` are in the same change. Tests: `crates/planner/tests/{serve_lowering,serve_replan_sim}.rs`, the unit tests of `replan/*`, `crates/tetraeder/tests/replan_cpu.rs`.

Choices the ADR left to the code:

- `recognize` reads the assign rules and the stage's device list, not a rebuilt plan, so it does not need the workload; it returns the plan's own knobs. A `Dense` plan with zero GPU layers keeps `[gpu0, cpu0]` in its stage, `Cpu` has `[cpu0]`.
- `candidates` falls back to `enumerate_cpu` when the GPU enumerator returns nothing (a VRAM shrink below every GPU placement), so there is always an executable plan on a box that has RAM.
- `Applies::project` changes `op_offload` only in plans that have a GPU in the stage; a CPU plan has no offload to switch.
- `predict_plan` adds an empty `cpu0` part when every layer is on the GPU: the cost model finds the embedding hop between the parts it is given, and an all-GPU plan has one part. `score::features` is unchanged.
- `correct_for_drift` scales rates by one factor, so the corrected step lands between the old prediction and the measurement when the per-op cost is a large part of the step (the per-op term is not scaled). A second drift trigger corrects the rest.
- The residual band is the largest bandwidth spread in the graph (0 for priors and files without spreads).
- Without a prediction (a device without rates) the plan runs, no replan thread and no admin listener start, and a warning says so. `--plan <file>` knobs the file leaves out are filled from the budget (threads), 512 and on, with a message each.
- The control thread and `PlannerSource` share one `Shared` (the planner's belief, the last median, the last proposal) instead of the source owning the input, so the regret gauge and the monitor read the corrected graph.
- `glue` functions return `Result` (a wrong `Lowered` variant is an error, not a panic). `SchedParts` in `serve.rs` is the part of `SchedulerConfig` that survives a reload; the loader builds a fresh config per load.
- The admin route parses bodies itself so every bad body is 400; `?wait=1` waits for the outcome without a timeout (a switch can take the drain timeout plus the load).
- No gateway change: `gateway/pump.rs` already turns a channel that closes without `Finished` or `Error` into an error for the client.
- vmalert reaches the admin listener from the compose stack only if `serve` listens beyond loopback (`--admin 0.0.0.0:8081`); the route has no authentication, so that is for trusted networks only.

Measured on the Windows dev VM (CPU, tiny MoE fixture, `replan_cpu`): `--plan auto` with `cpu_only.toml` at 2 threads picks `cpu`, threads 2, ubatch 2048 on libllama and ubatch 512 on the builder (the builder projection). A `device_change` after the file drops to 1 thread switches (drain, load, health) in 33 ms on libllama and 17 ms on the builder; greedy output over 16 tokens is identical before and after on both engines; a second trigger is rejected by the dwell. These are tiny-model numbers; the box benchmark above is still open.

Open (part B, needs `track-r-a6000-gap` and `track-s-kv-rotation` in main): the op offload switch of libllama and the builder prefill chunk from the plan (`Applies::all()`), and `Scheduler::shutdown` joining the worker so a GPU restart does not race the old backend's memory release.

## Implementation notes (part B)

Where: `ggml-sys` (`ContextParams.op_offload`), `engine` (`ContextConfig.op_offload`, default on), `scheduler` (`SchedulerConfig.prefill_chunk`, `MoeBackend::with_chunk`, `load_moe_chunked`, `Scheduler::shutdown`), `tetraeder` (`replan::{glue, start, engine, source}`). `serve` now builds its input with `Applies::all()`. Tests: `crates/scheduler/tests/replan_part_b.rs`, the unit tests of `replan::glue`, `crates/tetraeder/tests/replan_cpu.rs`.

- The plan's libllama `op_offload` reaches the context through `glue::llama_options`; the plan flag file `--plan <file>` and `--plan auto` honour it. Plain `serve` flags keep libllama's default (on).
- The builder's ubatch is both the scheduler's prefill chunk and the backend's chunk (`glue::builder_chunk`). Plain `serve` keeps `PREFILL_CHUNK` and `MOE_CHUNK`.
- `Scheduler::shutdown` lets the running job finish, then joins the worker. `SwitchableEngine::restart` calls it when it holds the last `Arc`, and waits up to 10 s for other holders; after that the worker still ends when the last holder drops.
- Cost model: `predict` skips the switch term for a plan with one part (nothing to cross), so a fully resident builder plan predicts without a link. `predict_plan` no longer adds an empty `cpu0` part; the embedding hop of such plans is not charged anymore (one alpha per token, a few microseconds).
