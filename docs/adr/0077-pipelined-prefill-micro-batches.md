# 0077. Pipelined prefill: micro-batches in one rank program (GPipe order)

Status: Proposed (2026-11-18). Phase 5. Amends ADR 0069 ("Rank graph": a rank graph may hold several micro-batches of one step) and implements its "Hazards" paragraph; nothing of 0069 is superseded. Companions: ADR 0068 (verdict), ADR 0070 (layer split, cost model), ADR 0075 (ranges are views).

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

## Context

- On `s2f-20261118` llama.cpp `-sm layer` read pp2048 1.36x (30B) and 1.51x (32B) faster than our `layer-2`; decode is about level. ADR 0068 wants layer-N to beat or tie `-sm layer`.
- libllama pipelines: with all layers on two or more GPUs it keeps several ubatches of a batch in flight across the stages (`llama-context.cpp:528`, `n_copies`). Our layer-N prefill runs the stages one after another: the planner writes `prefill_micro_batches = 1` (`planner/src/multi.rs`), `Topology::from_plan` drops the knob, the bench cuts the prompt into `--ubatch` steps, each a full program with a sync and a logits read, and the builders build one micro-batch per step.
- The executor already has what a pipeline needs: async ranges and copies from one thread, a copy event the destination stream waits on, one sync at the end. ADR 0069 "Hazards" decided one destination tensor per in-flight micro-batch, so no write-after-read fence is needed. The cost model (`costmodel::sim`) already models `prefill_micro_batches` chunks pipelined over the stages.

## Decision

### The knob reaches the builders through the topology

- `plan::Topology` gains `prefill_micro_batches: u32` (1: no pipelining), copied from `plan.knobs` by `from_plan`; `Topology::single` and hand-built topologies take 1. The builders read it from the topology, so plan files, `serve --plan` and `bench --plan` carry it with no new option.
- The planner (`multi.rs`) sets it only for shapes with two or more stages: `ceil(PROMPT / UBATCH)`, 4 for the judged pp2048 at ubatch 512 (ADR 0068). One-stage shapes keep 1. The Phase 3 search and `lower.rs` (one stage, `Sched`) are unchanged; `lower.rs` still refuses the knob for what it lowers.

### Which steps pipeline, and the chunking

- A step pipelines when the topology has two or more stages, `prefill_micro_batches = M > 1`, the step has one input that returns only its last token's logits, and that input has two or more tokens. Every other step (decode, batched sequences, `all_logits`, one stage) builds exactly as today.
- Chunking: the step's `n` tokens are cut into `min(M, n)` micro-batches whose sizes differ by at most one, the first ones larger (`plan::micro_batch_sizes`). This is `sim.rs`'s rule (`prompt / chunks + (c < prompt % chunks)`), so engine and cost model cut alike.
- Each micro-batch gets the step plan the same chunk would get as a step of its own (`KvPool::plan_chunks`: positions, rows, `n_kv` and mask of chunk j planned after chunks 0..j, lengths unchanged on return; the step commits once at the end, so ADR 0065's rollback contract holds). So a pipelined step computes what the same chunks run as serial steps compute.
- The bench sends `M x ubatch` tokens per step when the plan's topology pipelines (M = 1: `ubatch` per step as today). pp2048 at ubatch 512 and M = 4 is one step of four 512-token micro-batches; M = 1 reproduces today's run exactly.

### One rank graph per rank, micro-batches in node order

- Per rank and step key the builder still builds one graph: micro-batch 0's nodes, then 1's, and so on, each cut at its stage's boundaries as today, plus one cut at its end. The executor's rules (one graph per rank, each range computed once, a rank's ranges in node order, a copy after its source range and before its readers) hold unchanged; no executor op is added. The key is the list of per-micro-batch `(n, n_kv, n_out)`.
- Per micro-batch builder state: each micro-batch has its own inputs per rank (tokens, positions, rows, mask, output rows; MoE: its hidden-state input), so the host uploads all of them before the program runs.
- Hops: one destination tensor per micro-batch and boundary (M per hop, not two). A double buffer would need a write-after-read fence: the copy runs on the source stream, and only the destination stream waits on it, so a copy into a reused buffer could overwrite a hop the destination is still reading. M buffers need no fence, no event binding and no `unsafe`; the dense hop is a graph input (`ggml_gallocr` allocates graph leafs first and frees them only after their last reader), the MoE hop a persistent input. Copy sources are graph outputs, never reused within the step.
- KV: a stage writes micro-batch j's rows before j+1's attention reads them because both are on the stage's stream in node order. Ranks of one stage keep their lock step through the all-reduces.
- Last stage: micro-batches before the last stop after their last layer's KV writes (no FFN, no head on their last layer, nothing read back). Logits come from the last micro-batch only, with output rows local to it.

### Issue order and sync

- Ops are built per (stage, micro-batch) group and issued in GPipe wavefront order: by `stage + micro-batch`, then later stages first, keeping the order inside a group (`executor::gpipe_sort`; History 2026-11-19). Stage s issues micro-batch m after m-1 and after stage s-1 issued m and its copy, and the copy into stage s for m after stage s issued m-1. Issue does not block on GPUs (ADR 0069 "No blocking ranges"), so the streams overlap stage s on micro-batch m with stage s+1 on m-1; the wavefront order keeps the overlap if one range does block on the host.
- One sync at the end of the program, of the ranks holding logits (as today). CPU ranks compute synchronously, so CI checks results, not overlap.

### Memory

Per rank and step, against one micro-batch per step: M sets of step inputs (the mask is `n_kv x n_mb` F16, 2 MiB for the last 512-token micro-batch at 2048 rows), M hop inputs and M copy sources of `n_embd x n_mb x 4` bytes (10 MiB each at `n_embd` 5120 and 512 tokens: 80 MiB at M = 4), and the workspace of one micro-batch, since intermediates of micro-batch m are freed before m+1 in the same graph. The planner's workspace term (`multi.rs`, `UBATCH x (logits + 64 x n_embd)`) is per ubatch and does not add the M hops; at M = 4 they are under 1% of a 24 GB card, so the term stays as it is.

### What the cost model must match

- Chunking as above (equal chunks, `min(M, prompt)`), entering stage 0 in order; stage s computes chunk m after chunk m-1 and after the hop of chunk m arrived.
- A hop per chunk and boundary: `alpha + n_embd x 4 x tokens / beta`, serialized on the link (`sim.rs` `send`), one copy per rank pair for `tp x pp`.
- The last stage computes no head for chunks before the last; `sim.rs` charges no head term per chunk today, which agrees. The analytic predictor (`predict`, `features_multi`) does not pipeline prefill; the cost model work (track `costmodel-v2`) owns that change. This ADR changes no cost model code.

### Metrics, panel, alert (Rule 6)

- `tetraeder.executor.micro_batches` (histogram, `{micro_batch}`): per prefill step of a plan with two or more stages, the micro-batches it ran (1 when it did not pipeline).
- `tetraeder.executor.micro_batches.planned` (gauge): the topology's `prefill_micro_batches`, recorded at load for plans with two or more stages.
- Panel "Executor: micro-batches per prefill step" (average against planned) in `deploy/grafana/dashboards/overview.json`.
- Alert `TetraederPrefillNotPipelined` (`deploy/alerts/engine.yml`): planned above 1 and prefill steps averaging below 1.5 micro-batches for 15 minutes (batched prefill or a wrong bench chunk keeps the stages serial).

### Tests first

- `plan`: `micro_batch_sizes` cuts like `sim.rs`; `from_plan` carries the knob.
- `kvcache`: `plan_chunks` equals plan-then-commit per chunk and leaves lengths unchanged.
- `executor`, `cpu`: `gpipe_sort` puts every (stage, micro-batch) once, after (s-1, m) and (s, m-1), and (s-1, m) after (s, m-1), keeping group order; a two-rank, three-micro-batch program in that order validates and equals one graph bit for bit; validation refuses one hop tensor shared by two micro-batches and a copy issued before its source range.
- `model`, `cpu`: a pipelined `layer-2` prefill (M = 2 and 4, and `tp2-pp2`) gives logits bit-equal to the same chunks as serial steps, then greedy decode bit-equal; dense and MoE tiny fixtures. One stage ignores the knob.
- `model`, `gpu2_pipelined_prefill_matches` (dense, Qwen3-8B, `TETRAEDER_REQUIRE_GPUS` as the other `gpu2_` tests): `layer-2` with M = 4 against M = 1 on the same chunks within `GPU_TOL`, then 16 greedy tokens; `gpu2_moe_pipelined_prefill_matches` the same on the MoE model of `gpu_sharded_moe.rs`.
- Box confirmation (next 2x session): the two `gpu2_` tests, then `tetraeder bench --engine dense|moe --plan layer2.toml -p 2048 -n 0 --ubatch 512` with `prefill_micro_batches = 1` and `= 4`, paired with llama.cpp `-sm layer -ub 512` in the same session (ADR 0068 pairing).

## Consequences

- Layer-N prefill can overlap stages as `-sm layer` does; whether it ties is decided on a box, not here.
- No executor op, no ggml change, no `unsafe`; the executor gains only a sort helper. One rank graph grows to M micro-batches (node budget times M).
- Batched prefill (several sequences in one step) does not pipeline; splitting a multi-input step across micro-batches is left until serving needs it (the alert shows it).
- A plan's step graph count and range count per prefill step grow with M; ranges per decode token are unchanged.
- ADR number: 0077 was the next free number on `main` and `track-p5-power-pp` when written; `track-costmodel-v2` was not pushed then.

## History

- 2026-11-19, amendment "later stages first in a wavefront" (`track-pp-gap`, data from `s2g-20261118-pp`): with the order `(s + m, s)` layer-2 pp2048 at M = 4 ran 0.84 to 0.87 of `-sm layer`. A copy enqueues the destination stream's wait when it is issued (ggml-cuda `cpy_tensor_async`: record on the source stream, `cudaStreamWaitEvent` on the destination stream), so the hop into stage s+1 for m+1, issued before stage s+1's range for m, made that range wait until stage s finished m+1: stage s+1 ran one micro-batch late and the last two after stage s ended, one extra stage time per step. libllama issues per ubatch (stage 0, copy, stage 1) and has no such wait. Decision: inside a wavefront later stages issue first (`(s + m, Reverse(s))`); every other rule above holds. The stage split (planner, head bytes in the decode-balanced objective: 28/20 on the 30B, 25/23 in libllama) is left to the planner.
