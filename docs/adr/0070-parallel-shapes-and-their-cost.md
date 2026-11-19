# 0070. Parallel shapes: layer split, tensor, expert and tensor-times-pipeline parallel, and how the cost model prices them

Status: Proposed (2026-11-15). Phase 5. Builds on ADR 0027 (plan IR), 0040 and 0056 (cost model, calibration), 0069 (executor). Adds patch `0004` to the queue (ADR 0011, Rule 8). Briefs: `docs/phase5-briefs.md`.

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

## Context

The plan IR already has stages with a device group, `tp` and `ep` (ADR 0027), and embed and head may have 1 or `tp` shards. It does not say which dimension a shard cuts, and experts always have one shard, so the expert split llama.cpp's `-sm tensor` uses cannot be written. The planner's search (`crates/planner/src/search/cost.rs`) prices tensor parallel with a ring all-reduce formula from link priors, prices expert parallel as one dispatch and combine hop, and only forms groups over links of kind `P2p`, Linux only. None of the three all-reduce paths at the pin is a ring over P2P (NCCL picks its own algorithm, the internal path stages two GPUs through host memory, the executor's fallback is a star; ADR 0069), and the profiler's P2P links carry priors only (`crates/profiler/src/discover.rs`). Vast RTX 3090 boxes rarely have peer access, so today's search would prune tensor parallel there.

Model facts the rules must hold for (GGUF, Q4_K_M; K-quants have 256-element blocks along `ne0`, the input dimension of a weight):

| | Qwen3-32B | Qwen3-30B-A3B |
|---|---|---|
| Layers, n_embd | 64, 5120 | 48, 2048 |
| Heads, KV heads, head dim | 64, 8, 128 | 32, 4, 128 |
| FFN | n_ff 25600 (100 blocks) | 128 experts, top 8, n_ff_exp 768 (3 blocks) |
| `wo` input | 8192 (32 blocks) | 4096 (16 blocks) |

## Decision

### Shard conventions (no new IR field)

A shard's dimension is fixed by the role (Megatron's scheme): in a stage of degree `tp`, rank i of the group (its position in `Stage::devices`) holds slice i.

| Role | Cut | Notes |
|---|---|---|
| `attn_norm`, `ffn_norm`, `q_norm`, `k_norm`, router, output norm | replicated | computed on every rank, same values |
| `wq`, `wk`, `wv` | columns (output rows, `ne1`), by heads | rank i: heads `[i*H/tp, (i+1)*H/tp)` and KV heads likewise |
| `wo` | rows (`ne0`), by the same heads | output is a partial sum |
| `ffn_gate`, `ffn_up`, and per expert `ffn_gate_exps`, `ffn_up_exps` | columns | |
| `ffn_down`, `ffn_down_exps` | rows, matching the gate and up slice | partial sum |
| `token_embd` | replicated | each rank gathers its own rows |
| `output` (head) | columns, by vocab | each rank's logit slice is read back; the host concatenates |
| KV cache | by KV heads | each rank stores only its KV heads; one row allocator for all ranks (ADR 0069) |

Row cuts are in whole quant blocks. Blocks are dealt as evenly as possible; when they do not divide, the larger share rotates over layers (rank `(layer + i) mod tp` gets the extra block first), as llama.cpp's split rotation does, so memory balances while each layer's time is set by its larger share. Every rank needs at least one block of every row-cut tensor, else the shape is pruned with the reason. On Qwen3-30B-A3B `tp = 2` cuts the experts 512 and 256 wide and `tp = 4` is pruned (3 blocks); Qwen3-32B divides evenly for 2 and 4. `tp` must divide the head and KV head counts (validator, ADR 0027).

**IR change, one rule:** in a stage with `ep = 1` and `tp > 1`, an expert has `tp` shards on the group's ranks (cut as above); with `ep = tp`, an expert has one shard. The validator and `expand` follow; `ep > 1` with `tp != ep` is rejected in Phase 5 (ADR 0069 capability table).

### The shapes

- **Layer split.** S stages of one GPU each, contiguous layer spans from the search's dynamic program (uneven allowed). Embedding on the first stage, head on the last. Hop: copy of the hidden state, `n_embd x M` F32, stage s to s+1. With one sequence the stages run one after another; with `prefill_micro_batches > 1` the program issues micro-batch m of stage s after m-1, and streams overlap them (ADR 0069 hazards).
- **Tensor parallel.** One stage over `tp` ranks. Per layer two ranges and two all-reduces: attention (norm, Q K V, Q/K norm, RoPE, KV write, flash attention over the rank's heads, `wo` partial) then all-reduce, residual add on every rank; FFN (norm, gate and up, swiglu, down partial) then all-reduce, residual. For a MoE layer the router runs on every rank on the same input, so routing is identical, and every rank computes its column share of each chosen expert. All-reduce bytes per layer: `2 x n_embd x M x 4`.
- **Expert parallel** (`tp = ep = g`). Attention as tensor parallel. Experts are whole and placed in contiguous spans: rank i holds experts `[i*E/g, (i+1)*E/g)` of every MoE layer, loaded as one `[.., E/g]` slot tensor (the builder's slot machinery, ADR 0033). After the router, each rank maps the chosen ids through a per-rank I32 table `map_i[E]` (local index, or -1 for experts elsewhere) with `get_rows`, runs `mul_mat_id` on its slot tensor (rows with id -1 are zero and cost nothing, patch 0004), applies the router weights and sums: a partial MoE output, then the all-reduce. No tokens move between devices: every rank already has every token's hidden state, and the combine is the all-reduce. All-to-all dispatch is not in Phase 5 (it pays only at large batches or across nodes, Phase 6). Placement by routing frequency is a later lever; it needs a per-layer expert permutation in the IR, which the contiguous rule avoids, and the cost model below already prices the imbalance contiguous spans cause.
- **Tensor times pipeline.** S stages, each a `tp` group (`tp2-pp2` on four GPUs). Inside a stage as tensor parallel; at a stage boundary every rank of stage s holds the full hidden state after the last all-reduce, so rank i of stage s copies to rank i of stage s+1, all pairs at once. The head is cut by vocab over the last stage's ranks.

Single GPU is a one-stage, one-rank plan on the same executor (ADR 0069), so it competes on equal terms.

### Patch 0004: CUDA `mul_mat_id` skips id -1

At the pin the CPU kernel writes a zero row for id -1 (`ggml-cpu.c`, `i02 == -1`) and CUDA does not: MMVQ indexes `ids` directly and the MMQ path's `mm_ids_helper` only compacts ids that match an expert, leaving other rows unwritten. Without the patch expert parallel would have to park foreign pairs on a local expert with weight 0 (ADR 0064 parking), which at prefill computes `(g-1)/g` of the expert work for nothing. The patch: every CUDA `mul_mat_id` path (MMVQ, MMQ through `mm_ids_helper`, MMF, the generic gather fallback) treats id -1 as no expert, reads no weights for it and writes zeros, as the CPU does. Upstream-clean (it matches existing CPU semantics); if b12274 or later upstream already has it, it is a backport and `docs/sources.md` says so. Tests: CPU against CUDA `mul_mat_id` with random -1 masks at 1, 8, 64 and 512 tokens and Q4_K, Q6_K, Q8_0 and F16 weights, zero rows exact; a `gpu_` timing check that time falls with the share of -1 ids. Until it lands, expert parallel runs on CPU ranks only.

As built (E4): MMVQ and MMVF write the zero row in the kernel and read no weights; MMQ and MMF never match -1 to an expert, so a small kernel (`mmid.cu`) zeroes those rows after the matmul; the generic fallback gathers them from one extra zero row. The ids helper is unchanged. `ggml-sys/build.rs` sets `cfg(tetraeder_patch_0004)` and the `DEP_GGML_PATCH_0004` metadata when `patches/series` lists the patch. Upstream (b12274, master 2026-11-15) has no such change, so the row in `docs/sources.md` is own code.

### Cost model

One formula, owned by `costmodel::predict` (as ADR 0056); the search's `stage_time` calls the same helpers.

**Features.** `RunFeatures` gains `ranks: Option<RankFeatures>` (serde default `None`; with `None` every prediction is bit-identical to today):

```text
RankFeatures { phases: Vec<Phase>, step_fixed: bool }
Phase { repeat: u32, parts: Vec<DevicePart>, ranges_per_rank: u32, after: Option<Coll> }
Coll = AllReduce { group: Vec<String>, bytes_per_token: f64 }
     | Copy { pairs: Vec<(String, String)>, bytes_per_token: f64 }
```

Each `Phase` is what runs concurrently on the ranks between two collectives: one `DevicePart` per rank with that rank's share of bytes, params, ops and KV (ADR 0040's fields). A tensor-parallel layer is two phases with `repeat = layers`; a stage hop is a phase's `Copy`.

**Step time** for M tokens in graph mode `mode`:

```text
T(M) = sum over phases p:  repeat_p * ( max over parts r of t(r, M, mode)
                                        + ranges_per_rank_p * c_range(mode)
                                        + t_coll(after_p, M) )
       + c_step
t(r, M, mode)  = today's per-device term (c0(mode) per op + max(FLOPs / F, bytes / B), KV reads)
t_coll(AllReduce, M) = ar_curve(group)(bytes_per_token * M)
t_coll(Copy, M)      = max over pairs of copy_curve(pair)(bytes_per_token * M)
```

`pp_tps` sums `T` over the prompt's ubatches (KV growing), `tg_tps` over the generated tokens, as `predict` does now.

**Shares per shape.** Column and row cuts give each rank its slice's bytes and params; uneven block cuts use the rank's actual share, so the max over parts carries the imbalance. Expert parallel: decode bytes of rank r are `E[distinct local experts touched] x expert bytes` and params the local pairs; prefill reads every local expert once per ubatch. Both use `E[max over ranks]`, computed from the per-expert routing frequencies by a fixed-seed Monte Carlo (1000 steps of top-k sampling without replacement, deterministic), or from a recorded routing trace when the fixture has one (ADR 0056's replay). The head phase adds the logit readback, `vocab / tp x 4 x outputs` bytes over each rank's host link.

**c0 per graph mode.** `c0(mode)` per op stays as calibrated (ADR 0007, 0040). New: `c_range(mode)`, the fixed cost of launching one range (one captured CUDA graph, or an eager launch), and `c_step` (input upload, the end-of-step sync and readback). Both are fitted from the anchors of ADR 0068 (truncated `single` and `tp-2` runs differ in ranges per token, which identifies `c_range`).

**Links and collectives, measured.**
- The profiler adds one link per GPU pair always: `LinkKind::P2p` when peer access is on, else a new `LinkKind::HostStaged`. Prior for `HostStaged`: alpha = 2 x the larger host-link alpha, beta = the smaller host-link beta, halved when both GPUs sit in one lane group (`pcie.rs`, ADR 0040's PCIe topology). Search groups over `P2p` or `HostStaged` links.
- The collective probe (`tetraeder comm-probe`, ADR 0068) measures `ar_curve` per group and `copy_curve` per pair: median time over 20 repeats at 4 KiB to 64 MiB in powers of 4, through the path the executor will use. `HardwareGraph` gains `collectives: Vec<CollectiveCurve { op, ranks, comm, points: [(bytes, seconds)] }>`, interpolated piecewise linear in bytes (as ADR 0056's disk curve), flat extrapolation in latency below the first point, linear in bytes above the last. Measured curves are never fitted. Without a curve the prior is today's `allreduce_time` with the link's alpha and beta (ring over the group) and `transfer_time` for copies.

**Planner.**
- `enumerate_multi(hw, layout, hints, phase)`: on `g` usable GPUs of one vendor (Linux, ADR 0008), the fixed set `single` (if it fits), `layer-g`, `tp-g`, `ep-g` (MoE), and on four GPUs `tp2-pp2`; each pruned shape listed with its reason (memory, divisibility, blocks). Knobs equal for all (ADR 0068). Deterministic, uses no calibration.
- The search's skeletons: groups over `P2p` or `HostStaged`, expert parallel as `tp = ep = g` (the `tp = 1, ep = g` group is dropped), block-cut pruning as above.
- Simulated fixtures (Rule 4): `crates/costmodel/fixtures/two_3090_hoststaged.toml` and `four_3090_hoststaged.toml` (priors from an RTX 3090 box, no peer access, two GPUs sharing a lane group on the 4x fixture), with prior collective curves. Planner tests run every shape on both and on the existing fixtures.

### Numerics (Rule 2)

Sharded steps split matmul reductions and add partial sums, so results differ from one device in the last bits. Parity: every shape against the one-rank run of the same builder, top-5 logits within `tol_batch` (`max(TOL_CPU, 1e-3 x largest logit)`, `crates/model/tests/common/mod.rs`) on CPU ranks, `TOL_GPU` on CUDA (on real models: layer shapes `GPU_TOL`, tensor and expert shapes the bound anchored on libllama's `-sm tensor` gap, ADR 0069 History and ADR 0071); greedy tokens equal except where the reference's top-2 margin is below ADR 0043's near-tie bound. Per split point (Rule 2): layer split at the first, middle and last layer boundary.

## Consequences

- Three program ops and a role table cover all four shapes; the IR gains one rule (expert shards follow `tp` when `ep = 1`), no new field.
- Tensor parallel on Qwen3-30B-A3B is lopsided (2:1 expert columns on two GPUs, impossible on four); the cost model sees it, and it is a reason expert parallel can win there.
- Collective costs come from measured curves of the path that actually runs, so NCCL, ggml's internal path and the fallback are priced alike.
- Contiguous expert spans can be imbalanced under skewed routing; the prediction includes it, and frequency placement waits for evidence that it matters.
- Patch 0004 is the only ggml change in Phase 5.

## As built (E3, 2026-11-16)
- `ModelLayout` carries `n_ff` and `n_ff_exp`; older layout files estimate them from bytes, rounded to 256.
- `c_range` (graphs and eager) and `c_step` live in `HardwareGraph.range_costs`, fitted only from anchors that carry `ranks`, with a near-zero pull (`LAMBDA_RANGES`; the normal pull left `c_step` 15% off on the synthetic fit). Same-class GPUs share one set of fitted rates. (No session directory yet.)
- Without a measured curve, a copy costs `alpha + n / beta` and an all-reduce is a ring over the group's links; probe curves (`tetraeder comm-probe`) replace the prior ones with the same op and ranks. GPU pairs without peer access get a `HostStaged` link (prior: twice the larger host alpha, the smaller host beta, halved when both share a lane group).
- `enumerate_multi` returns `MultiCandidate { id, gpus, plan, notes }`, not `Candidate`, so the single-device enums and their exhaustive matches stay unchanged. The embedding is one shard on the first stage's rank 0 (the plan IR has no replicated unit).
- Priors only (fixtures `two_3090_hoststaged`, `four_3090_hoststaged`): Qwen3-32B single 34.6 tok/s, tp-2 41.0, layer-2 34.2; Qwen3-30B-A3B single 113, layer-2 109, tp-2 76.5, ep-2 76.8; tp-4 for the 30B pruned (3 blocks over 4 ranks of the expert down rows). S2's anchors decide the real numbers. (No session directory yet.)

## Revised after S1 (2026-11-16, `bench/results/20261116gpu3/s1`)
- At the pin MMQ already skips every tile of an expert without tokens and `mm_ids_helper` never matches -1, so the expert-parallel saving comes from the experts that get no tokens on a rank; "without it, EP wastes (g-1)/g of expert compute at prefill" holds against remapping remote slots to local experts, not against MMQ (`20261116gpu3`).
- The patch stays for: defined zero rows for -1 on MMQ and MMF; no weight reads for -1 on MMVQ and MMVF (decode), where unpatched kernels read expert -1 out of bounds; no assert on -1 in the generic path; `mm_ids_helper` entries left unwritten for -1 point at row 0 (else MMQ's gate/up scatter wrote through stale pool indices).
- The timing test is time against the number of local experts (g = 1 to 8) compared with remapping; random -1 ids leave nearly every expert with a token and cannot show the saving. K-quant parity uses 2e-2 (q8_1 against q8_K activations); q8_0 and f16 keep 5e-3.

## History
- 2026-11-18, cost model version 2 (`MODEL_VERSION` 2). The judged prediction stays this analytic model, made before the candidates run (owner, ADR 0068 History; ADR 0076 not adopted). The S2f diagnosis found its errors (`bench/results/s2f-20261118/README.md`, "Cost model diagnosis"); the numbers of the fix are in the same README, "Cost model version 2". The rules:
  - **Expert rate.** Routed-expert matmuls (`mul_mat_id`) have their own rate `F_exp` per GPU class (`ComputeSpec::expert_flops_per_s`). It is fitted when a ranks anchor has expert work; without it they run at `F`. `DevicePart` carries the expert bytes and params apart from the dense ones.
  - **Roofline per op class.** A part costs `c0 ops + max(2 p M / F, W / B) + max(2 p_exp M / F_exp, W_exp / B) + max(a M ctx / F, kv ctx / B)`, not one roofline over the sums.
  - **Output rows.** The head runs on the step's output rows only: one per step, in the bench and in the judged request. So does the dense builder's last FFN (`qwen3/ranked.rs`); the MoE shard builder runs its last FFN on every row (`qwen3moe_shard.rs`). A phase with `out_rows` prices a prefill step at those rows.
  - **Experts at prefill.** A prefill step reads every expert of the rank, its column share on `tp` and its span on `ep`; decode reads the top-k share. Single and `tp` now do what `ep` already did.
  - **Single on the ranks path.** `single` is a one-rank plan with one range, `c_step` and the logit readback, like every other shape.
  - **All-reduce in a step.** It costs `t(n) - (1 - s) t(0)`: `t` is the probe curve (else the ring prior), `t(0)` its zero-byte time, and `s` the fitted `RangeCosts::allreduce_latency_share`. The probe syncs every rank around each call, which a step does not. The byte part stays as measured, the BF16 switch included (the probe sends F32 tensors through the executor's path). `s` is one parameter, fitted from ADR 0068's `tp-2` anchors and applied to every all-reduce group (`tp-4` and `ep-4` inherit it); it is 1 when no anchor has an all-reduce.
  - **Not taken:** one factor on the whole all-reduce time (the owner's first wording, "one fitted scale on the all-reduce time"). On both sessions it leaves every `tp-2` anchor's pp too fast and every tg too slow: one factor cannot fit 4 MiB and 8 KiB messages at once.
  - **Host link held.** With ranks anchors and none on the CPU, the fit holds the host link and the CPU's `c0` at the prior. Only the logit readback crosses that link, so fitting it turned the link into a step cost.
  - **Not identified:** `c_range` against `s`. Every `tp` phase has one range and one all-reduce, so single and `tp-2` anchors cannot separate them. The weak pull on `c_range` decides, and it fits near zero.
  - **Pipelined prefill** (ADR 0077). With two or more stages and `prefill_micro_batches = M > 1`, a prefill step of `M x ubatch` tokens is cut as `plan::micro_batch_sizes`. Stage s computes chunk c after chunk c - 1 and after the chunk's hop arrived. Each hop costs the copy curve (else `alpha + n_embd x 4 x tokens / beta`) and is serialized per boundary. The fill is S - 1 chunk times of idle on the first and last stages. The output-row phases and `c_step` follow once per step. With M = 1 the stages run in turn, as before.
  - Anchors stay ADR 0068's (`single` and `tp-2` at L/4 and L/2). Runs without ranks are unchanged, so the ledger moves to version 2 with the same predictions. Offline replica: `scripts/costmodel_replica.py`. It reproduces version 1 to the digit and writes the version-2 fits that `crates/planner/tests/costmodel_v2.rs` checks against Rust.
- 2026-11-19, layer split balances the prefill step (`track-split-prefill`, data from `s2g-20261118-pp`). The layer-N and `tp x pp` stage cut minimizes the slowest stage of one pipelined prefill step, not decode bytes. A stage costs, per ubatch, the larger of its matmul flops over the device rate (routed experts at top-k) and its weight reads over the bandwidth, times the ubatches of the judged prompt (ADR 0077), plus the output head read once, since the head runs one row per step. A single decode stream runs the stages in turn, so its total does not depend on the cut and the head's per-token decode cost does not move it. `output_bytes` stays in the memory check of the last stage. The decode-bytes objective gave 28/20 on Qwen3-30B-A3B (stage 0 the prefill bottleneck); the new rule gives 24/24 there and 32/32 on Qwen3-32B (was 33/31); libllama cuts 25/23.
