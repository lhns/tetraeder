# 0076. Judged predictions from per-shape lines through truncated anchors

Status: Not adopted (owner, 2026-11-18): judged predictions stay the planner's analytic predictions before the candidates run (ADR 0068); `scripts/costmodel_lines.py` stays as a report-only check. Drafted 2026-11-18. Phase 5. Changes ADR 0070's "Cost model" for the judged predictions and ADR 0068's anchor plan ("Calibration, separate from candidates"); ADR 0007 and 0070's analytic model stay for the planner's search and for shapes without anchors. Diagnosis and numbers: `bench/results/s2f-20261118/README.md`, section "Cost model diagnosis". Offline check: `scripts/costmodel_lines.py`.

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

## Context

- The S2f rehearsal missed every candidate by more than the 15% line, on a fit whose own anchor band was already far above it.
- **The analytic model's structure is wrong in four places.** One FLOP rate prices dense and expert matmuls, though expert matmuls at prefill run several times slower. The head is charged for every prefill row, though the builders compute only the output rows. MoE prefill reads only the top-k share of the experts in the `single` and `tp-2` features, though every expert is read. `single` is priced on the pre-ranks path without step or readback terms. The fit also drove `c_range` and `c_step` to zero.
- **Rates depend on the shape.** The S2f GPUs ran under a power cap (inferred from peak draw and the 0x4 bit; no limit was recorded), and the s2e host under a higher one. Under a cap, a GPU's clock follows how the shape loads it: one GPU running alone is slowed most, and a layer-split stage that idles half the time much less. No set of rates shared across shapes fits `single`, `tp-2` and `layer-2` decode together, whatever form the refits over the `single` and `tp-2` anchors took. Rates fitted on one host do not carry to another.
- **The anchors themselves extrapolate.** Per shape and rate, time is linear in the layer count. On S2f, a line through the L/4 and L/2 medians predicts every full-model `single` and `tp-2` rate within a few percent.
- Only `single` and `tp-2` are anchored. `--layer-limit` keeps a prefix of the plan's stages (`features_multi`, `qwen3moe_shard.rs`), so a truncated `layer-2` plan is stage 0 alone. The dense builder rejects a layer limit on more than one stage (`qwen3/ranked.rs`).

## Decision

1. **Judged prediction.** For each candidate (model, shape) of a case, and for each rate: `T(L') = t_step + L' * t_layer`. It is fitted by least squares on that shape's own anchors from the same session, and evaluated at the model's L: `pp_tps = P / T_pp(L)`, `tg_tps = N / T_tg(L)`. Each line has two parameters. `t_step` is the fixed cost per request part (embedding, head, readback, sync). `t_layer` is the cost of one layer at the clocks, power and collective path the shape actually gets. No parameter is shared across shapes, models or boxes. `tetraeder-plan predict` writes these numbers into the same file, under the same rule: written before the first candidate runs, and hashed. The planner's judged pick (ADR 0068 "Winner") is the candidate with the smallest `T` by these lines; the analytic model's pick is reported beside it.
2. **Anchor plan.**
   - Every candidate shape of the case is anchored at L/4, L/2 and 3L/4: `single`, `layer-g`, `tp-g`, `ep-g` and `tp2-pp2`. Two rounds, pp2048 and tg128, as now.
   - An anchor at L' runs the candidate's shape planned for a model of L' layers. It is the same enumeration on the layout truncated to L', so spans are re-solved and the head sits on the last stage. Every stage then keeps its share, and so its duty.
   - All three points enter the fit. The summary also reports a holdout: the line through L/4 and L/2 predicting 3L/4. The holdout is reported in `verdict-notes.json` and not judged.
   - Anchors stay truncated, so no anchor equals a candidate (ADR 0068).
3. **The analytic model stays the planner's prior and is not judged.** It keeps ranking shapes for the search and prices shapes without anchors. It is fitted per box over both models, and its errors against the candidates go into `verdict-notes.json`. Five fixes, one new parameter:
   - the head, and the last layer's FFN for MoE, priced on the output rows at prefill;
   - MoE prefill reads every local expert, as `ep` already does;
   - a separate expert-matmul rate `F_exp` per GPU class;
   - a roofline per op class (dense matmul, expert matmul, attention, head), summed over the part;
   - `single` priced on the ranks path, so `c_step` and the readback apply to every shape.

   The collective probe curves stay measured and unfitted. They are F32 tensors through the executor's path, so the BF16 switch is already in them (`comm_probe.rs`).
4. **Power recorded.** `env.json` gains, per GPU, `power.limit`, `power.default_limit` and `power.max_limit`. A box below its default limit is accepted (common on Vast), and its anchors carry the effect.
5. `MODEL_VERSION` becomes 2 (ADR 0056 ledger).

Not taken:
- **An analytic power term** (speed `min(1, a / duty)` per GPU). It fits the `single` and `tp-2` anchors closely and still misses `layer-2` tg far outside the band. The cap's time constant would also need a parameter, since layer-2 prefill gains little and decode a lot. Identifying either needs `layer-2` anchors, and with those anchors the lines predict directly.
- **Shared rates across boxes.** Power limits differ by host.

## Consequences

- The judged numbers become each shape's measured extrapolation rather than the analytic model's output. The exit test then checks that truncated runs extrapolate and that the pick is right. The analytic model's quality is reported, not judged. Owner question: is that the reading of "predictions within 15%" that Phase 5 wants? Default: yes.
- **Unverified for `layer-2` and `ep-2` until a box session runs their anchors.** Truncating a `layer-2` plan shortens its stages' busy periods, at L/4 to a quarter of the full model's. Under a power cap this may favour the truncated runs. The 3L/4 holdout shows whether it does.
- Rates never transfer across boxes; the procedure does, and every session refits. Evidence so far covers one box: S2f's `single` and `tp-2`.
- Anchor runs grow from 16 to about 42 per 2x session (30B: 4 shapes, 32B: 3, times 3 truncations and 2 rounds). That is roughly 10 more minutes of box time.
- Code needed, not yet built:
  - planner: anchor plans for truncated layouts, predict from lines, the five analytic fixes;
  - builders: a topology of L' layers with `--layer-limit L'`, and multi-stage layer limits in the dense builder;
  - harness: the anchor list per shape and the power fields;
  - summary: the holdout and the analytic errors in `verdict-notes.json`.

  `scripts/costmodel_lines.py` is the reference for the line fit.
- Separate from the cost model: llama.cpp's `-sm layer` pipelines a batch's ubatches across GPUs, and our `layer-2` runs `prefill_micro_batches = 1`. Until layer split pipelines its prefill, the 30B case misses ADR 0068's baseline gate whatever the predictions say.
