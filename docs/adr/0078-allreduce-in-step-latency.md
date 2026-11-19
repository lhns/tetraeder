# 0078. Cost model 3: an all-reduce in a step pays its own fitted latency, not a share of the probe's

Status: Accepted (owner, 2026-11-19; proposed 2026-11-19). Phase 5. Amends ADR 0070 (cost model, "All-reduce in a step" of its 2026-11-18 History); nothing else of 0070 changes. Companions: ADR 0068 (anchors, reopen rule), ADR 0077 (pipelined prefill).

Measurements live in the session READMEs: the diagnosis and the before/after table are in `bench/results/s3b-20261119/README.md`, "Cost model diagnosis (v3)".

## Context

- Version 2 prices an all-reduce in a step at `t(n) - (1 - s) t(0)`: the probe curve minus a fitted share of its zero-byte time. `s` is fitted on the `tp-2` anchors and applied to every group.
- On one host (s3 and s3b, host 20210) `s` moved from 0.445 to 0.259 while `s x t(0)` of the anchored pair stayed near 12 us: the anchors identify an in-step latency, and `s` only absorbs the noise of the probe's floor (`t(0)` of the anchored pair 27.9 against 43.4 us). Multiplied into the 4-rank floor, that noise moved the `ep-4` decode prediction by a third between the two sessions with the measured rate unchanged.
- How that latency grows with the ranks is not identified by ADR 0068's anchors, which all-reduce over two ranks only. A ring's step count `2 (p - 1)` at the anchored per-step latency is an upper bound: it over-prices `tp-4` on the dense 32B, which has no routing term, beyond the 15% line.

## Decision

- **In-step latency.** An all-reduce over `p` ranks in a step costs `max(t(n) - t(0), 0) + a + 2 (p - 1) b`: the probe curve's byte part, the BF16 switch included, plus a fitted latency. `t(0)` is the curve's first point (else the ring prior's `2 (p - 1) alpha`).
- `a` (`RangeCosts::allreduce_latency_s`) is fitted whenever an anchor has an all-reduce, with the weak pull of the range costs (it is a fixed cost of a few percent of a step). The anchors identify `c_range + a` per phase, not the split (ADR 0070's "Not identified"), which no prediction of a phase with one range and one all-reduce depends on. `b` (`allreduce_step_s`) is fitted only when the anchors' all-reduces span two or more group sizes; otherwise it is absent and every group pays `a` (ADR 0070's inheritance rule without the probe floor).
- `s` is no longer fitted. Graphs fitted before (no `a`) predict as version 2, so committed calibrations and the ledger keep their numbers; `MODEL_VERSION` is 3.
- **Owner question (ADR 0068 reopen rule):** `b` needs an anchor over a second group size. The proposal is a truncated `tp-4` anchor on Qwen3-32B at L/4 and L/2 on 4-GPU boxes. Until then `b` is absent.

## Not decided here (owner)

- Expert-parallel prefill on Qwen3-30B-A3B stays over-predicted by about a third. The bench feeds byte ids 0 to 255 (`tetraeder bench`), and the slowest rank's expert work it implies is about two thirds of the layer's at both `ep-2` and `ep-4`, which uniform routing (one half, one quarter) cannot give. ADR 0070 already names the input: per-layer routing frequencies of the judged token stream; `tetraeder-plan predict` passes none. Recording them is a calibration input, not a fit to candidates.
- `tp2-pp2` prefill measures near the stages run in turn (32B) or about half the GPipe overlap (30B) while `layer-4` overlaps as modeled. No mechanism is identified offline; the measurement that decides it is `tp2-pp2` at `prefill_micro_batches` 1 against 4 on one box. An overlap factor is not introduced: it would be a fit to one candidate.

## Consequences

- An in-step latency fitted on the anchored pair no longer moves with the probe's floor; cross-session predictions of the same host agree.
- 4-rank all-reduces are priced at the 2-rank latency until a second group size is anchored, which under-prices them if the latency grows with the ranks (the `ep-4` decode miss stays open between that and routing skew).
- The planner search shares `coll_time`, so its all-reduce costs follow on fitted graphs; prior fixtures (no range costs) are unchanged.
- Offline replica: `scripts/costmodel_replica.py --version 3` (the default).

## History
- 2026-11-19, owner decision of ADR 0068 (History 2026-11-19) built (`track-p5-routing`). Rules:
  - **Routing input.** The calibration stage records the router's choices over the judged token stream (`tetraeder bench --engine moe --routing-out`, one pass of the pp2048 and tg128 tokens on GPU 0 without a plan, since the sharded load records none) into `routing-<model>.json`: selection counts per layer and expert, the pp rows (`prefill`) and the tg tokens (`decode`) apart. `tetraeder-plan predict --routing` prices expert parallel from them as ADR 0070 defines: `prefill` sets the busiest rank's share of a prefill step's routed rows, `decode` the Monte Carlo of a decode step's busiest rank. Each layer is scaled to mean 1, so equal counts give the uniform price exactly. A file whose layers or experts do not match the layout is an error, not uniform routing. Single, layer, `tp` and `tp x pp` shapes do not read it, and neither does `calibrate`: its anchors cut experts by columns. The predict file carries `routing_sha256`.
  - **tp-4 anchor.** On 4-GPU sessions the anchors add the dense model's `tp-4` (the session's plan) at L/4 and L/2, so the all-reduces span two group sizes and `b` is fitted; 2-GPU sessions are unchanged.
  - **tp2-pp2 overlap.** An optional, unjudged stage (`PP_OVERLAP`, on for 4-GPU rehearsals) benches the session's `tp2-pp2` plan at `prefill_micro_batches` 1 and 4, two rounds interleaved, into `pp-overlap-<model>.json`. No model change before it is read.
  - Offline check (synthetic routing only, no recorded routing exists yet): `bench/results/s3b-20261119/README.md`, "Routing input and tp-4 anchor (offline)".
