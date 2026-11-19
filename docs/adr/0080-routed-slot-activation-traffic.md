# 0080. Cost model 4: the routed slots' activation bytes, on every expert-parallel rank

Status: Proposed (2026-11-19). Phase 5. Extends ADR 0070 (cost model, ranks path) and ADR 0078 (cost model 3); nothing of either changes. Companion: ADR 0068 (verdict rule, 15% line).

Measurements live in `bench/results/s3c-20261119/README.md`, "Diagnosis of the misses (offline, ADR 0080)", with the before/after table over s2f, s2g, s3, s3b and s3c.

## Context

- On s3c the ep-4 prefill of Qwen3-30B-A3B stays over-predicted by 0.188 with the recorded routing priced in (ADR 0078 History). The busiest rank holds 0.391 of the routed rows on average; the measured step implies about 0.55.
- Version 3 prices a part as matmul rooflines plus `c0` per op; it has no term for activation bytes. A MoE layer fans each token out to k slots (`qwen3moe::weighted_experts`, `sum_experts`): gate and up outputs, swiglu, the quantized inputs, the down output, its weighting and the sum over slots are k times the token's width. On a single GPU these bytes are absorbed into the fitted expert rate.
- An expert-parallel rank computes every slot of every token: ids of experts held elsewhere are -1 (patch 0004), the MMQ quantize kernels still run over them and the weighting and sum are full-size graph ops. Version 3 scales all expert time by the busiest rank's row share, so it scales these bytes down too. A tp rank halves only the n_ff_exp part.

## Decision

- **Routed-slot bytes.** `DevicePart::expert_slot_bytes_per_token` counts, per MoE layer and token, `k (25.125 n_ff_local + 22.25 n_embd) + 12 (k - 1) n_embd` bytes: per slot and expert column gate and up written (8), swiglu read and written (12), the down input read and quantized to Q8_1 (5.125); per slot and `n_embd` row the down output (4), its weighting (8) and the gate and up input quantization (10.25); the sum over slots two reads and a write per added slot. `n_ff_local` is the rank's column share of `n_ff_exp` (tp), the whole width with whole experts (single, layer, ep). Expert parallel pays all slots on every rank. Dense models have none.
- They cost `bytes x M / B` per part, added to the matmul terms (separate kernels, so not inside a roofline max). No new fitted parameter: the byte count comes from the layout, `B` is the fitted memory rate.
- **Versioning.** `RangeCosts::expert_slot_traffic` marks graphs fitted by cost model 4; a fit with ranks sets it. Graphs without it (all committed calibrations before this ADR) predict as version 3, so the ledger and the committed v2 and v3 replica graphs keep their numbers. `MODEL_VERSION` is 4; the replica's `--version 4` is the default.
- The byte counts follow the builder's graph as of 27c6e00; a builder change that fuses these ops (for example gate and up, or the weighting into the down matmul) changes them and is a cost model change.

## Not decided here (owner)

- **tp2-pp2 prefill** stays over by 0.45 (32B) and 0.24 (30B). The stage price is right (micro-batches 1 within 0.02 of the pp-overlap stage), the GPipe overlap is not: measured 24% and 53% of the modeled saving, while layer-4 overlaps as modeled. All-reduce contention is ruled out by bandwidth arithmetic. The deciding measurement is an executor trace of tp2-pp2 at micro-batches 4 (host time per op, then device time per op); if the one issuing thread serializes the tp stages, the fix is in the executor, not the model. No overlap factor.
- **30B layer-4 decode** stays over by 0.17: about 0.2 ms of device time per rank per token that the model lacks, on MoE on host 20210 only (not on the 32B, not on s2g, not on tp2-pp2's hop). The deciding measurement is a sync-mode trace with GPU clocks sampled at 20 ms, layer-4 against single.
- **ep-4 decode** sits at -0.150 (version 3: -0.148); its four-rank decode all-reduces are over-priced, which this ADR does not touch.

## Consequences

- ep-4 prefill on the 30B: +0.188 to +0.070 (s3c), +0.353 to +0.170 (s3b, uniform routing), ep-2 on 2-GPU sessions +0.096 and +0.111 to -0.020 and +0.025; the 30B tp-2 anchors' residuals shrink. No candidate of s2f, s2g, s3b or s3c moves across the 15% line except s3c ep-4 pp (in); s3c ep-4 tg moves 0.002 to the line.
- `F_exp` rises (s3c 24.5 to 33.5 TFLOP/s), since the fitted expert time no longer carries the slots.
- Rust and replica agree bit for bit on the version-4 graphs committed in `bench/results/s2f-20261118/costmodel-v4/` (`crates/planner/tests/costmodel_v2.rs`); `bench/results/s3c-20261119/costmodel-v4/fit-s3c.json` holds the s3c fit with its routing record.
- ADR number: 0080 was the next free number on `main` and the fetched branches when written.
