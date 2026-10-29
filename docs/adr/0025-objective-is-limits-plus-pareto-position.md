# 0025. The objective is hard limits plus a position on the speed and quality front

Status: Proposed (design; implemented in Phase 3a)

## Context

Users want to say "fast" or "good" and how much slow hardware the engine may use. A single weighted score (for example 0.7 speed, 0.3 quality) cannot be set sensibly: tok/s and KL divergence have unrelated scales, and the right weight changes with the model and the hardware. With quantization a planner choice (ADR 0024), every candidate plan has a speed, a quality loss and a resource cost.

## Decision

- Inputs are **limits**: `target_tok_s` (and `p95_ttft_ms`, memory), an optional quality floor (`max_quality_loss` as KL divergence or `min_format`), and device rules (allow, forbid, prefer, and a cost per device or node).
- Among plans that meet all limits the planner decides in this order: quality first, then resource cost (fewest and cheapest devices), then extra speed. So a target tok/s alone is enough: the planner finds the best-quality plan that reaches it on the cheapest hardware that can.
- The planner also computes the **Pareto front** of speed against quality for the current limits. An optional `tradeoff` value from 0 (fastest plan) to 1 (best-quality plan) picks a point on that front. It is a view on the front, not a weight.
- If no plan meets the limits, the planner says which limit fails, returns the closest plan and shows the front.

## Consequences

- The simulator and cost model must score quality loss and resource cost next to latency and throughput.
- `plan explain` shows the front and where the chosen plan sits; `plan what-if` shows how it moves with other hardware.
- The tie-break order (quality, then cost, then speed) is the default; a hint can change it later if a real workload needs that.
