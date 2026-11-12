# 0063. GenAI semconv request metrics from the gateway for every engine; one tier transfer metric family

Status: Proposed (2026-11-12). Builds on ADR 0012 (one OTel data path), ADR 0022 (gateway telemetry) and ADR 0046 (real engines behind the gateway).

## Context

Most request metrics already exist: `gen_ai.server.time_to_first_token`, `gen_ai.server.time_per_output_token` and `gen_ai.server.request.duration` are recorded by the gateway (`crates/gateway/src/gateway/pump.rs` and `cost.rs`) around the `Engine` trait, so the builder, libllama and a proxied llama-server get the same numbers by construction. The SLO rules in `deploy/alerts/slo.yml` (ttft within 2 s, tpot within 100 ms, availability) and the dashboard's TTFT and TPOT panels read them. The gaps:

- Two of three histograms lacked `gen_ai.operation.name`, which the semantic conventions require on every server metric.
- No gateway test covered streaming, the Anthropic route, or the TPOT value.
- The dashboard panels aggregated over models.
- Bytes moved between tiers are spread over `tetraeder.tier.read.bytes` (device, tier), `tetraeder.expert.load.duration` (tier, source), `tetraeder.expert.admitted.bytes` and `tetraeder.expert.prefetch.wasted`, none of which names both ends of a move. Admission uploads (ADR 0059) and a future remote KV tier would each add one more.

## Decision

1. **Request metrics stay in the gateway.** Names and units follow the OpenTelemetry GenAI semantic conventions (`gen_ai.server.time_to_first_token`, `gen_ai.server.time_per_output_token`, `gen_ai.server.request.duration`, seconds). All three carry `gen_ai.request.model` and `gen_ai.operation.name`; `request.duration` adds `error.type` on failure. Unknown model names from clients are reported as `unknown` (bounded labels, as before).
   - TTFT: one per request that produced a token, measured from request start (queue wait included) to the first token event. Streamed or not, OpenAI or Anthropic route, the pump records it.
   - TPOT: one value per request, the mean interval between output tokens after the first (`(last - first) / (n - 1)`), none for one-token replies. The semconv describes per-token latency; a per-request mean keeps the recording cost off the per-token path and is what the SLO bucket rule `le=0.1` reads.
   - `crates/engine` records the same metrics for its own `generate` path (CLI and bench). The gateway does not call it, so a request is never counted twice.
   - Tokens: `tetraeder.tokens` stays. `gen_ai.client.token.usage` is a client-side metric with a different meaning, and the conventions define no server token counter.
2. **SLO alert.** The existing `ttft` SLI in `slo.yml` already burns on TTFT above 2 s with the page and ticket windows. No second rule is added. The dashboard panels now show p50 and p95 per model.
3. **One transfer family.** `tetraeder.tier.transfer.bytes` (counter, By) and `tetraeder.tier.transfer.duration` (histogram, s) with attributes `tetraeder.transfer.source` and `tetraeder.transfer.destination` (`disk`, `ram`, `vram`, `remote`) and `tetraeder.transfer.purpose` (`expert_miss`, `prefetch`, `admission`, `kv`). The duration is recorded only where the caller times the transfer. Emitted through `Metrics::tier_transfer(bytes, took, Tier, Tier, Purpose)`. `tetraeder.tier.read.bytes` stays unchanged until its two dashboard panels and the detector move over.
   - Mapping of existing counters, for the follow-up that emits from crates/model and crates/memory (out of scope here):

   | Existing | Transfer series |
   |---|---|
   | `tier.read.bytes{tier=nvme}` | source `disk`, destination `ram` or `vram` (where the bytes land), purpose by caller |
   | `tier.read.bytes{tier=ram}` | source `ram`, destination `vram`, purpose `expert_miss` or `prefetch` |
   | `expert.load.duration{tier,source}` | `tier.transfer.duration` with the same ends |
   | `expert.admitted.bytes` | source `ram`, destination `vram`, purpose `admission` |
   | `expert.prefetch.wasted` | stays (a waste measure, not a transfer) |
   | `link.bytes` (planned, plan.md) | stays: device to device over a link, not a tier move |

   - Call sites to move (follow-up): `crates/memory/src/metrics.rs` (`tier_read_bytes` at the read completion, `expert_load_duration` at slot fill, `expert_admitted`), the prefetch issue path and admission upload in `crates/model/src/experts.rs`, and the KV prefix restore in the scheduler.

## Consequences

- `Metrics::time_per_output_token` and `request_duration` take the operation name; `crates/engine` passes `text_completion`.
- Panels: TTFT and TPOT p50 and p95 per model (legend by `gen_ai_request_model`), "Tier transfer bandwidth" with mean transfer time. No alert rule is added; the telemetry catalog test lists the two new names.
- Until the call sites move, the transfer series are empty on a live engine and the panel shows no data.
- Tests: `crates/gateway/tests/genai.rs` (fake engine, both routes, streamed and not, TPOT value, one-token reply, rejected request); the catalog test in `crates/telemetry/src/setup.rs`.
