# 0022. No OpenTelemetry exemplars for now

Status: Accepted

## Context

The plan wants exemplars from slow histogram buckets to traces, if opentelemetry-rust supports them well enough (Phase 0 spike). We tested opentelemetry_sdk 0.33.1 with opentelemetry-otlp 0.33.1: `crates/telemetry/tests/exemplars.rs` records 100 values into a histogram inside a sampled span and reads the exported data point.

Result: the data point has zero exemplars. The data model (`HistogramDataPoint::exemplars()`) and the OTLP mapping in opentelemetry-proto exist, but the SDK aggregators always fill an empty list, and there is no feature flag or exemplar filter/reservoir configuration. So neither the SDK nor OTLP export carries trace-linked exemplars today.

## Decision

Do not build on exemplars. Link metrics to traces by other means: slow-request traces are kept by tail sampling in the collector, and dashboards pivot on time range and the `gen_ai.request.model` attribute. The test asserts zero exemplars and fails once the SDK produces them, which is our signal to revisit.

Not chosen: a custom reservoir or a post-processing exporter that adds exemplars (our own code on an unstable SDK API), and Prometheus client exemplars (a second metrics path, against ADR 0012).

## Consequences

No click-through from a slow bucket to a trace in Grafana at first. Revisit on each opentelemetry-rust upgrade by running the test.
