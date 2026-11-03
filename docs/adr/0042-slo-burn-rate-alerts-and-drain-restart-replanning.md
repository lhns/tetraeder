# 0042. SLO burn-rate alerts and drain-and-restart replanning

Status: Proposed (Phase 3a)

## Context

docs/plan.md "Alerts" asks for multi-window burn-rate alerts on time to first token, time per output token and errors instead of one alert per component; `deploy/alerts/engine.yml` defers them to Phase 3a. ADR 0009 makes replanning drain-and-restart until Phase 7. The planner's "Replan online" step needs a trigger, a switch rule and the stability hints (`max_switches_per_hour`, freeze windows).

## Decision

- **SLOs** in `deploy/alerts/slo.yml` (vmalert): per SLI a good-event ratio from the existing histograms and counters: TTFT at or below a threshold bucket, TPOT at or below a threshold bucket, requests without an engine error. Default objective 99% over 30 days. Recording rules give the error ratio over 5m, 30m, 1h, 6h, 1d and 3d. Alerts follow the multi-window pattern: page at burn rate 14.4 over 1h and 5m, page at 6 over 6h and 30m, ticket at 1 over 3d and 6h. A planner alert fires when the predicted versus measured residual stays above the band (recalibrate).
- **Replanning** in `control`: a `Replanner` takes triggers (residual drift, SLO burn, load shift, device change), asks the planner for a new plan and its predicted gain, and switches only when gain over the horizon exceeds switch cost times a safety factor (default 2), the stability hints allow it and the last switch is older than a minimum dwell (default 10 minutes). A switch runs through an `EngineControl` trait: stop admitting, wait for in-flight requests up to a timeout, restart with the new plan, health check, resume; a failed restart rolls back to the old plan. Cheap moves (knobs only) are preferred over reloads when both are offered.
- **Metrics:** `tetraeder.replan.total` (trigger, outcome), `tetraeder.replan.duration`, `tetraeder.plan.residual`, `tetraeder.plan.regret`; a dashboard panel for predicted versus measured and replans.

## Consequences

- Requests are refused (backpressure) while a switch drains, so a switch costs availability; the dwell and hysteresis keep that rare.
- Live reshard (Phase 7) replaces the restart step behind the same trait.
