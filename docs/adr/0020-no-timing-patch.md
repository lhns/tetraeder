# 0020. No timing patch: whole-step timing with CUDA graphs in production, per-op timing in calibration runs without graphs, CUPTI or rocprof from outside in profiling mode

Status: Accepted

## Context

Timing inside captured graphs is unusable and disabling graphs changes what is measured.

## Decision

No timing patch: whole-step timing with CUDA graphs in production, per-op timing in calibration runs without graphs, CUPTI or rocprof from outside in profiling mode.

## Consequences

Two c0 values per op in the cost model.
