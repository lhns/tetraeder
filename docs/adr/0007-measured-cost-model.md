# 0007. Cost model from measured per-op curves, c0 plus f(M), with c0 per graph mode

Status: Accepted

## Context

Fixed per-op cost dominates decode at batch 1; ranking matters more than absolute error.

## Decision

Cost model from measured per-op curves, c0 plus f(M), with c0 per graph mode; planner scored by regret and rank correlation over a defined candidate set.

## Consequences

Calibration runs, residual tracking, roofline only as a prior.
