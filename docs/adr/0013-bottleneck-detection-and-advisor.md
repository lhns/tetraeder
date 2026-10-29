# 0013. Bottleneck detection checks the environment first, picks the regime (load or batch 1), and outputs a ranked list with confidence or "balanced"

Status: Accepted

## Context

Demand explains throughput under load, critical path explains batch 1; a detector without false-positive control is not trustworthy.

## Decision

Bottleneck detection checks the environment first, picks the regime (load or batch 1), and outputs a ranked list with confidence or "balanced"; the advisor acts by policy with verification and rollback.

## Consequences

Fault-injection fixtures with no-fault and two-fault controls.
