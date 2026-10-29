# 0006. One placement plan IR

Status: Accepted

## Context

Every mode is a shape of one structure; the planner must not propose what cannot run.

## Decision

One placement plan IR; plan diffs compile to migration DAGs; each phase declares which fields it can execute.

## Consequences

Validator, property tests, per-phase capability table.
