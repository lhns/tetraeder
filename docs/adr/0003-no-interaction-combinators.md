# 0003. No interaction combinators, HVM, Bend2 or e-graph search

Status: Accepted

## Context

Dense matrix math with known shapes; these runtimes do not use tensor cores; Bend2 is experimental; e-graph gains are small here.

## Decision

No interaction combinators, HVM, Bend2 or e-graph search.

## Consequences

Their order-independent local rewriting survives only as the design rule for migration DAGs.
