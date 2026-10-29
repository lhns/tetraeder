# 0016. An own executor replaces ggml's backend scheduler in Phase 5

Status: Accepted

## Context

ggml issues splits sequentially from one host thread; tensor parallel needs concurrent per-device subgraphs with NCCL group semantics.

## Decision

An own executor replaces ggml's backend scheduler in Phase 5.

## Consequences

The all-reduce patch is only useful together with it.
