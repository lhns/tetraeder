# 0011. Ggml is modified only through a private ordered patch queue

Status: Accepted

## Context

ggml changes almost daily; nothing is upstreamed for now.

## Decision

ggml is modified only through a private ordered patch queue; one pin, a llama.cpp commit that vendors ggml; never a long-lived fork.

## Consequences

One weekly rebase job for libllama binding and patches; patches kept clean so upstreaming stays possible later.
