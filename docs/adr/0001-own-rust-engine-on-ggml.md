# 0001. Own Rust engine on ggml, reached by a strangler path from libllama

Status: Accepted

## Context

Tiered memory, expert streaming, one plan for all modes and deep observability need control no existing engine gives.

## Decision

Own Rust engine on ggml, reached by a strangler path from libllama; vLLM or SGLang only as an external process behind the gateway.

## Consequences

libllama for dense models first; own pieces replace it phase by phase.
