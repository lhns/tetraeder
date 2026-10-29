# 0015. Own graph builders in Rust per family

Status: Accepted

## Context

llama.cpp's builders are internal C++ without a stable API.

## Decision

Own graph builders in Rust per family; libllama stays as parity reference and fallback.

## Consequences

Parity tests against libllama per family and backend.
