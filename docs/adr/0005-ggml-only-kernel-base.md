# 0005. Inside the engine, ggml is the only kernel base

Status: Accepted

## Context

Three kernel ecosystems have incompatible KV and quant layouts; FlashInfer has no stable C ABI.

## Decision

Inside the engine, ggml is the only kernel base; no FlashInfer or CUTLASS.

## Consequences

Peak NVIDIA throughput comes only from an external vLLM or SGLang process (ADR 0001).
