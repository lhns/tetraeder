# 0018. Transport is chosen by measurement in Phase 0 (quinn, iroh, TCP with TLS, RDMA where present)

Status: Accepted

## Context

Userspace QUIC costs latency and CPU per hop, which matters for per-layer expert round trips.

## Decision

Transport is chosen by measurement in Phase 0 (quinn, iroh, TCP with TLS, RDMA where present).

## Consequences

ADR records the measured numbers.
