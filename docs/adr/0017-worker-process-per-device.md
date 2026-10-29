# 0017. One worker process per device

Status: Accepted

## Context

A sticky CUDA error or driver fault must not take down the whole engine.

## Decision

One worker process per device.

## Consequences

IPC between workers through shared memory; supervisor restarts a failed worker.
