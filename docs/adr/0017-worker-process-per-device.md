# 0017. One worker process per device

Status: Accepted; superseded in part by ADR 0069 (2026-11-15): one worker process per collective group, the whole box by default, instead of one per device.

## Context

A sticky CUDA error or driver fault must not take down the whole engine.

## Decision

One worker process per device.

## Consequences

IPC between workers through shared memory; supervisor restarts a failed worker.
