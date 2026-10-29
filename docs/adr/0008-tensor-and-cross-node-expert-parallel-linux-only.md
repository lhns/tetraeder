# 0008. Tensor parallel and cross-node expert parallel are Linux only

Status: Accepted

## Context

NCCL, RCCL, RDMA and io_uring do not exist on Windows; single-machine expert parallel needs none of them.

## Decision

Tensor parallel and cross-node expert parallel are Linux only; expert parallel inside one Windows machine is allowed.

## Consequences

Windows nodes are clients, workers or pipeline stages.
