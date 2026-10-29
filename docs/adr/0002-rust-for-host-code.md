# 0002. Rust for all host code

Status: Accepted

## Context

Performance lives in kernels; one static binary for Windows and Linux, memory safety, no GC pauses, the needed crates.

## Decision

Rust for all host code; kernels stay in C, C++ and CUDA.

## Consequences

`ggml-sys` FFI; no Python at runtime.
