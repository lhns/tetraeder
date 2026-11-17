# 0075. Executor ranges are views of the rank graph (patch 0005)

Status: Proposed (2026-11-17). Amends ADR 0069 (range graphs).

## Context

- The executor cut each rank graph into ranges by filling a new `ggml_cgraph` per range with `ggml_graph_add_node` (ADR 0069). Such a graph has an empty visited set, so `ggml_node_get_use_count` (`ggml-impl.h:706` at b12426) reads 0 for every node and `ggml_can_fuse` and `ggml_can_fuse_subgraph` refuse every fusion. The scheduler's splits are `ggml_graph_view` slices (`ggml-backend.cpp:1901`) that share the source graph's use counts and visited set, so they fuse.
- On CUDA the fused kernels are not all bit-equal to the separate ops. The Q/K norm and RoPE fusion (`rms_norm_mul_rope_f32`, `rope.cu:729`) sums the squares of a 128-wide head with 256 scalar threads; the separate `rms_norm_f32` takes the float4 path with 32 threads (`norm.cu:452-468`). The norm scales differ in the last bits. So the one-rank executor was not bit-equal to the scheduler on the S2 box (`bench/results/s2-20261117-step0/phase5/`, max delta 0.729 over the 6-token prompt of Qwen3-8B Q4_K_M), while CPU CI passed: the CPU's only fusion (`rms_norm` with `mul`, `ggml-cpu.c:3089`) computes `x * scale * w` as the separate ops do.
- Every executor range also ran unfused on CUDA (norm, rope, MMVQ gate/up/GLU and bias fusions), which slows each rank against libllama's `-sm tensor` in the step-0 check (ADR 0069).
- `ggml_graph_view` is internal (`ggml-impl.h`, no `GGML_API`) and returns the graph by value; `struct ggml_cgraph` is opaque in the public headers.

## Decision

- Patch 0005 adds `GGML_API struct ggml_cgraph * ggml_graph_view_new(ctx, cgraph, i0, i1)` to `ggml.h` and `ggml.c`: a graph object in `ctx` holding `ggml_graph_view(cgraph, i0, i1)`. Eight lines, no change to any existing function.
- `Graph::range` and `RangeGraphs` build ranges with it. A range shares the rank graph's node array, use counts and visited set, so a backend fuses in a range what it would fuse in the whole graph, and never across a cut: a node read in a later range keeps that use in its count.
- Not taken: mirroring `struct ggml_cgraph` in Rust (a silent layout break on a pin move); a C shim against `ggml-impl.h` (the same internal dependency, outside the patch queue's checks).

## Consequences

- The one-rank executor runs the scheduler's fused kernels. Bit-equality with the scheduler is still not guaranteed for every shape: the scheduler also calls the backend's `graph_optimize`, which on CUDA adds allocation dependencies so the fused gate/up/GLU MMQ kernel (prefill above 8 rows) can run; under the executor's `Gallocr` the memory-range check may refuse that fusion. `gpu_one_rank_is_bit_equal_to_the_scheduler` (6-token prompt, MMVQ) checks the case that matters for the step-0 numbers.
- CUDA graph capture is unchanged: ranges have no `uid` either way and are keyed by their first node.
- Test: `ranges_fuse_as_the_whole_graph` (`crates/ggml-sys/tests/exec_bindings.rs`, CPU): the CPU's `rms_norm` plus `mul` fusion leaves the norm's own output unwritten in the whole graph and in a range that holds both nodes; a range filled with `ggml_graph_add_node` writes it.
- One more patch to rebase on a pin move; upstream-clean, not upstreamed (Rule 8).
