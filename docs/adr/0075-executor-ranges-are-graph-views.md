# 0075. Executor ranges are views of the rank graph (patch 0005)

Status: Proposed (2026-11-17). Amends ADR 0069 (range graphs). The GPU parity gates, including `gpu_one_rank_is_bit_equal_to_the_scheduler`, passed on the S2d box with patch 0005 (`s2d-20261118-step0`).

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

## Context

- The executor cut each rank graph into ranges by filling a new `ggml_cgraph` per range with `ggml_graph_add_node` (ADR 0069). Such a graph has an empty visited set, so `ggml_node_get_use_count` (`ggml-impl.h:706` at b12426) reads 0 for every node and `ggml_can_fuse` and `ggml_can_fuse_subgraph` refuse every fusion. The scheduler's splits are `ggml_graph_view` slices (`ggml-backend.cpp:1901`) that share the source graph's use counts and visited set, so they fuse.
- On CUDA the fused kernels are not all bit-equal to the separate ops. The Q/K norm and RoPE fusion (`rms_norm_mul_rope_f32`, `rope.cu:729`) sums the squares of a 128-wide head with 256 scalar threads; the separate `rms_norm_f32` takes the float4 path with 32 threads (`norm.cu:452-468`). The norm scales differ in the last bits. So the one-rank executor was not bit-equal to the scheduler on the S2 box (`s2-20261117-step0`), while CPU CI passed: the CPU's only fusion (`rms_norm` with `mul`, `ggml-cpu.c:3089`) computes `x * scale * w` as the separate ops do.
- Every executor range also ran unfused on CUDA (norm, rope, MMVQ gate/up/GLU and bias fusions), which slows each rank against libllama's `-sm tensor` in the step-0 check (ADR 0069).
- `ggml_graph_view` is internal (`ggml-impl.h`, no `GGML_API`) and returns the graph by value; `struct ggml_cgraph` is opaque in the public headers.

## Decision

- Patch 0005 adds `GGML_API struct ggml_cgraph * ggml_graph_view_new(ctx, cgraph, i0, i1)` to `ggml.h` and `ggml.c`: a graph object in `ctx` holding `ggml_graph_view(cgraph, i0, i1)`. Eight lines, no change to any existing function.
- `Graph::range` and `RangeGraphs` build ranges with it. A range shares the rank graph's node array, use counts and visited set, so a backend fuses in a range what it would fuse in the whole graph, and never across a cut: a node read in a later range keeps that use in its count.
- Not taken: mirroring `struct ggml_cgraph` in Rust (a silent layout break on a pin move); a C shim against `ggml-impl.h` (the same internal dependency, outside the patch queue's checks).

## Consequences

- The one-rank executor fuses as the scheduler does. Bit-equality is still not guaranteed for every shape: the scheduler also calls the backend's `graph_optimize` (`ggml-cuda.cu:5507`), which on CUDA adds allocation dependencies (for the fused gate/up/GLU MMQ kernel, shared experts and top-k MoE), so its layout differs from the executor's `Gallocr`, and the memory-range check that guards the MMVQ and MMQ GLU fusions may pass on one side and refuse on the other. `gpu_one_rank_is_bit_equal_to_the_scheduler` (6-token prompt and 64 decode steps) decides it on a box.
- MoE: expert-parallel ranges now fuse too (`mul_mat_id` with GLU, top-k MoE, the weighted expert sum), with patch 0004's id -1 rows, a combination no path ran before. Patch 0004's zero rows are written by the same MMVQ kernel in its fused variant; `gpu_sharded_moe` must pass before expert parallel is trusted; it did not run on S2 (stopped at the dense failures) and first passed on the S2d box (`s2d-20261118-step0`).
- CUDA graph capture is unchanged: ranges have no `uid` either way and are keyed by their first node.
- Test: `ranges_fuse_as_the_whole_graph` (`crates/ggml-sys/tests/exec_bindings.rs`, CPU): the CPU's `rms_norm` plus `mul` fusion leaves the norm's own output unwritten in the whole graph and in a range that holds both nodes; a range filled with `ggml_graph_add_node` writes it.
- One more patch to rebase on a pin move; upstream-clean, not upstreamed (Rule 8).
