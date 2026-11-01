# 0036. The CPU backend keeps one threadpool; a step with every expert resident is one graph

Status: Accepted (Phase 2)

## Context

The run of 2026-11-01 (ADR 0035, Consequences) measured the builder all on CPU at 4.1 tok/s decode and 9.8 tok/s prefill on Qwen3-30B-A3B, where llama.cpp on the same box is several times faster. `tests/cpu_bench.rs` reproduces it on a synthetic 48-layer top-8 Q4_K_M MoE (bench/results/20261101-local): at 1 thread the builder matches libllama (decode ratio 0.97); at 8 threads it decodes at 0.09x libllama, 10 times slower than its own 1-thread run, with only about 9 ms per token in graph building and allocation. The time is inside `sched.compute`.

`Backend::cpu` set a thread count but no threadpool, and ggml is built without OpenMP. Then `ggml_graph_compute` creates a disposable threadpool for every graph: it spawns and joins `n - 1` threads per call. A step runs `n_layer + 1` segments (ADR 0030), so 48 thread creations and joins per token and thread count. libllama in the comparison and llama-bench attach a pool once.

With every expert resident the segments buy nothing: the host only copies the router's ids back into the next segment. They still cost a context, a graph build, a scheduler allocation and a compute launch per layer, and on a GPU a synchronization per layer (the resident GPU builder decoded 45 tok/s against libllama's 67 on the same box).

## Decision

- **`Backend::cpu` owns a ggml threadpool** for its lifetime (`ggml_threadpool_new`, `ggml_backend_cpu_set_threadpool`), freed after the backend. Every CPU compute, segments and group graphs, reuses its threads. No numeric change: the thread count is the same.
- **One graph per step when every expert is resident** (no streamer). It is the segments' ops in the same order, with the router's ids and weights feeding the expert matmuls directly instead of through the state slots. This is ADR 0030's whole-step graph for the case with no misses by construction; streaming keeps one segment per MoE layer. With `record_routing` the ids and weights of every layer are extra graph outputs. `segments_run` and `tetraeder.step.segments` count 1 per such step.
- `Qwen3Moe::take_profile` returns a host-side time breakdown (build, allocation, compute, readback, streamer) for the benchmark.

## Consequences

- Supersedes ADR 0030's one segment per MoE layer when every expert is resident.
- Local result (bench/results/20261101-local): the threadpool alone takes the builder at 8 threads from 0.25 to 2.5 tok/s decode; after both changes the 1-thread ratio to libllama is 1.14 decode and 1.08 prefill. The whole-step graph's gain is within this VM's noise; it removes 48 launches and synchronizations per token, which matters on a GPU and is measured on the box.
- Same logits bit for bit: the resident builder is the reference of the exactness tests (multipass, cpu_experts, slots), which compare it against the segmented streaming paths.
- The CPU policy (ADR 0035) still runs one segment per layer: its host work between layers is real. A RAM tier holding every expert could also become one graph with a device-side slot table; that waits for measurements on the box.
- The node order is part of the numerics on CUDA: ggml-cuda fuses by matching consecutive nodes. The first whole-step graph reached each router only through the expert matmuls' ids, so the order was `SOFT_MAX, ARGSORT, VIEW, MUL_MAT_ID...` and the topk-moe fusion (`ggml_cuda_topk_moe_fusion`, which needs `SOFT_MAX, RESHAPE, ARGSORT, VIEW, GET_ROWS, RESHAPE, SUM_ROWS, CLAMP, DIV, RESHAPE`) did not fire. libllama and the segments fuse it, so on the RTX 3090 the resident path differed from both while `GGML_CUDA_DISABLE_FUSION=1` made all three equal. The graph now expands each layer's router weights before the expert FFN, as `build_moe_ffn` does ("call early so that topk-moe can be used"). `graph_order` in `qwen3moe.rs` checks without a GPU that the step graph's ops equal the segments' ops in order and that every router and expert sum has the shape the CUDA matchers need.
- Not covered: llama.cpp repacks Q4_K weights for AVX2 (`q4_K_8x8_q8_K`, extra buffer type `CPU_REPACK`), the builder's CPU buffers do not. That changes numerics and is a separate decision, measured on the box with `tests/cpu_bench.rs`.
