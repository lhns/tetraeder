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

## Implementation note: CUDA graphs and the resident step (2026-11-01)

On the RTX 3090 the resident builder decoded about 104 tok/s against libllama's 127 (`tests/real_moe.rs`, `resident` records routing, a confound handled on another branch). Reading ggml-cuda at the pin (b11976) shows why the builder never gets CUDA graphs:

- ggml-cuda keys a captured graph by the address of the split's first node (`ggml_cuda_graph_get_key`, ggml-cuda.cu 2803) and replays it only after a warmup call in which nothing changed (`ggml_backend_cuda_graph_compute`, 4923-4951). "Nothing changed" is a `memcmp` of every node's whole `ggml_tensor` struct (data, view offset, src pointers) and its sources' data, `ne` and `nb` (`ggml_cuda_graph_update_required`, 2807-2849). A graph with the same `uid` as last time skips the comparison (2813); `ggml_backend_sched` gives every split a new `uid` on each `alloc_graph` (ggml-backend.cpp 1751).
- The builder stored K and V with `cpy(cur, view(cache, ..., n_past * row))` (qwen3moe.rs, `attention_and_router`). The view's data pointer moves one row per decode step, so the properties change on every call, warmup never completes and every step runs eagerly, about 2,500 kernel launches. This holds whatever else is reused. In addition the step graph lived in a fresh `Context` per step, so the key and every src pointer only stay equal if the allocator happens to return the same block.
- libllama writes the cache with `ggml_set_rows(cache, cur, idxs)` with the row indices as an input (llama-kv-cache.cpp 1355, 1390), so the topology and every pointer are constant, and reuses the previous graph and its scheduler allocation when the shapes allow (llama-context.cpp 1539-1576): same `uid`, no rebuild, no split, no comparison.
- Not a blocker: quantized `MUL_MAT_ID` at batch 1 takes MMVQ and does not need a stream sync (`ggml_cuda_mul_mat_id_needs_sync`, 2078-2096), so the step graph is capturable.

Changes:

- K and V are stored with `set_rows(cache, cur, pos)`; `pos` already holds the step's cache rows. Same values (the same F32 to F16 conversion), segments and whole-step graph alike. libllama uses I64 indices and a VIEW, which also lets CUDA fuse RoPE into the store (`ggml_cuda_should_fuse_rope_set_rows`); that fusion does not change values and is not needed for graph capture.
- `LoadOptions::reuse_step_graph` (default off until the box run) keeps the whole-step graph, its context and its scheduler allocation while the token count, padded KV length, `last_only` and routing recording stay the same: a decode step only writes inputs and computes. `ggml-sys` gains `OwnedGraph`, which holds every context its tensors live in, so computing a kept graph stays sound.
- `tests/gpu_graphs.rs` and `scripts/graphs_check.sh` measure decode on the box: libllama and the builder (reuse off and on, routing not recorded), each process with CUDA graphs on, off (`GGML_CUDA_DISABLE_GRAPHS=1`) and with `GGML_CUDA_GRAPH_DEBUG=1`, whose captures and uid reuses the test counts from ggml's log.

Expected on the box: libllama clearly slower with graphs off; the builder with reuse off about as fast with graphs on as off if the context address changes between steps; with reuse on, one capture per shape, a uid reuse on every later decode step, and decode close to libllama. If libllama is not faster with graphs on, CUDA graphs are not the gap and the decode profile (build, allocation, readback) is the next place to look.
