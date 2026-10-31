# 0030. Own Qwen3 MoE graph builder: safe ggml surface in ggml-sys, builder in model, ggml's scheduler, one segment per MoE layer

Status: Accepted (Phase 2)

## Context

ADR 0014 and 0015 ask for an own builder for one MoE family (Qwen3 MoE, CUDA and CPU) so experts can live in slot tensors that the engine fills. libllama keeps full expert tensors and runs top-k inside its graph. Until Phase 5 (ADR 0016) ggml's backend scheduler issues the graph. `unsafe` is allowed only in `ggml-sys` and `memory`.

plan.md says a step runs as one graph when every expert hits. At the pin (b11976) quantized `MUL_MAT_ID` at batch 1 takes the MMVQ path and does not block CUDA graphs (`ggml_cuda_mul_mat_id_needs_sync`), so graphs are possible in principle. But a whole-step graph needs every routed expert of every layer resident: Qwen3-30B-A3B routes 8 of 128 experts in each of 48 layers, so at an 80% per-expert hit rate a layer hits fully about 17% of the time and a step practically never. Experts computed on the CPU (llama.cpp's `-ot` setup, the baseline to beat) split the graph at every MoE layer anyway.

## Decision

- **Safe ggml surface** in `ggml_sys::ggml`: context, tensor handles tied to their context's lifetime, the ops a Qwen3 MoE layer needs (as libllama's `build_moe_ffn` and attention use them), backend and buffer handles, and `ggml_backend_sched` (reserve, alloc, compute, eval callback). Raw pointers do not leave `ggml-sys`.
- **Builder** in `tetraeder-model` (no `unsafe`): reads hyperparameters and the tensor index from GGUF, fills `plan::ModelDesc`, builds the graph op for op like libllama for `qwen3moe`, and owns a simple contiguous per-layer KV cache until Phase 4.
- **One graph segment per MoE layer.** The step is built as segments that end at a layer's router top-k. Between segments the host reads the top-k ids (a few bytes), maps experts to slots, starts loads for misses, and launches the next segment with the remapped ids as an input tensor. Each segment can be a CUDA graph. This is the correct baseline; it pays what `-ot` already pays.
- **Whole-step graphs are a later optimization, gated by measurement:** a device-side expert-to-slot table (`get_rows` on a table tensor) and a miss bitmap read once per step, re-running only layers that missed. Built only if the measured per-layer full-hit rate makes it pay (roughly 98% or more); the hit-rate metrics below decide.
- **Parity:** against libllama on the same GGUF, per backend: top-k logits per position within a tolerance, and identical greedy sequences, with all experts resident and with experts streamed. Local tests use a synthetic random-weight `qwen3moe` GGUF (2 layers, 8 experts, top-2, tiny dims) written by `cargo xtask tiny-moe` with the vendored gguf-py; the real model runs on Vast.
- **First failing test:** the tiny fixture's logits from the builder match libllama's on CPU.

## Consequences

- plan.md's "sync only on a miss" becomes "one segment per MoE layer, whole-step graphs once hit rates allow".
- `tetraeder.step.duration` counts segments per step; ADR 0020 whole-step timing still holds per step.
- The engine gains a second backend path next to libllama; the gateway's engine trait (ADR 0028) serves both.
- Rebases must keep the builder in step with `build_moe_ffn`; the parity test catches drift.

## Implementation notes (step 1)

- Segment `l` runs the expert FFN of layer `l - 1`, the residual, then attention and the router of layer `l`, so a step is `n_layer + 1` segments. Activations that cross a boundary (residual, normed input, router weights, ids) live in two alternating state slots in backend memory; only the ids travel to the host and back.
- The GGUF header (hyperparameters, tensor index, expert slab ranges) is read by a small Rust parser in `tetraeder-model`, so the model crate needs no `unsafe` and no ggml gguf bindings. Tensor data is read by offset into backend buffers.
- Attention uses `flash_attn_ext` with an F16 cache laid out like libllama's (`[n_embd_kv, n_ctx]` per layer), K and V stores entering the graph before the attention.
- Parity tolerance and results: see `crates/model/tests/parity.rs`.
