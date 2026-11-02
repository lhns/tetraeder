# 0039. No AVX2 repack of the builder's CPU weights for now

Status: Accepted (Phase 2). Revisit for CPU-only prefill (Phase 9 breadth) or if ggml's repack buffer becomes offloadable.

## Context

llama.cpp allocates CPU-resident Q4_K weights in ggml's `CPU_REPACK` extra buffer type, which interleaves 8 rows (`q4_K_8x8_q8_K`, AVX2) on write; the builder's CPU buffers and RAM tier are plain. ADR 0036 left this open. Measured on the box (bench/results/20261101/cpu-bench-real.json, EPYC 7K62, 11 threads, Qwen3-30B-A3B Q4_K_M, CPU only): the builder against libllama (which repacks) is 0.82x on prefill and 1.30x on decode (`builder_cpupolicy` 0.78x and 1.22x).

Reading ggml at the pin (b11976):

- `CPU_REPACK`'s `set_tensor` asserts a whole-tensor write (`offset == 0`, `size == nbytes`), but a view's `init_tensor` gets its own repack traits, and each expert's slab maps to the same bytes in the repacked layout (rows interleave in groups of 8 within one expert). So per-slot writes work through a 3D view of one slot without a ggml patch.
- The repack buffer type has no `is_host`, so `ggml_backend_sched` never op-offloads an op whose weights sit there (ggml-backend.cpp, the `op_offload && ggml_backend_buffer_is_host` check). The exit regime's prefill depends on op offload: 67.8 tok/s with it over a 32K prompt, 18.0 without over 2K (exit-main and exit-cpu-prefill-120520.json).
- The repacked kernels change rounding, so the CPU groups would no longer be bitwise equal to the resident builder (ADR 0034, 0035) unless the resident CPU weights repack too.

## Decision

Keep the builder's CPU weights and RAM tier unpacked. In the GPU-plus-RAM regime that Phase 2 targets, repacking would turn off op offload for the RAM tier (prefill several times slower) to speed up batch-1 CPU groups, where the builder already decodes 1.3x libllama on CPU. The CPU-only prefill gap (0.82x) is real but outside the Phase 2 exit test.

## Consequences

- No ggml patch and no new numerics; the exactness tests stay as they are.
- If CPU-only prefill matters later, the measured path is: repack per slot through views, only for `BackendChoice::Cpu` and only with op offload off, gated on `tests/cpu_bench.rs` showing prefill gains without the decode loss.
