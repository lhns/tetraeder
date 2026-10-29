# 0026. Phase 1 engine interface: safe libllama wrapper in ggml-sys, engine loop in a new engine crate

Status: Accepted

## Context

Phase 1 drives libllama (b11976, vendored) for dense models: load, tokenize, prefill, decode with sampling, stream, placement through llama.cpp parameters, step timing into telemetry, and a bench adapter for the exit test (within 5% of llama-bench at the same parameters, production telemetry overhead under 2%). `unsafe` is allowed only in `ggml-sys` and `memory` (AGENTS rule 7). The `executor` crate is reserved for the Phase 5 executor and `model` for own graph builders.

What the b11976 C API does and does not offer:

- Placement: `n_gpu_layers`, `split_mode` (none, layer, row, tensor), `main_gpu`, `tensor_split`, `devices`, `tensor_buft_overrides` (regex pattern to buffer type), `load_mode` (auto, none, mmap, mlock, mmap+mlock, direct I/O; replaces `use_mmap`/`use_mlock`).
- Per-tensor *type* overrides exist only in `llama_model_quantize_params.tt_overrides`, not at load. Converting at load is Phase 2 (ADR 0024).
- The memory breakdown per device (`llama_get_memory_breakdown`) is not in `llama.h`; only `llama_model_size` and device free/total memory are public.

## Decision

- **`ggml-sys`** keeps the raw bindings in `ggml_sys::raw` (re-exported at the root, lints relaxed only there) and gains a safe module `ggml_sys::llama`: `Model`, `Context`, `Batch`, `Sampler`, `ThreadPool`, `ModelParams`, `ContextParams`, device and buffer-type lookup by name, `quantize` with per-tensor type overrides, and GGUF tensor-type listing. Every handle frees itself on drop; params structs start from libllama's defaults; buffers that libllama keeps pointers to (override patterns, tensor split) live as long as the `Model`. libllama logs go to `tracing`. `llama_backend_init` runs once; the backend is never freed in-process.
- **`tetraeder-engine`** (new crate, no `unsafe`) owns the generation loop: typed `Placement` and `ContextConfig`, `Engine::load`, `generate` with greedy or seeded random sampling that streams UTF-8-safe text pieces through a callback, and `bench_prompt`/`bench_gen` that mirror llama-bench's timed loops (random tokens, BOS first if the model adds one, no sampler, synchronize after the prompt and after every generated token, memory cleared between repeats, a threadpool attached like llama-bench does).
- **Step timing:** one step is one `llama_decode` call followed by `llama_synchronize`, timed on the host (ADR 0020). Prefill is chunked by `n_batch`. `graphs` is on when layers are offloaded to CUDA and `GGML_CUDA_DISABLE_GRAPHS` is unset. Steps feed the ring and `tetraeder.step.duration`; requests feed TTFT, TPOT, request duration and `tetraeder.tokens`; weights bytes feed `tetraeder.memory.used{kind=weights}`. Traces: one `tracing` span per request (`gen_ai.operation.name`, `gen_ai.request.model`, usage counts) with `prefill` and `decode` child spans, no span per decode step.
- **Telemetry modes:** `off` records nothing; `on` records metrics, ring and spans into in-process providers, plus OTLP export only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. The overhead pair compares off and on without an endpoint.
- **CLI:** `tetraeder run --model ... --prompt ...` streams a completion; `tetraeder bench` takes the harness workload (`-p`, `-n`, `-b`, `--ubatch`, `-t`, `-r`) and prints llama-bench's JSON row shape, so the harness parses both engines with one parser. Placement flags are the same for both commands: `--ngl`, `--split-mode`, `--main-gpu`, `--tensor-split`, `--override-tensor <regex>=<buffer type>`, `--load-mode`.
- **Bench harness:** `EngineKind::Tetraeder` gets an argument builder; invocation and parsing are shared with llama-bench.
- **Parity:** greedy text of `llama-simple` (built from the vendored tree by `cargo xtask llama-tools`) is the reference; a bench-format golden (tokens plus top-k logits) guards regressions. Both recorded by `cargo xtask golden` on CPU.

## Consequences

- One safe surface over libllama for every later caller; the Phase 2 own builders reuse `ggml-sys` but not `tetraeder-engine`'s libllama path.
- Per-tensor type choice at load needs a quantize-to-cache step (Phase 2), not a load flag.
- Per-device KV and workspace bytes are not reported until a patch or the own KV cache (Phase 4) provides them.
- The exit test (within 5%, overhead under 2%) needs a paired GPU run; CPU pairs run locally. The overhead pair is reported, not gated, until a band below the box noise is measured.
- When llama-bench's timed loop changes on a rebase, `bench_prompt`/`bench_gen` must follow.
