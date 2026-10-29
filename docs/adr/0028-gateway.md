# 0028. Gateway: axum, a narrow token-level engine trait, minijinja templates, llguidance constraints

Status: Accepted

## Context

Phase 3b (plan rows 32, 33, 44, 47) needs OpenAI- and Anthropic-compatible HTTP APIs with streaming, chat templates from GGUF metadata, tool calls, structured output, cancellation, backpressure and per-request cost accounting. The engine (Phase 1, libllama) is built in parallel, so the gateway needs a contract it can be tested against now and that a small Phase 1 adapter can implement later.

Two gateway features need more than "text in, text out": token counting (`count_tokens`, context checks, usage) needs the tokenizer, and structured output needs a token mask applied before every sampling step. llama.cpp's sampler chain accepts custom samplers (`llama_sampler_i::apply` sees the candidate array), so a mask from outside costs one small custom sampler in the adapter.

## Decision

**HTTP stack.** axum 0.8 on tokio, SSE through `axum::response::sse`. One router serves both APIs; `GET /v1/models` answers in the Anthropic shape when the request carries an `anthropic-version` header, else in the OpenAI shape. Routes: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/models/{id}`, `/v1/messages`, `/v1/messages/count_tokens`. No auth yet (Phase 9); keys are accepted and ignored.

**Engine trait** (`crates/gateway/src/engine.rs`), tokens in, tokens out:

```rust
pub trait Engine: Send + Sync + 'static {
    fn models(&self) -> Vec<ModelInfo>;            // id, context length, GGUF chat template, bos/eos strings
    fn vocab(&self, model: &str) -> Result<Arc<Vocab>, EngineError>;   // bytes per token id, end-of-generation ids
    fn tokenize(&self, model: &str, text: &str) -> Result<Vec<TokenId>, EngineError>; // special tokens parsed, no BOS added
    fn generate(&self, req: GenerateRequest) -> Result<mpsc::Receiver<GenEvent>, EngineError>;
}
```

`GenerateRequest` carries the prompt tokens, sampling parameters, an optional `Box<dyn TokenConstraint>` (`mask()` before each step, `commit(token)` after) and a `CancellationToken`. The engine sends `GenEvent::Token(id)` per sampled token and ends with `GenEvent::Finished { reason, stats }` or `GenEvent::Error`. `EngineStats` holds prompt, cached and generated token counts plus `gpu_seconds` and `bytes_streamed` (weights moved between tiers), both `Option` so the engine fills what it measures. The channel is bounded: a slow client slows the engine instead of buffering without limit. The engine stops when the token is cancelled or the receiver is dropped.

Detokenization happens in the gateway from `Vocab` bytes (UTF-8 assembled across tokens), so the trait has no `detokenize`. The Phase 1 adapter is: `models` and `vocab` from GGUF metadata and `llama_token_to_piece`, `tokenize` via `llama_tokenize(parse_special=true, add_special=false)`, and a decode loop on a thread with one custom sampler that applies the mask. The mask hook is the only part beyond glue; it is kept because the alternative (llama.cpp's own llguidance or GBNF sampler) would put grammar construction inside the engine and duplicate the tool-call formats the gateway already owns.

**Admission and backpressure.** A semaphore bounds requests in generation; a bounded wait queue sits in front. A full queue is rejected at once: 429 `rate_limit_exceeded` on the OpenAI surface, 529 `overloaded_error` on the Anthropic surface, both with `Retry-After`. Queue depth is a gauge (`tetraeder.scheduler.queue.depth`). Prompt plus `max_tokens` beyond the context length is a 400 before admission. Phase 4's scheduler replaces the semaphore with continuous batching behind the same trait.

**Cancellation.** Each generation holds a drop guard that cancels its token. A client disconnect drops the SSE body stream or the handler future, which fires the guard; stop sequences and finished tool calls cancel the same way.

**Chat templates.** minijinja 3 with `minijinja-contrib` pycompat (Python string and dict methods used by HF templates), loop controls, `tojson`, and the HF extras `raise_exception` and `strftime_now`. Context: `messages`, `tools`, `add_generation_prompt`, `bos_token`, `eos_token`, plus per-request kwargs (`enable_thinking`). Messages are normalised to the HF shape first (assistant `tool_calls` with `arguments` as a JSON object, `tool` role messages, `reasoning_content`). Rendering is checked against Python jinja2 on templates from llama.cpp's corpus (`vendor/llama.cpp/models/templates`).

**Tool calls.** Tools are rendered by the template. Output parsing is per family, picked from the template text: Hermes and Qwen (`<tool_call>` JSON), Qwen3-Coder (`<tool_call><function=...>` XML), Llama 3.x (bare JSON or `<|python_tag|>`), Mistral (`[TOOL_CALLS]`). Unknown templates fall back to Hermes. `<think>` blocks become Anthropic `thinking` blocks and OpenAI `reasoning_content`. Forced tool use (`required`, `any`, a named tool) is enforced by a grammar of the family's call syntax.

**Structured output.** llguidance (pure Rust, builds on Windows and Linux; checked) turns `response_format` (OpenAI) and `output_config.format` (Anthropic) JSON schemas, and forced tool calls, into a `TokenConstraint` over the engine's `Vocab`.

**Telemetry.** Per request: `gen_ai.server.request.duration` (with `error.type`), `time_to_first_token`, `time_per_output_token`, `tetraeder.tokens`, and `tetraeder.request.cost` per resource (GPU-seconds, bytes streamed) and caller. One span per request named `{gen_ai.operation.name} {gen_ai.request.model}` with GenAI attributes (usage, finish reasons, response id). A `CostSink` hook receives every finished request's cost record for external accounting. The caller is the OpenAI `user` or Anthropic `metadata.user_id` field until auth exists.

## Consequences

- The Phase 1 adapter implements four methods and one custom sampler; the gateway is tested end to end against a fake engine with a byte-level vocabulary that honours the mask.
- Tool-call families beyond the four listed need a parser each; models with other formats get plain text.
- Images, audio, embeddings, `n > 1` and logprobs are rejected with 400 until Phase 9.
- `count_tokens` and usage are exact for the local tokenizer, not for the remote vendor's.
- No `tetraeder serve` command until an engine exists.
