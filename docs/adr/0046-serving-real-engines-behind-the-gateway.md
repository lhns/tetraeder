# 0046. Serving: one-sequence scheduler over logits, Rust sampling, prefix reuse, request log

Status: Accepted (Phase 3b). Partly supersedes ADR 0028 (mask hook, context check); Phase 4 replaces the scheduler loop.

## Context

ADR 0028 built the gateway against a fake engine and left `tetraeder serve` for when an engine exists. Two exist now: the libllama engine (`crates/engine`, dense models) and the Qwen3 MoE builder (`crates/model`), which only maps tokens to logits and has no tokenizer or sampler. Phase 3b's exit test also asks for a week of Claude Code use, which sends 15K to 30K tokens of system prompt and tools on every turn, `max_tokens` of 32K, its own `claude-*` model ids, and occasional mid-conversation `system` messages.

## Decision

**Backend trait at the logits level** (`crates/scheduler`, the crate the plan gives continuous batching in Phase 4): `n_ctx()`, `n_past()`, `truncate(n)` (keep the first `n` cached positions), `eval(tokens) -> last-position logits` (prefill chunked inside). Implemented by the libllama engine (new `Engine::eval`/`truncate` that go through its timed `step`, plus `llama_memory_seq_rm` in `ggml-sys`) and by `Qwen3Moe` (new `truncate`, which only lowers `n_past`: later positions are masked and overwritten). The scheduler does not call `tetraeder_engine::Engine::generate`, which records request metrics the gateway already records.

**Sampling in Rust over the logits.** Supersedes ADR 0028's custom libllama sampler: the MoE builder has no libllama sampler, and one sampler for both backends keeps their greedy output comparable. Per step: `TokenConstraint::mask()` sets disallowed logits to minus infinity, then greedy (argmax, lowest id on ties, as libllama) or temperature with top-k and top-p and a seeded SplitMix64; then `commit(token)`. Defaults when the API sends none are llama-server's (temperature 0.8, top-k 40, top-p 0.95), overridable per `serve` flag.

**Tokenizer and vocab from libllama for both backends**: the loaded model for dense, a `vocab_only` load for the MoE builder. `Vocab.tokens` from `token_to_piece(special = true)`, `special` from the control attribute, `eog` from `is_eog`; chat template from GGUF `tokenizer.chat_template` unless `--chat-template-file` overrides it.

**One worker thread per model**, owning the backend (loaded on that thread), taking requests from a queue. The gateway's semaphore stays at one request in generation. Per token: sample, send `GenEvent::Token` with `blocking_send` (bounded channel: backpressure), stop on cancel or a closed receiver, eval the token.

**Prefix reuse of the last sequence.** Before prefill the worker keeps the longest common prefix of the cached tokens and the new prompt (at most `prompt - 1`, so one token is evaluated for logits), truncates the rest and evaluates only the suffix. `EngineStats.cached_tokens` reports the kept length, which the gateway already meters as cache hits. This is what makes agent turns, which resend the whole conversation, cost only the new tokens. A prefix tree over many sequences is Phase 4 (`kvcache`).

**Stats.** `gpu_seconds`: summed eval wall time on a GPU backend, `None` on CPU. `bytes_streamed`: the delta of the builder's expert `bytes_read`, `None` for libllama.

**Gateway changes for real clients.**
- `max_tokens` beyond the remaining context is clamped to it (finish reason `length` when the context fills), as llama-server does; supersedes ADR 0028's 400. A prompt that alone fills the context stays a 400.
- `--alias` names the served model; `--default-model` routes unknown model ids (Claude Code's `claude-*` ids) to it. Responses name the served model.
- Anthropic `role: "system"` messages inside `messages` become user turns wrapped in `<system>` tags (chat templates allow one leading system message).
- The `x-tetraeder-generation-id` response header carries the generation id on both APIs, for the request log.

**Request log** (`serve --request-log <file>`): JSON Lines, one line per request to the generation and `count_tokens` endpoints, written when the response body ends: `v` (1), `ts_ms` (arrival, Unix ms), `method`, `path`, `query`, `headers` (only `anthropic-version`, `anthropic-beta`, `user-agent`; never keys or authorization), `status`, `stream`, `request` (the body as JSON), `first_byte_ms`, `total_ms`, `generation_id`, `usage` (`prompt_tokens`, `cached_tokens`, `completion_tokens`), `cost` (`gpu_seconds`, `bytes_streamed`), `error_type`, and `response` (the JSON body, or the raw SSE text when streamed). `scripts/gateway_replay.py` resends the `request` bodies with the logged inter-arrival times (scaled, or as fast as allowed) and records per-request TTFT and usage: the Phase 4 scheduler is measured on the same traffic.

**`tetraeder serve`**: `-m <gguf>` with the `run` placement flags, `--engine auto|llama|moe` (auto: the builder for GGUF architecture `qwen3moe`, else libllama), `--ctx`, `--host`, `--port`, `--alias`, `--default-model`, sampling defaults, `--max-queued`, `--request-log`, `--chat-template-file`. `--fake` serves the gateway's fake engine (no GGUF) for the SDK conformance suite in CI; its script answers with the text after the last `FAKE:` in the prompt's last user turn, else echoes that turn.

**Telemetry.** Gateway metrics as ADR 0028; engine step metrics from each backend. New: `tetraeder.scheduler.batch.size` (histogram, 1 until Phase 4) and `tetraeder.kv.prefix.hits` (counter, requests that reused a prefix). Dashboard panels for prompt tokens served from the prefix cache and batch size; alert `GatewayStalled` when requests wait but no output tokens are produced for 5 minutes.

## Consequences

- Concurrency above one queues: throughput at concurrency 4 equals concurrency 1 until Phase 4's continuous batching; llama-server with `-np 4` batches. The benchmark reports this gap as Phase 4-bound.
- Sampling costs a pass over the vocabulary in Rust per token (about 0.1 ms greedy at 152K tokens); measured in the paired benchmark against llama-server.
- Prefix reuse helps one conversation at a time; two interleaved conversations evict each other.
- The request log holds prompts and code in clear text; it is opt-in and local.
