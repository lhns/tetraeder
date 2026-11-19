# 0046. Serving: one-sequence scheduler over logits, Rust sampling, prefix reuse, request log

Status: Accepted (Phase 3b). Partly supersedes ADR 0028 (mask hook, context check); Phase 4 replaces the scheduler loop (partly superseded by ADR 0065: backend trait, scheduler loop, one request in generation). The Phase 3b proxy bar is met: all three clauses in `20261110d`; in `20261111d` the cache and non-200 clauses, the tasks clause over all runs under the owner's 2026-11-11 rule. The first proxy run (`20261103`) missed the cache and 400 clauses, fixed by ADR 0049.

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

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

**Request log** (`serve --request-log <file>`): JSON Lines, one line per request to the generation and `count_tokens` endpoints, written when the response body ends: `v` (1), `ts_ms` (arrival, Unix ms), `method`, `path`, `query`, `headers` (only `anthropic-version`, `anthropic-beta`, `user-agent`; never keys or authorization), `status`, `stream`, `request` (the body as JSON), `first_byte_ms`, `total_ms`, `generation_id`, `usage` (`prompt_tokens`, `cached_tokens`, `completion_tokens`), `cost` (`gpu_seconds`, `bytes_streamed`), `tool_outcomes` (the parser's tool-call outcomes of the request as counts keyed `outcome/class`, e.g. `repaired/no_opener`; `{}` without tool calls), `error_type`, and `response` (the JSON body, or the raw SSE text when streamed). `scripts/gateway_replay.py` resends the `request` bodies with the logged inter-arrival times (scaled, or as fast as allowed) and records per-request TTFT and usage: the Phase 4 scheduler is measured on the same traffic.

**`tetraeder serve`**: `-m <gguf>` with the `run` placement flags, `--engine auto|llama|moe` (auto: the builder for GGUF architecture `qwen3moe`, else libllama), `--ctx`, `--host`, `--port`, `--alias`, `--default-model`, sampling defaults, `--max-queued`, `--request-log`, `--chat-template-file`. `--fake` serves the gateway's fake engine (no GGUF) for the SDK conformance suite in CI; its script answers with the text after the last `FAKE:` in the prompt's last user turn, else echoes that turn.

**Telemetry.** Gateway metrics as ADR 0028; engine step metrics from each backend. New: `tetraeder.scheduler.batch.size` (histogram, 1 until Phase 4) and `tetraeder.kv.prefix.hits` (counter, requests that reused a prefix). Dashboard panels for prompt tokens served from the prefix cache and batch size; alert `GatewayStalled` when requests wait but no output tokens are produced for 5 minutes.

## Consequences

- Concurrency above one queues: throughput at concurrency 4 equals concurrency 1 until Phase 4's continuous batching; llama-server with `-np 4` batches. The benchmark reports this gap as Phase 4-bound.
- Sampling costs a pass over the vocabulary in Rust per token (about 0.1 ms greedy at 152K tokens); measured in the paired benchmark against llama-server. (No session directory yet.)
- Prefix reuse helps one conversation at a time; two interleaved conversations evict each other.
- The request log holds prompts and code in clear text; it is opt-in and local.

## Measured

The first measurements (`20261103`: gateway bench on an RTX 3090 and the first Claude Code proxy run): SDK conformance passes against the fake in CI and against both engines on the box, and Claude Code's first request is answered; at concurrency 1 `serve` is level with llama-server, at concurrency 4 it queues (Phase 4-bound); the proxy run lost cache share to side requests evicting the single cached sequence, and got 400s for `tool_reference` blocks (both fixed in ADR 0049).

## Addendum (2026-11-08): the week of use became a proxy run

On 2026-11-03 the user replaced the exit test's week of Claude Code use with a proxy run of a few hours: the same tasks against Tetraeder and llama-server on one box, ABBA. The bar: tasks solved at least llama-server's, prompt tokens served from cache at least its share, and no 400 for a request llama-server accepts. The plan's Phase 3b exit test says so now. The first run (`20261103`) missed the cache and 400 parts; ADR 0049 fixed both. Since `20261110d` the harness sets `CLAUDE_CODE_ATTRIBUTION_HEADER=0` for both engines (`scripts/claude_code_proxy.py`), so the per-session billing header is not in the prompt start (ADR 0057 addendum).

## Addendum (2026-11-09): the A6000 decode gap

The first proxy run's `tetraeder serve` had no `--engine`, so `auto` served the Qwen3 MoE builder (all experts resident), not libllama; other settings matched llama-server's. Its decode gap did not grow with depth, so it was a fixed per-token host cost; the 3090 gateway bench sampled greedily, Claude Code's requests take the temperature path (`20261103`).

Per-token host costs the code shows, fixed (tests in `sample.rs`, `qwen3moe.rs`, `scheduler.rs`):

- Sampling with temperature collected all 152K logits as candidates and ran `select_nth_unstable` for top-k 40: 2 ms per token on the local VM. Now one pass that rejects a logit with one compare against the k-th best (libllama uses a heap partial sort for k <= 128); same candidates, same order, same tokens per seed. (No session directory yet.)
- The builder's KQ mask had 32 padding rows (`[n_kv, pad(n, 32)]`); libllama's is `[n_kv, n]` at the pin. At 36K positions the decode step built, converted and uploaded 2.4 MB instead of 74 KB; the conversion (`flat_map` into a byte vector in `ggml-sys`) alone took 2 ms locally. Now `[n_kv, n]`. (No session directory yet.)
- `ggml-sys` tensor reads and writes copied through a byte vector; they now read into and write from the typed slice, and the builder backend reads each step's 151K logits into one kept buffer instead of a new 600 KB vector.

Not proven, left as measurements: graph rebuild and state reallocation every 256 positions, synchronous input uploads from pageable memory, per-token telemetry in `finish_step`, the gateway's per-token work. `TETRAEDER_DECODE_PROFILE=1` makes serve report per request the decode steps, eval, sample, hand-off and gateway time and the backend's counters (builder step profile and whole-step graph builds, libllama `n_reused`) in the request log and as `tetraeder.scheduler.decode.stage`; `scripts/claude_code_proxy.sh` turns it on and its summary adds `server_decode` (llama-server from its timing lines). `scripts/decode_gap.sh` reproduces the gap without Claude Code: llama-bench `-d` and both serve engines with the serve sampler at depths 0, 20K and 100K (`crates/scheduler/tests/decode_gap.rs`).

## Addendum (2026-11-11): owner decision on the tasks clause

The plan's clause reads "tasks solved at least as many as llama-server" for one run. Across the proxy runs per-run differences stay within one task and no task fails for Tetraeder alone, so at 20 tasks per run the clause as written is a coin flip (`20261111d`, which also lists the earlier runs).

Decision (owner, 2026-11-11, post hoc; the original text stays in the plan): the tasks clause is met when Tetraeder is no worse than llama-server beyond the run-to-run spread, judged over all runs, with no task failing for Tetraeder alone. Met over all runs. The cache and non-200 clauses are met on both current-harness runs (`20261110d`, `20261111d`). Tool-error rate is not a clause and flips between runs.
