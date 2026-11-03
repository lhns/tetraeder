# 0049. Multi-entry prefix cache in host RAM; lenient Anthropic input

Status: Accepted (Phase 3b). Extends ADR 0046; Phase 4's `kvcache` prefix tree replaces the cache.

## Context

The Claude Code proxy run (bench/results/20261103/claude-code-proxy, ADR 0046 "Measured") found two gaps against llama-server on the same model and tasks:

- Prompt tokens served from cache: 85% against 94%. ADR 0046 keeps one sequence; Claude Code interleaves side requests with a different prefix, which evict the main conversation (some requests reused 22 tokens). llama-server keeps evicted prompts in host RAM: its logs show `selected slot by LCP similarity` before every task and `making room for prompt cache entry, removing oldest entry (size = 1261 to 6760 MiB)` (server-llama-server-r1.log), the prompt cache of `--cache-ram` (default 8192 MiB, llama.cpp PR 16391).
- 12 requests (4 bursts of 3) got a 400 in about 2 ms that llama-server accepted: rejected while converting or rendering the request. ADR 0028 rejects images, unknown content blocks and assistant prefill; llama-server accepts more.

## Decision

**Prefix cache.** The scheduler keeps, besides the live sequence in the KV cache, up to a byte budget of saved sequences in host RAM (`serve --cache-ram <MiB>`, default 8192 like llama-server, 0 off), least recently used evicted first. Per request:

1. `live` = longest common prefix of the live sequence and the prompt; `best` = the longest common prefix over the saved entries.
2. Before the live sequence loses 512 or more tokens (truncation or restore), it is saved, unless an entry already holds it; an entry that is a prefix of it is replaced.
3. If `best` exceeds `live` by at least 64 tokens, that entry is restored into the KV cache.
4. Then truncate to the common prefix and evaluate the rest, as in ADR 0046.

Backends gain `snapshot() -> Snapshot` (the cached rows as host bytes plus their tokens) and `restore(&Snapshot)`. libllama: `llama_state_seq_get_size/get_data/set_data` for sequence 0 (new ggml-sys bindings, Rule 7). The MoE builder: copies of rows `0..n_past` of every layer's K and V tensors through the backend tensor get/set, then `n_past = n`. `EngineStats.cached_tokens` counts tokens from either source.

Metrics: `tetraeder.kv.prefix.hits` gains `source` (`live`, `ram`); new `tetraeder.kv.prefix.evictions` (counter), `tetraeder.kv.prefix.bytes` (gauge, host bytes held), `tetraeder.kv.prefix.restore.duration` (histogram). Dashboard panel for hits by source, evictions and bytes; alert `PrefixCacheThrash` when evictions outnumber RAM hits for 15 minutes.

**Lenient Anthropic input.** What llama-server accepts is accepted: unknown content block types are dropped with a warning; image and document blocks become a text placeholder (document text sources become their text); `tool_result` content lists keep their text, `is_error` is kept as an `Error:` prefix; empty text blocks and `cache_control` anywhere are ignored. A final assistant message is a prefill: the prompt is rendered without the generation prompt and cut after the prefill text (llama-server's `continue_final_message`); the response continues it. Template render errors are fixed by normalizing messages before rendering, not by rejecting them. Each shape found in Claude Code traffic gets a conformance test.

## Consequences

- Side requests no longer cost the main conversation its prefix; a switch costs one snapshot copy (about 98 KB per token for Qwen3-30B-A3B, 2 GB at 20K tokens) instead of a re-prefill.
- Host RAM use grows by up to `--cache-ram`; it is reported as a memory metric.
- Supersedes ADR 0028's 400 for prefill, images and unknown blocks.
