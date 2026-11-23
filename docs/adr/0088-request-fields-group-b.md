# 0088. Request and response fields of llama-server (parity group B)

Status: Accepted (2026-11-23, under ADR 0083). After M2. Matrix: [docs/notes/llamacpp-parity.md](../notes/llamacpp-parity.md). Tree node `serving-parity-fields`.

## Context

ADR 0083 asks that every matrix field is honoured or rejected with 400. Group A (sampling) is done. Group B is what is left of the request and response fields: optional `model`, token counts, log-probabilities, `n`, grammars, `system_fingerprint`, `timings`, `reasoning_effort`. The pinned llama-server (b12426) fixes the shapes; most choices below follow from its source.

## Decision

1. **Token count.** `completion_tokens` counts every sampled token, the end-of-generation token included (llama-server's `n_decoded`). The gateway adds one when the engine ends on EOS. Metrics and the request log keep counting emitted tokens.
2. **Log-probabilities** are the softmax of the raw logits (llama-server's default `get_token_probabilities`, before bias, grammar, mask and chain), the sampled token and the top `n_probs`, as `content[]` with `id`, `token`, `bytes`, `logprob`, `top_logprobs[]`. The scheduler computes them in `sample_one` and sends a `GenEvent::Probs` before the token's `Token` event (also for the end-of-generation token), so all three engines share it. `post_sampling_probs: true` is a 400. The brief asked for the probabilities after the token mask; llama-server reports the raw ones, and parity wins.
3. **`n` above 1** is n sequences from the gateway, not a scheduler feature: one spec per choice with `seed_offset` i (seed + i when seeded, as llama-server's child tasks), their slots taken in one `acquire_many` so two such requests cannot each hold half, and `n` above the parallel slots is a 400 (llama-server's hard limit). Usage: prompt and timings of the first choice, completions summed.
4. **Grammar** is libllama's grammar sampler, applied as `common_sampler_sample` does: the chain picks, the grammar rejects or accepts that one token, on rejection the grammar runs first and the chain again; only generated tokens are accepted into the grammar. Plain greedy and speculation are off for such requests. A grammar is checked in the gateway (`Engine::check_grammar`) so a bad one is a 400 before generation.
5. **`json_schema`** is converted by llama.cpp's own `json_schema_to_grammar`, compiled from five files of `common/` with a 100-line C wrapper (`crates/ggml-sys/shim/schema.cpp`) because libcommon is not built. The wrapper copies three string helpers of `common.cpp`; `docs/sources.md` has the row. `response_format` keeps the gateway's own token mask.
6. **Fields that are not implemented** are rejected with 400 when set to a non-default value (`return_tokens`, `timings_per_token`, `response_fields`, `t_max_predict_ms`, `n_indent`, `lora`, `adaptive_target`, `post_sampling_probs`, `grammar_lazy`), so clients that send the defaults work. `cache_prompt`, `id_slot`, `n_keep`, `n_discard`, `n_cache_reuse`, `return_progress`, `sse_ping_interval` and `verbose` are accepted without effect, each for a reason in the matrix.
7. **`system_fingerprint`** is `tetraeder-<version>`; **`timings`** follow `server_slot_stats` from per-request timestamps in the scheduler.

## Consequences

- Clients see one more completion token when a response ends on EOS, and `logprobs`, `n`, `grammar`, `json_schema` and `timings` work instead of failing.
- The tokens of a grammar request equal llama-server's only while the chain and the grammar see the same candidates; the conformance case compares them on the tiny fixture.
- The wrapper builds C++ with the `cc` crate on every OS; a compiler is already needed for ggml.
- Open: `reasoning_format`, `reasoning_budget_tokens`, `continue_final_message`, `add_generation_prompt`, `prefill_assistant`, the Anthropic `extra_params` passthrough and `draft_n` timings are still ignored.
