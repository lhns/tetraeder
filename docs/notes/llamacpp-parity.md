# Feature parity with llama-server: gap matrix

Target and rules: ADR 0083 (llama-server at the pin b12426, fc504f91; a request that works there works against `tetraeder serve` on the same GGUF; above-backend features on all three engines). Speed is in `docs/features.md`, not here. Inventory taken 2026-11-22 from the pinned sources in `vendor/llama.cpp` and the gateway code; the full table with file:line citations is the appendix.

## Groups and order

Each row of the appendix belongs to one group. A row is done when its conformance case passes against the baseline llama-server on the tiny fixture (tree node `serving-parity-harness`), when it is scoped out below with a reason, or when named unit tests cover it (table "Covered by unit tests"). A request that works on llama-server and differs here is a strict gap case under node `serving-parity-remaining`.

| Group | What | Tree node | When |
|---|---|---|---|
| A. Sampling | Chain order as llama-server (penalties, dry, top_n_sigma, top_k, typical_p, top_p, min_p, xtc, temperature, then the draw); min_p (default 0.05), repeat/presence/frequency penalties, typical_p, xtc, dry, top_n_sigma, mirostat, dynatemp, logit_bias, ignore_eos, min_keep, `samplers`; random seed when none is given; the model's suppress tokens | `serving-parity-sampling` | **done**, see the appendix rows |
| B. Request and response fields | Every matrix field honoured or rejected with 400, never dropped; logprobs and top_logprobs; `n` above 1; GBNF `grammar` and top-level `json_schema`; `reasoning_effort` (and Anthropic `output_config.effort`); `timings` in responses; `cache_prompt` accepted (`id_slot`, `n_keep` and `n_discard` act since ADR 0090) | `serving-parity-fields` | **done**, see the appendix rows (ADR 0088) |
| C. Endpoints | `/health`, Prometheus `/metrics`, `/props`, `/tokenize`, `/detokenize`, `/apply-template`, native `/completion`, unprefixed `/models` and `/chat/completions`, `/v1/responses`, `/infill` and `suffix`, input-token counting, `--api-key` | `serving-parity-endpoints` | **done**, see the appendix rows (ADR 0089) |
| D. What libllama has but the gateway does not expose | embeddings and rerank (libllama engine first), KV cache types (`-ctk`, `-ctv`), KV offload (`-nkvo`), context shift, slot save and restore, `--cpu-moe` and `--n-cpu-moe`, model download by name | `serving-parity-libllama` | **done**, see the appendix rows (ADR 0090); model download is refused, the GPU runs are open; `GET /slots` since ADR 0093 |
| E. Own ADR first (owner sees this matrix before work starts) | LoRA (plan feature 30), several models and router mode (29), multimodal, audio, TTS, transcription and realtime (34), other GPU backends (14) | `serving-parity-owner-scope` | owner decision |

The conformance cases are `tests/conformance/parity/cases.yaml` (Linux `test` job; a row with `expect: gap <node>` is xfail(strict) and flips to `pass` in the change that closes it). Run locally: build llama-server from the patched `vendor/llama.cpp` into `target/llama-server` (CPU only: `cmake -S vendor/llama.cpp -B target/llama-server -DGGML_CUDA=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_UI=OFF`, then `cmake --build target/llama-server --config Release --target llama-server`), `cargo build -p tetraeder`, then `uv run pytest -q tests/conformance/parity -rx` (`LLAMA_SERVER_BIN`, `TETRAEDER_BIN`; the cases skip without the binaries). The baseline is the patched tree: its patches touch ggml CPU internals and must not change CPU tokens, so a case where tokens differ is reported. Sampler rows that temperature 0 cannot show (min_p, typical_p, top_n_sigma, xtc, dry, mirostat, dynatemp, min_keep, `samplers` order, random seeding) stay with the unit tests of `serving-parity-sampling`.

Harness result (audit of ADR 0083 decision 6, 2026-11-23): the appendix has 165 rows. 75 have a conformance case (112 cases, 146 runs over auto, dense, moe and llama-moe, with `logprobs-after-reuse` added 2026-11-24; 106 cases expect `pass`, 6 are strict gap cases of node `serving-parity-remaining`), 74 are scoped out with a reason and 16 are covered by named unit tests. `test_parity_lib.py` fails when a row has none of the three. CI history: the first run (CI run 44983973508, Linux) found the differences the comparison now tolerates or that were fixed (the `model` field, the end-of-generation token in `completion_tokens`, `ping` events in the Anthropic stream); group C closed with CI run 45116789518 and group D with CI run 45145783401 (722 passed, 2 skipped, none xfail). The cases added by the audit were written without a local llama-server, so their first run is the next CI run. Greedy text and finish reasons of llama-server and `tetraeder serve` agree on auto, dense and moe. Differences the comparison tolerates are listed in `tests/conformance/parity/parity_lib.py` (extra null fields, U+FFFD runs). On CUDA (A100, `regression-gate-20261123`) 135 passed, 6 xfailed and 4 failed, all libllama-engine cases (repeat-penalty, seeded-sampler-names, logprobs, logprobs-shape); tetraeder's logprobs varied between runs; tree node `serving-cuda-parity`.

CUDA and prompt-cache reuse (`serving-cuda-parity`, diagnosed on an A40, `bench/results/stream-predict-20261124/`): the CUDA failures come from reuse state, not from tetraeder. llama-server alone is stable across fresh servers, but its second identical request on one server returns other logits (first-token logprob -1.8332, then -1.8114; a third value, -1.8150, is the fresh evaluation), so a prompt evaluated fresh and one served from the prompt cache differ on CUDA in both engines, and which one a case sees depends on the cases before it. The harness therefore pins reuse off for every case of mode `text`, `logprobs` or `stream` (`reuse: true` opts out): tetraeder `--no-prefix-reuse --cache-ram 0`, llama-server `--cache-ram 0` and `cache_prompt: false` in the request (`servers.py`, `test_parity.py`). One case keeps reuse on, `logprobs-after-reuse`: exact on CPU (1e-3), and on CUDA (`PARITY_CUDA=1`, set by the box scripts) within a stated band of 0.03, from the measured spread of 0.022 (-1.8332, -1.8114, -1.8150). CPU CI stays exact everywhere; the CUDA band is not confirmed until a box run.

## Scoped out (ADR 0083)

First cell = the appendix row (exact), so the harness test can match it. Reasons: out of scope by ADR 0083 point 4 (LoRA, multimodal, audio and realtime, router mode), not a request feature (server flags, backend choice), a duplicate of another row, or rejected with 400 by design.

| Row | Reason |
|---|---|
| Web UI (`--ui`) | not a request-level feature; any OpenAI-compatible UI works against the gateway |
| /cors-proxy, /tools (MCP/agent features) | agent features of the bundled UI, not of the inference API |
| GCP compatibility routes | a hosting adapter, not used by the clients this project serves |
| TLS | terminate at a reverse proxy; the gateway listens on localhost by default |
| GET /v1/stream, POST /v1/streams/lookup (resumable streams) | revisit if a client needs it; not part of the OpenAI or Anthropic API |
| Speed | judged in Phases 2, 4 and 5 (`docs/features.md`) |
| GET /v1/models/{id} | a Tetraeder extra; llama-server has no such route |
| POST /v1/chat/completions/control (`reasoning_end`) | control channel of the reasoning budget; open with `reasoning_budget_tokens` (node `serving-parity-remaining`), no signal on a fixture without a thinking template |
| POST /v1/audio/transcriptions, /audio/transcriptions, /tts, /v1/systemone | audio, ADR 0083 point 4 (plan feature 34, group E) |
| **[new @pin]** POST /v1/audio/speech, /audio/speech (OpenAI TTS) | audio, ADR 0083 point 4 (plan feature 34, group E) |
| **[new @pin]** WebSocket /v1/realtime | realtime, ADR 0083 point 4 (plan feature 34, group E) |
| GET/POST /lora-adapters | LoRA, ADR 0083 point 4 (plan feature 30, group E) |
| Router mode: POST /models, /models/load, /models/unload, GET /models/sse | several models, ADR 0083 point 4 (plan feature 29, group E) |
| Tetraeder-only | no llama-server counterpart |
| content parts: image_url / input_audio (needs `--mmproj`) | multimodal, ADR 0083 point 4 (group E); refused with 400 here |
| adaptive_target / adaptive_decay | the adaptive-p sampler is not supported; a target of 0 or more is rejected with 400 (`crates/gateway/src/sampling.rs`, test `unsupported_fields_are_rejected_unless_at_their_default`) |
| return_tokens | `true` is rejected with 400 by design (`crates/gateway/src/sampling.rs`); `false` is the default on both |
| return_progress / sse_ping_interval | accepted without effect (prompt progress chunks and the ping interval do not change any token or field a client reads) |
| timings_per_token / verbose / response_fields / t_max_predict_ms / n_indent | rejected with 400 when set to anything but the default (`crates/gateway/src/sampling.rs`) |
| reasoning_format (request) | ignored; open in node `serving-parity-remaining`, no signal on a fixture without a thinking template |
| reasoning_budget_tokens / thinking_budget_tokens | ignored; open in node `serving-parity-remaining`, no signal on a fixture without a thinking template |
| suffix | rejected with 400 on the chat and completions routes; fill-in-the-middle is `/infill` (case `infill`) |
| lora (per request) | LoRA, ADR 0083 point 4 (plan feature 30, group E); a non-empty list is rejected with 400 |
| speculative.* per request | `#if 0` at the pin, llama-server ignores it too |
| Anthropic: extra_params passthrough (min_p, seed ...) | llama-server's extension outside the Anthropic API; clients of that API do not send it |
| Backend (GPU) sampling | the host sampler is the only one; it changes no token |
| Request log (prompts) | Tetraeder has its own `--request-log`; a flag, not a request feature |
| Parallel slots | server flag (`--parallel`); the slot count is not part of a request |
| Context size | server flag (`--ctx`) |
| Batch sizes | server flags; no output change |
| Flash attention | server flag; no output change (case `kv-types` runs with it on) |
| Unified KV / per-slot | server flag; always unified here |
| Prompt cache | server flags of the prefix cache; a hit changes no output |
| Slot save/restore | duplicate of the `/slots` row (cases `slot-save`, `slot-restore`, `slot-erase`) |
| Chat template | server flag; the template comes from the GGUF or `--chat-template-file` |
| chat-template-kwargs | server flag; the same values are sent per request (case `chat-template-kwargs`) |
| Reasoning | server flags; the request fields are rows of their own |
| Tools/MCP server side (**[new @pin]** `--allow-unauthenticated-tools` :3447, `--media-allow-private-urls` :3679) | agent features of the bundled UI, not of the inference API |
| Embeddings / rerank | duplicate of the `/embedding` and `/rerank` rows (group D cases) |
| LoRA | LoRA, ADR 0083 point 4 (plan feature 30, group E) |
| Multimodal | multimodal, ADR 0083 point 4 (group E) |
| Speculative / draft | server flags; speculation changes no token and has no request field |
| Grammar flags | server flags with defaults; the same text is sent per request (cases `grammar-gbnf`, `json-schema-top-level`) |
| API key | duplicate of the `API key auth` row (case `api-key`) |
| Metrics/props/slots endpoints | duplicate of the `/metrics`, `/props` and `/slots` rows |
| Host/port | server flags |
| CORS, path, api-prefix | server flags; the gateway listens on localhost by default |
| HTTP threads, timeout, SSE ping | server flags; no output change |
| GPU layers | backend placement; no output change |
| Split mode | backend placement; no output change |
| Tensor split / main GPU | backend placement; no output change |
| Load mode | backend placement; no output change |
| Threads | backend placement; no output change |
| Multi-model router | several models, ADR 0083 point 4 (plan feature 29, group E) |
| Planner / replan / worker isolation | Tetraeder-only |
| Pin-only flags | flags of llama-server's own backend (`--moe-cache-cpu-share`, `--ple`, `--shard`) |
| Fake engine | test only |
| `qwen3moe` | engine selection (section 5) is Tetraeder's own; the cases run each engine (`engines`) |
| `qwen3` | engine selection (section 5) is Tetraeder's own; the cases run each engine (`engines`) |
| `llama`, `gemma3`, anything else | engine selection (section 5) is Tetraeder's own; libllama takes them |
| unreadable GGUF / no arch key | engine selection (section 5) is Tetraeder's own; an error before the port opens |
| Embeddings | section 6 duplicate of the `/embedding` row |
| Reranking | section 6 duplicate of the `/rerank` row |
| Tokenize / detokenize | section 6 duplicate of the `/tokenize` row |
| apply-template | section 6 duplicate of the `/apply-template` row |
| /props | section 6 duplicate of the `/props` row |
| /slots | section 6 duplicate of the `/slots` row |
| /metrics | section 6 duplicate of the `/metrics` row |
| /health | section 6 duplicate of the `/health` row |
| Infill (FIM) | section 6 duplicate of the `/infill` row |
| Multimodal (mtmd) | multimodal, ADR 0083 point 4 (group E) |
| LoRA adapters | LoRA, ADR 0083 point 4 (plan feature 30, group E) |
| Prompt caching | the prefix cache changes no output (cases `cache-prompt-accepted`, `usage-fields`) |
| Continuous batching | server behaviour; no output change |
| Structured output | section 6 duplicate of the `grammar` and `json_schema` rows |
| Quota / auth / multi-tenant | Tetraeder-only attribution (`caller`); llama-server has api keys only (case `api-key`) |

## Covered by unit tests

The Rust tests run in every CI job. A row is listed here when no live comparison is possible or needed: the behaviour is inside Tetraeder, or the baseline would need a model other than the tiny fixtures. The test column names files and test functions.

| Row | Test |
|---|---|
| messages (chat) | `crates/gateway/tests/openai.rs` (`chat_prompt_uses_the_template`), `crates/gateway/tests/serving.rs` (`system_message_in_messages_becomes_user_text`); live in `chat-greedy` |
| content parts: text | `crates/gateway/tests/openai.rs` (`chat_completion_shape`); live in `chat-greedy` |
| tools | `crates/gateway/tests/openai.rs` (`tool_call_response`, `tool_history_reaches_the_template`) |
| parallel_tool_calls | `crates/gateway/tests/anthropic.rs` (`disable_parallel_tool_use_cancels_after_the_first_call`), `crates/gateway/tests/openai.rs` (`single_tool_call_cancels_the_engine`) |
| tool call output | `crates/gateway/tests/toolcall.rs` (Hermes, Qwen XML, Llama 3 and Mistral parsers) |
| reasoning/thinking content | `crates/gateway/tests/openai.rs` (`reasoning_content_is_split`), `crates/gateway/tests/toolcall.rs` |
| user | `crates/gateway/tests/gateway.rs` (`cost_record_per_request`) |
| Anthropic: metadata.user_id | `crates/gateway/tests/gateway.rs` (`cost_record_per_request`) |
| Anthropic: thinking blocks in history | `crates/gateway/tests/anthropic.rs` (`thinking_blocks`), `crates/gateway/tests/lenient.rs` (`thinking_blocks_and_unknown_blocks`) |
| Anthropic: image/document blocks | `crates/gateway/tests/lenient.rs` (`image_and_document_blocks_become_text`); images themselves are multimodal, group E |
| **[new @pin]** Anthropic: output_config.effort -> reasoning_effort | `crates/gateway/tests/fields.rs` (`anthropic_model_is_optional_and_effort_is_read`) |
| Anthropic: tool defs `input_schema`, `defer_loading`/`tool_reference` | `crates/gateway/tests/lenient.rs` (`deferred_tools_are_rendered`, `tool_reference_in_tool_result_becomes_text`) |
| Anthropic stop_reason | `crates/gateway/tests/anthropic.rs` (`stop_sequence_and_max_tokens`); live in `anthropic-greedy` |
| Anthropic usage | `crates/gateway/tests/anthropic.rs` (`message_shape`, `count_tokens_matches_usage`); live in `anthropic-greedy` |
| Error format | `crates/gateway/tests/openai.rs` (`errors_use_the_openai_shape`), `crates/gateway/tests/anthropic.rs` (`errors_use_the_anthropic_envelope`) |
| Chain construction (sampling.cpp:299-444) | `crates/scheduler/src/sample.rs` (`default_chain_is_llama_servers`, `disabled_stages_left_out_change_nothing`, `greedy_keeps_stages_that_can_remove_the_best_token`, `logit_bias_and_ignore_eos_ban_tokens`), `crates/scheduler/tests/sampler_dry.rs` |

## Differences that change outputs today

Group A is closed (node `serving-parity-sampling`): the host sampler is libllama's own chain in llama-server's order and defaults, so the old differences (temperature before top-p, no min_p, counter seed, SplitMix64) are gone. This changes sampled outputs at temperature above 0 for existing clients. Still different:

- Group B is closed (node `serving-parity-fields`): the fields of the matrix are honoured, rejected with 400 when set to a non-default value, or accepted without effect for a stated reason (appendix rows). Fields no row names are still ignored, as llama-server ignores its unknown ones. Still open from group B: `reasoning_format`, `reasoning_budget_tokens`, `continue_final_message`, `add_generation_prompt`, `prefill_assistant` and the Anthropic `extra_params` passthrough are ignored; `response_format` keeps its own token mask (not llama.cpp's grammar), so its exact tokens can differ from llama-server's. The audit's strict gap cases and these open items are tracked in node `serving-parity-remaining`.
- A request `logit_bias` with text keys is rejected with 400; llama-server tokenizes them.
- `samplers` names `infill` and `adaptive_p`, and `adaptive_target` and `adaptive_decay`, are not supported; an unknown sampler name is a 400 here, a skipped entry there.
- `repeat_last_n` and `dry_penalty_last_n` are capped at the context of a sequence (`SampleParams::resolve`). The history never exceeds it, so no output changes; llama-server does not cap them and libllama allocates a ring buffer of the size asked for.
- `--api-key-file` or `--api-key` that yields no key (an empty file, blank lines) stops the server at startup; llama-server starts without authentication. Lines of the key file are trimmed here.
- A backend that cannot cut a suffix (a recurrent model) takes a saved RAM or disk entry only when the prompt starts with the whole entry (ADR 0087 addendum).
- Speculation (`--spec`) treats a request as greedy when nothing but the argmax matters; the model's suppress tokens are not part of that test.

Greedy outputs (temperature 0) are not affected by the change.

## Appendix: inventory (2026-11-22)

### 0. Source status

- Verified against `vendor/llama.cpp` at **fc504f91d8cf (b12426)** plus the 6 Tetraeder patches (ggml files show as modified; `tools/server`, `common/` are upstream). All llama-server citations below are to this tree. The first draft of this file was made at b11976 (2e0d86be); rows that differ between b11976 and the pin are marked **[changed @pin]** or **[new @pin]**. Rows without a mark are the same in substance at both (line numbers moved).
- Caveat on the diff: b11976 was read in full for routes, request schema, chat parse, task JSON and arg.cpp flags. `common/sampling.cpp` and the b11976 sampler default chain were only read via `common.h`, so sampler rows in 2b are marked new only where the b11976 text showed a difference.
- Notation: `L:` = `vendor/llama.cpp/tools/server/`, `C:` = `vendor/llama.cpp/common/`, `T:` = `crates/`. Tetraeder cites are unchanged by this update.
- Tetraeder gateway structs have no `#[serde(deny_unknown_fields)]` (grep over crates/gateway/src: none), so any field not in the struct is **silently ignored**. Only fields in the structs can be rejected.

### Summary of changes between b11976 and the pin (server side)
- **[new @pin] routes:** `POST /v1/audio/speech`, `/audio/speech` (`server.cpp:192-193`, handler `post_speech_oai` context.cpp:6432), WebSocket `/v1/realtime` (`server.cpp:194`), `DELETE /v1/stream` (`server.cpp:234`), GCP/Vertex compat routes (`ctx_http.register_gcp_compat()`, `server.cpp:222`). Slot scheduling moved to `server-slot-sched.{h,cpp}` (context.cpp:1132,1448; the b12426 note in ADR 0065 "#32154").
- **[new @pin] Anthropic:** `output_config.effort` becomes `reasoning_effort` (`server-chat.cpp:662-668`). Tetraeder reads only `output_config.format` (`anthropic.rs:373-388`): effort is ignored.
- **[new @pin] flags:** `--allow-unauthenticated-tools` (arg.cpp:3447), `--media-allow-private-urls` (3679), `--moe-cache-cpu-share` (2758), `--ple/--shard` (3036). **[removed @pin]:** `-dt/--defrag-thold` (matches ADR 0065 line 213).
- **Unchanged in substance:** the whole completion schema (`server-schema.cpp:13-470` has identical field lines; only the TTS schema at :540+ changed: adds `speed`, `loudness`), result JSON (usage, finish_reason, timings), chat parse logic, error format, API key middleware, `n_parallel` auto = 4.
- Other changes inside this range that are not server-visible (CUDA FP8 #28413, stream-k mm tiles) are out of scope here.

### 1. HTTP endpoints

llama-server route table: `L:server.cpp:174-319` (core) and `:555-558` (router mode only). Tetraeder router: `T:gateway/src/server.rs:52-62`.

| Capability | llama-server evidence | Tetraeder status | Engine notes |
|---|---|---|---|
| GET /health, /v1/health (public, no key) | server.cpp:174-175; handler server-context.cpp:5673 | **done** (node `serving-parity-endpoints`, ADR 0089): `{"status": "ok"}`, public with `--api-key`; 503 `Loading model` while the engine is not ready (`serve` binds after the load, so only the replanner's switch shows it). Case `health` | n/a |
| GET /metrics (Prometheus, `--metrics`) | server.cpp:176; context.cpp:5686; names `llamacpp:*` via server-task.cpp:1641-1643 | **done** (node `serving-parity-endpoints`, ADR 0089): `--metrics` serves llama-server's `llamacpp:*` names from gateway counters, 501 without it. Not exported (no such number here): `n_decode_total`, `n_busy_slots_per_decode`, `spec_decode_*`. The OTLP metrics are unchanged. Case `metrics` | all |
| GET /props, POST /props (`--props` for POST) | server.cpp:177-178; context.cpp:5823,5835 | **done** (node `serving-parity-endpoints`, ADR 0089): GET gives `default_generation_settings`, `total_slots`, `model_alias`, `model_path`, `chat_template`, `bos_token`, `eos_token`, `build_info` and the constant fields; `model_ftype`, `media_marker`, `chat_template_caps` and `chat_template_tool_use` are left out. POST is 501 (no `--props`: nothing to change). Case `props` | n/a |
| GET /models, /v1/models | server.cpp:179-180; context.cpp:6349 | **done** (node `serving-parity-endpoints`, ADR 0089): `/models` is the same handler as `/v1/models` (our body, not llama-server's `models` plus `data`). Case `models-unprefixed` | all |
| GET /v1/models/{id} | not registered in llama-server | supported (server.rs:57) | all |
| POST /completion, /completions (native, many fields) | server.cpp:181-182; context.cpp:5925 | **done** (node `serving-parity-endpoints`, ADR 0089): native request (`prompt` as string, token array, mixed array or `{prompt_string}`; `n_predict` and its aliases, `n_cmpl` or `n`, `stop`, `n_probs`, all sampling fields) and response (`content`, empty `tokens`, `stop_type`, `stopping_word`, `tokens_predicted`, `tokens_evaluated`, `tokens_cached`, `generation_settings`, `timings`), stream without `[DONE]`. A batch of prompts and `n_predict` 0 are 400. Case `native-completion` | n/a |
| POST /v1/completions | server.cpp:183; context.cpp:5937 | **supported, narrow** (server.rs:53; openai.rs:487; single string prompt only, openai.rs:496-507) | all |
| POST /chat/completions, /v1/chat/completions | server.cpp:184-185; context.cpp:5949 | **done** (node `serving-parity-endpoints`, ADR 0089): `/chat/completions` is the same handler. Case `chat-unprefixed` | all |
| POST /v1/chat/completions/control (`reasoning_end`) | server.cpp:186; context.cpp:5973-5983 | missing | n/a |
| POST /v1/responses, /responses (OpenAI Responses API) | server.cpp:187-188; context.cpp:6010 | **done** (node `serving-parity-endpoints`, ADR 0089): converted to a chat body as `server_chat_convert_responses_to_chatcmpl` does; `response` object and `response.*` events, no `[DONE]`; `previous_response_id`, `input_file` and images are 400. Case `responses` | n/a |
| POST /v1/audio/transcriptions, /audio/transcriptions, /tts, /v1/systemone | server.cpp:189-191,204 | missing | n/a |
| **[new @pin]** POST /v1/audio/speech, /audio/speech (OpenAI TTS) | server.cpp:192-193; context.cpp:6432 | missing | n/a |
| **[new @pin]** WebSocket /v1/realtime | server.cpp:194 | missing | n/a |
| POST /v1/messages (Anthropic) | server.cpp:195; context.cpp:6307; converter server-chat.cpp:391 | **supported** (server.rs:54; anthropic.rs:466) | all |
| POST /v1/messages/count_tokens | server.cpp:213; context.cpp:6329 | supported (server.rs:55; anthropic.rs:707; renders template and tokenizes, gateway.rs:352-360) | all |
| POST /v1/chat/completions/input_tokens, /responses/input_tokens | server.cpp:209-212; context.cpp:5969,6032 | **done** (ADR 0093): the four routes (with and without `/v1`) render and tokenize the body as its generation route does and answer `{input_tokens, object: "response.input_tokens"}`. Cases `input-tokens-chat`, `input-tokens-responses` | n/a |
| POST /infill | server.cpp:196; context.cpp:5847 | **done** (node `serving-parity-endpoints`, ADR 0089): FIM prompt of `format_prompt_infill` (repo-level extra chunks, 3:1 prefix and suffix share of 2048 tokens); a model without FIM tokens gets llama-server's 501. The tiny fixtures have none, so the case compares that error; the layout is tested in `crates/gateway/tests/endpoints.rs`. `suffix` on `/v1/completions` is still rejected | n/a |
| POST /embedding, /embeddings, /v1/embeddings (needs `--embeddings`) | server.cpp:197-199; context.cpp:6416,6420,6691 | **done** (node `serving-parity-libllama`, ADR 0090): `--embeddings` (alias `--embedding`) starts a second libllama context on the model; `/embedding` and `/embeddings` answer `[{index, embedding: [[...]]}]`, `/v1/embeddings` the OpenAI list object (float or base64); inputs as `tokenize_input_prompts`, `embd_normalize` default 2; pooling `none` gives one vector per token and is a 400 on the OpenAI route; 501 `Start it with --embeddings` without the flag. Cases `embeddings-*`, `embedding-legacy-route` (vectors within 1e-3) | libllama; builders refuse `--embeddings` |
| POST /rerank, /reranking, /v1/rerank, /v1/reranking (`--rerank`) | server.cpp:200-203; context.cpp:6424 | **done** (ADR 0090): `--reranking` (alias `--rerank`) = rank pooling; the four routes answer Jina's shape or TEI's with `texts`; the `rerank` chat template or `[BOS] q [EOS] [SEP] d [EOS]`; 501 `Start it with --reranking` without it. The tiny fixtures have no rank head: libllama scores with the pooled hidden state and both servers compute the same value, which the `rerank-jina` and `rerank-tei` cases compare at 1e-5; the `rerank-unsupported` cases compare the error path | libllama; builders refuse |
| POST /tokenize, /detokenize | server.cpp:205-206; context.cpp:6361,6402 | **done** (node `serving-parity-endpoints`, ADR 0089): `content` (string or mixed array), `add_special`, `parse_special`, `with_pieces`; `tokens`. Cases `tokenize`, `detokenize` | n/a |
| POST /apply-template | server.cpp:207; context.cpp:6334 | **done** (node `serving-parity-endpoints`, ADR 0089): the chat route's prompt as `{"prompt"}`. Case `apply-template` | n/a |
| GET/POST /lora-adapters | server.cpp:215-216; context.cpp:6525,6553 | missing | n/a |
| GET /slots, POST /slots/:id?action=save/restore/erase | server.cpp:218-219; context.cpp:5748,5790,5807-5809 | **`POST` done** (ADR 0090): `--slot-save-path DIR`, slot `i` is sequence `i`, `id_slot` pins a request to it; save writes tokens and the KV snapshot to a file of this server's format (`n_saved`, `n_written`, `timings.save_ms`), restore makes the slot an idle prefix, erase frees it; 501 without the flag, 400 `Invalid slot ID`, `Invalid action`, `Invalid filename`. **`GET /slots` done** (ADR 0093): one entry per slot with `id`, `n_ctx`, `speculative` false and `is_processing`, `?fail_on_no_slot=1` is 503 when none is idle, `/props` says `endpoint_slots` true. Known difference: the gateway admits by count, not by slot id, so `is_processing` marks the first busy-count slots, and the last task's fields (`id_task`, `params`, `next_token`, `prompt`, `generated`) are not given; case `slots-get` compares the status | `GET` all; `POST` libllama, builders refuse |
| GET /v1/stream, POST /v1/streams/lookup (resumable streams) | server.cpp:232-234 (**[new @pin]** adds `DELETE /v1/stream`) | missing | n/a |
| /cors-proxy, /tools (MCP/agent features) | server.cpp:284-319 | missing | n/a |
| Router mode: POST /models, /models/load, /models/unload, GET /models/sse | server.cpp:555-558 | missing (one model per process; `--alias`, `--default-model`, `--accept-any-model`, serve.rs:56-63) | n/a |
| API key auth (`--api-key`, Bearer) | arg.cpp:3529; server-http.cpp:347-359 | **done** (node `serving-parity-endpoints`, ADR 0089): `--api-key` (comma list, `LLAMA_API_KEY`) and `--api-key-file`; `Authorization` with or without `Bearer `, else `X-Api-Key`; only `/health` is public; 401 `Invalid API Key`. Case `api-key` | n/a |
| Web UI (`--ui`) | arg.cpp:3486 | missing | n/a |
| Tetraeder-only | n/a | admin listener with `--plan`: `POST /admin/replan`, `POST /api/v2/alerts`, `GET /admin/plan` on `--admin 127.0.0.1:8081` (serve.rs:184-186) | planner (replanner) only |

### 2. Request fields

### 2a. llama-server (completion schema shared by /completion, /v1/completions, /v1/chat/completions, /v1/messages)

Parsed in `L:server-schema.cpp:make_llama_cmpl_schema` (lines 13-470), evaluated at `:eval_llama_cmpl_schema` (~480-520). Chat requests go through `oaicompat_chat_params_parse` (`L:server-common.cpp:1489-1700`), which copies every remaining body key through to the schema (server-common.cpp:1690-1697), so native fields work on the OAI endpoints. `null` values count as absent (schema.cpp `has_value`). Out-of-range values are clamped (soft limits) or rejected (hard limits).

| Field | llama-server evidence | Tetraeder status (OpenAI chat / completions / Anthropic) | Engine notes |
|---|---|---|---|
| model | context.cpp meta->model_name | **done** (node `serving-parity-fields`): optional, as there: a request without `model` gets the served model; a named unknown id still resolves per `--default-model`/`--accept-any-model`. Case `model-optional` | all |
| messages (chat) | common.cpp:1552-1583 | supported: roles system/developer/user/assistant/tool (openai.rs:158-207); other roles rejected (openai.rs:208-212) | all |
| content parts: text | common.cpp:1580 | supported (openai.rs:103) | all |
| content parts: image_url / input_audio (needs `--mmproj`) | common.cpp:1580 (`oaicompat_content_load_media` 1415) `oaicompat_content_load_media`; arg.cpp:2549 | **rejected** on OpenAI (openai.rs:106-110 "only text"); Anthropic image block becomes the text `[image]` (anthropic.rs:60) | n/a (no mtmd) |
| prompt (completions): string / array / token ids / mixed / multimodal | common.cpp:1084-1250 (`tokenize_mixed` 1084, `tokenize_input_prompts` ~1191) | only one string or one-string array; batches and token arrays rejected (openai.rs:496-507) | all |
| temperature | schema.cpp:118 (clamp >= 0) | supported (openai.rs:358; anthropic.rs:422); default 0.8 (serve.rs:66) | all (scheduler sampler, sample.rs:35-44) |
| top_k | schema.cpp:91 | supported, OpenAI extension (openai.rs:55,360); default 40 (serve.rs:68); a value below 0 clamps to 0 and above `INT32_MAX` to it, as the soft limits there | all |
| top_p | schema.cpp:95 | supported (openai.rs:359); default 0.95 (serve.rs:70) | all |
| min_p | schema.cpp:99; **default 0.05** (common.h:239) | **done** (node `serving-parity-sampling`): parsed and applied (`sampling.rs`, `sample.rs`); default 0.05, as llama-server | all |
| top_n_sigma | schema.cpp:103 | **done** (node `serving-parity-sampling`): parsed and applied, default off | all |
| xtc_probability / xtc_threshold | schema.cpp:106,110 | **done** (node `serving-parity-sampling`): parsed and applied, seeded with the request seed | all |
| typical_p | schema.cpp:114 | **done** (node `serving-parity-sampling`): parsed and applied, default off | all |
| dynatemp_range / dynatemp_exponent | schema.cpp:122,125 | **done** (node `serving-parity-sampling`): parsed and applied through `llama_sampler_init_temp_ext` | all |
| repeat_penalty / repeat_last_n | schema.cpp:132,128 | **done** (node `serving-parity-sampling`): parsed and applied; the prompt tokens count as history, as in llama-server. Case `repeat-penalty` | all |
| frequency_penalty / presence_penalty (OpenAI standard names) | schema.cpp:135,138 | **done** (node `serving-parity-sampling`): parsed on both OpenAI routes and applied. Cases `presence-penalty`, `frequency-penalty` | all |
| dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, dry_sequence_breakers | schema.cpp:141-155,244 | **done** (node `serving-parity-sampling`): parsed and applied through libllama's DRY sampler on the tokenizer model's vocabulary; works on all three engines (they share the scheduler's tokenizer model). `dry_base` below 1.0 means the server default; breakers must be 1 to 64 distinct strings | all |
| mirostat / mirostat_tau / mirostat_eta | schema.cpp:159-166 | **done** (node `serving-parity-sampling`): parsed and applied (1 and 2); the chain is temperature plus Mirostat, as in llama-server | all |
| adaptive_target / adaptive_decay | schema.cpp:169,173 | `adaptive_target` of 0 or more is rejected with 400 (the adaptive-p sampler is not supported), `adaptive_decay` is ignored; both at their defaults are accepted | all |
| samplers (order) | schema.cpp:472 | **done** (node `serving-parity-sampling`): parsed (names, aliases or the letter string) and honoured in list order. `infill` and `adaptive_p` are rejected with 400, as is any unknown name (llama-server logs and skips it) | all |
| seed | schema.cpp:178 (-1 random) | **done** (node `serving-parity-sampling`): seeded requests use libllama's `dist` sampler (mt19937), the same RNG as llama-server, so seeded outputs can match (case `seeded-sampling`); unseeded requests draw a random seed per request. The value is cast to `uint32_t` as llama-server does (`sampling::libllama_seed`): -1 and 4294967295 (`LLAMA_DEFAULT_SEED`) are random, other values wrap modulo 2^32. `/v1/messages` accepts `seed` too. Server default seed: `--seed` | all |
| logit_bias | schema.cpp:401 | **done** (node `serving-parity-sampling`): both forms (`{id: bias}` and `[[id, bias]]`, `false` bans); text keys and string tokens are rejected with 400 (llama-server tokenizes them); ids outside the vocabulary are dropped, as there. Case `logit-bias` | all |
| ignore_eos | schema.cpp:442 | **done** (node `serving-parity-sampling`): bans every end-of-generation token. Case `ignore-eos` | all |
| min_keep | schema.cpp:185 | **done** (node `serving-parity-sampling`): parsed and passed to every truncating sampler | all |
| n_predict / max_tokens / max_completion_tokens | schema.cpp:46 (aliases) | supported: `max_completion_tokens` then `max_tokens` (openai.rs:357); completions and chat default: no limit but the context, as `n_predict` -1 (gateway.rs:88); Anthropic `max_tokens` optional (anthropic.rs:29,421; llama-server defaults it to 4096, chat.cpp:639-643) | all |
| n_keep, n_discard, context_shift | schema.cpp:56,60; arg.cpp:1720 | **done** (ADR 0090): `n_keep` and `n_discard` are read and `--context-shift` shifts a full context (libllama only, see the Context shift row); without the flag they have no effect, the prompt is never truncated and over-length prompts fail with `context_length_exceeded` 400 (error.rs:14-16,50-51; engine.rs:155-157) | libllama; the builders refuse the flag |
| n (n_cmpl, multiple choices) | schema.cpp:64 (1..n_parallel) | **done** (node `serving-parity-fields`): `n` choices through the gateway, each its own sequence with its own constraint and parser (the prompt is shared through prefix reuse where it lands on one sequence); choice i samples with seed + i when seeded, as llama-server's child tasks do; `n` of 0 or above the parallel slots is a 400 (llama-server's hard limit 1..n_parallel); the slots of one request are taken together, so two `n` requests cannot deadlock; usage has one prompt and the summed completions; streams interleave chunks by `index`. Case `n-choices` | all |
| stop | schema.cpp:454; chat copy common.cpp:1513-1518 | supported: string or array (openai.rs:126-143); Anthropic `stop_sequences` (anthropic.rs:33,425); matched in gateway, held-back suffix (decode.rs:50-60) | all |
| stream | schema.cpp:25 | supported (openai.rs:60,367; anthropic.rs:38) | all |
| stream_options.include_usage | schema.cpp:28 | chat: the empty-choices chunk with `usage`. Completions: ignored, the finish chunk always carries `usage` (llama-server's final result is the non-stream body) | all |
| cache_prompt | schema.cpp:33 | accepted without effect: prefix reuse is automatic and a hit changes no output; `--no-prefix-reuse` is server-wide | all |
| n_cache_reuse | schema.cpp:69 | accepted without effect (own prefix cache) | all |
| id_slot | context.cpp:5301 | **done** (ADR 0090): pins the request to sequence `id_slot` (slot `i` of `--parallel`); it waits while that sequence runs and does not hold up other requests; an id out of range wraps around the slots (`get_slot_by_id`: 99 on two slots is slot 1) | all |
| return_tokens | schema.cpp:36 | `true` is rejected with 400; `false` is accepted | n/a |
| return_progress / sse_ping_interval | schema.cpp:39,42 | accepted without effect; SSE keep-alive fixed 15 s (openai.rs) | n/a |
| timings_per_token / verbose / response_fields / t_max_predict_ms / n_indent | schema.cpp:19,22,80,76,52 | `timings_per_token`, `response_fields`, `t_max_predict_ms` and `n_indent` are rejected with 400 when set to anything but their default; `verbose` is accepted without effect | n/a |
| n_probs, logprobs, top_logprobs, post_sampling_probs | schema.cpp:181,192; common.cpp:1679-1688 | **done** (node `serving-parity-fields`): chat `logprobs: true` with `top_logprobs` (default 20), completions `logprobs: N`, and `n_probs` on both; the probabilities are the softmax of the raw logits (before the token mask, bias, grammar and the sampling chain), which is what llama-server reports by default; `post_sampling_probs: true` is rejected with 400; `top_logprobs` without `logprobs` is 400, as is `logprobs` with tools and a stream. Shape as there: `content[]` of `id`, `token`, `bytes`, `logprob`, `top_logprobs[]`; the end-of-generation token has an entry with no text; no entry for a token that leaves an incomplete UTF-8 character; a stop string drops the entries of its tokens in non-streaming responses. Streams carry them on the chunks. Case `logprobs`, compared with a tolerance | all |
| grammar (GBNF) | schema.cpp:271 | **done** (node `serving-parity-fields`): `grammar` on chat, completions and Anthropic requests runs libllama's grammar sampler on the tokenizer model's vocabulary, applied as `common_sampler_sample` does (chain first, grammar as a rejection check, then the grammar before the chain); it composes with the tool-call token mask (the mask applies first); a grammar that does not parse is a 400 before generation starts. `grammar_lazy: true` is rejected. Case `grammar-gbnf` | all |
| json_schema (native field) | schema.cpp:271; common.cpp:1520-1523 | **done** (node `serving-parity-fields`): `json_schema` is converted to GBNF by llama.cpp's own converter (`common/json-schema-to-grammar.cpp` at the pin, built through a shim in `crates/ggml-sys/shim`, `docs/sources.md`); `{}` means any object; with both fields on a completion the grammar wins (as there), on chat both is a 400. Case `json-schema-top-level` | all |
| response_format: json_object / json_schema | common.cpp:1525-1540 | supported (openai.rs:284-300; Anthropic `output_config.format` json_schema only, anthropic.rs:373-388) via llguidance mask | all |
| tools | common.cpp:1503-1508 (needs `--jinja`, on by default, common.h:675) | supported, function tools only (openai.rs:224-250: other types rejected) | all |
| tool_choice auto/none/required/named | common.cpp:1588 -> `common_chat_tool_choice_parse_oaicompat` | supported (openai.rs:252-282; Anthropic auto/any/none/tool, anthropic.rs:337-371). Forced choices use a grammar (ADR 0057 line 13: only forced choice builds a grammar; `auto` is parsed after the fact) | all |
| parallel_tool_calls | common.cpp:1592 | supported (openai.rs:353; Anthropic `disable_parallel_tool_use`, anthropic.rs:348-351) | all |
| chat_template_kwargs | common.cpp:1626-1630 | supported (openai.rs:70,354) | all |
| reasoning_format (request) | common.cpp:1618-1621; schema.cpp:313 | ignored | all |
| reasoning_effort | common.cpp:1643-1650 | **done** (node `serving-parity-fields`): `none` sets `enable_thinking` false, any other value goes to the template as `reasoning_effort` | all |
| reasoning_budget_tokens / thinking_budget_tokens | common.cpp:1664-1676; schema.cpp:354 | ignored; Anthropic `thinking.type` enabled/adaptive/disabled sets `enable_thinking` kwarg, `budget_tokens` ignored (anthropic.rs:394-407); llama-server maps `thinking.budget_tokens` (chat.cpp:650-657) | all |
| continue_final_message / add_generation_prompt / prefill_assistant | common.cpp:1593-1610 | partial: Anthropic trailing assistant message becomes `prefill` (anthropic.rs, `Prompt::Chat.prefill`, openai.rs:350 passes `None`) | all |
| echo | schema.cpp:325 (chat), OAI completions | `echo` is ignored on `/v1/completions` (the pin never calls its `echo` check there, `server-common.cpp:1268` is dead code); on chat it means continue-the-final-message, a gap with `continue_final_message` | all |
| suffix | /infill | rejected (openai.rs:493-495) | n/a |
| lora (per request) | schema.cpp:231 | a non-empty list is rejected with 400 | n/a |
| user | not used for sampling (metadata only) | parsed, used as cost/quota key `caller` (openai.rs:67,364) | all |
| speculative.* per request | schema.cpp:202-227 are `#if 0` | n/a | n/a |
| Anthropic: system | server-chat.cpp (converter 391-684) | supported (anthropic.rs:31) | all |
| Anthropic: metadata.user_id | chat.cpp:668-674 | supported as `caller` (anthropic.rs:427-432) | all |
| Anthropic: thinking blocks in history | chat.cpp:460,585 | supported -> `reasoning_content` (anthropic.rs:220-226; redacted_thinking dropped) | all |
| Anthropic: image/document blocks | chat.cpp (media) | image -> `[image]`, document -> its text (anthropic.rs:60-69); unknown blocks dropped with a warning (anthropic.rs:70-77) | all |
| **[new @pin]** Anthropic: output_config.effort -> reasoning_effort | server-chat.cpp:662-668 | **done** (node `serving-parity-fields`): read as `reasoning_effort` | n/a |
| Anthropic: extra_params passthrough (min_p, seed ...) | chat.cpp:676-682 | missing | n/a |
| Anthropic: top_k / top_p / temperature | chat.cpp:646 | supported (anthropic.rs:34-36) | all |
| Anthropic: tool defs `input_schema`, `defer_loading`/`tool_reference` | chat.cpp:600-615 | supported (anthropic.rs:330; tool_reference text, anthropic.rs:56) | all |

### 2b. Sampler defaults and order at the pin (`common/common.h`, `common/sampling.cpp`)

| Item | llama-server at the pin | Tetraeder |
|---|---|---|
| Defaults (`common_params_sampling`, common.h:231-260) | seed random (`LLAMA_DEFAULT_SEED`, :232), top_k 40 (:237), top_p 0.95 (:238), **min_p 0.05** (:239), xtc_probability 0 (:240), **xtc_threshold 0.10** (:241), typical_p 1.0 (:242), temp 0.80 (:243), penalty_last_n 64 (:246), repeat 1.0, freq 0, presence 0 (:247-249), dry_multiplier 0 (:250), adaptive_target -1 (:254), mirostat 0 (:256), top_n_sigma -1 (:257), ignore_eos false (:260), min_keep 0 (:236), n_probs 0 (:235) | **done**: `SampleParams::default` has all of these (`crates/scheduler/src/sample.rs`); the `adaptive_*` and `n_probs` fields are not supported |
| Default sampler order (common.h:268-278) | penalties, dry, top_n_sigma, top_k, typical_p, top_p, min_p, xtc, temperature, then `dist` (seeded) | **done**: libllama's chain built in this order, ending in `dist` (`sample.rs`) |
| Chain construction (sampling.cpp:299-444) | grammar sampler (output-format / tool-call grammars prefilled with the template's generation prompt, :299-313); reasoning-budget sampler (:315-328); `logit_bias` merged with the **model's suppress tokens** (`llama_vocab_get_suppress_tokens`, -inf; :328-342); chain per `samplers` list; ends in `greedy` when temp == 0 (or top_k == 1) and n_probs == 0 (:404-407), else `dist(seed)`; mirostat 1/2 replaces the chain with temp + mirostat (:409-415) | **done** except the grammar and reasoning-budget samplers (Tetraeder masks tokens through its own constraints) and `n_probs`: logit_bias plus the model's suppress tokens, the `samplers` list, greedy when the list ends in temperature 0 or `top_k` 1, Mirostat 1 and 2. Stages that cannot change a token with the given settings are left out |
| Sampler names accepted in `samplers` (sampling.cpp:964-980) | dry, top_k, top_p, top_n_sigma, typ_p, min_p, temperature, xtc, infill, penalties, adaptive_p | **done** for dry, top_k, top_p, top_n_sigma, typ_p, min_p, temperature, xtc, penalties; `infill` and `adaptive_p` are rejected with 400 |
| Backend (GPU) sampling | `backend_sampling` field / `-bs` flag, disabled with grammar or reasoning budget (sampling.cpp:424-436) | n/a (host sampler over a logits row) |
| RNG | `std::mt19937` seeded from the chain seed (sampling.cpp:444) | **done** (node `serving-parity-sampling`): libllama's `dist` sampler, the same mt19937; unseeded requests draw a random seed. Case `seeded-rng` |

Sampler facts for Tetraeder (all engines share one host-side sampler over the logits row): fields are temperature, top_k, top_p, seed, plus the constraint mask (`T:scheduler/src/sample.rs:35-44,63-115`; `T:gateway/src/engine.rs:57-65` `SamplingParams` is only `max_tokens, temperature, top_p, top_k, seed`). Temperature <= 0 is greedy, lowest id wins ties (sample.rs:65-66,150-161). Scheduler fills defaults from `--temp/--top-k/--top-p` (scheduler.rs:1277-1283; serve.rs:65-71). Speculative decoding (`spec/mod.rs`): `Sampling` has only temperature+seed (spec/mod.rs:67-77), the scheduler drafts only for **greedy, unconstrained** sequences (scheduler.rs:1620-1633); `--spec` is off by default.

### 3. Response features

| Capability | llama-server evidence | Tetraeder status | Engine notes |
|---|---|---|---|
| usage (prompt/completion/total, `prompt_tokens_details.cached_tokens`) | server-task.cpp:357-364 | **done** (node `serving-parity-fields`): `completion_tokens` counts every sampled token, the end-of-generation token included, as llama-server's `n_decoded` does (also Anthropic `output_tokens`); the request log and the decode metrics keep counting emitted tokens. Case `usage-eos-count` | all |
| timings (`prompt_n`, `prompt_ms`, `predicted_per_second`, `cache_n`, `draft_n`, ...) | server-common.cpp:146-165; task.cpp:400,448,509 | **done** (node `serving-parity-fields`): `timings` on non-streaming responses and on the last stream chunk (the usage chunk when there is one), from the scheduler's per-request timestamps: `prompt_n` is the uncached prompt tokens, `prompt_ms` runs from the first admission to the first sampled token, `predicted_ms` from there to the last token, and the rates count the steps after the first token. `draft_n` fields are not reported. Case `timings` | all |
| finish_reason chat | task.cpp:407-416: length / stop / tool_calls | same three (chat.rs:111-119); stop sequence maps to `stop` | all |
| finish_reason completions | task.cpp:374-377 | stop / length (openai.rs:540) | all |
| `stop_type`, `stopping_word`, `truncated`, `tokens_cached`, `generation_settings` (native) | task.cpp:337-349 | **done** (node `serving-parity-endpoints`, ADR 0089): in the native response; `generation_settings` is the server defaults plus the request's values, fields the engine lacks carry llama-server's defaults | n/a |
| logprobs shape | task.cpp:274-300 (`content[]` with token, bytes, logprob, top_logprobs; native `completion_probabilities`) | **done** (node `serving-parity-fields`): see `n_probs`; the shape is the same on completions and chat | n/a |
| system_fingerprint | task.cpp:389 | **done** (node `serving-parity-fields`): `tetraeder-<version>` on every response and chunk (llama-server writes its build, `b12426-fc504f91d`; a client treats the string as opaque). Case `system-fingerprint` | n/a |
| streaming chat chunk | task.cpp:456-510 (`chat.completion.chunk`, per-diff deltas, incremental tool-call argument deltas, optional `timings`, `prompt_progress`) | `chat.completion.chunk`: first chunk `{role, content:""}`, text/reasoning deltas, tool calls emitted **whole in one delta** with `index` (openai.rs:401-485), final empty delta with finish_reason, optional usage chunk with `choices: []`, `[DONE]`; mid-stream errors as an `error` data frame without `[DONE]` (openai.rs:475-479) | all |
| tool call output | `msg.to_json_oaicompat` (common_chat parsers, many families under `common/`) | parsed by gateway from text: Hermes `<tool_call>`, Qwen XML `<function=`, Llama3 JSON/`<|python_tag|>`, Mistral `[TOOL_CALLS]`; family detected from chat template, unknown -> Hermes (toolcall.rs:25-56). Arguments typed as strings of JSON (openai.rs:320-326) | all (family from template) |
| reasoning/thinking content | `--reasoning-format` default deepseek -> `message.reasoning_content` (arg.cpp:3730-3740; common.h:686) | `reasoning_content` field, split on `<think>`/`</think>` when template contains `<think>` (gateway.rs:337, toolcall.rs:12-13); Anthropic `thinking` block with empty `signature` (anthropic.rs:544-549,587,607). Only the `deepseek` mode; no `none` / `deepseek-legacy` choice | all |
| Anthropic stop_reason | task.cpp:735-737,788: max_tokens / end_turn / tool_use | end_turn / max_tokens / stop_sequence / tool_use (anthropic.rs:436-443) | all |
| Anthropic usage | task.cpp (anthropic) | input_tokens (minus cached), output_tokens, cache_creation 0, cache_read = cached (anthropic.rs:445-452), the formula of llama-server; the reused-prefix count differs by design, so cases compare `stop_reason` and not `input_tokens` | all |
| Error format | server-common.cpp:98-140: `{code,message,type}`, 501 for not-supported, 503 | OpenAI `{error:{type,code,...}}` and Anthropic errors with `request-id` (error.rs:49-102); overload 429, Anthropic 529 (error.rs:66,100); `Retry-After` (gateway.rs:35) | all |
| Request log (prompts) | `--log-prompts-dir` arg.cpp:3946 | `--request-log FILE` JSONL, authorization never logged (serve.rs:104-106; reqlog.rs:7) | all |

### 4. Server flags

llama-server flags: `C:arg.cpp` (line is the flag-name line). Tetraeder: `T:tetraeder/src/serve.rs` and `T:tetraeder/src/engine_args.rs`.

| Behaviour | llama-server | Tetraeder `serve` | Engine notes |
|---|---|---|---|
| Model | `-m` arg.cpp:3027; `-hf` :3062; `-mu` :3046 | `-m` only; `-hf`, `-hff` and `--hf-repo` are **refused** with a pointer to download the GGUF and pass `-m` (no download client in the workspace; ADR 0090); no `--model-url` | all |
| Parallel slots | `-np` :2515 (server default auto = 4, kv_unified true; arg.cpp:1382; server.cpp:129-133) | `--parallel` default 4 (`PARALLEL`, scheduler.rs:30; serve.rs:74-75); with `--plan` replanner exactly 1 | libllama, dense, moe |
| Context size | `-c` :1618 (per-slot split unless `-kvu`) | `--ctx` default 32768, one shared pool for all sequences (serve.rs:47-51) | all |
| Batch sizes | `-b` :1648, `-ub` :1655 | `--step-tokens` (default 2048), `--ubatch`; `-b` ignored (serve.rs:35-38,76-81) | libllama uses n_batch/n_ubatch; builders use step tokens |
| KV cache types | `-ctk/-ctv` :2417,2430 | **done** (ADR 0090): `--cache-type-k/-v` (`-ctk/-ctv`) with llama-server's names set `type_k`/`type_v` on the libllama context; `kv_format` reports the pair for the prefix cache. Own builders: a pair `--kv` allows (`q8_0/q8_0`, `q8_0/q4_0`, `f16`, `bf16`, `q4_0`) maps onto `--kv`, any other is refused with the allowed list | libllama; builders by `--kv` |
| KV offload (`-nkvo`) | `-kvo/-nkvo` :2394 | **done** (ADR 0090, ADR 0084 step 0): `--no-kv-offload` (`-nkvo`) and `--kv-offload` set `offload_kqv` on the libllama context; the own builders refuse the flag (host KV there is ADR 0084 step 1). Not run on a GPU yet (CI has none) | libllama; builders refuse |
| Flash attention | `-fa` :1734 | `--flash-attn auto/on/off` (engine_args.rs:54-56) | libllama; builders own kernels |
| Unified KV / per-slot | `-kvu` :1704 | always unified (serve.rs:635; load.rs:27-30) | all |
| Context shift | `--context-shift` :1720 | **done** (ADR 0090): `--context-shift` (default off, as at the pin) drops `n_discard` positions after `n_keep` (request fields, llama-server's arithmetic) and moves the rest down when a sequence is one position from a full context; a shifted sequence is freed at its end, not reused; backends that cannot shift (builders, workers, recurrent models) fail at start | libllama; others refuse |
| Prompt cache | `--cache-prompt` :3611, `--cache-reuse` :3619, `--cache-ram` :1695, `--cache-idle-slots` :1712, `--slot-prompt-similarity` :3847 | `--cache-ram` MiB default 8192 (serve.rs:97-99), `--cache-disk <dir>` and `--cache-disk-mib` disk tier, default off (ADR 0087), `--no-prefix-reuse` (serve.rs:100-103) | all |
| Slot save/restore | `--slot-save-path` :3651 | **done** (ADR 0090): see the `/slots` row | libllama; builders refuse |
| Chat template | `--chat-template` :3801, `--chat-template-file` :3813, `--jinja` :3722 (default on) | `--chat-template-file` (serve.rs:107-109), template from GGUF otherwise (minijinja, template.rs:1-8). No built-in named templates | all |
| chat-template-kwargs | `--chat-template-kwargs` :3571 | per request only | all |
| Reasoning | `--reasoning-format` :3730, `--reasoning` :3741, `--reasoning-budget` :3771, `--reasoning-effort` :3759 | none (always deepseek-style split) | all |
| Tools/MCP server side (**[new @pin]** `--allow-unauthenticated-tools` :3447, `--media-allow-private-urls` :3679) | `--tools` :3426, `--mcp-servers-*` :3454-3462, `--agent` :3470 | missing | n/a |
| Embeddings / rerank | `--embeddings` :3514, `--rerank` :3521, `--pooling` :2293 | **done** (ADR 0090): `--embeddings`/`--embedding`, `--reranking`/`--rerank`, `--pooling none|mean|cls|last|rank` (model default when unset); refused with `--plan`, `--worker`, `--fake` and on the builders | libllama; builders refuse |
| LoRA | `--lora` :2947, `--lora-scaled` :2957, `--lora-init-without-apply` :3854, `--control-vector` :2972 | missing (no lora in crates; only ggml-sys build.rs mention) | n/a |
| Multimodal | `--mmproj` :2549, `--mmproj-url` :2557, `--image-*` :2600-2623 | missing | n/a |
| Speculative / draft | `--spec-type` :4325, `--spec-draft-model/-md` :4317, `--spec-ngram-*` :4335-4491, `--draft*` :4463, `--spec-default` :4807 | `--spec ngram|draft:GGUF`, `--spec-max-batch` (default 1), `--spec-max-draft` (default 6, max 6) (serve.rs:82-93,598-618); lone greedy unconstrained requests only | draft model runs on libllama; works with all engines |
| Grammar flags | `--grammar` :2248, `--json-schema` :2262 (server defaults) | missing | n/a |
| Sampling defaults | `--temp` :1995, `--top-k` :2004, `--top-p` :2012, `--min-p` :2020, penalties :2058-2106, dry :2106-2141, mirostat :2201, `--samplers` :1965, `--seed` :1974, `--ignore-eos` :1988, `-l/--logit-bias` :2226 | **done** for `--temp`, `--top-k`, `--top-p`, `--min-p`, `--typical`, `--top-nsigma`, `--xtc-*`, `--repeat-*`, `--presence-penalty`, `--frequency-penalty`, `--dry-*` (breakers: `--dry-sequence-breaker`, repeatable), `--mirostat*`, `--dynatemp-*`, `--samplers` (`;` separated), `--seed`, `--ignore-eos`; not `-l/--logit-bias` or `-s` | all |
| API key | `--api-key` :3529, `--api-key-file` :3540 | **done** (ADR 0089): `--api-key`, `--api-key-file`, `LLAMA_API_KEY` | n/a |
| TLS | `--ssl-key-file` :3557, `--ssl-cert-file` :3564 | missing | n/a |
| Metrics/props/slots endpoints | `--metrics` :3629, `--props` :3636, `--slots` :3643 | `--metrics` and `/props` **done** (ADR 0089), OTLP via env as before; `--props` (nothing to change) and `--slots` missing (`GET /slots` is always on, as by default there) | n/a |
| Host/port | `--host` :3323, `--port` :3339 | `--host 127.0.0.1`, `--port 8080` (serve.rs:52-55) | all |
| CORS, path, api-prefix | :3353-3396 | missing | n/a |
| HTTP threads, timeout, SSE ping | `--threads-http` :3604, `-to` :3589, `--sse-ping-interval` :3597 | `--max-queued` 16 (serve.rs:94-96; gateway.rs:30-50 stream_buffer 32); SSE keep-alive 15 s fixed | all |
| GPU layers | `-ngl` :2781 | `--ngl/--n-gpu-layers` default -1 (engine_args.rs:28-29) | libllama; builders: 0 = CPU else GPU `--main-gpu` (serve.rs:538-560) |
| Split mode | `-sm` :2799 (none/layer/row/tensor) | `--split-mode none/layer/row/tensor` (engine_args.rs:30-32) | libllama; builders only through `--plan FILE` with executor ranks (serve.rs:152-155,437-462) |
| Tensor split / main GPU | `-ts` :2823, `-mg` :2850 | `--tensor-split`, `--main-gpu` (engine_args.rs:33-37) | same as above |
| Tensor overrides | `-ot` :2725, `-cmoe` :2731, `-ncmoe` :2738 | **done** (ADR 0090): `--cpu-moe` (`-cmoe`) and `--n-cpu-moe N` (`-ncmoe`) add llama-server's patterns (`\.ffn_(up\|down\|gate\|gate_up)_(ch\|)exps`, per layer `blk\.N`) after `--override-tensor`, on libllama. The MoE builder refuses them with a pointer to `--expert-slots`, `--ram-slots` and `--miss`, the dense builder too | libllama honours overrides; builders refuse the CPU-MoE flags and `--ot` is not applied |
| Load mode | `-lm` :2662, mlock/mmap flags | `--load-mode auto/none/mmap/mlock/mmap-mlock/direct-io` (engine_args.rs:42-44) | libllama |
| Threads | `-t` :1496, `-tb` :1506 | `-t` (engine_args.rs:46-47) | all |
| Multi-model router | `--models-dir` :3686 etc. | missing | n/a |
| Planner / replan / worker isolation | none | `--plan auto|FILE`, `--hw`, `--plan-*`, `--worker`, `--admin`, `--replan-*` (serve.rs:147-199) | planner: libllama and MoE builder; dense only with plan FILE |
| Pin-only flags | `--moe-cache-cpu-share` :2758 and `--ple/--shard` :3036 are **[new @pin]**; `-dt/--defrag-thold` is **[removed @pin]** | none | n/a |
| Fake engine | none | `--fake` (serve.rs:111-112) | test only |

### 5. Model architectures and engine selection

`serve::loader` (`T:tetraeder/src/serve.rs:399-432`) reads `general.architecture` via `tetraeder_scheduler::architecture` (`T:scheduler/src/load.rs:131-136`) only for `--engine auto`:

- `--engine auto` (default, serve.rs:45-46,405-410): arch `qwen3moe` -> own MoE builder (`load_moe`, load.rs:40-62); **every other arch -> libllama** (`load_llama`, load.rs:20-37). The enum doc says the same: "The Qwen3 MoE builder for GGUF architecture `qwen3moe`, else libllama" (serve.rs:26).
- `--engine llama`: libllama regardless of arch (serve.rs:403).
- `--engine moe`: forced MoE builder regardless of arch (serve.rs:404); the builder rejects non-`qwen3moe` GGUFs: `Hparams::from_gguf` errors "architecture `X`, expected `qwen3moe`" (`T:model/src/hparams.rs:65-71`).
- `--engine dense`: forced dense builder (serve.rs:401,426-427); rejects non-`qwen3`: `DenseHparams::from_gguf` errors "architecture `X`, expected `qwen3`" (`T:model/src/qwen3.rs:84-91`). **`auto` never selects dense**: "`auto` keeps libllama" for `qwen3` (serve.rs:30). Dense must be asked for explicitly.
- `--plan auto` (replanner): `qwen3moe` -> builder candidates, else libllama (`T:tetraeder/src/replan/start.rs:51-67`); `--engine dense` with `--plan auto` bails (start.rs:54-58); `--engine dense --plan FILE` runs on the executor (serve.rs:441-448).
- No arch allow-list or rejection exists in the serve path for libllama: `load_llama` calls `Engine::load` with no architecture check (load.rs:20-37). So `llama`, `gemma3`, `qwen3` (under auto), etc. are **routed to libllama, not rejected**; whether they load is libllama's decision. Hence llama-server's architecture coverage (175 `LLM_ARCH_*` names in `vendor/llama.cpp/src/llama-arch.cpp` at the pin, e.g. `llama` :11, `qwen3` :38, `qwen3moe` :40, `gemma3` :63) is inherited by `tetraeder serve` through libllama (the same vendored libllama), minus whatever the gateway cannot do (no mmproj).
- The tokenizer is always libllama (`vocab_only`) for the builders (load.rs:46-56,108-117).
- Gateway behaviour per architecture is decided by the chat template text, not the arch: tool format by `ToolFormat::detect` (toolcall.rs:38-56), `<think>` handling when the template contains `<think>` (gateway.rs:337).

| `general.architecture` | `--engine auto` | Forced engines | Evidence |
|---|---|---|---|
| `qwen3moe` | own MoE builder | `moe` ok; `llama` ok; `dense` errors | serve.rs:405-410; hparams.rs:65-71; qwen3.rs:85-91 |
| `qwen3` | libllama | `dense` ok (own builder); `moe` errors; `llama` ok | serve.rs:30,405-410; qwen3.rs:39-40 |
| `llama`, `gemma3`, anything else | libllama (not rejected) | `moe`/`dense` error with "expected ..." | load.rs:20-37; serve.rs:402-411 |
| unreadable GGUF / no arch key | error "cannot read the model architecture" (auto path only) | | serve.rs:406-408 |

Note: the planner candidates for architectures other than qwen3/qwen3moe come from `plan::layout` (`T:tetraeder/src/plan.rs:63`, reads arch); not traced here.

### 6. Other features

| Capability | llama-server | Tetraeder | Engine notes |
|---|---|---|---|
| Embeddings | `--embeddings`, /v1/embeddings (context.cpp:6416-6424,6691-6760) | **done**, see `/v1/embeddings` above | libllama |
| Reranking | `--rerank`, /rerank (arg.cpp:3521; server.cpp:200-203) | **done**, see `/rerank` above | libllama |
| Tokenize / detokenize | server.cpp:205-206; context.cpp:6361,6402 (`add_special`, `parse_special`, `with_pieces`) | **done** (ADR 0089): `/tokenize` and `/detokenize`; `count_tokens` (Anthropic) is unchanged | all |
| apply-template | context.cpp:6334 | **done** (ADR 0089): `/apply-template` | n/a |
| /props | context.cpp:5823 | **done** (ADR 0089): `/props` | n/a |
| /slots | context.cpp:5748 | **done**, see `/slots` above | `GET` all, `POST` libllama |
| /metrics | context.cpp:5686 | **done** (ADR 0089): `/metrics` behind `--metrics`, OTLP push as before | n/a |
| /health | context.cpp:5673 | **done** (ADR 0089): `/health`, `/v1/health` | n/a |
| Infill (FIM) | context.cpp:5847 (`input_prefix`, `input_suffix`, `input_extra`) | **done** (ADR 0089): `/infill` | n/a |
| Multimodal (mtmd) | `--mmproj`; server.cpp mtmd ctx (context.cpp:22-23,50-51,268) | missing; OpenAI image parts rejected (openai.rs:106-110); Anthropic image becomes `[image]` | n/a |
| LoRA adapters | /lora-adapters; `lora` per request (schema.cpp:231) | missing | n/a |
| Prompt caching | slot prompt cache, `cache_prompt` | internal prefix cache + RAM save/restore (`PrefixCache`, scheduler lib.rs:21; `--cache-ram`); reported as `cached_tokens` | all |
| Continuous batching | on by default | `--parallel`, step token budget (serve.rs:72-81) | all |
| Structured output | json_schema -> GBNF | json_schema -> llguidance mask (constraint.rs) | all |
| Quota / auth / multi-tenant | api keys only | `caller` attribution (openai.rs:364; anthropic.rs:427) and request-log cost; no enforcement in serve (GatewayConfig has no key/quota fields, gateway.rs:29-40) | all |
