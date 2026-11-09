# 0051. No prompt checkpoints; tool-call outcomes counted, empty calls dropped

Status: Proposed (Phase 3b). Extends ADR 0049 (prefix cache) and ADR 0028 (tool-call parser). Briefs: `docs/gateway-briefs.md`.

## Context

The plan's levers table names two gateway ideas: Strata's prompt checkpoints at message boundaries and near the tail, and Unsloth's tool-call repair. The only evidence is the first Claude Code proxy run (bench/results/20261103/claude-code-proxy, ADR 0046 "Measured"): per request token counts, cached tokens, stop reason and errors from `scripts/anthropic_proxy.py`, plus Claude Code's session results. It holds no request or response bodies: the request logs (`tetraeder-r*.reqlog.jsonl`) stayed on the destroyed box (clear-text prompts, ADR 0046), and llama-server logs no parse outcomes.

**Cache share by class** (`scripts/claude_code_cache_classes.py`; conversations inferred from token counts, 64-token slack):

| Uncached prompt tokens, share of all prompt tokens | Tetraeder | llama-server |
|---|---|---|
| First request of a conversation (Claude Code's prompts share 21 to 23 tokens) | 2.60% | 2.70% |
| Turn suffix (new turns, re-rendered output) | 1.79% | 2.59% |
| Mid-prompt divergence, shared part already cached | 0.91% | 0.58% |
| Evicted by another conversation | 9.25% | 0.51% |
| **Served from cache, measured** | **85.4%** | **93.6%** |
| Served from cache with ADR 0049, estimated | 94.3% | 94.1% |
| Ceiling of any prefix cache (first and suffix are new tokens) | 95.6% | 94.7% |

The eviction loss is three sessions (r1 feature-late-fee, r1 and r2 rename-compute-total) where Claude Code alternated its main conversation (11K tokens and growing) with a subagent (5K and growing), each request re-prefilled from token 22. ADR 0049 restores the other conversation from host RAM; 8.8 of the 9.25 points come back.

**Tool calls.** No HTTP error beyond the 12 `tool_reference` 400s that ADR 0049 fixed (`protocol_errors` 15 = those 12 plus 3 sessions at Claude Code's turn limit; llama-server has 3 such sessions too). Stop reason `tool_use` on 460 of 483 Tetraeder and 586 of 624 llama-server responses. One parser artifact: the last response of tetraeder-r1-write-shipping-tests (209 tokens, `end_turn`) ended with a bare `<tool_call>` before end of generation; `step_tagged` found an empty block and emitted it as text, which ended up in Claude Code's result. The task passed. No llama-server result shows a marker. Arguments Claude Code rejects (schema validation, unknown tool names) come back as `is_error` tool results in the next request body, so they are unmeasured for both engines.

## Decision

**No prompt checkpoints.** The KV rows of a prefix do not depend on later tokens in a full-attention model, so ADR 0046 (truncate the live sequence to the longest common prefix) and ADR 0049 (restore a saved sequence, then truncate) already reuse the longest common prefix of any live or saved sequence at token granularity: editing or branching a long history keeps everything before the edit. Checkpoints only pay where state cannot be cut at an arbitrary token: recurrent or linear-attention layers, sliding-window ring buffers, caches kept in blocks. No served model has such state (qwen3moe builder, libllama dense Qwen3). On this traffic they could touch only the divergence class (0.9%), where the shared part is already cached (3,486 of 18,332 tokens after the `tool_reference` retries, 17,226 of 22,424 in r2 write-shipping-tests). Revisit with the first hybrid or sliding-window model, in Phase 4's `kvcache` prefix tree. If restore time shows up in `tetraeder.kv.prefix.restore.duration`, copy only the rows up to the common prefix; that needs no checkpoints.

The cache bar of the Phase 3b exit is llama-server's share in the same run (93.6% here, quoted as 94%), not a fixed number: the estimate beats it by 0.7 points, within what one run moves.

**Tool-call outcomes are counted** by the gateway's output parser, per call: outcome `parsed`, `repaired` or `failed`, and a class. `parsed`: `ok`, or `unknown_tool` (name not in the request's tools; passed through, the client reports it, as llama-server does). `failed`: `unclosed` (output ends inside the call), `bad_json`, `bad_xml`; the raw text still goes out as text, as today. Counter `tetraeder.gateway.tool_calls` with `tetraeder.outcome` and `tetraeder.class`.

**One repair, for the one class seen:** a call opener whose body is empty or whitespace (`<tool_call>` with or without `</tool_call>`, Mistral `[TOOL_CALLS]`, Llama 3 `<|python_tag|>`) is dropped, counted `repaired`/`empty_call`. The finish reason follows from the calls that remain.

**The rerun measures the rest on both engines.** The proxy counts per request, without storing bodies: `is_error` tool results in the request's last user turn by class (`input_validation`, `unknown_tool`, `other`), and from the response the tool calls, names not in the request's tools, and text containing `<tool_call>`, `<function=` or `<parameter=`. The summary reports them per engine. A repair for another class is designed only when the rerun shows it on Tetraeder in at least 1% of tool calls or more often than on llama-server, with fixtures extracted from the rerun's request log (only the failing outputs, never prompts). Not seen and not designed: a missing `</parameter>` or `</function>`, JSON inside XML parameters, fuzzy tool-name matching, type coercion beyond the schema.

Panel "Tool calls by outcome" (rate by outcome and class) in the gateway rows of the overview dashboard; alert `ToolCallFailures` when failed plus repaired calls exceed 2% of tool calls over 30 minutes, for 15 minutes (a template or model regression).

## Consequences

- Nothing to build for the prefix cache; the rerun shows whether ADR 0049 reaches llama-server's share. Risk: the 8 GiB default holds about 87K tokens of the 30B's f16 KV (98,304 bytes per token); the run's largest prompt was 70.8K, so a long main conversation plus a subagent can evict. `PrefixCacheThrash` fires then; raise `--cache-ram` if the box has the RAM.
- Tool-call repair stays one rule until data asks for more; every later rule comes with a fixture and a class label.
- The proxy's new counts make both engines' tool-call failures comparable without keeping prompts.

## Implementation notes

Built (briefs G1 and G2).

- Parser (`crates/gateway/src/toolcall.rs`): `OutputParser::take_outcomes()` returns `CallOutcome { outcome, class }`, one per decided call. Classes as built: `parsed/ok`, `parsed/unknown_tool`, `repaired/empty_call`, `failed/unclosed`, `failed/bad_json`, `failed/bad_xml`. `Step::Fail` carries the class, `Step::Drop` is the empty-call repair (block consumed, nothing emitted).
- A tagged block that fails to parse and has no `</tool_call>` is `unclosed`; with the closing tag it is `bad_xml` (QwenXml content starting with `<function=`) or `bad_json`. A tagged block without the closing tag that parses stays `parsed/ok`, as before. Llama 3 is `unclosed` when the output ends inside a JSON value, else `bad_json`; Mistral is `unclosed` when the JSON is partial at the end.
- Behaviour change: a bare `[TOOL_CALLS]` at the end of output was text, now it is dropped (the Mistral case of the repair). The one existing test asserting it (`mistral_malformed_is_text`) lost that input.
- Gateway: counter `tetraeder.gateway.tool_calls` (`{call}`, attributes `tetraeder.outcome`, `tetraeder.class`), recorded by `Run::record_outcomes` after every push and after `finish`. Panel "Tool calls by outcome" (overview.json, id 39, appended after the prefix cache panel) and alert `ToolCallFailures`.
- Proxy (`scripts/anthropic_proxy.py`): the Observer keeps per-block text only while the response streams and writes counts, never bodies: `tool_uses`, `tool_errors` by class, `tool_names_unknown`, `leaked_markers`. Leaked markers are found on the whole text block, so delta boundaries need no tail buffer. The `is_error` patterns (`InputValidationError`, `No such tool available`) are unchecked against Claude Code's real wording until the rerun's request log; adjust then.
- Summary: per-engine totals, `tool_error_rate`, a `terr` table column; `claude_code_proxy.sh` also writes `cache_classes.txt`. `scripts/toolcall_fixtures.py` extracts failing outputs from a request log (reuses the Observer).
