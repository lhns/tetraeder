# 0057. Qwen XML tool arguments: Python literals typed by the schema

Status: Proposed (Phase 3b). Extends ADR 0051 (tool-call outcomes) and ADR 0028 (tool-call parser).

## Context

The second Claude Code proxy run (bench/results/20261110/claude-code-proxy, box 59998776, RTX A6000, Qwen3-Coder-30B-A3B Q4_K_M, 10 tasks x 2 reps) gave Tetraeder 12/20 tasks and an 18.5% tool-error rate (100 of 541 calls), llama-server 13/20 and 5.6% (26 of 466). Pairing each `is_error` tool result in the request logs with the `tool_use` it answers:

| Class (proxy) | Count | What it is |
|---|---|---|
| input_validation | 52 | Edit `replace_all` sent as the string `"False"` |
| input_validation | 4 | Edit `replace_all` as `"True"` |
| input_validation | 2 | ScheduleWakeup `stop` as `"True"` |
| input_validation | 2 | ReportFindings `findings` (array) as the string of a Python `repr` list (`[{'category': ...}]`) |
| other | 29 | Bash exit code 1 (pytest failures, scripts) |
| other | 4 | Bash refused by Claude Code's rm check (multi-line `python -c`) |
| other | 7 | Read on a directory or missing file, Edit string not found or a no-op edit |
| unknown_tool | 1 | `Explore` (not in the request's tools) |

The request log holds one more `other` result (40) than the proxy counted (39). The log confirms the proxy's `is_error` patterns, which ADR 0051 left unchecked.

All 60 validation errors are one cause: Qwen3-Coder writes XML parameter values as Python literals (`True`, `False`, `None`, single-quoted strings), which the parser in `toolcall.rs` could not read as JSON, so it kept the text as a string. The "other" errors are the model's work on the task, the same kind llama-server has (26). The five Tetraeder sessions with `is_error` (llama-server: one) all ended at Claude Code's 40-turn limit, and each had 4 to 13 validation errors: the model retried the same Edit with the same `"False"`.

llama-server never sends these: llama.cpp (`common/parsers/qwen3-coder.cpp` at the pin) samples tool calls under a lazy grammar triggered by `<tool_call>` (or `<function=NAME>`) whose non-string parameters must be JSON of the schema's type, so the model cannot write `False` there. The gateway builds a grammar only for forced tool choice (ADR 0028); Claude Code uses `auto`. The prompts do not differ here: both template engines print a JSON boolean through `| string` as `False` (the tool's `<default>False</default>` and earlier calls in the history), which likely primes the model to write it. llama-server in this run was the pin build (2e0d86b, env.json). llama.cpp's own Python-literal reader (`normalize_quotes_to_json` in `common/chat-peg-parser.cpp`) serves other formats, not Qwen3-Coder.

## Decision

`convert_param` keeps its rule (string-typed or untyped parameters stay text) and, for a parameter whose schema types exclude `string`, tries in order: JSON; a whole Python literal (str, int, float, `True`/`False`/`None`, list, tuple, dict with string keys, Python escapes, trailing comma); `true`/`false` in any case. The first result that fits a schema type wins; otherwise the text stays a string, as before. Own code, nothing copied.

Calls stay counted `parsed/ok`: the call is well formed and the value has the schema's type. Counting them `repaired` would put about 11% of calls into the `ToolCallFailures` alert, which is meant for template or model regressions.

Not done: a lazy grammar for `auto` tool choice, as llama.cpp does. It would also stop unknown tool names, but it masks every token after the trigger and needs a decode benchmark on a box (Rule 3). Revisit if the next run still shows validation errors or unknown names above llama-server.

## Consequences

- The 58 boolean calls of the run become schema-valid; the 2 ReportFindings calls get the right type (their item fields are not checked here). With all else equal the tool-error rate drops from 18.5% to about 7.4% (40 of 541) at best, near llama-server's 5.6%; converted Edits can still fail for task reasons. The turn-limit sessions lose their retry loops. The next proxy run measures it.
- A string parameter that the model fills with `False` stays `"False"`; only the schema decides.
- Union types with `string` (`["string", "number"]`) and `anyOf` schemas still keep text; llama.cpp tries JSON first there. No fixture shows them yet.

## Addendum: rerun 20261110b (box 60050186, A40)

Tasks 16/20 on both engines; tool errors 44 of 369 calls (11.9%) against llama-server's 27 of 550 (4.9%). No `input_validation` errors are left. Pairing each `is_error` result in the request logs with its `tool_use`:

| Proxy class | Count | What it is |
|---|---|---|
| other | 37 | Bash exit code non-zero: 31 of 80 pytest runs, 6 of 47 python runs |
| other | 5 | Edit string not found (2), Edit no-op (1), Read on a missing file (1) or a directory (1) |
| unknown_tool | 2 | `Explore` (a subagent type of `Agent`, not a tool), parsed correctly |

The 42 "other" errors are task outcomes; llama-server's 27 cannot be split from the committed data (no request log). The proxy now classes them `bash_exit` and `file`, and the summary adds `call_error_rate` (input_validation plus unknown_tool per tool call), the part the gateway answers for.

**Bare `<function=` opener.** The one leaked marker (tetraeder-r2-rename-compute-total, first response) was a call without `<tool_call>`: `<function=Bash>...</function>\n</tool_call>`. The parser sent it as text, Claude Code ended the session after one request and the task failed. llama.cpp's grammar triggers on `<function=NAME>` as well. Decision: in the Qwen XML format, `<function=NAME>` starts a call when NAME is one of the request's tools and `<parameter=` or `</function>` follows; it ends at `</function>` (an orphan `</tool_call>` after it is consumed) or at `</tool_call>`. Otherwise only the opener is text and scanning goes on, so prose that mentions `<function=` costs no later call. Outcome `repaired`/`no_opener`, counted in `ToolCallFailures` like `empty_call` (1 of 369 calls here). Test: `qwen_xml_call_without_opener_from_run_fixture`, from `toolcall-fixtures-r2.json`.

**Unknown names.** `Explore` twice against none on llama-server meets this ADR's condition for revisiting a lazy grammar under `auto`. Still not done: it needs a decode benchmark on a box (Rule 3).
