# 0057. Qwen XML tool arguments: Python literals typed by the schema

Status: Accepted (2026-11-15: built). Earlier: Proposed (Phase 3b). Extends ADR 0051 (tool-call outcomes) and ADR 0028 (tool-call parser). The rerun (`20261110b`) left no validation errors; the decisions it caused are in the addendum.

Measurements live in the session READMEs (`bench/results/<session>/README.md`), not here.

## Context

The second Claude Code proxy run (`20261110`, `claude-code-proxy`, RTX A6000, Qwen3-Coder-30B-A3B Q4_K_M) gave Tetraeder fewer solved tasks than llama-server and a tool-error rate about three times higher. Pairing each `is_error` tool result in the request logs with the `tool_use` it answers: most were `input_validation` errors, Edit `replace_all` and ScheduleWakeup `stop` sent as the strings `"False"` or `"True"` and ReportFindings `findings` (an array) as the string of a Python `repr` list; the rest were the model's work on the task (Bash exit codes, Bash refused by Claude Code's rm check, Read or Edit misses) and one unknown tool (`Explore`, not in the request's tools). The request log confirms the proxy's `is_error` patterns, which ADR 0051 left unchecked.

All validation errors are one cause: Qwen3-Coder writes XML parameter values as Python literals (`True`, `False`, `None`, single-quoted strings), which the parser in `toolcall.rs` could not read as JSON, so it kept the text as a string. The "other" errors are the model's work on the task, the same kind llama-server has. The Tetraeder sessions with validation errors ended at Claude Code's 40-turn limit, retrying the same Edit with the same `"False"`.

llama-server never sends these: llama.cpp (`common/parsers/qwen3-coder.cpp` at the pin) samples tool calls under a lazy grammar triggered by `<tool_call>` (or `<function=NAME>`) whose non-string parameters must be JSON of the schema's type, so the model cannot write `False` there. The gateway builds a grammar only for forced tool choice (ADR 0028); Claude Code uses `auto`. The prompts do not differ here: both template engines print a JSON boolean through `| string` as `False` (the tool's `<default>False</default>` and earlier calls in the history), which likely primes the model to write it. llama-server in this run was the pin build (2e0d86b, env.json). llama.cpp's own Python-literal reader (`normalize_quotes_to_json` in `common/chat-peg-parser.cpp`) serves other formats, not Qwen3-Coder.

## Decision

`convert_param` keeps its rule (string-typed or untyped parameters stay text) and, for a parameter whose schema types exclude `string`, tries in order: JSON; a whole Python literal (str, int, float, `True`/`False`/`None`, list, tuple, dict with string keys, Python escapes, trailing comma); `true`/`false` in any case. The first result that fits a schema type wins; otherwise the text stays a string, as before. Own code, nothing copied.

Calls stay counted `parsed/ok`: the call is well formed and the value has the schema's type. Counting them `repaired` would put a large share of calls (`20261110`) into the `ToolCallFailures` alert, which is meant for template or model regressions.

Not done: a lazy grammar for `auto` tool choice, as llama.cpp does. It would also stop unknown tool names, but it masks every token after the trigger and needs a decode benchmark on a box (Rule 3). Revisit if the next run still shows validation errors or unknown names above llama-server.

## Consequences

- The boolean calls of the run become schema-valid; the ReportFindings calls get the right type (their item fields are not checked here). The tool-error rate was expected to fall near llama-server's; converted Edits can still fail for task reasons, and the turn-limit sessions lose their retry loops (estimate in `20261110`; the rerun `20261110b` measures it).
- A string parameter that the model fills with `False` stays `"False"`; only the schema decides.
- Union types with `string` (`["string", "number"]`) and `anyOf` schemas still keep text; llama.cpp tries JSON first there. No fixture shows them yet.

## Addendum: rerun 20261110b (box 60050186, A40)

No `input_validation` errors are left; the remaining tool errors are task outcomes (Bash exit codes, Edit and Read misses) and `Explore` called as a tool, parsed correctly (`20261110b`). The proxy now classes the task outcomes `bash_exit` and `file`, and the summary adds `call_error_rate` (input_validation plus unknown_tool per tool call), the part the gateway answers for.

**Bare `<function=` opener.** The one leaked marker (tetraeder-r2-rename-compute-total, first response) was a call without `<tool_call>`: `<function=Bash>...</function>\n</tool_call>`. The parser sent it as text, Claude Code ended the session after one request and the task failed. llama.cpp's grammar triggers on `<function=NAME>` as well. Decision: in the Qwen XML format, `<function=NAME>` starts a call when NAME is one of the request's tools and `<parameter=` or `</function>` follows; it ends at `</function>` (an orphan `</tool_call>` after it is consumed) or at `</tool_call>`. Otherwise only the opener is text and scanning goes on, so prose that mentions `<function=` costs no later call. Outcome `repaired`/`no_opener`, counted in `ToolCallFailures` like `empty_call`. Test: `qwen_xml_call_without_opener_from_run_fixture`, from `toolcall-fixtures-r2.json`.

**Unknown names.** `Explore` on Tetraeder against none on llama-server (`20261110b`) meets this ADR's condition for revisiting a lazy grammar under `auto`. Still not done: it needs a decode benchmark on a box (Rule 3).
