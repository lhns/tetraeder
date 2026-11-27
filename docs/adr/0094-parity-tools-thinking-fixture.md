# 0094. A parity fixture with tool calls and thinking (node serving-parity-tool-template)

Status: Accepted (advisor, delegated by the owner 2026-11-27). Follows ADR 0093, whose open items needed a template that renders tools and `<think>`.

## Context

The parity suite gives both servers `tests/fixtures/chatml.jinja`, which renders no tools and no reasoning. Forced `tool_choice`, `response_format` with tools or a thinking template, the continuation on a thinking template and the reasoning controls could not be compared.

## Decision

1. `tests/fixtures/qwen3-tools.jinja`, written for this repo (not copied; no `docs/sources.md` row), uses Qwen3's markers: tools in `<tools>`, calls as `<tool_call>{"name", "arguments"}</tool_call>`, results in `<tool_response>`, reasoning of the turns after the last user message in `<think>\n..\n</think>\n\n`, `enable_thinking: false` closes an empty think block. Its tool preamble is minimal: the tiny models have 512 context and one token per byte. llama-server's analysis reads it as `JSON_NATIVE` tools and `TAG_BASED` reasoning.
2. A case selects it with `--chat-template-file {fixtures}/qwen3-tools.jinja` in `llama_args` and `tetraeder_args`; `servers.py` then gives no default template. Cases run on `auto` only.
3. Two gateway fixes the new cases found: tools reach the template with `description` always present (empty when absent), as llama-server's `common_chat_tools_to_json_oaicompat` gives them; the continuation of a message with reasoning on a thinking template is cut from the template's own rendering of the message (its think markers) instead of literal `<think>` and `</think>`. Without reasoning the literal construction stays, now with Qwen3's markers (`<think>\n`, `\n</think>\n\n`) as llama-server finds them.
4. Differences left as strict gap cases (node `serving-parity-tool-template` until a follow-up node takes them):
   - Forced `tool_choice` (required, named): on a thinking template llama-server's parser is `reasoning? content? tool-calls`, so free text may precede the call and noise text stays content; the gateway's mask starts the call at once. Kept: the mask is what forces calls on real templates. The chatml case `gap-tool-choice-required` stays too (there llama-server builds no grammar at all); both differences are llama-server's.
   - `response_format` with tools or on a thinking template: llama-server's grammar has the think block (and the tool call) in front and its first token is the code fence; the gateway keeps its token mask, which ends the reply sooner.
   - `reasoning_format` (llama-server `none` leaves the reasoning in `content` with its tag) and `reasoning_budget_tokens` (forces the end of the think block after N tokens) are ignored.
   - `POST /v1/chat/completions/control` is 404 (llama-server: 400 "missing completion id"; it steers a running reasoning budget).

## Consequences

- The thinking continuation and tools rendering are compared, not only pinned by a fake-engine test.
- The gap cases fail for their own reasons and flip to `pass` when a follow-up closes them.
