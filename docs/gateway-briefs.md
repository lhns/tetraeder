# Implementation briefs for ADR 0051

Two self-contained briefs. Every design decision is in ADR 0051; read it first, then AGENTS.md. The prefix cache gets no change (ADR 0051: no prompt checkpoints). Box runs are on hold: no brief creates a box.

All briefs: failing tests first, then code. Every local cargo command goes through the shared lock with one job: `L=/c/Users/pierr/AppData/Local/Temp/claude/cargo-host.lock; until mkdir "$L" 2>/dev/null; do sleep 20; done; CARGO_BUILD_JOBS=1 cargo <args>; rmdir "$L"`. Done means `cargo build --workspace`, `cargo test --workspace`, `cargo clippy --workspace -- -D warnings`, `cargo fmt --check` pass (Rule 11; use CI on the `track-*` branch when the lock starves local builds), `python -m pytest tests/test_alert_rules.py tests/test_claude_code_proxy.py` passes, and ADR 0051 gets an "Implementation notes" section with what was built and any deviation. Commit only if your brief says so.

G1 and G2 share no files and can run in parallel.

---

## Brief G1: tool-call outcomes and the empty-call repair (gateway)

**Goal.** The output parser counts every tool call by outcome and class, drops empty call openers instead of emitting them as text, and the gateway records the counts as a metric with a dashboard panel and an alert (Rule 6).

**Read first.** ADR 0051; ADR 0028 (parser); `crates/gateway/src/toolcall.rs` (all of it: `drain_block`, `step_tagged`, `step_mistral`, `step_llama`, `Step`); `crates/gateway/src/gateway/pump.rs` (`segments`, the `Finished` arm); `crates/gateway/src/gateway.rs` `GatewayMetrics` (line 178); `crates/gateway/tests/toolcall.rs` (helpers `Case`, `random_chunks`, `norm`); `crates/gateway/tests/gateway.rs` `gen_ai_metrics_are_recorded` (in-memory exporter pattern); `deploy/alerts/gateway.yml`; the "Prefix cache (host RAM)" panel in `deploy/grafana/dashboards/overview.json` (pattern for a new panel).

**Decisions already made (do not revisit).**
- `toolcall.rs`: `pub struct CallOutcome { pub outcome: &'static str, pub class: &'static str }` and `OutputParser::take_outcomes(&mut self) -> Vec<CallOutcome>` (drains). One entry per call the parser decided on:
  - `parsed`/`ok`; `parsed`/`unknown_tool` when the name is not in the parser's `tools` (the call still goes out unchanged).
  - `repaired`/`empty_call`: the block's body is empty or whitespace. Tagged formats: the text between `<tool_call>` and `</tool_call>` or the end of output trims to empty. Mistral: nothing but whitespace after `[TOOL_CALLS]` at the end of output. Llama 3: the block trims to empty or to `<|python_tag|>` alone at the end of output. The block is consumed and nothing is emitted (no text, no call).
  - `failed`/`unclosed`: output ends inside the call (`step_tagged` without `</tool_call>` at finish; Mistral or Llama 3 JSON still partial at finish). `failed`/`bad_xml`: QwenXml content starting with `<function=` that `parse_xml_call` rejects. `failed`/`bad_json`: every other parse failure. Failed calls keep today's behaviour: their raw text is emitted as text.
- Carry the class in `Step::Fail` (e.g. `Step::Fail(usize, &'static str)`) and add `Step::Drop(usize)` for the repair; `drain_block` records the outcome. Do not restructure the state machine.
- Chunking must not change outcomes: same outcomes for any split of the input.
- Gateway: `GatewayMetrics` gains `tool_calls: Counter<u64>`, name `tetraeder.gateway.tool_calls`, unit `{call}`, attributes `tetraeder.outcome` and `tetraeder.class` (string literals in the gateway crate, next to the counter). `Run` in `pump.rs` records `take_outcomes()` after every `p.push` and after `p.finish()` (a small helper on `Run`).
- Finish reason: unchanged logic; after a dropped empty call with no other call the finish is `Stop` (`end_turn`), and the response no longer carries the marker.
- Panel "Tool calls by outcome" in the gateway rows of `overview.json`: `sum by (tetraeder_outcome, tetraeder_class) (rate(tetraeder_gateway_tool_calls_total[$__rate_interval]))`, unit ops, with a description naming ADR 0051. Place it next to "Prefix cache (host RAM)"; do not move other panels.
- Alert in `deploy/alerts/gateway.yml`, group `gateway`:
  ```yaml
  # Tool calls the parser had to repair or could not parse (ADR 0051): a template or model regression.
  - alert: ToolCallFailures
    expr: |
      sum(rate(tetraeder_gateway_tool_calls_total{tetraeder_outcome=~"failed|repaired"}[30m]))
        / sum(rate(tetraeder_gateway_tool_calls_total[30m])) > 0.02
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "More than 2% of tool calls fail to parse or need repair"
  ```
  `tests/test_alert_rules.py` must accept it (it checks that the series is emitted by the code; the counter name string must appear in a crate's `src`).

**Tests first** (`crates/gateway/tests/toolcall.rs` unless noted; each also through `random_chunks` like the existing cases):
1. `qwen_xml_trailing_empty_opener_is_dropped`, the fixture from the run (tetraeder-r1-write-shipping-tests, last response): QwenXml with one tool, input `"The test suite is now ready to ensure the shipping module works correctly and to catch any future changes that might break existing functionality.\n<tool_call>"` gives exactly one text segment equal to the input without `<tool_call>`, and outcomes `[repaired/empty_call]`.
2. `empty_closed_call_is_dropped`: Hermes `"<tool_call>\n</tool_call>"` and QwenXml `"<tool_call> </tool_call>after"` drop the call (the second keeps `after` as text).
3. `mistral_and_llama3_empty_calls_are_dropped`: `"[TOOL_CALLS]  "` and `"<|python_tag|>"` at end of output.
4. `outcomes_by_class`: one case each for `parsed/ok`, `parsed/unknown_tool`, `failed/unclosed` (QwenXml `<tool_call>\n<function=read>\n<parameter=path>\na` then end), `failed/bad_json` (Hermes `<tool_call>{"name": </tool_call>`), `failed/bad_xml` (QwenXml `<tool_call><function=read><parameter=path>x</tool_call>`); the failed ones still emit their raw text, as the existing tests expect.
5. `crates/gateway/tests/gateway.rs`: `tool_call_outcomes_are_recorded`, in-memory exporter like `gen_ai_metrics_are_recorded`, a fake engine whose output is a complete call (see `FakeEngine` in `src/fake.rs` for a fixed output and a Hermes or Qwen template fixture), asserts one point of `tetraeder.gateway.tool_calls` with `tetraeder.outcome=parsed`, `tetraeder.class=ok`.
6. Existing toolcall tests stay green unchanged, except any that asserted a bare empty opener comes out as text: change those to the new behaviour and say so in the notes.

**Done.** The Rule 11 set and the pytest files above pass; ADR 0051 "Implementation notes" lists the outcome classes as built.

**Out of scope.** Any other repair (ADR 0051 lists the classes not designed); the proxy (G2); the prefix cache.

---

## Brief G2: tool-call evidence in the proxy run, both engines (scripts)

**Goal.** The Claude Code proxy run reports tool-call failures per engine without storing bodies, and a script turns a kept request log into small committable fixtures, so the rerun (docs/m1-open.md item 3) decides whether any further repair class is worth building.

**Read first.** ADR 0051; `scripts/anthropic_proxy.py` (`Observer`), `scripts/claude_code_proxy_summary.py` (`engine_stats`, `summarize`, `format_table`), `scripts/claude_code_proxy.py` (where the request log is written, line 392), `tests/test_claude_code_proxy.py` (summary and proxy tests), `crates/gateway/src/reqlog.rs` (request log fields), `scripts/claude_code_cache_classes.py` (the cache-class report of ADR 0051).

**Decisions already made (do not revisit).**
- `Observer.__init__` parses the request once and keeps: `tool_names` (names in `tools`), and per class the count of `tool_result` blocks with `is_error: true` in the last user message. Class from the result text (string content, or the text blocks of list content): `input_validation` if it contains `InputValidationError`, `unknown_tool` if it contains `No such tool available`, else `other`. Check these two strings against the first rerun's request log and adjust the patterns if Claude Code words them differently; keep the `other` bucket.
- From the response (SSE and JSON): `tool_uses` (count of `tool_use` content blocks), `tool_names_unknown` (tool_use names not in `tool_names`), `leaked_markers` (text deltas or text blocks containing `<tool_call>`, `<function=` or `<parameter=`; count blocks, across delta boundaries by keeping the last 16 characters of the previous text delta of the block).
- New log fields: `tool_uses`, `tool_errors` (object class to count, empty object when none), `tool_names_unknown`, `leaked_markers`. Old logs lack them; the summary treats missing as 0.
- Summary per engine: `tool_uses_total`, `tool_errors_by_class`, `tool_names_unknown_total`, `leaked_markers_total`, and `tool_error_rate` (sum of tool errors / tool uses, null when no tool uses). Keep `protocol_errors` as it is. `format_table` gains one column `terr` (tool errors).
- `scripts/claude_code_proxy.sh` also runs `scripts/claude_code_cache_classes.py "$out" > "$out/cache_classes.txt"` after the summary.
- New `scripts/toolcall_fixtures.py <reqlog.jsonl> <out.json>`: for every logged response whose text contains one of the markers above, or whose tool_use names a tool not in the request's tools, writes `{generation_id, stop_reason, tools: [names], output}` where `output` is only the assistant output of that response (text and tool_use blocks, from the SSE text or JSON body). Never writes request messages or system prompts. The rerun commits the fixtures file, not the request log.
- Python stdlib only, as the existing scripts.

**Tests first** (`tests/test_claude_code_proxy.py`, following the existing proxy and summary tests):
1. An Observer fed a request whose last user turn has two `tool_result` blocks with `is_error: true` (one `<tool_use_error>InputValidationError: ...`, one other) and a streamed response with one `tool_use` of an unknown name and a text delta split as `"done <tool"` + `"_call>"` records `tool_errors == {"input_validation": 1, "other": 1}`, `tool_uses == 1`, `tool_names_unknown == 1`, `leaked_markers == 1`.
2. The same for a non-streamed JSON response.
3. `engine_stats` on rows with and without the new fields gives the totals and `tool_error_rate`; rows without them count 0.
4. `toolcall_fixtures.py` on a two-line synthetic request log keeps the one response with a marker and writes no field from the request except tool names.

**Done.** `python -m pytest tests/test_claude_code_proxy.py tests/test_gateway_scripts.py` passes; `bash -n scripts/claude_code_proxy.sh`; running the summary on bench/results/20261103/claude-code-proxy still works (old logs) and prints `terr` 0. Note the rerun steps in docs/m1-open.md item 3: fetch `*.reqlog.jsonl` with the results but do not commit them; run `toolcall_fixtures.py` on each and commit the fixtures; read `cache_classes.txt` against ADR 0051's table.

**Out of scope.** Gateway code (G1); deciding a new repair class (that follows the rerun's numbers, ADR 0051's rule).
