"""Unit tests of the harness's own comparison logic and of the case file (run everywhere, no servers)."""

import pathlib
import re
from types import SimpleNamespace

import pytest
import yaml
from parity_lib import (
    DEFAULT_IGNORE,
    compare,
    diff_sequences,
    diff_shapes,
    filter_shape,
    generated_text,
    get_path,
    parse_sse,
    shape,
    stream_shape,
    stream_text,
)

ROOT = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent


def ans(status=200, body=None, events=None):
    return SimpleNamespace(status=status, body=body, events=events or [])


def test_shape_records_paths_and_types_not_values():
    a = shape({"id": "x", "n": 1, "f": 2.5, "ok": True, "none": None, "l": [{"k": "v"}, {"k": "w"}], "e": []})
    assert a == {
        "": {"dict"},
        "id": {"str"},
        "n": {"number"},
        "f": {"number"},
        "ok": {"bool"},
        "none": {"null"},
        "l": {"list"},
        "l[].k": {"str"},
        "l[]": {"dict"},
        "e": {"list", "empty"},
    }
    assert shape({"a": 1}) == shape({"a": 99})


def test_shape_unions_list_element_types():
    assert shape({"l": [1, "a", None]})["l[]"] == {"number", "str", "null"}


def test_diff_names_missing_extra_and_type_changes():
    base, other = shape({"a": 1, "b": "x", "c": [1]}), shape({"a": "1", "d": 1, "c": []})
    lines = diff_shapes(base, other)
    assert "type of a: baseline number, tetraeder str" in lines
    assert "missing in tetraeder: b (str)" in lines
    assert not any("d" == line.split(": ")[1].split(" ")[0] for line in lines)  # extra paths are not reported
    assert any(line.startswith("missing in tetraeder: c[]") for line in lines)
    assert any(line.startswith("type of c:") for line in lines)
    assert diff_shapes(base, base) == []


def test_null_only_baseline_path_matches_any_type_but_must_exist():
    base = shape({"a": None, "b": None})
    assert diff_shapes(base, shape({"a": "x", "b": 1})) == []
    assert diff_shapes(base, shape({"a": "x"})) == ["missing in tetraeder: b (null)"]
    assert diff_shapes(shape({"a": "x"}), shape({"a": None})) == ["type of a: baseline str, tetraeder null"]


def test_replacement_character_runs_are_one():
    base = ans(200, {"choices": [{"text": "a��"}]})
    other = ans(200, {"choices": [{"text": "a�"}]})
    assert compare({"mode": "text"}, base, other) == []
    assert compare({"mode": "text"}, base, ans(200, {"choices": [{"text": "b�"}]})) != []
    cut = ans(200, {"choices": [{"text": "a"}]})  # a sequence cut by the token limit, held back by one server
    assert compare({"mode": "text"}, cut, other) == []


def test_default_ignore_drops_ids_clocks_and_timings():
    sh = shape({"id": "a", "created": 1, "timings": {"prompt_n": 3}, "usage": {"id": 2}, "choices": [{"index": 0}]})
    kept = filter_shape(sh)
    assert "id" not in kept and "created" not in kept and not any(p.startswith("timings") for p in kept)
    assert "usage.id" in kept  # only the top-level path is ignored, not every key of that name
    assert "choices[].index" in kept
    assert "timings" in DEFAULT_IGNORE


def test_only_overrides_ignore_and_restricts():
    sh = shape({"timings": {"prompt_n": 3}, "usage": {"total_tokens": 4}})
    assert set(filter_shape(sh, only=("timings",))) == {"timings", "timings.prompt_n"}
    assert set(filter_shape(sh, only=("usage.total_tokens",))) == {"usage.total_tokens"}


def test_prefix_match_is_on_path_boundaries():
    sh = shape({"idx": 1, "id": "x"})
    assert set(filter_shape(sh, ignore=("id",))) == {"", "idx"}


def test_get_path_and_generated_text():
    body = {"choices": [{"message": {"content": "hi"}}]}
    assert get_path(body, "choices[0].message.content") == "hi"
    assert get_path(body, "choices[1]") is None and get_path(body, "x.y") is None
    assert generated_text(body) == "hi"
    assert generated_text({"choices": [{"text": "t"}]}) == "t"
    assert generated_text({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}) == "ab"
    assert generated_text({"content": "native"}) == "native"
    assert generated_text("not json") is None


def test_status_mismatch_stops_the_comparison():
    assert compare({"mode": "text"}, ans(200, {"a": 1}), ans(404, {"error": "x"})) == [
        "status: baseline 200, tetraeder 404 ({\"error\": \"x\"})"
    ]
    assert compare({"mode": "status"}, ans(200, {"a": 1}), ans(200, {"b": 1})) == []


def test_text_mode_compares_shape_then_text():
    base = ans(200, {"choices": [{"message": {"content": "a"}}], "usage": {"prompt_tokens": 3}})
    same = ans(200, {"choices": [{"message": {"content": "a"}}], "usage": {"prompt_tokens": 9}})
    diff = ans(200, {"choices": [{"message": {"content": "b"}}], "usage": {"prompt_tokens": 3}})
    assert compare({"mode": "text"}, base, same) == []
    assert compare({"mode": "text"}, base, diff) == ["text: baseline 'a', tetraeder 'b'"]
    assert compare({"mode": "shape"}, base, diff) == []


def test_paths_restrict_the_shape():
    base = ans(200, {"a": 1, "extra": {"x": 1}})
    other = ans(200, {"a": 1})
    assert compare({"mode": "shape"}, base, other) != []
    assert compare({"mode": "shape", "paths": ["a"]}, base, other) == []
    assert compare({"mode": "shape", "ignore": ["extra"]}, base, other) == []


def test_value_paths_compare_listed_paths_only():
    base, other = ans(200, {"u": {"n": 3, "m": 1}}), ans(200, {"u": {"n": 4, "m": 2}})
    assert compare({"mode": "shape", "value_paths": ["u.n"]}, base, other) == [
        "value at u.n: baseline 3, tetraeder 4"
    ]


def test_exact_mode_and_text_bodies():
    assert compare({"mode": "exact"}, ans(200, "ok"), ans(200, "ok")) == []
    assert compare({"mode": "exact"}, ans(200, "ok"), ans(200, "no"))[0].startswith("body:")
    assert compare({"mode": "shape"}, ans(200, {"a": 1}), ans(200, "<html>"))  # json against text differs


RAW = (
    'data: {"id":"1","choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
    'data: {"id":"1","choices":[{"delta":{"content":"he"}}]}\n\n'
    'data: {"id":"1","choices":[{"delta":{"content":"llo"}}]}\n\n'
    ": keep-alive\n\n"
    'data: {"id":"1","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def test_parse_sse_and_text():
    ev = parse_sse(RAW)
    assert [d if isinstance(d, str) else "json" for _, d in ev] == ["json"] * 4 + ["[DONE]"]
    assert stream_text(ev) == "hello"


def test_parse_sse_named_events():
    ev = parse_sse('event: message_start\ndata: {"type":"message_start"}\n\nevent: ping\ndata: {"type":"ping"}\n\n')
    assert [n for n, _ in ev] == ["message_start", "ping"]


def test_stream_shape_collapses_repeated_deltas():
    ev = parse_sse(RAW)
    seq = stream_shape(ev)
    # role chunk, content delta (twice, collapsed), finish chunk, DONE
    assert [n for n, _ in seq] == ["message"] * 4
    assert len(seq) == 4


def test_stream_compare_flags_missing_final_chunk_and_text():
    base = ans(200, None, parse_sse(RAW))
    short = ans(200, None, parse_sse(RAW.replace("data: [DONE]\n\n", "")))
    assert compare({"mode": "stream"}, base, base) == []
    lines = compare({"mode": "stream"}, base, short)
    assert any(line.startswith("event count") for line in lines)
    other_text = ans(200, None, parse_sse(RAW.replace('"llo"', '"xx"')))
    assert compare({"mode": "stream"}, base, other_text) == ["streamed text: baseline 'hello', tetraeder 'hexx'"]
    assert compare({"mode": "stream", "compare_text": False}, base, other_text) == []


SSE_PING = (
    'event: a\ndata: {"x":1}\n\n'
    'event: ping\ndata: {"type":"ping"}\n\n'
    'event: b\ndata: {"y":1}\n\n'
)


def test_dropped_events_are_left_out_of_the_sequence():
    ev = parse_sse(SSE_PING)
    assert [n for n, _ in stream_shape(ev, drop=("ping",))] == ["a", "b"]
    only_a = ans(200, None, ev[:1] + ev[2:])
    assert compare({"mode": "stream", "drop_events": ["ping"]}, ans(200, None, ev), only_a) == []
    assert compare({"mode": "stream"}, ans(200, None, ev), only_a) != []


def test_diff_sequences_reports_first_divergence():
    a = [("m", {"x": {"str"}})]
    b = [("m", {"y": {"str"}})]
    assert "event 0" in diff_sequences(a, b)[0]
    assert diff_sequences(a, a) == []


def lp_body(entries):
    return {"choices": [{"logprobs": {"content": entries}}]}


def lp_entry(tok, logprob, top):
    return {
        "id": ord(tok),
        "token": tok,
        "bytes": [ord(tok)],
        "logprob": logprob,
        "top_logprobs": [{"id": ord(t), "token": t, "bytes": [ord(t)], "logprob": lp} for t, lp in top],
    }


def test_logprobs_mode_allows_float_noise_but_not_other_tokens():
    case = {"mode": "logprobs"}
    base = ans(200, lp_body([lp_entry("a", -0.5, [("a", -0.5), ("b", -1.5)])]))
    near = ans(200, lp_body([lp_entry("a", -0.5004, [("a", -0.5004), ("b", -1.4996)])]))
    assert compare(case, base, near) == []
    far = ans(200, lp_body([lp_entry("a", -0.6, [("a", -0.6), ("b", -1.5)])]))
    assert "logprob: baseline -0.5, tetraeder -0.6" in compare(case, base, far)[0]
    other_top = ans(200, lp_body([lp_entry("a", -0.5, [("a", -0.5), ("c", -1.5)])]))
    assert "top_logprobs ids" in compare(case, base, other_top)[0]
    other_tok = ans(200, lp_body([lp_entry("z", -0.5, [("a", -0.5), ("b", -1.5)])]))
    assert any(".id:" in d or ".token:" in d for d in compare(case, base, other_tok))
    assert compare({"mode": "logprobs", "tolerance": 0.2}, base, far) == []


def test_logprobs_mode_needs_entries_and_equal_length():
    case = {"mode": "logprobs"}
    e = lp_entry("a", -0.5, [("a", -0.5)])
    assert any("no logprobs" in d for d in compare(case, ans(200, lp_body([])), ans(200, lp_body([e]))))
    assert "entries: baseline 1, tetraeder 2" in compare(case, ans(200, lp_body([e])), ans(200, lp_body([e, e])))[0]
    gone = ans(200, {"choices": [{"logprobs": None}]})
    assert any("logprobs: missing in tetraeder" in d for d in compare(case, ans(200, lp_body([e])), gone))


def test_vectors_mode_allows_float_noise_but_not_other_numbers_or_sizes():
    case = {"mode": "vectors", "vector_paths": ["data[0].embedding"]}
    mk = lambda v: ans(200, {"data": [{"embedding": v}]})  # noqa: E731
    assert compare(case, mk([0.1, 0.2]), mk([0.1004, 0.1996])) == []
    assert "largest difference" in compare(case, mk([0.1, 0.2]), mk([0.1, 0.25]))[0]
    assert "2 numbers" in compare(case, mk([0.1, 0.2]), mk([0.1]))[0]
    # a list of per-token vectors is not the same as one flat vector with as many numbers
    assert compare(case, mk([[0.1], [0.2]]), mk([0.1, 0.2]))
    assert compare(case, mk([[0.1], [0.2]]), mk([[0.1], [0.2]])) == []
    assert "no numbers" in compare(case, mk([]), mk([]))[0]
    assert any("no numbers" in d for d in compare(case, mk([0.1]), mk("x")))
    assert compare({"mode": "vectors"}, mk([0.1]), mk([0.1]))[0].startswith("the case lists no vector_paths")
    assert compare({**case, "tolerance": 0.1}, mk([0.1, 0.2]), mk([0.1, 0.25])) == []


# The case file: every row is an exact appendix row, every gap names a tree node.

CASES = yaml.safe_load((HERE / "cases.yaml").read_text(encoding="utf-8"))["cases"]
MATRIX = (ROOT / "docs" / "notes" / "llamacpp-parity.md").read_text(encoding="utf-8").split("## Appendix", 1)[1]
ROWS = {m.group(1).strip() for m in re.finditer(r"^\| ([^|]+?) \|", MATRIX, re.M)}
NODES = set(re.findall(r"^- id: (\S+)", (ROOT / "docs" / "roadmap" / "nodes.yaml").read_text(encoding="utf-8"), re.M))


def test_case_ids_are_unique():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_case_fields(case):
    assert case["group"] in "ABCD"
    assert case["row"] in ROWS, f"row {case['row']!r} is not a first cell of the appendix"
    assert re.fullmatch(r"(GET|POST) /\S*", case["endpoint"])
    assert case["mode"] in {"status", "shape", "text", "logprobs", "stream", "exact", "vectors"}
    assert set(case["engines"]) <= {"auto", "dense", "moe", "llama-moe"} and case["engines"]
    if case["mode"] == "vectors":
        assert case["vector_paths"]
    for step in case.get("setup", ()):
        assert re.fullmatch(r"(GET|POST) /\S*", step["endpoint"])
    if "tetraeder_refuses" in case:
        # a refusal case starts no baseline and compares no answer; it names the text the refusal must carry
        assert case["tetraeder_refuses"] and case.get("tetraeder_args")
    exp = case["expect"]
    assert exp == "pass" or (exp.startswith("gap ") and exp[4:] in NODES), exp


# The converse (ADR 0083 decision 6): every appendix row has a case, a scope-out, or a unit-test entry.

NOTE = (ROOT / "docs" / "notes" / "llamacpp-parity.md").read_text(encoding="utf-8")


def table_rows(text: str) -> list[list[str]]:
    """Rows of the markdown tables in `text` as cell lists, without header and separator lines."""
    rows, lines = [], text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("| "):
            continue
        if i + 1 < len(lines) and re.match(r"^\|[-| :]+\|$", lines[i + 1]):
            continue  # header line
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip()[1:-1])]
        if all(set(c) <= set("-: ") for c in cells):
            continue  # separator
        rows.append(cells)
    return rows


def section(title: str) -> str:
    m = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## )", NOTE, re.M | re.S)
    assert m, f"section {title!r} missing in the matrix"
    return m.group(1)


APPENDIX_ROWS = [r[0] for r in table_rows(MATRIX)]
SCOPED_OUT = {r[0]: r for r in table_rows(section("Scoped out (ADR 0083)"))}
UNIT_TESTED = {r[0]: r for r in table_rows(section("Covered by unit tests"))}
CASE_ROWS = {c["row"] for c in CASES}


def test_every_appendix_row_has_a_case_a_scope_out_or_a_unit_test():
    open_rows = [r for r in APPENDIX_ROWS if r not in CASE_ROWS and r not in SCOPED_OUT and r not in UNIT_TESTED]
    assert not open_rows, f"{len(open_rows)} rows without case, scope-out or unit test: {open_rows}"


def test_scope_outs_and_unit_tests_name_appendix_rows_and_exist():
    rows = set(APPENDIX_ROWS)
    for name in {**SCOPED_OUT, **UNIT_TESTED}:
        if name in ("Speed", "GCP compatibility routes"):
            continue
        assert name in rows, f"{name!r} is not a first cell of the appendix"
    for name, cells in UNIT_TESTED.items():
        assert len(cells) >= 2 and cells[1], name
        paths = [p for p in re.findall(r"`([^`]+)`", cells[1]) if "/" in p]
        assert paths, f"{name!r} names no test file"
        for path in paths:
            assert (ROOT / path.split("::")[0]).exists(), f"{name!r}: {path} does not exist"
    for name, cells in SCOPED_OUT.items():
        assert len(cells) >= 2 and cells[1], f"{name!r} has no reason"
