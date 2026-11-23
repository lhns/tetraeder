"""Comparison logic of the llama-server conformance harness (ADR 0083). Pure functions, no servers.

A case sends one request to llama-server (the baseline) and to `tetraeder serve`, and compares the
two answers by HTTP status, JSON shape (paths and value types, never values) and, where the case
asks for it, the generated text or chosen values.
"""

from __future__ import annotations

import json
import re

# Paths dropped from every shape: per-request ids and clocks, and fields whose absence is a matrix
# row of their own (system_fingerprint, timings), so they do not hide every other case behind them.
# A case that wants one of them lists it in `paths`, which overrides this list.
DEFAULT_IGNORE = ("id", "created", "created_at", "system_fingerprint", "timings", "__verbose")

NUMBER = "number"


def type_name(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return NUMBER
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    return "dict"


def shape(value, path: str = "", out: dict[str, set[str]] | None = None) -> dict[str, set[str]]:
    """Map every key path of a JSON value to the set of value types seen there.

    List elements share one path `a[]`; an empty list is `a: {"list", "empty"}` so that `[]` and a
    populated list differ. Numbers are one type (JSON does not tell 1 from 1.0).
    """
    out = {} if out is None else out
    if isinstance(value, dict):
        out.setdefault(path, set()).add("dict")
        for k, v in value.items():
            shape(v, f"{path}.{k}" if path else k, out)
    elif isinstance(value, list):
        out.setdefault(path, set()).add("list")
        if not value:
            out[path].add("empty")
        for v in value:
            shape(v, path + "[]", out)
    else:
        out.setdefault(path, set()).add(type_name(value))
    return out


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith((prefix + ".", prefix + "["))


def filter_shape(sh: dict[str, set[str]], only=(), ignore=DEFAULT_IGNORE) -> dict[str, set[str]]:
    """Keep the paths under `only` (all when empty); otherwise drop those under `ignore`."""
    keep = {}
    for p, t in sh.items():
        if only:
            if not any(_under(p, o) for o in only):
                continue
        elif any(_under(p, i) for i in ignore):
            continue
        keep[p] = t
    return keep


def diff_shapes(base: dict[str, set[str]], other: dict[str, set[str]]) -> list[str]:
    """What a client written against the baseline would trip over, as short lines.

    A path the baseline has must exist in tetraeder's answer with the same types; a baseline path that
    is only ever null says nothing about the type, so it matches any. Paths only tetraeder has (extra
    nulls such as `logprobs`, `refusal`) are not reported: a client of the baseline ignores them.
    """
    lines = []
    for p in sorted(base):
        label = p or "<root>"
        if p not in other:
            lines.append(f"missing in tetraeder: {label} ({'/'.join(sorted(base[p]))})")
        elif base[p] != other[p] and base[p] != {"null"}:
            lines.append(f"type of {label}: baseline {'/'.join(sorted(base[p]))}, tetraeder {'/'.join(sorted(other[p]))}")
    return lines


def get_path(value, path: str):
    """Value at a dotted path with `[i]` indexes (`choices[0].message.content`); None when absent."""
    cur = value
    for part in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if part.startswith("["):
            i = int(part[1:-1])
            if not isinstance(cur, list) or i >= len(cur):
                return None
            cur = cur[i]
        else:
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
    return cur


def norm_text(t):
    """Runs of U+FFFD as one and none at the end: the tiny models write bytes that are not UTF-8, and the servers
    differ in how many replacement characters they put for one broken sequence and in whether a sequence cut by
    the token limit is held back (llama-server) or written as U+FFFD (tetraeder)."""
    return None if t is None else re.sub("�+", "�", t).removesuffix("�")


TEXT_PATHS = ("choices[0].message.content", "choices[0].text", "content")


def generated_text(body) -> str | None:
    """The generated text of an OpenAI chat or completions body, an Anthropic message or a native completion."""
    if not isinstance(body, dict):
        return None
    for p in TEXT_PATHS:
        v = get_path(body, p)
        if isinstance(v, str):
            return v
        if isinstance(v, list):  # Anthropic content blocks
            return "".join(b.get("text", "") for b in v if isinstance(b, dict) and b.get("type") == "text")
    return None


def parse_sse(raw: str) -> list[tuple[str, object]]:
    """Events of a server-sent stream as (event name, data); `[DONE]` stays a string, so does non-JSON data."""
    events = []
    for block in re.split(r"\r?\n\r?\n", raw.strip()):
        name, data = "message", []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
            # ":" comment lines (keep-alives) and other fields carry no event
        if not data:
            continue
        text = "\n".join(data)
        try:
            events.append((name, json.loads(text)))
        except ValueError:
            events.append((name, text))
    return events


def stream_shape(events, only=(), ignore=DEFAULT_IGNORE, drop=()) -> list[tuple[str, dict[str, set[str]]]]:
    """Event sequence shape: (event name, filtered shape of its data), consecutive equal events collapsed;
    events named in `drop` (Anthropic `ping`) are left out."""
    seq: list[tuple[str, dict[str, set[str]]]] = []
    for name, data in events:
        if isinstance(data, str):
            sh = {"": {data if data == "[DONE]" else "str"}}
        else:
            sh = filter_shape(shape(data), only, ignore)
        if name in drop:
            continue
        if not seq or seq[-1] != (name, sh):
            seq.append((name, sh))
    return seq


def diff_sequences(base, other) -> list[str]:
    """Differences of two collapsed event sequences: the first diverging event, then the length."""
    lines = []
    for i, ((bn, bs), (on, os_)) in enumerate(zip(base, other)):
        detail = diff_shapes(bs, os_)
        if bn != on or detail:
            lines.append(f"event {i}: baseline {bn!r}, tetraeder {on!r}" + ("; " + "; ".join(detail) if detail else ""))
            break
    if len(base) != len(other):
        lines.append(f"event count: baseline {len(base)}, tetraeder {len(other)}")
    return lines


def stream_text(events) -> str:
    """Concatenated text deltas of an OpenAI, Anthropic or native stream."""
    out = []
    for _, d in events:
        if not isinstance(d, dict):
            continue
        for p in ("choices[0].delta.content", "choices[0].text", "delta.text", "content"):
            v = get_path(d, p)
            if isinstance(v, str):
                out.append(v)
                break
    return "".join(out)


def compare(case: dict, base, other) -> list[str]:
    """Differences between the baseline answer and tetraeder's, as lines; empty means they conform.

    `base` and `other` have `status`, `body` (parsed JSON, else text) and `events` (parsed SSE of a
    stream case). The case's `mode` picks what is compared beyond the status:
    status (nothing more) | shape | text (shape and generated text) |
    logprobs (shape, then tokens and candidates equal and log-probabilities within `tolerance`) |
    vectors (shape, then the numbers under each of `vector_paths` equal in count and within `tolerance`) |
    stream (event sequence and streamed text) | exact (the whole body).
    `value_paths` adds equal values at those paths to any JSON mode.
    `paths` restricts the shape to those prefixes, `ignore` adds paths to DEFAULT_IGNORE.
    """
    mode = case.get("mode", "text")
    only = tuple(case.get("paths", ()))
    ignore = DEFAULT_IGNORE + tuple(case.get("ignore", ()))
    if base.status != other.status:
        line = f"status: baseline {base.status}, tetraeder {other.status}"
        err = other.body if other.status >= 400 else base.body if base.status >= 400 else None
        return [line + (f" ({_clip(err, 200)})" if err else "")]
    lines: list[str] = []
    if mode == "status":
        return lines
    if mode == "exact":
        if base.body != other.body:
            lines.append(f"body: baseline {_clip(base.body)}, tetraeder {_clip(other.body)}")
        return lines
    if mode == "stream":
        drop = tuple(case.get("drop_events", ()))
        lines += diff_sequences(stream_shape(base.events, only, ignore, drop), stream_shape(other.events, only, ignore, drop))
        bt, ot = norm_text(stream_text(base.events)), norm_text(stream_text(other.events))
        if case.get("compare_text", True) and bt != ot:
            lines.append(f"streamed text: baseline {bt!r}, tetraeder {ot!r}")
        return lines
    structured = (dict, list)
    if isinstance(base.body, structured) or isinstance(other.body, structured):
        lines += diff_shapes(filter_shape(shape(base.body), only, ignore), filter_shape(shape(other.body), only, ignore))
    elif type(base.body) is not type(other.body):
        lines.append(f"body kind: baseline {type(base.body).__name__}, tetraeder {type(other.body).__name__}")
    if mode == "text":
        bt, ot = norm_text(generated_text(base.body)), norm_text(generated_text(other.body))
        if bt != ot:
            lines.append(f"text: baseline {bt!r}, tetraeder {ot!r}")
    if mode == "logprobs":
        lines += logprob_diffs(base, other, case.get("tolerance", LOGPROB_TOLERANCE))
    if mode == "vectors":
        lines += vector_diffs(base, other, case.get("vector_paths", ()), case.get("tolerance", VECTOR_TOLERANCE))
    for p in case.get("value_paths", ()):
        bv, ov = get_path(base.body, p), get_path(other.body, p)
        if bv != ov:
            lines.append(f"value at {p}: baseline {bv!r}, tetraeder {ov!r}")
    return lines


LOGPROB_TOLERANCE = 1e-3
# Embeddings of the same GGUF on both servers differ by float summation order only (different batch shapes, thread
# counts); the vectors are unit length or per-token activations well below 10, so 1e-3 absolute is far above that noise
# and far below a wrong pooling, normalization or token list.
VECTOR_TOLERANCE = 1e-3


def flatten_numbers(v) -> list[float] | None:
    """All numbers of a nested list in order; None when something else is inside."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return [float(v)]
    if isinstance(v, list):
        out: list[float] = []
        for x in v:
            f = flatten_numbers(x)
            if f is None:
                return None
            out += f
        return out
    return None


def vector_diffs(base, other, paths, tol: float = VECTOR_TOLERANCE) -> list[str]:
    """For each path: the same list structure (per-token embeddings are a list of vectors) and every number within `tol`."""
    if not paths:
        return ["the case lists no vector_paths, so it compares nothing"]
    lines = []
    for p in paths:
        bv, ov = get_path(base.body, p), get_path(other.body, p)
        bf = flatten_numbers(bv)
        if not bf:
            lines.append(f"{p}: the baseline has no numbers there ({_clip(bv)}), so the case compares nothing")
            continue
        of = flatten_numbers(ov)
        if of is None:
            lines.append(f"{p}: tetraeder has no numbers there ({_clip(ov)})")
        elif len(bf) != len(of) or shape(bv) != shape(ov):
            lines.append(f"{p}: baseline {len(bf)} numbers {sorted(shape(bv))[:3]}, tetraeder {len(of)} numbers {sorted(shape(ov))[:3]}")
        else:
            worst = max(abs(a - b) for a, b in zip(bf, of))
            if worst > tol:
                lines.append(f"{p}: largest difference {worst:.6f} above the tolerance {tol}")
    return lines


def logprob_diffs(base, other, tol: float = LOGPROB_TOLERANCE) -> list[str]:
    """Differences of `choices[0].logprobs.content` of two answers: the same tokens (id, text, bytes) and the
    same top candidates in the same order, with log-probabilities within `tol` (the two servers add floats in
    a different order, so the values are not bit equal)."""
    b = get_path(base.body, "choices[0].logprobs.content")
    o = get_path(other.body, "choices[0].logprobs.content")
    if not isinstance(b, list) or not b:
        return ["the baseline gave no logprobs, so the case compares nothing"]
    if not isinstance(o, list):
        return ["logprobs: missing in tetraeder"]
    lines = []
    if len(b) != len(o):
        lines.append(f"logprobs entries: baseline {len(b)}, tetraeder {len(o)}")
    for i, (be, oe) in enumerate(zip(b, o)):
        for k in ("id", "token", "bytes"):
            if be.get(k) != oe.get(k):
                lines.append(f"logprobs[{i}].{k}: baseline {be.get(k)!r}, tetraeder {oe.get(k)!r}")
        if abs(be["logprob"] - oe.get("logprob", float("inf"))) > tol:
            lines.append(f"logprobs[{i}].logprob: baseline {be['logprob']}, tetraeder {oe.get('logprob')} (tolerance {tol})")
        bt, ot = be.get("top_logprobs", []), oe.get("top_logprobs", [])
        if [t["id"] for t in bt] != [t["id"] for t in ot]:
            lines.append(f"logprobs[{i}].top_logprobs ids: baseline {[t['id'] for t in bt]}, tetraeder {[t.get('id') for t in ot]}")
            continue
        for j, (bj, oj) in enumerate(zip(bt, ot)):
            if bj["token"] != oj.get("token") or bj["bytes"] != oj.get("bytes"):
                lines.append(f"logprobs[{i}].top_logprobs[{j}] text: baseline {bj['token']!r}, tetraeder {oj.get('token')!r}")
            if abs(bj["logprob"] - oj["logprob"]) > tol:
                lines.append(f"logprobs[{i}].top_logprobs[{j}].logprob: baseline {bj['logprob']}, tetraeder {oj['logprob']} (tolerance {tol})")
    return lines


def _clip(v, n=160) -> str:
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True)
    return s if len(s) <= n else s[:n] + "..."
