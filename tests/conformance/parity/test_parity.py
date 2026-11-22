"""llama-server conformance (ADR 0083, docs/notes/llamacpp-parity.md).

Each case in cases.yaml goes to llama-server (the baseline, built from the pinned and patched
vendor tree, CPU only) and to `tetraeder serve` on the same tiny GGUF. A case with `expect: pass`
must conform; `expect: gap <tree node>` is xfail(strict): a gap that starts passing fails the run,
and the case is flipped to `pass` in the same change.

Needs the llama-server binary (LLAMA_SERVER_BIN or target/llama-server/bin) and the tetraeder
binary (TETRAEDER_BIN or target/debug); without them the tests skip, or fail when
TETRAEDER_REQUIRE_PARITY=1 (CI). Run locally: see docs/notes/llamacpp-parity.md.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from dataclasses import dataclass

import httpx
import pytest
import yaml
from parity_lib import compare, parse_sse
from servers import FAMILY, ServerStartError, find_llama_server, find_tetraeder, start_llama, start_tetraeder

HERE = pathlib.Path(__file__).resolve().parent
CASES = yaml.safe_load((HERE / "cases.yaml").read_text(encoding="utf-8"))["cases"]
REQUIRE = os.environ.get("TETRAEDER_REQUIRE_PARITY") == "1"
TIMEOUT = 120.0


@dataclass
class Answer:
    status: int
    body: object
    events: list


def missing(what: str, hint: str):
    msg = f"skip: no {what} ({hint})"
    if REQUIRE:
        pytest.fail(msg.replace("skip:", "required but missing:"))
    pytest.skip(msg)


@pytest.fixture(scope="session")
def llama_bin():
    p = find_llama_server()
    if p is None:
        missing("llama-server binary", "set LLAMA_SERVER_BIN or build it into target/llama-server, see docs/notes/llamacpp-parity.md")
    return p


@pytest.fixture(scope="session")
def tetraeder_bin():
    p = find_tetraeder()
    if p is None:
        missing("tetraeder binary", "run `cargo build -p tetraeder` or set TETRAEDER_BIN")
    return p


class Pool:
    """Servers started on first use and kept for the session, keyed by (kind, key, extra args)."""

    def __init__(self, llama_bin, tetraeder_bin, log_dir):
        self.llama_bin, self.tetraeder_bin, self.log_dir = llama_bin, tetraeder_bin, log_dir
        self.running = {}

    def get(self, kind, key, extra=()):
        k = (kind, key, tuple(extra))
        if k not in self.running:
            tag = f"-{len(self.running)}"
            if kind == "llama":
                self.running[k] = start_llama(self.llama_bin, key, self.log_dir, extra, tag)
            else:
                self.running[k] = start_tetraeder(self.tetraeder_bin, key, self.log_dir, extra, tag)
        return self.running[k]

    def close(self):
        for s in self.running.values():
            s.stop()


@pytest.fixture(scope="session")
def pool(llama_bin, tetraeder_bin, tmp_path_factory):
    p = Pool(llama_bin, tetraeder_bin, tmp_path_factory.mktemp("parity"))
    yield p
    p.close()


def send(base_url: str, case: dict, body: dict | None = None) -> Answer:
    method, path = case["endpoint"].split(" ", 1)
    headers = dict(case.get("headers", {}))
    body = case.get("body") if body is None else body
    # tetraeder requires `model` (a row of its own, case model-optional); every other case names it
    if isinstance(body, dict) and body.keys() & {"messages", "prompt", "input"} and not case.get("no_model"):
        body = {"model": "tiny", **body}
    kw = {"json": body} if body is not None else {}
    with httpx.Client(timeout=TIMEOUT) as c:
        if case["mode"] == "stream":
            with c.stream(method, base_url + path, headers=headers, **kw) as r:
                raw = r.read().decode("utf-8", errors="replace")
                if r.status_code != 200:
                    return Answer(r.status_code, _json_or_text(raw), [])
                return Answer(r.status_code, None, parse_sse(raw))
        r = c.request(method, base_url + path, headers=headers, **kw)
        return Answer(r.status_code, _json_or_text(r.text), [])


def _json_or_text(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


def probed_body(pool, case) -> dict:
    """The request body with a field derived from the baseline (the tiny model's text and token ids are not
    known in advance): `stop_ascii` stops on printable ASCII characters of what the baseline generates for the body; `ban_first_token`
    biases the baseline's first token for `prompt` down."""
    probe = case["probe"]
    base = pool.get("llama", FAMILY[case["engines"][0]]).base_url
    if probe["kind"] == "stop_ascii":
        r = send(base, case, case["body"]).body["choices"][0]
        out = r.get("message", {}).get("content") or r.get("text") or ""
        n = probe["length"]
        # after the first character, so that text is generated before the stop
        piece = next((out[i : i + n] for i in range(1, len(out) - n + 1) if re.fullmatch(r"[!-~]{%d}" % n, out[i : i + n])), None)
        if piece is None:
            pytest.skip(f"baseline text {out!r} has no {n} printable ASCII characters in a row")
        return {**case["body"], "stop": [piece]}
    if probe["kind"] == "ban_first_token":
        native = {"endpoint": "POST /completion", "mode": "shape"}
        r = send(base, native, {"prompt": case["body"]["prompt"], "n_predict": 1, "temperature": 0, "return_tokens": True})
        return {**case["body"], "logit_bias": {str(r.body["tokens"][0]): -100}}
    raise ValueError(probe)


def params() -> list:
    out = []
    for case in CASES:
        for engine in case["engines"]:
            marks = []
            if case["expect"].startswith("gap "):
                node = case["expect"][4:]
                marks.append(pytest.mark.xfail(strict=True, reason=f"gap, tree node {node}", raises=(AssertionError, ServerStartError)))
            out.append(pytest.param(case, engine, id=f"{case['id']}[{engine}]", marks=marks))
    return out


@pytest.mark.parametrize("case,engine", params())
def test_case(case, engine, pool):
    base_srv = pool.get("llama", FAMILY[engine], case.get("llama_args", ()))
    body = probed_body(pool, case) if "probe" in case else None
    try:
        other_srv = pool.get("tetraeder", engine, case.get("tetraeder_args", ()))
    except ServerStartError as e:
        raise AssertionError(f"tetraeder did not start with {case.get('tetraeder_args')}: {str(e)[:300]}") from e
    base = send(base_srv.base_url, case, body)
    other = send(other_srv.base_url, case, body)
    diffs = compare(case, base, other)
    assert not diffs, f"{case['row']} ({case['endpoint']}):\n  " + "\n  ".join(diffs)
