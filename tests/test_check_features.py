"""docs/features.md check: evidence paths exist, ratio numbers appear verbatim in the cited session files, node ids
exist in the tech tree."""

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load():
    spec = importlib.util.spec_from_file_location("check_features", SCRIPTS / "check_features.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


HEAD = "| # | Feature | State and node | Engine compared | Paired ratio | Scope | Evidence |\n| --- | --- | --- | --- | --- | --- | --- |\n"


def tree(tmp_path, row, session_text="builder 0.943 and 1.02 here\n"):
    (tmp_path / "docs" / "roadmap").mkdir(parents=True)
    (tmp_path / "docs" / "roadmap" / "nodes.yaml").write_text("nodes:\n- id: node-a\n", encoding="utf-8")
    s = tmp_path / "bench" / "results" / "s1"
    s.mkdir(parents=True)
    (s / "README.md").write_text(session_text, encoding="utf-8")
    (tmp_path / "docs" / "plan.md").write_text("plan\n", encoding="utf-8")
    (tmp_path / "docs" / "features.md").write_text("intro 9.99\n\n" + HEAD + row + "\n", encoding="utf-8")
    return tmp_path


def run(tmp_path, row, **kw):
    return load().check(tree(tmp_path, row, **kw))


def test_ok(tmp_path):
    row = "| 25 | f | done: `node-a` | llama-server | c4 0.943, 1.02 | scope 0.5 | `bench/results/s1/README.md` |"
    assert run(tmp_path, row) == []


def test_number_missing(tmp_path):
    row = "| 25 | f | done: `node-a` | e | c4 0.944 | s | `bench/results/s1/README.md` |"
    errs = run(tmp_path, row)
    assert len(errs) == 1 and "0.944" in errs[0]


def test_number_is_not_a_prefix_match(tmp_path):
    row = "| 25 | f | done: `node-a` | e | 0.94 | s | `bench/results/s1/README.md` |"
    assert any("0.94" in e for e in run(tmp_path, row))


def test_fraction(tmp_path):
    row = "| 32 | f | done: `node-a` | e | 15/20 | s | `bench/results/s1/README.md` |"
    assert run(tmp_path, row, session_text="solved 15/20 tasks\n") == []
    assert run(tmp_path / "x", row, session_text="solved 15/21\n") != []


def test_missing_evidence_path(tmp_path):
    row = "| 25 | f | done: `node-a` | e | 0.943 | s | `bench/results/nope/README.md` |"
    assert any("nope" in e for e in run(tmp_path, row))


def test_number_only_counts_in_session_files_not_docs(tmp_path):
    row = "| 25 | f | done: `node-a` | e | 7.77 | s | `docs/plan.md`, `bench/results/s1/README.md` |"
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    tree(tmp_path, row)
    (tmp_path / "docs" / "plan.md").write_text("7.77\n", encoding="utf-8")
    assert any("7.77" in e for e in load().check(tmp_path))


def test_unknown_node(tmp_path):
    row = "| 25 | f | done: `node-zzz` | e | 0.943 | s | `bench/results/s1/README.md` |"
    assert any("node-zzz" in e for e in run(tmp_path, row))


def test_not_measured_row_needs_only_existing_evidence(tmp_path):
    row = "| 1 | f | not measured: `node-a` | Strata | not measured | reason | `docs/plan.md` |"
    assert run(tmp_path, row) == []


def test_feature_number_must_be_a_list_of_ints(tmp_path):
    row = "| x | f | done: `node-a` | e | 0.943 | s | `bench/results/s1/README.md` |"
    assert any("feature" in e.lower() for e in run(tmp_path, row))


def test_no_table_is_an_error(tmp_path):
    tree(tmp_path, "")
    (tmp_path / "docs" / "features.md").write_text("nothing\n", encoding="utf-8")
    assert load().check(tmp_path) != []


def test_real_page_is_green():
    assert load().check(ROOT) == []
