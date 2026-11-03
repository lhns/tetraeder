"""Fails unless tests/ (hidden tests excluded) has at least --min test functions whose source matches --pattern."""

import argparse
import ast
import pathlib
import re
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--min", type=int, default=1)
    args = ap.parse_args()
    rx = re.compile(args.pattern)
    found = []
    for path in sorted(pathlib.Path("tests").rglob("test_*.py")):
        if path.name.startswith("test_hidden"):
            continue
        src = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
                if rx.search(ast.get_source_segment(src, node) or ""):
                    found.append(f"{path.as_posix()}::{node.name}")
    print(f"{len(found)} matching tests: {found}")
    if len(found) < args.min:
        print(f"need at least {args.min} test functions matching {args.pattern!r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
