"""At least --min new test functions that exercise --module, and the module itself unchanged."""

import argparse
import ast
import pathlib
import subprocess
import sys


def count_tests(sources: dict[str, str]) -> int:
    n = 0
    for src in sources.values():
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--min", type=int, default=3)
    ap.add_argument("--unchanged", required=True)
    args = ap.parse_args()
    short = args.module.rsplit(".", 1)[-1]
    now = {
        p.as_posix(): p.read_text(encoding="utf-8")
        for p in pathlib.Path("tests").rglob("test_*.py")
        if not p.name.startswith("test_hidden")
    }
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", "HEAD", "tests"], capture_output=True, text=True).stdout
    before = {
        n: subprocess.run(["git", "show", f"HEAD:{n}"], capture_output=True, text=True).stdout
        for n in names.split()
        if n.endswith(".py") and pathlib.Path(n).name.startswith("test_")
    }
    new = count_tests(now) - count_tests(before)
    touching = [n for n, s in now.items() if args.module in s or f"import {short}" in s or f"{short} import" in s]
    print(f"{new} new test functions; files touching {args.module}: {touching}")
    if new < args.min:
        print(f"need at least {args.min} new test functions")
        return 1
    if not touching:
        print(f"no test file imports {args.module}")
        return 1
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", args.unchanged]).returncode != 0:
        print(f"{args.unchanged} was modified")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
