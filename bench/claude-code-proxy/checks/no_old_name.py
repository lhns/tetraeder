"""Fails if a name still appears in any .py file of the repo (hidden tests excluded)."""

import argparse
import pathlib
import re
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    rx = re.compile(rf"\b{re.escape(args.name)}\b")
    hits = []
    for path in pathlib.Path(".").rglob("*.py"):
        if ".git" in path.parts or path.name.startswith("test_hidden"):
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if rx.search(line):
                hits.append(f"{path.as_posix()}:{i}")
    print(f"{len(hits)} remaining uses of {args.name}: {hits[:10]}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
