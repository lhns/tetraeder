#!/usr/bin/env python3
"""Keep docs/features.md honest.

Parses its tables (header starts with "| # |"). Per row:
- the feature cell is integers separated by commas;
- every backticked path in the evidence cell exists;
- every decimal or fraction in the ratio cell (0.943, 15/20) appears verbatim in a cited file under
  bench/results/ (a number in docs/ does not count; integers are not checked);
- every backticked node id in the state cell exists in docs/roadmap/nodes.yaml.
Exit 1 and list the problems otherwise.
"""

import pathlib
import re
import sys

NUM = re.compile(r"\d+\.\d+|\d+/\d+")
TICKS = re.compile(r"`([^`]+)`")
NODE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)+$")


def node_ids(root):
    text = (root / "docs" / "roadmap" / "nodes.yaml").read_text(encoding="utf-8")
    return set(re.findall(r"^- id: (\S+)\s*$", text, re.M))


def tables(text):
    """Yield (line number, cells) for every data row of a table whose header starts with '| # |'."""
    in_table = False
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("| # |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not s.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in s.strip("|").split(" | ")]
        if set(s) <= set("|- "):
            continue
        yield n, cells


def has_number(text, num):
    return re.search(r"(?<![\d.])" + re.escape(num) + r"(?!\d)", text) is not None


def check(root):
    root = pathlib.Path(root)
    page = root / "docs" / "features.md"
    errs = []
    if not page.is_file():
        return ["docs/features.md is missing"]
    nodes = node_ids(root)
    rows = 0
    for n, cells in tables(page.read_text(encoding="utf-8")):
        rows += 1
        where = f"docs/features.md:{n}"
        if len(cells) != 7:
            errs.append(f"{where}: {len(cells)} cells, want 7")
            continue
        feature, _, state, _, ratio, _, evidence = cells
        if not re.fullmatch(r"\d+(, \d+)*", feature):
            errs.append(f"{where}: feature cell {feature!r} is not a list of feature numbers")
        for nid in TICKS.findall(state):
            if NODE.match(nid) and nid not in nodes:
                errs.append(f"{where}: node {nid} is not in docs/roadmap/nodes.yaml")
        texts = []
        for p in TICKS.findall(evidence):
            f = root / p
            if not f.is_file():
                errs.append(f"{where}: evidence path {p} does not exist")
            elif p.startswith("bench/results/"):
                texts.append(f.read_text(encoding="utf-8", errors="replace"))
        for num in NUM.findall(ratio):
            if not any(has_number(t, num) for t in texts):
                errs.append(f"{where}: ratio {num} is not in the cited bench/results files")
    if rows == 0:
        errs.append("docs/features.md has no table rows")
    return errs


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    errs = check(root)
    for e in errs:
        print(e)
    print("features check: " + ("FAIL" if errs else "ok"))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
