"""NOTES.md must exist, name the key parser functions, and the code must be untouched."""

import pathlib
import subprocess
import sys

REQUIRED = ["parse_invoice_text", "split_headers", "parse_item", "ParseError"]
OPTIONAL = ["parse_number", "parse_customer", "parse_issued", "parse_terms", "parse_discount_field"]


def main() -> int:
    notes = pathlib.Path("NOTES.md")
    if not notes.is_file():
        print("NOTES.md is missing")
        return 1
    text = notes.read_text(encoding="utf-8")
    if len(text) < 400:
        print(f"NOTES.md is too short ({len(text)} chars)")
        return 1
    missing = [n for n in REQUIRED if n not in text]
    if missing:
        print(f"NOTES.md does not mention: {missing}")
        return 1
    opt = [n for n in OPTIONAL if n in text]
    if len(opt) < 3:
        print(f"NOTES.md mentions only {len(opt)} of the field parsers: {opt}")
        return 1
    diff = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "invoicing", "tests", "legacy"])
    if diff.returncode != 0:
        print("source or tests were modified; the task was to write notes only")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
