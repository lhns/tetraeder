"""parsing.py must have a new helper used at least 3 times and at most 2 'missing field' messages left."""

import ast
import pathlib
import subprocess
import sys

ORIGINAL = {
    "split_headers", "parse_number", "parse_customer", "parse_issued", "parse_terms", "parse_item",
    "parse_discount_field", "parse_invoice_text",
}


def main() -> int:
    path = pathlib.Path("invoicing/parsing.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    msgs = [
        n for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and "missing field" in n.value
    ]
    if len(msgs) > 2:
        print(f"{len(msgs)} 'missing field' message literals left, expected at most 2")
        return 1
    helpers = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)} - ORIGINAL - {"__init__"}
    calls: dict[str, int] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in helpers:
                calls[name] = calls.get(name, 0) + 1
    print(f"new helpers and call counts: {calls}")
    if not any(c >= 3 for c in calls.values()):
        print("no new helper function is called at least 3 times")
        return 1
    # behaviour of the public error messages must be unchanged
    sys.path.insert(0, ".")
    from invoicing.parsing import ParseError, parse_invoice_text

    base = "Invoice: A\nCustomer: B\nIssued: 2024-01-01\nItem: x | 1 | 1\n"
    for field in ("Invoice", "Customer", "Issued"):
        text = "\n".join(l for l in base.splitlines() if not l.startswith(field))
        try:
            parse_invoice_text(text)
        except ParseError as e:
            if str(e) != f"missing field: {field}":
                print(f"message changed for {field}: {e}")
                return 1
        else:
            print(f"no error for missing {field}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
