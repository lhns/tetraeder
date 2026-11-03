from datetime import date

import pytest

from invoicing.discounts import Discount
from invoicing.parsing import ParseError, parse_invoice_text

GOOD = """
# a comment
Invoice: INV-001
Customer: Acme Corp
Issued: 2024-01-15
Terms: net30
Discount: 10%
Item: Widget | 2 | $4.50 | std
Item: Gadget | 1 | 25 | reduced
"""


def test_parse_full_invoice():
    inv = parse_invoice_text(GOOD)
    assert inv.number == "INV-001"
    assert inv.customer == "Acme Corp"
    assert inv.issued == date(2024, 1, 15)
    assert inv.discount == Discount("percent", 1000)
    assert [(i.description, i.qty, i.unit_cents, i.tax_code) for i in inv.items] == [
        ("Widget", 2, 450, "std"),
        ("Gadget", 1, 2500, "reduced"),
    ]


def test_defaults():
    inv = parse_invoice_text("Invoice: A\nCustomer: B\nIssued: 01.02.2024\nItem: x | 1 | 1")
    assert inv.terms == "net30"
    assert inv.discount is None
    assert inv.items[0].tax_code == "std"


@pytest.mark.parametrize("field", ["Invoice", "Customer", "Issued"])
def test_missing_field(field):
    text = "\n".join(l for l in GOOD.splitlines() if not l.startswith(field + ":"))
    with pytest.raises(ParseError, match=f"missing field: {field}"):
        parse_invoice_text(text)


def test_no_items():
    with pytest.raises(ParseError, match="Item"):
        parse_invoice_text("Invoice: A\nCustomer: B\nIssued: 2024-01-01")


@pytest.mark.parametrize(
    "line",
    ["Item: only | 2", "Item: x | 0 | 5", "Item: x | two | 5", "Item: x | 1 | abc"],
)
def test_bad_items(line):
    with pytest.raises(ParseError):
        parse_invoice_text("Invoice: A\nCustomer: B\nIssued: 2024-01-01\n" + line)


def test_bad_header_line_reports_line_number():
    with pytest.raises(ParseError, match="line 2"):
        parse_invoice_text("Invoice: A\nnonsense")


def test_bad_terms_and_date():
    base = "Invoice: A\nCustomer: B\nItem: x | 1 | 1\n"
    with pytest.raises(ParseError, match="terms"):
        parse_invoice_text(base + "Issued: 2024-01-01\nTerms: forever")
    with pytest.raises(ParseError, match="date"):
        parse_invoice_text(base + "Issued: tomorrow")
