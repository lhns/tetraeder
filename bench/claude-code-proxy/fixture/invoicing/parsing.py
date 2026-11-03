"""Parser for the plain-text invoice format.

    Invoice: INV-001
    Customer: Acme Corp
    Issued: 2024-01-15
    Terms: net30
    Discount: 10%
    Item: Widget | 2 | $4.50 | std
"""

from datetime import date

from invoicing.dates import parse_date
from invoicing.discounts import Discount, parse_discount
from invoicing.models import Invoice, LineItem
from invoicing.money import parse_money

VALID_TERMS_PREFIXES = ("net", "eom", "receipt")


class ParseError(ValueError):
    pass


def split_headers(text: str) -> tuple[dict[str, str], list[str]]:
    """Split the text into header fields and raw item lines."""
    headers: dict[str, str] = {}
    items: list[str] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ParseError(f"line {lineno}: expected 'Key: value', got {line!r}")
        key = key.strip().title()
        if key == "Item":
            items.append(value.strip())
        else:
            headers[key] = value.strip()
    return headers, items


def parse_number(headers: dict[str, str]) -> str:
    value = headers.get("Invoice")
    if value is None or not value.strip():
        raise ParseError("missing field: Invoice")
    return value.strip()


def parse_customer(headers: dict[str, str]) -> str:
    value = headers.get("Customer")
    if value is None or not value.strip():
        raise ParseError("missing field: Customer")
    return value.strip()


def parse_issued(headers: dict[str, str]) -> date:
    value = headers.get("Issued")
    if value is None or not value.strip():
        raise ParseError("missing field: Issued")
    try:
        return parse_date(value)
    except ValueError as e:
        raise ParseError(str(e)) from e


def parse_terms(headers: dict[str, str]) -> str:
    value = headers.get("Terms")
    if value is None or not value.strip():
        return "net30"
    terms = value.strip().lower()
    if not terms.startswith(VALID_TERMS_PREFIXES):
        raise ParseError(f"unknown terms: {value!r}")
    return terms


def parse_item(raw: str) -> LineItem:
    parts = [p.strip() for p in raw.split("|")]
    if len(parts) not in (3, 4):
        raise ParseError(f"item needs 3 or 4 fields, got {len(parts)}: {raw!r}")
    description, qty_text, price_text = parts[:3]
    code = parts[3] if len(parts) == 4 else "std"
    if not qty_text.isdigit() or int(qty_text) == 0:
        raise ParseError(f"bad quantity: {qty_text!r}")
    try:
        unit = parse_money(price_text)
    except ValueError as e:
        raise ParseError(str(e)) from e
    return LineItem(description, int(qty_text), unit, code)


def parse_discount_field(headers: dict[str, str]) -> Discount | None:
    value = headers.get("Discount")
    if value is None or not value.strip():
        return None
    try:
        return parse_discount(value)
    except ValueError as e:
        raise ParseError(str(e)) from e


def parse_invoice_text(text: str) -> Invoice:
    """Parse one invoice; raises ParseError with a readable message."""
    headers, raw_items = split_headers(text)
    if not raw_items:
        raise ParseError("missing field: Item")
    return Invoice(
        number=parse_number(headers),
        customer=parse_customer(headers),
        issued=parse_issued(headers),
        terms=parse_terms(headers),
        discount=parse_discount_field(headers),
        items=[parse_item(r) for r in raw_items],
    )
