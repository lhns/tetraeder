"""JSON persistence for invoices."""

import json
from pathlib import Path

from invoicing.discounts import Discount
from invoicing.dates import parse_date
from invoicing.models import Invoice, LineItem


def invoice_to_dict(inv: Invoice) -> dict:
    return {
        "number": inv.number,
        "customer": inv.customer,
        "issued": inv.issued.isoformat(),
        "terms": inv.terms,
        "discount": None if inv.discount is None else {"kind": inv.discount.kind, "value": inv.discount.value},
        "items": [
            {"description": i.description, "qty": i.qty, "unit_cents": i.unit_cents, "tax_code": i.tax_code}
            for i in inv.items
        ],
    }


def invoice_from_dict(data: dict) -> Invoice:
    disc = data.get("discount")
    return Invoice(
        number=data["number"],
        customer=data["customer"],
        issued=parse_date(data["issued"]),
        terms=data.get("terms", "net30"),
        discount=None if disc is None else Discount(disc["kind"], disc["value"]),
        items=[
            LineItem(i["description"], i["qty"], i["unit_cents"], i.get("tax_code", "std"))
            for i in data.get("items", [])
        ],
    )


def save_invoices(path: Path, invoices: list[Invoice]) -> None:
    path.write_text(json.dumps([invoice_to_dict(i) for i in invoices], indent=2) + "\n", encoding="utf-8")


def load_invoices(path: Path) -> list[Invoice]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("invoice file must contain a JSON list")
    return [invoice_from_dict(d) for d in raw]
