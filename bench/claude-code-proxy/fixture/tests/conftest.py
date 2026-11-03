from datetime import date

import pytest

from invoicing.discounts import Discount
from invoicing.models import Invoice, LineItem
from invoicing.storage import save_invoices


@pytest.fixture
def sample_invoices():
    return [
        Invoice(
            "INV-002",
            "Zeta Ltd",
            date(2024, 2, 10),
            "eom",
            None,
            [LineItem("Consulting", 3, 12_000), LineItem("Books", 2, 1_999, "reduced")],
        ),
        Invoice(
            "INV-001",
            "Acme Corp",
            date(2024, 1, 15),
            "net30",
            Discount("percent", 1000),
            [LineItem("Widget", 2, 450), LineItem("Gadget", 1, 2_500)],
        ),
        Invoice("INV-003", "Acme Corp", date(2024, 3, 1), "receipt", Discount("fixed", 500), [LineItem("Fee", 1, 5_000, "zero")]),
    ]


@pytest.fixture
def invoice_file(tmp_path, sample_invoices):
    path = tmp_path / "invoices.json"
    save_invoices(path, sample_invoices)
    return path
