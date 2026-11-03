from datetime import date

from invoicing.fees import late_fee
from invoicing.models import Invoice, LineItem


def invoice():
    # due 2024-01-31, total 12 000 with tax
    return Invoice("A", "c", date(2024, 1, 1), "net30", None, [LineItem("x", 1, 10_000)])


def test_no_fee_before_or_on_due_date():
    assert late_fee(invoice(), date(2024, 1, 31)) == 0


def test_fee_per_started_period():
    assert late_fee(invoice(), date(2024, 2, 1)) == 180
    assert late_fee(invoice(), date(2024, 3, 1)) == 180
    assert late_fee(invoice(), date(2024, 3, 2)) == 360


def test_custom_rate():
    assert late_fee(invoice(), date(2024, 2, 1), monthly_rate_bp=200) == 240
