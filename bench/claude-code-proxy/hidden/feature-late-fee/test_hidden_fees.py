from datetime import date

from invoicing.discounts import Discount
from invoicing.fees import late_fee
from invoicing.models import Invoice, LineItem


def inv(terms="net30", discount=None):
    # issued 2024-01-01, net30: due 2024-01-31; total 10 000 net + 20% tax = 12 000
    return Invoice("A", "c", date(2024, 1, 1), terms, discount, [LineItem("x", 1, 10_000)])


def test_not_overdue_is_free():
    assert late_fee(inv(), date(2024, 1, 31)) == 0
    assert late_fee(inv(), date(2024, 1, 1)) == 0


def test_first_period_one_to_thirty_days():
    assert late_fee(inv(), date(2024, 2, 1)) == 180  # 1.5% of 12 000
    assert late_fee(inv(), date(2024, 3, 1)) == 180  # 30 days late


def test_second_and_third_period():
    assert late_fee(inv(), date(2024, 3, 2)) == 360  # 31 days late
    assert late_fee(inv(), date(2024, 4, 30)) == 540  # 90 days late
    assert late_fee(inv(), date(2024, 5, 1)) == 720  # 91 days late


def test_custom_rate_and_discount():
    assert late_fee(inv(), date(2024, 2, 5), monthly_rate_bp=200) == 240
    # 10% discount: 9 000 net + 1 800 tax = 10 800; 1.5% = 162
    assert late_fee(inv(discount=Discount("percent", 1000)), date(2024, 2, 5)) == 162


def test_eom_terms():
    i = inv("eom")  # due 2024-01-31
    assert late_fee(i, date(2024, 2, 1)) == 180
