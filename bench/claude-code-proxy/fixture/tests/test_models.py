from datetime import date

from invoicing.discounts import Discount, discount_amount, parse_discount
from invoicing.models import Invoice, LineItem, Totals, compute_total, line_net
from invoicing.tax import rate_for, tax_for_line, tax_for_lines

import pytest


def test_line_net():
    assert line_net(LineItem("x", 3, 250)) == 750


def test_compute_total_without_discount():
    inv = Invoice("A", "c", date(2024, 1, 1), items=[LineItem("a", 2, 1000), LineItem("b", 1, 500, "reduced")])
    assert compute_total(inv) == Totals(subtotal=2500, discount=0, tax=425, total=2925)


def test_compute_total_percent_discount_reduces_tax():
    inv = Invoice("A", "c", date(2024, 1, 1), discount=Discount("percent", 1000), items=[LineItem("a", 1, 10_000)])
    assert compute_total(inv) == Totals(10_000, 1_000, 1_800, 10_800)


def test_compute_total_discount_spread_over_lines():
    inv = Invoice(
        "A",
        "c",
        date(2024, 1, 1),
        discount=Discount("fixed", 1000),
        items=[LineItem("a", 1, 6000), LineItem("b", 1, 4000, "zero")],
    )
    t = compute_total(inv)
    assert t.discount == 1000
    assert t.tax == 1080  # (6000 - 600) * 20%
    assert t.total == 10_080


def test_compute_total_empty():
    assert compute_total(Invoice("A", "c", date(2024, 1, 1))) == Totals(0, 0, 0, 0)


def test_tax_rates():
    assert rate_for("std") == 2000
    with pytest.raises(ValueError):
        rate_for("exotic")
    assert tax_for_line(333, "std") == 67
    assert tax_for_lines([(1000, "std"), (1000, "reduced"), (1000, "zero")]) == 250


def test_parse_discount():
    assert parse_discount("10%") == Discount("percent", 1000)
    assert parse_discount("12.5%") == Discount("percent", 1250)
    assert parse_discount("$5.00") == Discount("fixed", 500)
    with pytest.raises(ValueError):
        parse_discount("120%")


def test_discount_amount_is_capped():
    assert discount_amount(300, Discount("fixed", 500)) == 300
    assert discount_amount(300, None) == 0
    assert discount_amount(1999, Discount("percent", 1000)) == 200
